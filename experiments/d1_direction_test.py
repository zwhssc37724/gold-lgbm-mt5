"""D1（日线）方向三分类实验：最后一条未验证的路。

背景（2026-08-30）：
- H1 direction3 全量 argmax 宏 AUC 0.539 不超基线，但置信度门控
  |p_long-p_short|>0.25 的多头腿命中率 72%（基线 39%，z=7.4）。
- D1 有 2011-11 至今 3,829 根真实日线（已验证与 H1 聚合完全一致），
  宏观特征（DXY/US10Y/VIX/GLD）本身就是日线级，在 D1 尺度咬合更好。

实验设计（防泄漏，与主管道同标准）：
1. 数据：MT5 原生 D1（拒绝合成）；快照 data/xauusd_d1_snapshot.parquet。
2. 特征：81 维（65 价格 + 16 宏观，build_features 周期无关 +
   build_macro_features 对齐到 D1 时间戳，日线值 shift(1) 天防泄漏）。
3. 标签：未来 5 根 D1 对数收益，自适应阈值 = ATR%(14) 近 24 根中位数
   （与 H1 direction3 同构；全样本分布 观望53%/看多26%/看空21%）。
4. 评估：walk-forward 5 窗口（滚动训练→滚动预测，train 尾部 purge 5 根），
   每窗口报告宏 AUC vs 基线（多数类准确率 + 类别分布概率的宏 AUC）。
5. 门控分层：|p_long-p_short| 分层命中率（复用 H1 的验证方法）。

运行：.venv/Scripts/python.exe experiments/d1_direction_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gold_model import config, mt5_client  # noqa: E402
from gold_model.features import _atr, build_features, build_labels_3class  # noqa: E402
from gold_model.macro_features import build_macro_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("d1_experiment")

D1_SNAPSHOT = config.DATA_DIR / "xauusd_d1_snapshot.parquet"
D1_HORIZON = 5  # 未来 5 个交易日
N_WINDOWS = 5

# 固定超参（不跑 Optuna：样本 ~3800，先看信号存不存在；有效再正式调参）
PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "seed": config.RANDOM_STATE,
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
}


def load_d1(use_snapshot: bool = True) -> pd.DataFrame:
    """拉取 MT5 原生 D1（拒绝合成），存快照保证可复现。"""
    if use_snapshot and D1_SNAPSHOT.exists():
        df = pd.read_parquet(D1_SNAPSHOT)
        logger.info("using D1 snapshot: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
        return df
    import MetaTrader5 as mt5

    with mt5_client._MT5_LOCK:
        if not mt5.initialize():
            raise RuntimeError("MT5 不可用：D1 实验需要真实数据")
        try:
            raw = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_D1, 0, 6000)
        finally:
            mt5.shutdown()
    if raw is None or len(raw) == 0:
        raise RuntimeError("MT5 未返回 D1 数据")
    df = pd.DataFrame(raw)[["time", "open", "high", "low", "close", "tick_volume", "spread"]].copy()
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    # 密度过滤（D1：正常年份 ~258 根/年，伪装数据会出现大量缺口）
    df = mt5_client.filter_dense_history(df, timeframe="D1")
    df.to_parquet(D1_SNAPSHOT)
    logger.info("D1 snapshot saved: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def build_d1_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """特征 + 自适应三分类标签。返回 X, y, fwd（前瞻收益，供门控分层）。"""
    X_price = build_features(df)
    X_macro = build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)

    y = build_labels_3class(df, horizon=D1_HORIZON, threshold=0)  # 自适应阈值
    fwd = np.log(df["close"].shift(-D1_HORIZON) / df["close"])

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    fwd = fwd[mask].reset_index(drop=True)
    # 末尾 horizon 根无完整标签，剔除
    X = X.iloc[:-D1_HORIZON]
    y = y.iloc[:-D1_HORIZON]
    fwd = fwd.iloc[:-D1_HORIZON]
    return X, y, fwd


def baseline_macro_auc(y_val: pd.Series) -> float:
    """基线：用验证集自身类别分布作为每类概率的宏 AUC（结构性上界）。"""
    n_class = 3
    dist = y_val.value_counts(normalize=True)
    proba = np.zeros((len(y_val), n_class))
    for c in range(n_class):
        proba[:, c] = dist.get(c, 0.0)
    try:
        return float(roc_auc_score(y_val, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def walk_forward_d1(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """扩展窗 walk-forward：5 窗口，train 尾部 purge horizon 根。"""
    n = len(X)
    win = n // (N_WINDOWS + 2)  # 训练起点约 2013 年初
    results = []
    for w in range(N_WINDOWS):
        tr_end = n - (N_WINDOWS - w) * win
        te_end = min(tr_end + win, n)
        tr_idx = np.arange(0, tr_end - D1_HORIZON)  # purge
        te_idx = np.arange(tr_end, te_end)
        if len(tr_idx) < 500 or len(te_idx) < 100:
            continue
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        model = lgb.train(PARAMS, dtrain, num_boost_round=400)
        proba = np.asarray(model.predict(X.iloc[te_idx]))
        y_te = y.iloc[te_idx].reset_index(drop=True)
        pred = proba.argmax(axis=1)
        try:
            auc = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        results.append({
            "窗口": w,
            "train_end": str(X.index[tr_idx[-1]]),
            "test_end": str(X.index[te_idx[-1]]),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "宏AUC": round(auc, 4),
            "基线宏AUC": round(baseline_macro_auc(y_te), 4),
            "准确率": round(float((pred == y_te.to_numpy()).mean()), 4),
            "多数类准确率": round(float((y_te == y_te.mode().iloc[0]).mean()), 4),
        })
        logger.info("WF window %d: AUC=%.4f base=%.4f acc=%.4f major=%.4f (%s ~ %s)",
                    w, results[-1]["宏AUC"], results[-1]["基线宏AUC"], results[-1]["准确率"],
                    results[-1]["多数类准确率"], results[-1]["train_end"], results[-1]["test_end"])
    return results


def gating_analysis(X: pd.DataFrame, y: pd.Series, fwd: pd.Series) -> dict:
    """置信度门控分层：收集 WF 样本外概率，按 |p_long-p_short| 分层看方向命中率。"""
    n = len(X)
    win = n // (N_WINDOWS + 2)
    all_proba, all_y, all_fwd = [], [], []
    for w in range(N_WINDOWS):
        tr_end = n - (N_WINDOWS - w) * win
        te_end = min(tr_end + win, n)
        tr_idx = np.arange(0, tr_end - D1_HORIZON)
        te_idx = np.arange(tr_end, te_end)
        if len(tr_idx) < 500 or len(te_idx) < 100:
            continue
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        model = lgb.train(PARAMS, dtrain, num_boost_round=400)
        proba = np.asarray(model.predict(X.iloc[te_idx]))
        all_proba.append(proba)
        all_y.append(y.iloc[te_idx].to_numpy())
        all_fwd.append(fwd.iloc[te_idx].to_numpy())
    proba = np.vstack(all_proba)
    y_true = np.concatenate(all_y)
    fwd_true = np.concatenate(all_fwd)
    conf = proba[:, 2] - proba[:, 0]

    out = {"样本数": int(len(y_true))}
    # 分层命中率：方向命中 = (真实非观望) 且 (符号一致)
    sig = fwd_true != 0  # 非观望
    for lo, hi in [(0.0, 0.15), (0.15, 0.25), (0.25, 1.0)]:
        m = (conf > lo) & (conf <= hi)
        if m.sum() < 10:
            out[f"conf({lo},{hi}]"] = {"n": int(m.sum()), "命中率": None}
            continue
        hit = float((np.sign(fwd_true[m]) == 1).mean())
        base = float(sig.mean())
        out[f"conf({lo},{hi}]"] = {
            "n": int(m.sum()),
            "多头命中率(收益>0)": round(hit, 4),
            "基线(全样本上涨占比)": round(base, 4),
        }
    # 高置信多头 / 高置信空头
    hi_long = conf > 0.25
    hi_short = conf < -0.25
    if hi_long.sum() >= 10:
        out["高置信多头>0.25"] = {
            "n": int(hi_long.sum()),
            "多头命中率": round(float((fwd_true[hi_long] > 0).mean()), 4),
            "平均前瞻收益%": round(float(fwd_true[hi_long].mean() * 100), 3),
        }
    if hi_short.sum() >= 10:
        out["高置信空头<-0.25"] = {
            "n": int(hi_short.sum()),
            "空头命中率": round(float((fwd_true[hi_short] < 0).mean()), 4),
            "平均前瞻收益%": round(float(fwd_true[hi_short].mean() * 100), 3),
        }
    return out


def main() -> None:
    df = load_d1()
    X, y, fwd = build_d1_dataset(df)
    dist = y.value_counts(normalize=True).sort_index().round(4).to_dict()
    logger.info("dataset: %d samples, %d features, 三类分布=%s", len(X), X.shape[1], dist)

    wf = walk_forward_d1(X, y)
    auc_mean = float(np.nanmean([w["宏AUC"] for w in wf]))
    base_mean = float(np.nanmean([w["基线宏AUC"] for w in wf]))
    logger.info("WF 宏AUC 均值=%.4f  基线均值=%.4f  差=%.4f", auc_mean, base_mean, auc_mean - base_mean)

    gating = gating_analysis(X, y, fwd)

    report = {
        "实验": "D1 方向三分类（最后一条路）",
        "时间": pd.Timestamp.now(tz="UTC").isoformat(),
        "数据": {"根数": len(df), "范围": [str(df["time"].iloc[0]), str(df["time"].iloc[-1])]},
        "标签": {"horizon": D1_HORIZON, "阈值": "自适应 ATR%(14) 近24根中位数", "分布": dist},
        "WalkForward": wf,
        "WF均值": {"宏AUC": round(auc_mean, 4), "基线宏AUC": round(base_mean, 4), "差": round(auc_mean - base_mean, 4)},
        "门控分层": gating,
    }
    out = ROOT / "data" / "reports" / "d1_direction_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n报告已保存 -> {out}")


if __name__ == "__main__":
    main()
