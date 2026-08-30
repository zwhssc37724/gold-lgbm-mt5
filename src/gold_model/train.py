"""训练黄金交易模型（LightGBM + Optuna 调参）。

支持两类任务：
  - 突破/扩张（二分类）：下一根 K 线振幅是否突破近 100 根中位数
  - 方向三分类：未来 24 根 H1 收益按自适应阈值划分为看空/观望/看多

本版本要点（相对旧版）：
  1. 数据快照：训练数据从 MT5 拉取后经过滤（剔除伪装成 H1 的日线历史）
     存 parquet 快照，保证可复现；重训时可用 --use-snapshot 直接读。
  2. 防泄漏切分：所有 train/val 边界 purge 掉 horizon 根（标签前瞻窗），
     CV 各折之间再 embargo 1 天，杜绝标签跨界污染。
  3. Walk-forward 评估：不再依赖单一 20% 测试窗，而是滚动多窗口
     报告均值±标准差，给出可信的样本外估计。
  4. 朴素基线：每个任务同时报告基线分数（breakout=前值/波动率动量，
     direction3=多数类），模型必须显著超过基线才算有效。
  5. 宏观特征：合并 macro_features（DXY/10Y/VIX/GLD，防泄漏对齐）。

用法（在工程根目录）：
    uv run gold-train --target breakout
    uv run gold-train --target direction3
    uv run gold-train --target breakout --use-snapshot   # 复用数据快照
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from gold_model import config, mt5_client
from gold_model.features import build_features, build_labels, build_labels_3class
from gold_model import macro_features

logger = logging.getLogger("gold_model.train")


@dataclass
class TargetSpec:
    """每种预测任务的 LightGBM 训练配置。"""

    name: str
    is_multiclass: bool
    num_class: int
    objective: str
    metric: str
    optuna_direction: str
    model_path: Path
    horizon: int  # 标签前瞻根数（purge 用）


def _spec_for(target: str) -> TargetSpec:
    if target == "breakout":
        return TargetSpec(
            name="breakout",
            is_multiclass=False,
            num_class=1,
            objective="binary",
            metric="auc",
            optuna_direction="maximize",
            model_path=config.BREAKOUT_MODEL_PATH,
            horizon=config.BREAKOUT_HORIZON,
        )
    if target == "direction3":
        return TargetSpec(
            name="direction3",
            is_multiclass=True,
            num_class=3,
            objective="multiclass",
            metric="multi_logloss",
            optuna_direction="minimize",  # 越低越好
            model_path=config.DIRECTION3_MODEL_PATH,
            horizon=config.DIRECTION3_HORIZON,
        )
    if target == "direction_d1":
        return TargetSpec(
            name="direction_d1",
            is_multiclass=True,
            num_class=3,
            objective="multiclass",
            metric="multi_logloss",
            optuna_direction="minimize",
            model_path=config.DIRECTION_D1_MODEL_PATH,
            horizon=config.DIRECTION_D1_HORIZON,
        )
    raise ValueError(f"unknown target: {target}")


# ---------------------------------------------------------------------------
# 数据集构建（快照 + 清洗 + 宏观特征）
# ---------------------------------------------------------------------------

def load_raw_bars(bars: int, use_snapshot: bool = False) -> pd.DataFrame:
    """拉取 H1 K 线：优先复用快照；否则 MT5 拉取→密度过滤→存快照。"""
    if use_snapshot and config.DATA_SNAPSHOT.exists():
        df = pd.read_parquet(config.DATA_SNAPSHOT)
        if mt5_client.is_market_open()["状态"] != "weekend" and df["time"].iloc[-1] < (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
        ):
            logger.info("snapshot is stale (>24h), refetching")
        else:
            logger.info("using snapshot: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
            return df
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, bars=bars)
    if mt5_client.is_synthetic(df):
        raise RuntimeError("MT5 不可用且无快照：拒绝用合成数据训练。请先连接 MT5 终端。")
    if config.DENSE_HISTORY:
        df = mt5_client.filter_dense_history(df, timeframe=config.TIMEFRAME)
    df = df.reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATA_SNAPSHOT)
    logger.info("snapshot saved: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def load_raw_bars_d1(bars: int = 6000, use_snapshot: bool = False) -> pd.DataFrame:
    """拉取原生 D1 K 线（2011 年至今 ~3800 根）：优先复用快照。"""
    if use_snapshot and config.DATA_SNAPSHOT_D1.exists():
        df = pd.read_parquet(config.DATA_SNAPSHOT_D1)
        if mt5_client.is_market_open()["状态"] != "weekend" and df["time"].iloc[-1] < (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
        ):
            logger.info("D1 snapshot is stale (>3d), refetching")
        else:
            logger.info("using D1 snapshot: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
            return df
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe="D1", bars=bars)
    if mt5_client.is_synthetic(df):
        raise RuntimeError("MT5 不可用且无 D1 快照：拒绝用合成数据训练。请先连接 MT5 终端。")
    if config.DENSE_HISTORY:
        df = mt5_client.filter_dense_history(df, timeframe="D1")
    df = df.reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATA_SNAPSHOT_D1)
    logger.info("D1 snapshot saved: %d bars, %s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def build_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """特征 + 标签（含宏观特征），返回对齐后的 X, y。"""
    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)

    if target == "direction3":
        y = build_labels_3class(df, threshold=0)  # 自适应阈值
    elif target == "direction_d1":
        y = build_labels_3class(df, horizon=config.DIRECTION_D1_HORIZON, threshold=0)
    else:
        y = build_labels(df, horizon=config.BREAKOUT_HORIZON, target=target)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    # 末尾 horizon 根没有完整标签，直接剔除
    X = X.iloc[: -_spec_for(target).horizon]
    y = y.iloc[: -_spec_for(target).horizon]
    return X, y


# ---------------------------------------------------------------------------
# 防泄漏切分：purge + embargo
# ---------------------------------------------------------------------------

def purged_splits(
    n: int,
    horizon: int,
    n_splits: int = 5,
    embargo_bars: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """扩展窗时序切分，train 尾部 purge horizon 根，折间再 embargo。

    对每个折 (train=[0, train_end), val=[train_end, val_end))：
      - train 实际用到 [0, train_end - horizon)（purge：train 尾部标签看不到 val 开头）
      - 上一折 val 结束点与下一折 train 开始天然分离（扩展窗），无需额外处理；
        embargo_bars 额外拉开 train/val 距离，吸收波动率相关。
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    first_train = int(n * 0.6)
    step = (n - first_train) // n_splits
    for k in range(n_splits):
        train_end = first_train + k * step
        val_end = min(train_end + step, n)
        train_idx = np.arange(0, max(0, train_end - horizon - embargo_bars))
        val_idx = np.arange(train_end, val_end)
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# 朴素基线
# ---------------------------------------------------------------------------

def baseline_score(X_val: pd.DataFrame, y_val: pd.Series, spec: TargetSpec) -> float:
    """与 Optuna 目标同口径的朴素基线。"""
    if spec.name == "breakout":
        # 波动率动量基线：最近一根 K 线的相对振幅 > 近 100 根中位数 → 预测扩张
        rng = X_val.get("hl_range")
        base = rng.rolling(config.BREAKOUT_LOOKBACK).median() if rng is not None else None
        if base is None or base.notna().sum() < 10:
            return 0.5
        pred = (rng > base.shift(1)).astype(float)
        mask = base.shift(1).notna()
        return float(roc_auc_score(y_val[mask], pred[mask]))
    # direction3：多数类基线（准确率口径）转成可比指标由调用方处理
    major = int(y_val.mode().iloc[0])
    pred = pd.Series(major, index=y_val.index)
    return float(accuracy_score(y_val, pred))


def baseline_direction3(X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """direction3 基线：多数类准确率 + 永远看多的 AUC。"""
    major = int(y_val.mode().iloc[0])
    acc_major = float((y_val == major).mean())
    n_class = int(y_val.max()) + 1
    # 每类概率 = 训练分布（用验证集自身分布近似上界）
    dist = y_val.value_counts(normalize=True)
    proba = np.zeros((len(y_val), n_class))
    for c in range(n_class):
        proba[:, c] = dist.get(c, 0.0)
    try:
        auc = float(roc_auc_score(y_val, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {"多数类": {0: "看空", 1: "观望", 2: "看多"}.get(major, major), "准确率": acc_major, "宏平均AUC": auc}


# ---------------------------------------------------------------------------
# Optuna 目标
# ---------------------------------------------------------------------------

def make_objective(X: pd.DataFrame, y: pd.Series, splits, spec: TargetSpec):
    def _run(trial: optuna.Trial) -> float:
        params: dict = {
            "objective": spec.objective,
            "metric": spec.metric,
            "verbosity": -1,
            "seed": config.RANDOM_STATE,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        if spec.is_multiclass:
            params["num_class"] = spec.num_class

        scores: list[float] = []
        for train_idx, val_idx in splits:
            dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            scores.append(booster.best_score["valid_0"][spec.metric])
        return float(np.mean(scores))

    return _run


def _best_sign(value: float, spec: TargetSpec) -> str:
    if spec.metric == "auc":
        return f"CV AUC={value:.5f}"
    if spec.metric == "multi_logloss":
        return f"CV 多分类对数损失={value:.5f}"
    return f"CV {spec.metric}={value:.5f}"


def _evaluate(spec: TargetSpec, y_true: pd.Series, proba: np.ndarray) -> dict:
    """统一的测试指标。"""
    result: dict = {"样本数": len(y_true)}
    if spec.is_multiclass:
        pred = proba.argmax(axis=1)
        result["准确率"] = float(accuracy_score(y_true, pred))
        try:
            macro_auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError:
            macro_auc = float("nan")
        result["宏平均AUC（OvR）"] = macro_auc
        dist = y_true.value_counts(normalize=True).sort_index().to_dict()
        result["三类分布"] = {int(k): round(v, 4) for k, v in dist.items()}
    else:
        result["AUC"] = float(roc_auc_score(y_true, proba))
        result["正样本率"] = round(float(y_true.mean()), 4)
    return result


# ---------------------------------------------------------------------------
# Walk-forward 评估
# ---------------------------------------------------------------------------

def walk_forward(
    spec: TargetSpec,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_windows: int = 4,
    train_frac: float = 0.75,
) -> dict:
    """滚动训练→滚动预测，报告多窗口指标均值±标准差。"""
    n = len(X)
    scores: list[dict] = []
    win_size = max(int(n * (1 - train_frac) / n_windows), 200)
    start = n - n_windows * win_size
    for w in range(n_windows):
        tr_end = start + w * win_size
        te_end = min(tr_end + win_size, n)
        if tr_end <= spec.horizon + 24 or te_end - tr_end < 100:
            continue
        tr_idx = np.arange(0, tr_end - spec.horizon)  # purge
        te_idx = np.arange(tr_end, te_end)
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        model = lgb.train(
            params, dtrain, num_boost_round=600,
            callbacks=[lgb.log_evaluation(0)],
        )
        proba = model.predict(X.iloc[te_idx])
        if spec.is_multiclass and proba.ndim > 1:
            proba = proba
        m = _evaluate(spec, y.iloc[te_idx], np.asarray(proba))
        m["窗口"] = w
        m["起点"] = str(X.index[te_idx[0]])
        scores.append(m)
        logger.info("walk-forward window %d: %s", w, {k: v for k, v in m.items() if k not in ("起点",)})
    if not scores:
        return {"窗口数": 0}
    key = "AUC" if not spec.is_multiclass else "宏平均AUC（OvR）"
    vals = [s[key] for s in scores if key in s and s[key] == s[key]]
    summary: dict = {"窗口数": len(scores), "逐窗口": scores}
    if vals:
        summary[f"{key}均值"] = float(np.mean(vals))
        summary[f"{key}标准差"] = float(np.std(vals))
    return summary


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train(bars: int, trials: int, target: str, use_snapshot: bool = False) -> dict:
    spec = _spec_for(target)
    if target == "direction_d1":
        df = load_raw_bars_d1(bars=6000, use_snapshot=use_snapshot)
    else:
        df = load_raw_bars(bars, use_snapshot=use_snapshot)
    X, y = build_dataset(df, target=target)

    if target in ("direction3", "direction_d1"):
        dist = y.value_counts(normalize=True).sort_index().to_dict()
        logger.info(
            "dataset ready: target=%s, %d samples, %d features, 三类分布=%s",
            target, len(X), X.shape[1], {int(k): round(v, 4) for k, v in dist.items()},
        )
    else:
        logger.info(
            "dataset ready: target=%s, %d samples, %d features, 正样本率=%.4f",
            target, len(X), X.shape[1], y.mean(),
        )

    # 时序切分：末 20% 作为不打扰的测试集
    test_start = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]
    logger.info("train=%d  test=%d (time-ordered split)", len(X_tr), len(X_te))

    # 朴素基线（测试窗口）
    if target == "breakout":
        base = baseline_score(X_te, y_te, spec)
        logger.info("naive baseline (vol-momentum) test AUC=%.5f", base)
    else:
        base_info = baseline_direction3(X_te, y_te)
        logger.info("naive baseline (majority) test acc=%.5f", base_info["准确率"])
        base = base_info["准确率"]

    splits = purged_splits(len(X_tr), horizon=spec.horizon)
    study = optuna.create_study(
        direction=spec.optuna_direction,
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(make_objective(X_tr, y_tr, splits, spec), n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(
        {
            "objective": spec.objective,
            "metric": spec.metric,
            "verbosity": -1,
            "seed": config.RANDOM_STATE,
        }
    )
    if spec.is_multiclass:
        best["num_class"] = spec.num_class
    logger.info("best %s  params=%s", _best_sign(study.best_value, spec), best)

    # 在训练窗口上用最后一折做早停重训
    train_idx, val_idx = splits[-1]
    dtrain = lgb.Dataset(X_tr.iloc[train_idx], label=y_tr.iloc[train_idx])
    dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr.iloc[val_idx], reference=dtrain)
    model = lgb.train(
        best, dtrain, num_boost_round=3000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    proba = model.predict(X_te)
    test_metrics = _evaluate(spec, y_te, proba)
    logger.info("test metrics: %s", test_metrics)
    test_metrics["朴素基线"] = round(base, 5)
    test_metrics["超越基线"] = bool(
        (test_metrics.get("AUC", 0) > base) if not spec.is_multiclass
        else (test_metrics.get("准确率", 0) > base)
    )

    # Walk-forward 样本外评估
    wf_params = dict(best)
    wf_params["num_boost_round"] = 600
    wf = walk_forward(spec, X, y, wf_params)
    logger.info("walk-forward: %s", {k: v for k, v in wf.items() if k != "逐窗口"})

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": list(X.columns),
        "params": best,
        "target": target,
        "is_multiclass": spec.is_multiclass,
        "num_class": spec.num_class,
        "symbol": config.SYMBOL,
        "timeframe": "D1" if target == "direction_d1" else config.TIMEFRAME,
        "horizon": spec.horizon,
        "threshold": "adaptive(ATR24 median)" if target in ("direction3", "direction_d1") else None,
        "adaptive_threshold": target in ("direction3", "direction_d1"),
        "train_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "cv_score": float(study.best_value),
        "test_metrics": test_metrics,
        "walk_forward": {k: v for k, v in wf.items() if k != "逐窗口"},
        "data_range": [str(df["time"].iloc[0]), str(df["time"].iloc[-1])],
        "feature_importance": dict(zip(X.columns, model.feature_importance("gain"))),
    }
    joblib.dump(bundle, spec.model_path)
    config.FEATURES_PATH.write_text(json.dumps(list(X.columns), ensure_ascii=False, indent=2))

    # Top-15 特征重要性
    imp = sorted(zip(X.columns, model.feature_importance("gain")), key=lambda kv: -kv[1])[:15]
    for name, gain in imp:
        logger.info("  特征 %-16s 增益=%.1f", name, gain)

    # 保存训练报告
    report = {
        "训练时间": pd.Timestamp.now(tz="UTC").isoformat(),
        "目标": target,
        "数据范围": bundle["data_range"],
        "最佳CV": {spec.metric: float(study.best_value)},
        "测试集指标": test_metrics,
        "WalkForward": wf,
        "模型路径": str(spec.model_path),
        "特征数量": len(X.columns),
        "训练样本数": len(X_tr),
        "测试样本数": len(X_te),
        "Top15特征": [{"名称": name, "增益": round(gain, 2)} for name, gain in imp],
    }
    report_path = config.MODEL_DIR / f"train_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("训练报告已保存 -> %s", report_path)

    result = {
        "目标": target,
        "最佳CV": {spec.metric: float(study.best_value)},
        "测试集指标": test_metrics,
        "WalkForward均值": wf.get("AUC均值") or wf.get("宏平均AUC（OvR）均值"),
        "WalkForward标准差": wf.get("AUC标准差") or wf.get("宏平均AUC（OvR）标准差"),
        "模型路径": str(spec.model_path),
        "报告路径": str(report_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练黄金交易模型")
    parser.add_argument("--bars", type=int, default=config.BARS)
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument(
        "--target",
        choices=["breakout", "direction3", "direction_d1"],
        default="breakout",
        help="breakout：二分类，预测下一根 K 线波动扩张；direction3：三分类，看空/观望/看多；"
             "direction_d1：日线三分类，未来 5 个交易日方向",
    )
    parser.add_argument("--use-snapshot", action="store_true", help="复用 data/ 下的数据快照（不重新拉取）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    result = train(bars=args.bars, trials=args.trials, target=args.target, use_snapshot=args.use_snapshot)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"模型已保存 -> {_spec_for(args.target).model_path}")


if __name__ == "__main__":
    main()
