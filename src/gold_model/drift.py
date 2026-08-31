"""特征漂移监控（PSI，Population Stability Index）。

动机：模型部署后特征分布相对训练快照的偏移没人看，模型"悄悄失效"不会报警。
每次训练时把训练特征的分位数参考存盘（models/drift_reference_<target>.json），
之后随时对比近期真实数据的特征分布。

用法：
    uv run gold-drift                     # 默认：direction3（H1 近 500 根 vs 参考）
    uv run gold-drift --target direction_d1 --bars 200
MCP: check_drift()

判读（惯例阈值）：
    PSI < 0.10   稳定
    0.10 ~ 0.25  轻度漂移（关注）
    > 0.25       显著漂移（建议重训）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gold_model import config, macro_features, mt5_client
from gold_model.features import build_features

logger = logging.getLogger(__name__)

PSI_BINS = 10
PSI_STABLE = 0.10
PSI_DRIFT = 0.25


def reference_path(target: str) -> Path:
    return config.MODEL_DIR / f"drift_reference_{target}.json"


# 周期性日历特征：相邻窗口相位平移是几何必然，不构成分布漂移，监控排除
CYCLIC_FEATURES = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def build_reference(X_train: pd.DataFrame, target: str, tail: int = 500) -> Path:
    """训练时存参考分布：分位边界 + 参考自身落箱比例 + 缺失率。

    关键设计（2026-08-31 修复）：
    1. 参考取**训练尾段**而非全史——市场非平稳，全史参考会让任何近期窗口
       都报"显著漂移"（实测训练数据自比全史仍有 48/81 超阈），报警器常响。
       漂移监控语义是「当前市场是否偏离模型最后见过的市场」。
    2. 落箱比例必须存**真实直方图**而非假设每箱 1/n_bins——宏观特征是日线级，
       24 根 H1 共享同值，大量并列值使分位分箱无法等分（并列日全部落入同箱），
       "每箱 10%"的假设对并列值特征恒不成立（实测 PSI=1.2 的假漂移来源）。
    """
    X_tail = X_train.tail(tail)
    ref = {}
    for col in X_tail.columns:
        if col in CYCLIC_FEATURES:
            continue
        s = X_tail[col].astype(float).dropna()
        if len(s) < 50:
            continue
        quantiles = np.quantile(s, np.linspace(0, 1, PSI_BINS + 1))
        bins = np.unique(quantiles)
        if len(bins) < 3:
            continue  # 常数特征无法分箱
        counts = np.histogram(s, bins=bins)[0]
        ref[col] = {
            # 边界必须存原始浮点（repr 完整精度）：round(x,10) 会把边界推到
            # 并列值另一侧，参考落箱分布随之改变（实测 42/46/46... → 0/46/69...）
            "bins": [float(b) for b in bins],
            "bin_props": [float(c) / len(s) for c in counts],
            "missing_rate": float(X_tail[col].isna().mean()),
        }
    ref["_meta"] = {
        "tail_bars": tail,
        "n_train_rows": len(X_train),
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path = reference_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    logger.info("drift reference saved -> %s (%d features, tail=%d)", path, len(ref) - 1, tail)
    return path


def _psi_from_ref(spec: dict, actual: np.ndarray) -> float:
    """PSI = Σ (a_i - e_i) * ln(a_i / e_i)，e 来自参考的真实落箱比例。"""
    eps = 1e-6
    bins = np.array(spec["bins"])
    e = np.array(spec["bin_props"], dtype=float)
    n_bins = len(bins) - 1
    if n_bins < 2 or len(e) != n_bins:
        return 0.0
    a_counts = np.histogram(actual, bins=bins)[0].astype(float)
    a = a_counts / max(a_counts.sum(), 1)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift(target: str = "direction3", bars: int = 500) -> dict:
    """近期真实数据特征分布 vs 训练参考。返回每特征 PSI + 汇总。

    target: direction3（H1）/ direction_d1（D1）/ breakout（H1）
    bars: 近期 K 线根数
    """
    path = reference_path(target)
    if not path.exists():
        return {
            "错误": f"参考文件不存在：{path}。请先重训 {target}（训练时会自动保存参考）。",
            "建议": f"uv run gold-train --target {target} --use-snapshot",
        }

    timeframe = "D1" if target == "direction_d1" else config.TIMEFRAME
    # 滚动特征预热：ma_bias_200/vol_168 需要 200+ 根；D1 一次拉多了也就 4000 根
    warmup = 300 if timeframe != "D1" else 250
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=timeframe, bars=bars + warmup)
    if mt5_client.is_synthetic(df):
        return {"错误": "MT5 数据不可用（合成数据），漂移检查需要真实行情。"}
    df = mt5_client.filter_dense_history(df, timeframe=timeframe)

    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
    X = X.tail(bars)  # 只取目标窗口（预热行只为让滚动特征有效）

    ref = json.loads(path.read_text(encoding="utf-8"))
    meta = ref.pop("_meta", {})
    rows = []
    for col, spec in ref.items():
        if col not in X.columns or col in CYCLIC_FEATURES:
            continue
        actual = X[col].astype(float).dropna().to_numpy()
        if len(actual) < 50:
            continue
        psi = _psi_from_ref(spec, actual)
        rows.append({"特征": col, "PSI": round(psi, 4)})

    if not rows:
        return {"错误": "无可比对特征（特征列不匹配？）"}

    rows.sort(key=lambda r: -r["PSI"])
    worst = rows[:10]
    n_drift = sum(1 for r in rows if r["PSI"] > PSI_DRIFT)
    n_watch = sum(1 for r in rows if PSI_STABLE < r["PSI"] <= PSI_DRIFT)
    if n_drift > 0:
        verdict = f"显著漂移：{n_drift} 个特征 PSI>{PSI_DRIFT}，建议重训"
    elif n_watch > 0:
        verdict = f"轻度漂移：{n_watch} 个特征 PSI>{PSI_STABLE}，持续关注"
    else:
        verdict = "稳定"

    return {
        "目标": target,
        "近期数据": f"{len(df)} 根 {timeframe}（{df['time'].iloc[0]} ~ {df['time'].iloc[-1]}）",
        "参考窗口": f"训练尾段 {meta.get('tail_bars', '?')} 根（建于 {str(meta.get('built_at', '?'))[:10]}）",
        "结论": verdict,
        "PSI阈值": {"稳定": PSI_STABLE, "漂移": PSI_DRIFT},
        "漂移特征数": {"显著": n_drift, "轻度": n_watch, "总特征": len(rows)},
        "Top10漂移": worst,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="特征漂移检查（PSI）")
    parser.add_argument("--target", default="direction3", choices=["breakout", "direction3", "direction_d1"])
    parser.add_argument("--bars", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = check_drift(target=args.target, bars=args.bars)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
