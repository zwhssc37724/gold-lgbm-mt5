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


def build_reference(X_train: pd.DataFrame, target: str) -> Path:
    """训练时存参考分位数（每特征 10 个分位点 + 缺失率）。"""
    ref = {}
    for col in X_train.columns:
        s = X_train[col].astype(float)
        ref[col] = {
            "quantiles": [round(float(q), 6) for q in np.quantile(s.dropna(), np.linspace(0, 1, PSI_BINS + 1))],
            "missing_rate": round(float(s.isna().mean()), 4),
        }
    path = reference_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    logger.info("drift reference saved -> %s (%d features)", path, len(ref))
    return path


def _psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    """PSI = Σ (a_i - e_i) * ln(a_i / e_i)，分箱边界来自参考分布。"""
    eps = 1e-6
    e_counts = np.histogram(expected, bins=bins)[0].astype(float)
    a_counts = np.histogram(actual, bins=bins)[0].astype(float)
    e = e_counts / max(e_counts.sum(), 1) + eps
    a = a_counts / max(a_counts.sum(), 1) + eps
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
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=timeframe, bars=bars)
    if mt5_client.is_synthetic(df):
        return {"错误": "MT5 数据不可用（合成数据），漂移检查需要真实行情。"}

    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)

    ref = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for col, spec in ref.items():
        if col not in X.columns:
            continue
        actual = X[col].astype(float).dropna().to_numpy()
        if len(actual) < 50:
            continue
        bins = np.array(spec["quantiles"])
        bins = np.unique(bins)  # 常数特征的分位边界会重复
        if len(bins) < 3:
            continue  # 常数特征无法分箱
        # 参考分布用同样的分箱边界（把参考分位本身当样本点重建直方图）
        expected = np.linspace(bins[0], bins[-1], 1000)
        psi = _psi(expected, actual, bins)
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
