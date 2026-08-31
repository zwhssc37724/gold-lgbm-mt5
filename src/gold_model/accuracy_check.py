"""预测对账：回看 prediction_ledger.jsonl，统计真实命中率与校准。

逻辑：
- breakout 记录：K线时间 t 的预测针对 t+1 根。命中 = (t+1 振幅 > 近100根中位数) 与
  (扩张概率≥0.5) 同真同假。
- direction3 记录：K线时间 t 的预测针对未来 24 根。命中 = 预测类别（按当根自适应阈值
  重算）与实际类别一致；另报告严格方向命中（看多且涨超阈值 / 看空且跌超阈值）。
- 校准：把概率分桶，比较预测概率与实际频率（reliability）。

用法：
    uv run gold-accuracy            # 默认近 7 天
    uv run gold-accuracy --days 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd

from gold_model import config, mt5_client
from gold_model.features import build_labels, build_labels_3class

logger = logging.getLogger("gold_model.accuracy_check")

LEDGER = config.DATA_DIR / "prediction_ledger.jsonl"


def load_ledger(days: int = 7) -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame()
    recs = []
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["ts"] = pd.to_datetime(df["ts"])
    df["kline_time"] = pd.to_datetime(df["kline_time"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df["ts"] >= cutoff].reset_index(drop=True)


def check_breakout(ledger: pd.DataFrame, klines: pd.DataFrame, kind: str = "breakout",
                   freq: str = "h") -> dict:
    """对账 breakout / breakout_m15 记录：用真实 K 线重算标签。"""
    recs = ledger[ledger["kind"] == kind].copy()
    if recs.empty:
        return {"记录数": 0}
    y = build_labels(klines, horizon=1, target="breakout")
    rng = (klines["high"] - klines["low"]) / klines["close"]
    base = rng.rolling(config.BREAKOUT_LOOKBACK).median()

    time_to_idx = pd.Series(np.arange(len(klines)), index=klines["time"].dt.floor(freq))
    hits, total, probs, actuals = [], 0, [], []
    for _, r in recs.iterrows():
        t = r["kline_time"]
        if pd.isna(t):
            continue
        idx = time_to_idx.get(t)
        if idx is None or idx + 1 >= len(klines) or pd.isna(base.iloc[idx + 1]):
            continue  # 该 K 线不在拉取范围或标签未形成
        actual = int(y.iloc[idx + 1])
        pred = int(r["probability"] >= 0.5)
        hits.append(actual == pred)
        probs.append(float(r["probability"]))
        actuals.append(actual)
        total += 1
    if total == 0:
        return {"记录数": len(recs), "可对账": 0, "说明": "K线数据不足或时间不匹配"}
    out = {
        "记录数": len(recs),
        "可对账": total,
        "命中率": round(float(np.mean(hits)), 4),
        "实际扩张占比": round(float(np.mean(actuals)), 4),
    }
    # 校准（概率分桶）
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    p, a = np.array(probs), np.array(actuals)
    cal = []
    for lo, hi in itertools.pairwise(bins):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() >= 3:
            cal.append({"概率区间": f"[{lo},{hi})", "样本": int(m.sum()),
                        "平均预测概率": round(float(p[m].mean()), 3),
                        "实际扩张频率": round(float(a[m].mean()), 3)})
    if cal:
        out["校准表"] = cal
    return out


def check_direction3(ledger: pd.DataFrame, klines: pd.DataFrame) -> dict:
    """对账 direction3 记录（自适应阈值重算实际类别）。"""
    recs = ledger[ledger["kind"] == "direction3"].copy()
    if recs.empty:
        return {"记录数": 0}
    y = build_labels_3class(klines, threshold=0)
    time_to_idx = pd.Series(np.arange(len(klines)), index=klines["time"].dt.floor("h"))
    hits, strict_hits, total = [], [], 0
    class_map = {"看空": 0, "观望": 1, "看多": 2}
    for _, r in recs.iterrows():
        t = r["kline_time"]
        if pd.isna(t):
            continue
        idx = time_to_idx.get(t)
        if idx is None or idx + config.DIRECTION3_HORIZON >= len(klines):
            continue
        actual = int(y.iloc[idx])
        pred = class_map.get(str(r.get("pred_class")), None)
        if pred is None:
            continue
        hits.append(actual == pred)
        # 严格方向命中：预测看多且实际看多（忽略观望）
        if pred in (0, 2) and actual == pred:
            strict_hits.append(True)
        elif pred in (0, 2):
            strict_hits.append(False)
        total += 1
    if total == 0:
        return {"记录数": len(recs), "可对账": 0, "说明": "24根K线窗口未走完或时间不匹配"}
    out = {
        "记录数": len(recs),
        "可对账": total,
        "命中率(含观望)": round(float(np.mean(hits)), 4),
    }
    if strict_hits:
        out["方向命中(剔除观望预测)"] = round(float(np.mean(strict_hits)), 4)
        out["方向预测数"] = len(strict_hits)
    return out


def check_direction_d1(ledger: pd.DataFrame, klines_d1: pd.DataFrame) -> dict:
    """对账 direction_d1 记录（D1，horizon=5 交易日）。"""
    recs = ledger[ledger["kind"] == "direction_d1"].copy()
    if recs.empty:
        return {"记录数": 0}
    y = build_labels_3class(klines_d1, horizon=config.DIRECTION_D1_HORIZON, threshold=0)
    time_to_idx = pd.Series(np.arange(len(klines_d1)), index=klines_d1["time"].dt.floor("D"))
    hits, total = [], 0
    class_map = {"看空": 0, "观望": 1, "看多": 2}
    for _, r in recs.iterrows():
        t = r["kline_time"]
        if pd.isna(t):
            continue
        idx = time_to_idx.get(t.floor("D") if hasattr(t, "floor") else t)
        if idx is None or idx + config.DIRECTION_D1_HORIZON >= len(klines_d1):
            continue  # 5 日窗口未走完
        actual = int(y.iloc[idx])
        pred = class_map.get(str(r.get("pred_class")), None)
        if pred is None:
            continue
        hits.append(actual == pred)
        total += 1
    if total == 0:
        return {"记录数": len(recs), "可对账": 0, "说明": "5交易日窗口未走完或时间不匹配"}
    return {
        "记录数": len(recs),
        "可对账": total,
        "命中率": round(float(np.mean(hits)), 4),
    }


def run(days: int = 7) -> dict:
    ledger = load_ledger(days)
    if ledger.empty:
        return {"说明": f"近 {days} 天无预测记录", "台账路径": str(LEDGER)}
    klines = mt5_client.get_klines(bars=2000)
    if mt5_client.is_synthetic(klines):
        return {"说明": "MT5 不可用，无法对账", "台账路径": str(LEDGER)}
    klines = mt5_client.filter_dense_history(klines)
    out = {
        "对账窗口": f"近 {days} 天",
        "breakout": check_breakout(ledger, klines),
        "direction3": check_direction3(ledger, klines),
    }
    # direction_d1 需要日线数据
    if (ledger["kind"] == "direction_d1").any():
        d1 = mt5_client.get_klines(timeframe="D1", bars=200)
        if not mt5_client.is_synthetic(d1):
            out["direction_d1"] = check_direction_d1(ledger, d1)
    # breakout_m15 需要 M15 数据
    if (ledger["kind"] == "breakout_m15").any():
        m15 = mt5_client.get_klines(timeframe="M15", bars=2000)
        if not mt5_client.is_synthetic(m15):
            out["breakout_m15"] = check_breakout(ledger, m15, kind="breakout_m15", freq="15min")
    out["台账路径"] = str(LEDGER)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="预测对账（命中率/校准）")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(days=args.days)
    import json as _json
    print(_json.dumps(report, indent=2, ensure_ascii=False))
    out = config.DATA_DIR / "reports" / f"accuracy_{pd.Timestamp.now():%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
