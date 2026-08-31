"""预测台账：记录每次预测，供 accuracy_check 对账。

原属于 serve_mcp.py，重构后独立为纯工具模块——MCP 服务端由外部
gold-trading MCP 提供，本项目仅保留训练/评估管线。
"""

from __future__ import annotations

import json
import logging
import threading

import pandas as pd

from gold_model import config

logger = logging.getLogger("gold_model.ledger")

PREDICTION_LEDGER = config.DATA_DIR / "prediction_ledger.jsonl"
_LEDGER_LOCK = threading.Lock()


def record_prediction(kind: str, result: dict) -> None:
    """把一次预测追加写入 JSONL 台账（失败不影响返回）。"""
    try:
        rec = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "kind": kind,
            "signal": result.get("信号"),
            "price": result.get("最新收盘价"),
            "kline_time": result.get("K线时间"),
        }
        if kind == "breakout":
            rec["probability"] = result.get("扩张概率")
        else:
            rec["prob_short"] = result.get("看空概率")
            rec["prob_flat"] = result.get("观望概率")
            rec["prob_long"] = result.get("看多概率")
            rec["pred_class"] = result.get("预测类别")
        with _LEDGER_LOCK:
            PREDICTION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with PREDICTION_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("failed to record prediction: %s", exc)