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
    """把一次预测追加写入 JSONL 台账（失败不影响返回）。

    去重：同 (kind, kline_time) 只记一条——哨兵 30 分钟轮询时，
    同一根 H1 K 线会被预测两次，重复记录会让对账样本虚胖。

    时区契约：kline_time 必须是 **MT5 服务器钟**（K线时间_服务器），
    与 mt5_client.get_klines 返回的时间同基准，accuracy_check 靠它
    做 join——展示用的 K线时间（北京时间）绝不能进台账。
    """
    try:
        rec = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "kind": kind,
            "signal": result.get("信号"),
            "price": result.get("最新收盘价"),
            "kline_time": result.get("K线时间_服务器") or result.get("K线时间"),
        }
        if kind in ("breakout", "breakout_m15"):
            rec["probability"] = result.get("扩张概率")
        else:
            rec["prob_short"] = result.get("看空概率")
            rec["prob_flat"] = result.get("观望概率")
            rec["prob_long"] = result.get("看多概率")
            rec["pred_class"] = result.get("预测类别")
        with _LEDGER_LOCK:
            PREDICTION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            key = (kind, str(rec.get("kline_time")))
            if key in _RECENT_KEYS:
                return  # 同一根 K 线已记录过
            _RECENT_KEYS.add(key)
            # 防御性裁剪：避免长跑进程内存缓慢增长
            if len(_RECENT_KEYS) > 10_000:
                _RECENT_KEYS.clear()
            with PREDICTION_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("failed to record prediction: %s", exc)


# 进程内去重（kind, kline_time）——哨兵 30 分钟轮询的防重复
_RECENT_KEYS: set[tuple[str, str]] = set()