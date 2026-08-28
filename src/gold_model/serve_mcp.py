"""HTTP MCP 服务，对外提供黄金交易模型预测能力。

启动方式（在工程根目录）：
    uv run gold-mcp

服务地址（streamable-http）：http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from gold_model import config, mt5_client
from gold_model.features import DIRECTION3_NAMES_CN, build_features

logger = logging.getLogger("gold_model.mcp")

mcp = MCPServer("gold-trading-model")

# 输出标签中英文映射
BREAKOUT_SIGNAL_CN = {
    "EXPECT_EXPANSION": "预期扩张",
    "EXPECT_COMPRESSION": "预期收敛",
    "NEUTRAL": "中性",
}
DIRECTION3_SIGNAL_CN = {
    0: "看空（开空仓）",
    1: "观望（不操作）",
    2: "看多（开多仓）",
}


@lru_cache(maxsize=4)
def _load_model(path_key: str):
    """惰性加载模型，结果缓存。path_key: 'breakout' 或 'direction3'。"""
    path = config.BREAKOUT_MODEL_PATH if path_key == "breakout" else config.DIRECTION3_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"模型未找到：{path}。请先运行 `uv run gold-train --target {path_key}`"
        )
    return joblib.load(path)


def _build_input_row(symbol: str, timeframe: str, bundle: dict) -> pd.DataFrame:
    needed = 300  # 滚动特征需要的历史长度
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=needed)
    X = build_features(df)
    feats = [c for c in bundle["features"] if c in X.columns]
    row = X[feats].iloc[-1:].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return row, df


@mcp.tool()
def get_quote(symbol: str = config.SYMBOL) -> dict:
    """获取某品种的 MT5 实时报价（买价/卖价/最新价）。"""
    return mt5_client.get_quote(symbol)


@mcp.tool()
def get_klines(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, bars: int = 500) -> list[dict]:
    """获取某品种的 MT5 K 线数据（OHLCV）。

    timeframe: M1/M5/M15/M30/H1/H4/D1；bars: 10..50000。
    """
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=bars)
    return _json_records(df)


@mcp.tool()
def predict(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """波动扩张预测（二分类）：下一根 K 线振幅是否突破近 100 根中位数。

    返回字段：
      - 目标：固定为 "breakout"（波动扩张）
      - 扩张概率：模型输出 0~1
      - 信号：预期扩张 / 预期收敛 / 中性
      - 最新价、报价、时间、模型
    """
    bundle = _load_model("breakout")
    row, df = _build_input_row(symbol, timeframe, bundle)
    proba = float(bundle["model"].predict(row)[0])
    raw_signal = "EXPECT_EXPANSION" if proba >= 0.6 else "EXPECT_COMPRESSION" if proba <= 0.4 else "NEUTRAL"
    quote = mt5_client.get_quote(symbol)
    return {
        "品种": symbol,
        "周期": timeframe,
        "目标": "波动扩张（breakout）",
        "扩张概率": round(proba, 4),
        "信号": BREAKOUT_SIGNAL_CN[raw_signal],
        "最新收盘价": round(float(df["close"].iloc[-1]), 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（二分类）",
    }


@mcp.tool()
def predict_direction_3class(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """方向三分类预测：未来 24 根 H1 K 线（约 1 个交易日）的方向 — 看空 / 观望 / 看多。

    标签定义（按未来 24 根对数收益率）：
      - 0 看空：  收益率 < -0.3%
      - 1 观望：  -0.3% ≤ 收益率 ≤ +0.3%
      - 2 看多：  收益率 > +0.3%

    返回字段：
      - 看空概率、观望概率、看多概率（三类概率分布）
      - 信号：概率最大的类别（带仓位建议）
      - 预测类别、看多减看空置信度、最新收盘价、实时报价、K线时间、模型
    """
    bundle = _load_model("direction3")
    row, df = _build_input_row(symbol, timeframe, bundle)
    proba_vec = bundle["model"].predict(row)[0]  # shape (3,)
    proba_vec = np.asarray(proba_vec, dtype=float)
    pred_class = int(np.argmax(proba_vec))
    quote = mt5_client.get_quote(symbol)
    confidence_long_short = float(proba_vec[2] - proba_vec[0])  # 看多减看空
    return {
        "品种": symbol,
        "周期": timeframe,
        "目标": "方向三分类（看空/观望/看多）",
        "看空概率": round(float(proba_vec[0]), 4),
        "观望概率": round(float(proba_vec[1]), 4),
        "看多概率": round(float(proba_vec[2]), 4),
        "信号": DIRECTION3_SIGNAL_CN[pred_class],
        "预测类别": DIRECTION3_NAMES_CN[pred_class],
        "看多减看空置信度": round(confidence_long_short, 4),
        "最新收盘价": round(float(df["close"].iloc[-1]), 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（三分类）",
        "预测周期": f"未来 {bundle.get('horizon', 24)} 根 K 线（{bundle.get('horizon', 24) * 60} 分钟）",
        "分类阈值": f"±{bundle.get('threshold', 0.003) * 100:.2f}%",
    }


def _json_records(df: pd.DataFrame) -> list[dict]:
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return v

    return [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("MCP 服务已启动，地址 http://%s:%d%s", config.MCP_HOST, config.MCP_PORT, config.MCP_PATH)
    mcp.run(
        transport="streamable-http",
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        streamable_http_path=config.MCP_PATH,
    )


if __name__ == "__main__":
    main()
