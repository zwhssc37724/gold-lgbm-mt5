"""HTTP MCP server exposing the gold trading model.

Run from the project root:
    uv run gold-mcp

Endpoints (MCP tools, streamable-http at http://127.0.0.1:8000/mcp):
    get_quote(symbol)        -> latest MT5 tick (or synthetic fallback)
    get_klines(symbol, tf, bars) -> OHLCV klines
    predict(symbol, tf)      -> LightGBM direction forecast for the next bar
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from gold_model import config, mt5_client
from gold_model.features import build_features

logger = logging.getLogger("gold_model.mcp")

mcp = MCPServer("gold-trading-model")

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"model not found at {config.MODEL_PATH}; run `uv run gold-train` first"
            )
        _MODEL = joblib.load(config.MODEL_PATH)
    return _MODEL


@mcp.tool()
def get_quote(symbol: str = config.SYMBOL) -> dict:
    """Get the latest real-time quote (bid/ask/last) for a symbol from MT5."""
    return mt5_client.get_quote(symbol)


@mcp.tool()
def get_klines(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, bars: int = 500) -> list[dict]:
    """Get OHLCV kline data for a symbol from MT5.

    timeframe: M1/M5/M15/M30/H1/H4/D1; bars: 10..50000.
    """
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=bars)
    return json_records(df)


@mcp.tool()
def predict(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """Run the LightGBM model on the latest kline and return the forecast.

    The saved model's target defines the meaning:
      - "breakout": probability that the next bar's range expands beyond the
        rolling 100-bar median (volatility expansion / breakout setup).
      - "direction": probability that the next bar closes up.
    """
    bundle = _load_model()
    target = bundle.get("target", "direction")
    needed: int = 300  # rolling features need history
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=needed)
    X = build_features(df)
    feats = [c for c in bundle["features"] if c in X.columns]
    row = X[feats].iloc[-1:].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    proba = float(bundle["model"].predict(row)[0])
    if target == "breakout":
        signal = "EXPECT_EXPANSION" if proba >= 0.6 else "EXPECT_COMPRESSION" if proba <= 0.4 else "NEUTRAL"
        prob_key = "probability_range_expansion"
    else:
        signal = "LONG" if proba >= 0.55 else "SHORT" if proba <= 0.45 else "NEUTRAL"
        prob_key = "probability_up"
    quote = mt5_client.get_quote(symbol)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "target": target,
        prob_key: round(proba, 4),
        "signal": signal,
        "last_close": float(df["close"].iloc[-1]),
        "quote": quote,
        "time": str(df["time"].iloc[-1]),
        "model": "lightgbm-optuna",
    }


def json_records(df: pd.DataFrame) -> list[dict]:
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
    logger.info("starting MCP server at http://%s:%d%s", config.MCP_HOST, config.MCP_PORT, config.MCP_PATH)
    mcp.run(
        transport="streamable-http",
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        streamable_http_path=config.MCP_PATH,
    )


if __name__ == "__main__":
    main()
