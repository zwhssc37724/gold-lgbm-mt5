"""MT5 data access: real-time quotes and klines, with a synthetic offline fallback.

If the local MetaTrader 5 terminal is installed and running, we pull real
XAUUSD data via the official ``MetaTrader5`` package. When the terminal is
unavailable (not installed / not running / not logged in), we fall back to a
deterministic synthetic gold price series that embeds a mild trend/momentum
regime so the full training pipeline can be validated end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_model import config

logger = logging.getLogger(__name__)

TIMEFRAMES = {
    "M1": ("M1", 1),
    "M5": ("M5", 5),
    "M15": ("M15", 15),
    "M30": ("M30", 30),
    "H1": ("H1", 60),
    "H4": ("H4", 240),
    "D1": ("D1", 1440),
}

_KLINE_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


def _mt5():
    """Import MetaTrader5 lazily; return None when unavailable."""
    try:
        import MetaTrader5 as mt5

        return mt5
    except Exception as exc:  # pragma: no cover - depends on terminal
        logger.warning("MetaTrader5 package unavailable: %s", exc)
        return None


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    time: str  # ISO format, broker/server time


def _tf_to_mt5(timeframe: str):
    mt5 = _mt5()
    if mt5 is None:
        return None
    return getattr(mt5, f"TIMEFRAME_{timeframe}", None)


def get_quote(symbol: str = config.SYMBOL) -> dict:
    """Return the latest tick for ``symbol`` as a plain dict (JSON-friendly)."""
    mt5 = _mt5()
    if mt5 is not None:
        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                # Try to enable the symbol in Market Watch first.
                mt5.symbol_select(symbol, True)
                tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                return {
                    "symbol": symbol,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "last": float(tick.last),
                    "time": pd.Timestamp(tick.time, unit="s", tz="UTC").isoformat(),
                    "source": "mt5",
                }
        except Exception as exc:
            logger.warning("MT5 quote failed (%s), using synthetic tick", exc)
        finally:
            if mt5 is not None:
                try:
                    mt5.shutdown()
                except Exception:  # pragma: no cover
                    pass
    # Fallback: synthetic tick consistent with the synthetic series.
    px = _synthetic_last_price(symbol)
    return {
        "symbol": symbol,
        "bid": round(px * 0.9999, 2),
        "ask": round(px * 1.0001, 2),
        "last": round(px, 2),
        "time": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "synthetic",
    }


def get_klines(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    bars: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV klines from MT5; fall back to a synthetic series."""
    bars = int(max(10, min(bars, config.BARS)))
    mt5 = _mt5()
    if mt5 is not None:
        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
            tf = _tf_to_mt5(timeframe)
            if tf is None:
                raise ValueError(f"unsupported timeframe: {timeframe}")
            mt5.symbol_select(symbol, True)
            raw = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            if raw is not None and len(raw) > 0:
                df = pd.DataFrame(raw)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                return df[_KLINE_COLUMNS].sort_values("time").reset_index(drop=True)
            raise RuntimeError("MT5 returned no rates")
        except Exception as exc:
            logger.warning("MT5 klines failed (%s), using synthetic data", exc)
        finally:
            if mt5 is not None:
                try:
                    mt5.shutdown()
                except Exception:  # pragma: no cover
                    pass
    return synthetic_klines(symbol=symbol, timeframe=timeframe, bars=bars)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

_SYNTH_CACHE: dict[str, pd.DataFrame] = {}


def synthetic_klines(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    bars: int = 50_000,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Deterministic synthetic gold klines with trend/momentum regimes.

    The series is a regime-switching random walk with momentum clustering and
    mild mean reversion, resembling H1 gold. It is only a pipeline stand-in,
    not a market simulator.
    """
    key = f"{symbol}:{timeframe}:{bars}:{seed}"
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]

    rng = np.random.default_rng(seed)
    n = int(bars)
    minutes = TIMEFRAMES.get(timeframe, ("H1", 60))[1]
    end = pd.Timestamp.utcnow().floor("h")
    times = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")

    # Regime-switching drift: momentum persists within a regime.
    drift = np.zeros(n)
    regime = 0
    for i in range(1, n):
        if rng.random() < 0.01:  # ~1% chance of regime change per bar
            regime = rng.choice([-1, 0, 1], p=[0.35, 0.3, 0.35])
        drift[i] = 0.9 * drift[i - 1] + regime * 6e-4 + rng.normal(0, 1e-4)

    # Volatility clustering (GARCH-like).
    vol = np.full(n, 0.0016)
    for i in range(1, n):
        vol[i] = 0.05 * 0.0015 + 0.90 * vol[i - 1] + 0.05 * abs(rng.normal()) * 0.002

    rets = drift + rng.normal(0, 1, n) * vol
    close = 2650.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread_amp = close * vol * rng.uniform(0.3, 1.0, n)
    high = np.maximum(open_, close) + spread_amp * np.abs(rng.normal(0, 1, n))
    low = np.minimum(open_, close) - spread_amp * np.abs(rng.normal(0, 1, n))
    volume = rng.integers(800, 9000, n)

    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
            "spread": rng.integers(10, 40, n),
        }
    )
    _SYNTH_CACHE[key] = df
    return df


def _synthetic_last_price(symbol: str) -> float:
    df = synthetic_klines(symbol=symbol, bars=10)
    return float(df["close"].iloc[-1])
