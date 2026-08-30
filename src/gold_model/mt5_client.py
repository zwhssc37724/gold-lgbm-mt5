"""MT5 data access: real-time quotes and klines, with a synthetic offline fallback.

Key guarantees:
- Thread safety: all MT5 calls go through one global lock (MetaTrader5 package is not
  thread-safe, and the MCP server may serve concurrent requests).
- Connection reuse: initialize once per process, never shutdown per call.
- Data provenance: every returned DataFrame carries ``df.attrs["source"]``
  ("mt5" / "synthetic") so downstream code (predictions!) can refuse synthetic data.
- Dense-history filter: brokers often backfill old H1 history with disguised daily
  bars (all at hour 00, 24-72h gaps). ``filter_dense_history`` drops that prefix.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

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

SOURCE_MT5 = "mt5"
SOURCE_SYNTH = "synthetic"

# ---------------------------------------------------------------------------
# Shared, locked MT5 connection
# ---------------------------------------------------------------------------

_MT5_LOCK = threading.RLock()
_MT5 = None  # module handle after first successful initialize


def _mt5():
    """Return the MetaTrader5 module with an initialized connection, or None.

    The connection is initialized lazily once per process and kept alive.
    Every caller must hold ``_MT5_LOCK``.
    """
    global _MT5
    try:
        import MetaTrader5 as mt5
    except Exception as exc:  # pragma: no cover - depends on terminal
        logger.warning("MetaTrader5 package unavailable: %s", exc)
        return None
    if not mt5.initialize():
        # Try one full re-init (recovers after terminal restart).
        try:
            mt5.shutdown()
        except Exception:
            pass
        if not mt5.initialize():
            logger.warning("MT5 initialize failed: %s", mt5.last_error())
            return None
    _MT5 = mt5
    return mt5


def mt5_available() -> bool:
    """Cheap health check under the lock."""
    with _MT5_LOCK:
        return _mt5() is not None


def _mark(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df.attrs["source"] = source
    return df


def is_synthetic(df: pd.DataFrame) -> bool:
    """True when the frame came from the synthetic fallback."""
    return df.attrs.get("source") == SOURCE_SYNTH


# ---------------------------------------------------------------------------
# Data quality: drop disguised-daily "H1" history
# ---------------------------------------------------------------------------

def filter_dense_history(df: pd.DataFrame, timeframe: str = config.TIMEFRAME) -> pd.DataFrame:
    """Drop the leading span of low-quality bars (daily bars posing as H1, etc.).

    Rule: count bars per calendar month. A month is "dense" when it holds at
    least 50% of the theoretical bar count for the timeframe (H1 ≈ 470+/mo;
    a real trading month never drops below half). The dense history starts at
    the first month of the final unbroken run of dense months. Disguised daily
    history (~21 bars/month for H1) fails this test.
    """
    if len(df) < 3 or df["time"].isna().all():
        return _mark(df, df.attrs.get("source", SOURCE_MT5))
    minutes = TIMEFRAMES.get(timeframe, ("H1", 60))[1]
    bars_per_day = 1440 // minutes
    min_month = max(2, int(0.5 * bars_per_day * 22))
    monthly = df.groupby(df["time"].dt.strftime("%Y-%m")).size()
    dense = monthly >= min_month
    if not dense.any():
        return _mark(df, df.attrs.get("source", SOURCE_MT5))
    # Final unbroken run of dense months.
    keep_from = None
    for ym, is_dense in reversed(list(dense.items())):
        if not is_dense:
            break
        keep_from = ym
    if keep_from is None:
        return _mark(df, df.attrs.get("source", SOURCE_MT5))
    mask = df["time"].dt.strftime("%Y-%m") >= keep_from
    dropped = int((~mask).sum())
    if dropped > 0:
        logger.info(
            "dropped %d low-quality leading bars (before %s) for %s",
            dropped,
            keep_from,
            timeframe,
        )
    return _mark(
        df[mask].reset_index(drop=True),
        df.attrs.get("source", SOURCE_MT5),
    )


# ---------------------------------------------------------------------------
# Market status
# ---------------------------------------------------------------------------

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


def is_market_open(symbol: str = config.SYMBOL) -> dict:
    """检测市场是否开市。

    返回字段：
      - 状态: open / closed / weekend / holiday / unknown
      - 当前时间: UTC
      - 下次开市: 预计下次开市时间（如果可计算）
      - 说明: 人类可读的状态描述
    """
    now_utc = datetime.now(UTC)
    weekday = now_utc.weekday()  # 0=Monday, 6=Sunday

    # 黄金外汇市场：周日 22:00 UTC 开市，周五 22:00 UTC 收市（冬令时）
    if weekday == 5:  # Saturday
        next_open = _next_sunday_22utc(now_utc)
        return {
            "状态": "weekend",
            "当前时间": now_utc.isoformat(),
            "下次开市": next_open.isoformat() if next_open else None,
            "说明": "周末休市，黄金市场周日 22:00 UTC 开市",
        }

    if weekday == 6 and now_utc.hour < 22:  # Sunday before open
        next_open = now_utc.replace(hour=22, minute=0, second=0, microsecond=0)
        return {
            "状态": "weekend",
            "当前时间": now_utc.isoformat(),
            "下次开市": next_open.isoformat() if next_open else None,
            "说明": "周末休市，今晚 22:00 UTC 开市",
        }

    # 工作日：尝试从 MT5 获取服务器时间来验证
    with _MT5_LOCK:
        mt5 = _mt5()
        if mt5 is not None:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    tick_time = pd.Timestamp(tick.time, unit="s", tz="UTC")
                    age_minutes = (pd.Timestamp.now(tz="UTC") - tick_time).total_seconds() / 60
                    if age_minutes > 5:
                        return {
                            "状态": "closed",
                            "当前时间": now_utc.isoformat(),
                            "数据时间": tick_time.isoformat(),
                            "数据延迟分钟": round(age_minutes, 1),
                            "说明": f"市场已关闭，最新数据来自 {round(age_minutes)} 分钟前",
                        }
                    return {
                        "状态": "open",
                        "当前时间": now_utc.isoformat(),
                        "数据时间": tick_time.isoformat(),
                        "说明": "市场开市，数据实时",
                    }
            except Exception as exc:
                logger.warning("MT5 market status check failed: %s", exc)

    return {
        "状态": "unknown",
        "当前时间": now_utc.isoformat(),
        "说明": "无法连接 MT5，市场状态未知",
    }


def _next_sunday_22utc(now: datetime) -> datetime | None:
    """计算下一个周日 22:00 UTC。"""
    days_ahead = 6 - now.weekday()  # days until Sunday
    if days_ahead <= 0:
        days_ahead += 7
    next_sunday = now + pd.Timedelta(days=days_ahead)
    return next_sunday.replace(hour=22, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Quotes & klines
# ---------------------------------------------------------------------------

def get_quote(symbol: str = config.SYMBOL) -> dict:
    """获取 ``symbol`` 的最新行情快照（返回中文键的字典）。"""
    with _MT5_LOCK:
        mt5 = _mt5()
        if mt5 is not None:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    mt5.symbol_select(symbol, True)
                    tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    tick_time = pd.Timestamp(tick.time, unit="s", tz="UTC")
                    age_seconds = (pd.Timestamp.now(tz="UTC") - tick_time).total_seconds()
                    data_fresh = age_seconds < 60
                    return {
                        "品种": symbol,
                        "买价": float(tick.bid),
                        "卖价": float(tick.ask),
                        "最新价": float(tick.last),
                        "时间": tick_time.isoformat(),
                        "数据来源": "MT5 实时" if data_fresh else "MT5 延迟",
                        "数据新鲜度": "实时" if data_fresh else f"延迟 {int(age_seconds)} 秒",
                        "市场状态": "开市" if data_fresh else "休市/延迟",
                    }
            except Exception as exc:
                logger.warning("MT5 quote failed (%s), using synthetic tick", exc)
    # Fallback: synthetic tick consistent with the synthetic series.
    px = _synthetic_last_price(symbol)
    return {
        "品种": symbol,
        "买价": round(px * 0.9999, 2),
        "卖价": round(px * 1.0001, 2),
        "最新价": round(px, 2),
        "时间": pd.Timestamp.now(tz="UTC").isoformat(),
        "数据来源": "合成数据",
        "数据新鲜度": "合成",
        "市场状态": "未知",
    }


def get_klines(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    bars: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV klines from MT5; fall back to a synthetic series.

    The returned frame carries ``attrs["source"]`` = "mt5" or "synthetic".
    Trading/prediction code MUST check this (see ``is_synthetic``).
    """
    bars = int(max(10, min(bars, config.BARS)))
    with _MT5_LOCK:
        mt5 = _mt5()
        if mt5 is not None:
            try:
                tf = _tf_to_mt5(timeframe)
                if tf is None:
                    raise ValueError(f"unsupported timeframe: {timeframe}")
                mt5.symbol_select(symbol, True)
                raw = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                if raw is not None and len(raw) > 0:
                    df = pd.DataFrame(raw)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    return _mark(
                        df[_KLINE_COLUMNS].sort_values("time").reset_index(drop=True),
                        SOURCE_MT5,
                    )
                raise RuntimeError("MT5 returned no rates")
            except Exception as exc:
                logger.warning("MT5 klines failed (%s), using synthetic data", exc)
    return _mark(synthetic_klines(symbol=symbol, timeframe=timeframe, bars=bars), SOURCE_SYNTH)


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

    Only a pipeline stand-in, never a market simulator.
    """
    key = f"{symbol}:{timeframe}:{bars}:{seed}"
    if key in _SYNTH_CACHE:
        df = _SYNTH_CACHE[key].copy()
        return _mark(df, SOURCE_SYNTH)

    rng = np.random.default_rng(seed)
    n = int(bars)
    minutes = TIMEFRAMES.get(timeframe, ("H1", 60))[1]
    end = pd.Timestamp.now(tz="UTC").floor("h")
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
    return _mark(df, SOURCE_SYNTH)


def _synthetic_last_price(symbol: str) -> float:
    df = synthetic_klines(symbol=symbol, bars=10)
    return float(df["close"].iloc[-1])
