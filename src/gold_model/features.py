"""Feature engineering for the gold trading model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def _bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ((close - ma) / (2 * sd)).fillna(0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the model feature matrix from OHLCV klines."""
    out = pd.DataFrame(index=df.index)
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    # Returns over multiple horizons
    for h in (1, 2, 3, 6, 12, 24, 48):
        out[f"ret_{h}"] = np.log(close / close.shift(h))

    # Candle geometry
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = ((close - open_) / rng).fillna(0.0)
    out["upper_wick"] = ((high - np.maximum(close, open_)) / rng).fillna(0.0)
    out["lower_wick"] = ((np.minimum(close, open_) - low) / rng).fillna(0.0)
    out["hl_range"] = np.log(high / low.replace(0, np.nan)).fillna(0.0)

    # Indicators
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_7"] = _rsi(close, 7)
    macd, macd_sig, macd_hist = _macd(close)
    out["macd"] = macd / close
    out["macd_signal"] = macd_sig / close
    out["macd_hist"] = macd_hist / close
    out["atr_14"] = _atr(df, 14) / close
    out["atr_7"] = _atr(df, 7) / close
    out["bb_pos_20"] = _bollinger_position(close, 20)

    # Moving average relations
    for p in (10, 20, 50, 100, 200):
        ma = close.rolling(p).mean()
        out[f"ma_bias_{p}"] = (close / ma - 1.0)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema_cross"] = ema20 / ema50 - 1.0

    # Volatility & momentum statistics
    logret = np.log(close / close.shift(1))
    for p in (12, 24, 72, 168):
        out[f"vol_{p}"] = logret.rolling(p).std()
        out[f"mom_{p}"] = close.pct_change(p)
    out["skew_24"] = logret.rolling(24).skew().fillna(0.0)
    out["kurt_24"] = logret.rolling(24).kurt().fillna(0.0)

    # Volume features
    if "tick_volume" in df.columns:
        vol = df["tick_volume"].astype(float)
        out["vol_ratio_24"] = (vol / vol.rolling(24).mean()).fillna(1.0)
        out["vol_chg"] = vol.pct_change().fillna(0.0).clip(-5, 5)

    # Autocorrelation of returns (regime fingerprint)
    out["autocorr_24"] = logret.rolling(24).apply(
        lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 else 0.0,
        raw=True,
    ).fillna(0.0)

    # Calendar features (hour of day / day of week)
    t = pd.to_datetime(df["time"])
    out["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 5)

    return out.replace([np.inf, -np.inf], 0.0)


def build_labels(df: pd.DataFrame, horizon: int = 1, target: str = "direction") -> pd.Series:
    """二分类标签。

    target="breakout":  下一根 K 线的相对振幅是否突破近 N 根中位数（波动扩张）。
    target="direction": 下一根 K 线收盘是否高于当前收盘。
    """
    close = df["close"]
    if target == "direction":
        fwd = np.log(close.shift(-horizon) / close)
        return (fwd > 0).astype(int)
    if target == "breakout":
        from gold_model import config  # 避免循环导入

        rng = (df["high"] - df["low"]) / close
        baseline = rng.rolling(config.BREAKOUT_LOOKBACK).median()
        return (rng.shift(-horizon) > baseline).astype(int)
    raise ValueError(f"unknown target: {target}")


def build_labels_3class(
    df: pd.DataFrame, horizon: int = 24, threshold: float = 0.003
) -> pd.Series:
    """方向三分类标签：0=看空（跌超阈值）、1=观望（在阈值内）、2=看多（涨超阈值）。"""
    from gold_model import config  # 延迟导入

    horizon = horizon or config.DIRECTION3_HORIZON
    threshold = threshold or config.DIRECTION3_THRESHOLD
    fwd = np.log(df["close"].shift(-horizon) / df["close"])
    labels = pd.Series(1, index=df.index, dtype=int)  # 默认观望
    labels[fwd > threshold] = 2  # 看多
    labels[fwd < -threshold] = 0  # 看空
    return labels


DIRECTION3_NAMES_CN = {0: "看空", 1: "观望", 2: "看多"}
