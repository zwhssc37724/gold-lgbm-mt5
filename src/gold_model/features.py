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

    # ====== 新增特征 ======

    # 支撑阻力距离特征
    for w in (10, 20, 50):
        recent_high = high.rolling(w).max()
        recent_low = low.rolling(w).min()
        out[f"dist_to_high_{w}"] = (recent_high - close) / close
        out[f"dist_to_low_{w}"] = (close - recent_low) / close
        out[f"range_position_{w}"] = (close - recent_low) / (recent_high - recent_low).replace(0, np.nan)

    # 斐波那契回撤位置（基于最近 50 根的高低点）
    fib_high = high.rolling(50).max()
    fib_low = low.rolling(50).min()
    fib_range = fib_high - fib_low
    fib_range = fib_range.replace(0, np.nan)
    out["fib_236"] = (close - (fib_high - 0.236 * fib_range)) / close
    out["fib_382"] = (close - (fib_high - 0.382 * fib_range)) / close
    out["fib_500"] = (close - (fib_high - 0.500 * fib_range)) / close
    out["fib_618"] = (close - (fib_high - 0.618 * fib_range)) / close

    # 趋势强度（ADX 简化版）
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = _atr(df, 14) * close  # 还原 ATR 为价格单位
    out["adx_plus"] = (plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr.replace(0, np.nan)).fillna(0.0)
    out["adx_minus"] = (minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr.replace(0, np.nan)).fillna(0.0)
    out["adx_diff"] = out["adx_plus"] - out["adx_minus"]

    # 价格动量加速度
    out["mom_accel_12"] = close.pct_change(12).diff()
    out["mom_accel_24"] = close.pct_change(24).diff()

    # 波动率比率（短期/长期）
    out["vol_ratio_12_168"] = (out["vol_12"] / out["vol_168"].replace(0, np.nan)).fillna(1.0)
    out["vol_ratio_24_72"] = (out["vol_24"] / out["vol_72"].replace(0, np.nan)).fillna(1.0)

    # K线实体动量
    out["body_mom_3"] = (out["body_ratio"].rolling(3).mean()).fillna(0.0)
    out["body_mom_6"] = (out["body_ratio"].rolling(6).mean()).fillna(0.0)

    # 上下影线比率（情绪指标）
    out["wick_ratio"] = (out["upper_wick"] / (out["lower_wick"] + 0.001)).fillna(1.0)

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
    """方向三分类标签：0=看空（跌超阈值）、1=观望（在阈值内）、2=看多（涨超阈值）。

    threshold 传 0（或 None）时启用自适应阈值：阈值 = 近 24 根 ATR(%) 中位数
    乘以 ADAPTIVE_THRESHOLD_ATR_MULT（config，默认 1.0）。金价波动放大时
    "观望带"同步放宽，标签分布不随波动率漂移。训练时应传 0 以逐样本自适应。
    """
    from gold_model import config  # 延迟导入

    horizon = horizon or config.DIRECTION3_HORIZON
    close = df["close"]
    fwd = np.log(close.shift(-horizon) / close)

    if not threshold:
        atr_pct = _atr(df, 14) / close
        thr_series = (
            atr_pct.rolling(24).median() * config.ADAPTIVE_THRESHOLD_ATR_MULT
        )
        # 逐样本自适应阈值；无效值回退到全局固定阈值
        thr_series = thr_series.where(
            thr_series.notna() & (thr_series > 0), config.DIRECTION3_THRESHOLD
        )
        labels = pd.Series(1, index=df.index, dtype=int)
        labels[fwd > thr_series] = 2
        labels[fwd < -thr_series] = 0
        return labels

    labels = pd.Series(1, index=df.index, dtype=int)  # 默认观望
    labels[fwd > threshold] = 2  # 看多
    labels[fwd < -threshold] = 0  # 看空
    return labels


DIRECTION3_NAMES_CN = {0: "看空", 1: "观望", 2: "看多"}
