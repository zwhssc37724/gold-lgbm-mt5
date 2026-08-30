"""宏观驱动特征：把 DXY / 10Y 名义收益率 / VIX / GLD 对齐到 XAUUSD H1 K 线。

防泄漏原则（关键）：
- 宏观序列是日线级（当日收盘值要到当天结束后才知道），所以对齐时必须用
  **前一日及更早** 的值（shift 1 天），绝不能把"当天"的值喂给当天的 H1 K 线。
- 对齐使用 ``pd.merge_asof(direction="backward")``：每根 H1 K 线只能看到
  时间戳 <= 其时间的最近一个宏观数据点——配合 shift(1) 双保险。

特征内容（对每个宏观序列 s）：
  - macro_{name}_ret5 / ret20 : 5/20 日对数收益
  - macro_{name}_z20          : 相对 20 日均值的 z-score
  - macro_{name}_ma20_bias    : 收盘相对 20 日均线的偏离

数据源 yfinance 拉取失败时返回空 DataFrame（训练管道会退化为纯价格特征，
而不是崩溃），落盘缓存在 data/macro_cache/ 下，有效期 12 小时。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gold_model import config

logger = logging.getLogger(__name__)

MACRO_SYMBOLS: dict[str, str] = {
    "dxy": "DX-Y.NYB",   # 美元指数
    "us10y": "^TNX",     # 美债 10 年名义收益率（%）
    "vix": "^VIX",       # VIX 恐慌指数
    "gld": "GLD",        # SPDR 黄金 ETF（价格代理，机构资金流）
}

CACHE_DIR = config.DATA_DIR / "macro_cache"
CACHE_TTL_HOURS = 12.0


# 宏观序列拉取起点：D1 训练数据从 2011-11 起，宏观序列必须覆盖更早（含 20 日预热窗）
MACRO_START = "2011-01-01"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('^', 'idx_').replace('.', '_')}.parquet"


def fetch_macro_series(
    symbol: str,
    period: str = "max",
    start: str = MACRO_START,
    cache_buster: str = "",
) -> pd.DataFrame | None:
    """拉取日线序列（Close + 时间索引），带 parquet 缓存（12h）。

    关键修复（2026-08-30）：旧版默认 period="5y"，D1 训练（2011 起）的前 10 年
    宏观列全是零填充的假数据。现在默认从 MACRO_START 拉全历史；缓存按
    (symbol, start, cache_buster) 键隔离，旧短缓存自动失效重拉。

    拉取失败时返回 None（调用方零填充，训练管道退化为纯价格特征）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{symbol}|{start}|{cache_buster}"
    cp = CACHE_DIR / f"{_cache_path(symbol).stem}_{abs(hash(cache_key)) & 0xFFFFFF:06x}.parquet"
    if cp.exists():
        age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cp.stat().st_mtime, unit="s", tz="UTC")).total_seconds() / 3600
        if age_h < CACHE_TTL_HOURS:
            try:
                return pd.read_parquet(cp)
            except Exception as exc:
                logger.warning("macro cache read failed for %s: %s", symbol, exc)
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(start=start, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        df = hist[["Close"]].copy()
        df.columns = ["close"]
        df.index = pd.to_datetime(df.index).tz_convert("UTC") if df.index.tz is not None else pd.to_datetime(df.index).tz_localize("UTC")
        df.index.name = "time"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # 完整性检查：拉到的起点不得晚于请求起点太多（防止 API 静默降级回短历史）
        first = df.index.min()
        if first > pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=30):
            logger.warning("macro series %s starts at %s, later than requested %s", symbol, first, start)
        df.to_parquet(cp)
        return df
    except Exception as exc:
        logger.warning("macro fetch failed for %s: %s", symbol, exc)
        return None


def _series_features(close: pd.Series, name: str) -> pd.DataFrame:
    """单个宏观序列的日线特征。"""
    out = pd.DataFrame(index=close.index)
    logc = np.log(close)
    out[f"macro_{name}_ret5"] = logc.diff(5)
    out[f"macro_{name}_ret20"] = logc.diff(20)
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out[f"macro_{name}_z20"] = ((close - ma20) / sd20.replace(0, np.nan)).fillna(0.0)
    out[f"macro_{name}_ma20_bias"] = (close / ma20 - 1.0).fillna(0.0)
    return out.replace([np.inf, -np.inf], 0.0)


def build_macro_features(kline_index: pd.Series) -> pd.DataFrame:
    """构建对齐到 XAUUSD H1 K 线时间戳的宏观特征矩阵。

    参数 kline_index: H1 K 线的 time 列（tz-aware UTC）。
    返回与 kline_index 等长的 DataFrame（拉取失败序列给全 0 列，不中断训练）。
    """
    kline_index = pd.to_datetime(pd.Series(kline_index))
    if kline_index.dt.tz is None:
        kline_index = kline_index.dt.tz_localize("UTC")
    else:
        kline_index = kline_index.dt.tz_convert("UTC")

    result = pd.DataFrame(index=np.arange(len(kline_index)))
    aligned_times = kline_index.reset_index(drop=True)

    for name, symbol in MACRO_SYMBOLS.items():
        cols = [f"macro_{name}_ret5", f"macro_{name}_ret20", f"macro_{name}_z20", f"macro_{name}_ma20_bias"]
        series = fetch_macro_series(symbol)
        if series is None or series.empty:
            logger.warning("macro series %s unavailable, zero-filled", name)
            for c in cols:
                result[c] = 0.0
            continue

        feats = _series_features(series["close"], name)
        # 防泄漏：日线收盘值 shift(1) —— 当天的 H1 K 线只能用昨天的宏观值
        feats = feats.shift(1)
        feats = feats.reset_index()
        feats.columns = ["time"] + cols
        left = aligned_times.to_frame("time").reset_index(drop=True)
        # 统一时间精度（merge_asof 要求 key dtype 完全一致）
        left["time"] = left["time"].astype("datetime64[ns, UTC]")
        feats["time"] = feats["time"].astype("datetime64[ns, UTC]")
        merged = pd.merge_asof(
            left,
            feats.dropna(subset=cols),
            on="time",
            direction="backward",
        )
        for c in cols:
            result[c] = merged[c].fillna(0.0).to_numpy()
    return result


def macro_coverage(kline_index: pd.Series) -> dict:
    """诊断：各宏观序列在给定 H1 时间戳上的真实覆盖率（非填充比例）。"""
    cov = {}
    for name, symbol in MACRO_SYMBOLS.items():
        series = fetch_macro_series(symbol)
        if series is None or series.empty:
            cov[name] = 0.0
            continue
        t = pd.to_datetime(pd.Series(kline_index))
        if t.dt.tz is None:
            t = t.dt.tz_localize("UTC")
        else:
            t = t.dt_tz_convert("UTC") if hasattr(t, "dt_tz_convert") else t.dt.tz_convert("UTC")
        # 简化覆盖率：时间戳 >= 序列首日期+1天 且 <= 序列末日期 的比例
        lo = series.index[0] + pd.Timedelta(days=1)
        hi = series.index[-1]
        cov[name] = float(((t >= lo) & (t <= hi)).mean())
    return cov
