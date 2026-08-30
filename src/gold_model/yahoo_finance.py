"""Yahoo Finance 数据源：黄金相关宏观指标。

提供以下数据：
- 黄金期货 (GC=F)
- 黄金 ETF (GLD)
- 美元指数 (DX-Y.NYB)
- 美债收益率 (^TNX)
- VIX 恐慌指数 (^VIX)
- 白银期货 (SI=F)
- 原油 (CL=F)

注意：Yahoo Finance 免费 API 有频率限制，建议配合缓存使用。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from gold_model import config

logger = logging.getLogger("gold_model.yahoo_finance")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "yahoo_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 缓存有效期（分钟）
CACHE_TTL = 30


def _get_cache_path(symbol: str, period: str) -> Path:
    """获取缓存文件路径。"""
    safe_symbol = symbol.replace("=", "_").replace("^", "_").replace("-", "_")
    return CACHE_DIR / f"{safe_symbol}_{period}.json"


def _is_cache_valid(path: Path) -> bool:
    """检查缓存是否过期。"""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age = datetime.now(UTC) - mtime
    return age < timedelta(minutes=CACHE_TTL)


def _fetch_yahoo(symbol: str, period: str = "1d") -> dict | None:
    """从 Yahoo Finance 获取数据。"""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            logger.warning("Yahoo Finance 无数据: %s", symbol)
            return None

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest

        return {
            "代码": symbol,
            "最新价": round(float(latest["Close"]), 2),
            "开盘价": round(float(latest["Open"]), 2),
            "最高价": round(float(latest["High"]), 2),
            "最低价": round(float(latest["Low"]), 2),
            "成交量": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
            "涨跌": round(float(latest["Close"] - prev["Close"]), 2),
            "涨跌幅": round(float((latest["Close"] / prev["Close"] - 1) * 100), 2),
            "时间": hist.index[-1].isoformat(),
            "数据来源": "Yahoo Finance",
        }

    except Exception as e:
        logger.error("Yahoo Finance 获取失败 %s: %s", symbol, e)
        return None


def get_yahoo_data(symbol: str, period: str = "1d", use_cache: bool = True) -> dict | None:
    """获取 Yahoo Finance 数据（带缓存）。

    参数：
      - symbol: 品种代码，如 "GC=F"（黄金期货）
      - period: 时间范围，如 "1d", "5d", "1mo"
      - use_cache: 是否使用缓存
    """
    cache_path = _get_cache_path(symbol, period)

    if use_cache and _is_cache_valid(cache_path):
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("缓存读取失败 %s: %s", cache_path, e)

    data = _fetch_yahoo(symbol, period)

    if data and use_cache:
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return data


# ---------------------------------------------------------------------------
# 黄金相关品种快捷方式
# ---------------------------------------------------------------------------

def get_gold_futures() -> dict | None:
    """COMEX 黄金期货。GC=F 在 Yahoo 免费接口已无数据（possibly delisted），
    失败时回落到 GLD ETF 作为黄金价格代理。"""
    gc = get_yahoo_data("GC=F")
    if gc is not None:
        return gc
    gld = get_yahoo_data("GLD")
    if gld is not None:
        gld["代码"] = "GLD（黄金价格代理）"
    return gld


def get_gold_etf() -> dict | None:
    """SPDR 黄金 ETF (GLD)。Yahoo 免费版可用，作为黄金价格的代理。"""
    return get_yahoo_data("GLD")


def get_dxy() -> dict | None:
    """美元指数。

    2026-08-30：Yahoo 免费 API 已不再返回 DX-Y.NYB（possibly delisted，
    yfinance 每次都会打一条 ERROR 日志），直接用 UUP（美元看涨 ETF）作为代理。
    若日后 Yahoo 恢复该符号，可改回先试 DX-Y.NYB 再回落。
    """
    uup = get_yahoo_data("UUP")
    if uup is not None:
        uup["代码"] = "UUP（美元指数代理）"
    return uup


def get_uup() -> dict | None:
    """美元看涨 ETF (UUP)。Yahoo 免费版可用，作为美元指数的代理。"""
    return get_yahoo_data("UUP")


def get_us10y() -> dict | None:
    """美国 10 年期国债收益率 (^TNX)。"""
    return get_yahoo_data("^TNX")


def get_vix() -> dict | None:
    """VIX 恐慌指数 (^VIX)。"""
    return get_yahoo_data("^VIX")


def get_silver_etf() -> dict | None:
    """白银 ETF (SLV)。Yahoo 免费版可用，作为白银价格的代理。"""
    return get_yahoo_data("SLV")


def get_oil_etf() -> dict | None:
    """原油 ETF (USO)。Yahoo 免费版可用，作为原油价格的代理。"""
    return get_yahoo_data("USO")


def get_spx() -> dict | None:
    """标普 500 指数 (^GSPC)。"""
    return get_yahoo_data("^GSPC")


def get_nasdaq() -> dict | None:
    """纳斯达克指数 (^IXIC)。"""
    return get_yahoo_data("^IXIC")


# ---------------------------------------------------------------------------
# 宏观指标聚合
# ---------------------------------------------------------------------------

def get_macro_indicators() -> dict:
    """获取黄金相关的宏观指标汇总。

    返回：
      - 美元指数（UUP，作为 DXY 代理）
      - 美债 10 年收益率
      - VIX 恐慌指数
      - 黄金 ETF（GLD）
      - 白银 ETF（SLV）
      - 原油 ETF（USO）
      - 金银比
      - 金油比
      - 黄金情绪分析
    """
    # get_dxy() 内部已处理 DXY→UUP 回落（含"UUP（美元指数代理）"标注）
    dxy = get_dxy()

    us10y = get_us10y()
    vix = get_vix()
    gold = get_gold_etf()  # 用 GLD 替代 GC=F
    silver = get_silver_etf()  # 用 SLV 替代 SI=F
    oil = get_oil_etf()  # 用 USO 替代 CL=F

    result = {
        "时间": datetime.now(UTC).isoformat(),
        "美元指数": dxy,
        "美债10年收益率": us10y,
        "VIX恐慌指数": vix,
        "黄金ETF": gold,
        "白银ETF": silver,
        "原油ETF": oil,
    }

    # 计算比率
    if gold and silver:
        result["金银比"] = round(gold["最新价"] / silver["最新价"], 2)
    if gold and oil:
        result["金油比"] = round(gold["最新价"] / oil["最新价"], 2)

    return result


def get_gold_sentiment() -> dict:
    """黄金情绪指标。

    基于多个维度综合判断黄金市场情绪：
      - DXY 趋势（美元强则黄金弱）
      - VIX 水平（恐慌则黄金避险需求上升）
      - 金银比（历史高位可能意味着黄金超买）
    """
    macro = get_macro_indicators()
    sentiment = {
        "时间": macro["时间"],
        "综合情绪": "中性",
        "各维度": {},
    }

    # DXY 分析
    dxy = macro.get("美元指数")
    if dxy:
        dxy_change = dxy.get("涨跌幅", 0)
        if dxy_change > 0.5:
            sentiment["各维度"]["美元指数"] = "强势上涨，利空黄金"
        elif dxy_change > 0:
            sentiment["各维度"]["美元指数"] = "小幅上涨，轻微利空"
        elif dxy_change < -0.5:
            sentiment["各维度"]["美元指数"] = "大幅下跌，利好黄金"
        elif dxy_change < 0:
            sentiment["各维度"]["美元指数"] = "小幅下跌，轻微利好"
        else:
            sentiment["各维度"]["美元指数"] = "持平"

    # VIX 分析
    vix = macro.get("VIX恐慌指数")
    if vix:
        vix_level = vix.get("最新价", 0)
        if vix_level > 30:
            sentiment["各维度"]["VIX"] = "恐慌情绪高涨，避险需求利好黄金"
        elif vix_level > 20:
            sentiment["各维度"]["VIX"] = "恐慌情绪上升，利好黄金"
        elif vix_level < 15:
            sentiment["各维度"]["VIX"] = "市场平静，避险需求低迷"
        else:
            sentiment["各维度"]["VIX"] = "恐慌情绪正常"

    # 金银比分析
    gold_silver_ratio = macro.get("金银比")
    if gold_silver_ratio:
        if gold_silver_ratio > 90:
            sentiment["各维度"]["金银比"] = "金银比历史高位，黄金可能超买"
        elif gold_silver_ratio > 80:
            sentiment["各维度"]["金银比"] = "金银比偏高，黄金相对白银偏贵"
        elif gold_silver_ratio < 60:
            sentiment["各维度"]["金银比"] = "金银比历史低位，黄金可能超卖"
        elif gold_silver_ratio < 70:
            sentiment["各维度"]["金银比"] = "金银比偏低，黄金相对白银便宜"
        else:
            sentiment["各维度"]["金银比"] = "金银比正常范围"

    # 综合判断
    bullish_count = sum(1 for v in sentiment["各维度"].values() if "利好" in str(v))
    bearish_count = sum(1 for v in sentiment["各维度"].values() if "利空" in str(v))

    if bullish_count > bearish_count:
        sentiment["综合情绪"] = "偏多"
    elif bearish_count > bullish_count:
        sentiment["综合情绪"] = "偏空"
    else:
        sentiment["综合情绪"] = "中性"

    return sentiment


# ---------------------------------------------------------------------------
# 历史数据获取（用于回测）
# ---------------------------------------------------------------------------

def get_historical_data(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """获取历史数据（用于回测或长期分析）。

    参数：
      - symbol: 品种代码
      - period: 时间范围，如 "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return None

        return hist

    except Exception as e:
        logger.error("获取历史数据失败 %s: %s", symbol, e)
        return None
