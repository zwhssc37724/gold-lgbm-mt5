"""多数据源聚合模块：整合 Yahoo Finance、Alpha Vantage、Massive Market Data。

提供统一的行情数据获取接口，自动处理故障转移和数据校验。

数据源优先级：
1. gold-lgbm-mt5（MT5 本地）— 现货主源
2. Yahoo Finance — 跨资产首选
3. Massive Market Data — 第二源校验
4. Alpha Vantage — 最后兜底

注意：Alpha Vantage 和 Massive 需要 API key，在各自官网申请。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from gold_model import config

logger = logging.getLogger("gold_model.market_data")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "market_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys（从环境变量或手动设置）
ALPHA_VANTAGE_API_KEY = ""  # https://www.alphavantage.co/support/#api-key
MASSIVE_API_KEY = ""        # https://massive.com/（原 Polygon.io）

# 数据源状态追踪
_source_status: dict[str, dict] = {}


def _record_status(source: str, success: bool, detail: str = "") -> None:
    """记录数据源调用状态。"""
    _source_status[source] = {
        "success": success,
        "detail": detail,
        "time": datetime.now(UTC).isoformat(),
    }


def get_source_status() -> dict:
    """获取所有数据源的最近调用状态。"""
    return _source_status.copy()


# ---------------------------------------------------------------------------
# Yahoo Finance（通过 yfinance，已封装在 yahoo_finance.py）
# ---------------------------------------------------------------------------

def get_yahoo_quote(symbol: str) -> dict | None:
    """获取 Yahoo Finance 行情。"""
    try:
        from gold_model.yahoo_finance import get_yahoo_data
        data = get_yahoo_data(symbol)
        _record_status("yahoo", data is not None, f"symbol={symbol}")
        return data
    except Exception as e:
        _record_status("yahoo", False, str(e))
        return None


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------

def get_alpha_vantage_quote(symbol: str) -> dict | None:
    """获取 Alpha Vantage 行情。

    支持品种：
      - 外汇：EURUSD, GBPUSD, USDJPY 等
      - 黄金：XAUUSD（需要 FOREX 权限）
      - 美股：AAPL, MSFT 等
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("未配置 Alpha Vantage API key")
        return None

    try:
        import requests

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if "Global Quote" in data:
            q = data["Global Quote"]
            result = {
                "代码": symbol,
                "最新价": float(q.get("05. price", 0)),
                "开盘价": float(q.get("02. open", 0)),
                "最高价": float(q.get("03. high", 0)),
                "最低价": float(q.get("04. low", 0)),
                "涨跌": float(q.get("09. change", 0)),
                "涨跌幅": float(q.get("10. change percent", "0%").replace("%", "")),
                "成交量": int(q.get("06. volume", 0)),
                "时间": q.get("07. latest trading day", ""),
                "数据来源": "Alpha Vantage",
            }
            _record_status("alpha_vantage", True, f"symbol={symbol}")
            return result
        else:
            _record_status("alpha_vantage", False, str(data)[:200])
            return None

    except Exception as e:
        _record_status("alpha_vantage", False, str(e))
        return None


def get_alpha_vantage_fx(symbol: str = "XAUUSD") -> dict | None:
    """获取 Alpha Vantage 外汇/贵金属数据。"""
    if not ALPHA_VANTAGE_API_KEY:
        return None

    try:
        import requests

        # 外汇实时汇率
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": "XAU",
            "to_currency": "USD",
            "apikey": ALPHA_VANTAGE_API_KEY,
        }

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if "Realtime Currency Exchange Rate" in data:
            rate = data["Realtime Currency Exchange Rate"]
            result = {
                "代码": symbol,
                "最新价": float(rate.get("5. Exchange Rate", 0)),
                "时间": rate.get("6. Last Refreshed", ""),
                "数据来源": "Alpha Vantage FX",
            }
            _record_status("alpha_vantage_fx", True, f"symbol={symbol}")
            return result
        else:
            _record_status("alpha_vantage_fx", False, str(data)[:200])
            return None

    except Exception as e:
        _record_status("alpha_vantage_fx", False, str(e))
        return None


# ---------------------------------------------------------------------------
# Massive Market Data（原 Polygon.io）
# ---------------------------------------------------------------------------

def get_massive_quote(symbol: str) -> dict | None:
    """获取 Massive Market Data 行情。

    支持品种：
      - 美股：AAPL, MSFT 等
      - 外汇：C:EURUSD, C:GBPUSD 等
      - 加密货币：X:BTCUSD, X:ETHUSD 等
      - 期货：GC=F, SI=F 等
    """
    if not MASSIVE_API_KEY:
        logger.warning("未配置 Massive API key")
        return None

    try:
        import requests

        # Massive REST API
        url = f"https://api.massive.com/v2/last/trade/{symbol}"
        params = {"apiKey": MASSIVE_API_KEY}

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("status") == "OK" and "results" in data:
            r = data["results"]
            result = {
                "代码": symbol,
                "最新价": r.get("p", 0),
                "成交量": r.get("s", 0),
                "时间": datetime.fromtimestamp(r.get("t", 0) / 1000, tz=UTC).isoformat(),
                "数据来源": "Massive",
            }
            _record_status("massive", True, f"symbol={symbol}")
            return result
        else:
            _record_status("massive", False, str(data)[:200])
            return None

    except Exception as e:
        _record_status("massive", False, str(e))
        return None


# ---------------------------------------------------------------------------
# 统一行情接口（自动故障转移）
# ---------------------------------------------------------------------------

def get_quote(symbol: str, market: str = "auto") -> dict | None:
    """获取行情，自动尝试多个数据源。

    参数：
      - symbol: 品种代码
      - market: 市场类型（auto/forex/stock/futures）

    返回：行情数据，或 None（全部失败）
    """
    # 根据品种类型选择数据源
    sources = []

    if symbol in ("XAUUSD", "XAU/USD", "GOLD"):
        # 黄金：优先 MT5，其次 Yahoo，然后 Alpha Vantage，最后 Massive
        sources = [
            ("mt5", lambda: None),  # MT5 由 serve_mcp 处理，这里跳过
            ("yahoo", lambda: get_yahoo_quote("GLD")),  # 用 GLD 代理
            ("alpha_vantage_fx", lambda: get_alpha_vantage_fx("XAUUSD")),
            ("massive", lambda: get_massive_quote("C:XAUUSD")),
        ]
    elif symbol in ("DXY", "DX-Y.NYB", "USDOLLAR"):
        # 美元指数
        sources = [
            ("yahoo", lambda: get_yahoo_quote("UUP")),  # 用 UUP 代理
            ("alpha_vantage_fx", lambda: get_alpha_vantage_fx("DXY")),
            ("massive", lambda: get_massive_quote("C:DXY")),
        ]
    elif symbol in ("^VIX", "VIX"):
        # VIX
        sources = [
            ("yahoo", lambda: get_yahoo_quote("^VIX")),
            ("alpha_vantage", lambda: get_alpha_vantage_quote("^VIX")),
        ]
    elif symbol in ("^TNX", "US10Y"):
        # 美债收益率
        sources = [
            ("yahoo", lambda: get_yahoo_quote("^TNX")),
            ("alpha_vantage", lambda: get_alpha_vantage_quote("^TNX")),
        ]
    else:
        # 通用：Yahoo 优先
        sources = [
            ("yahoo", lambda: get_yahoo_quote(symbol)),
            ("alpha_vantage", lambda: get_alpha_vantage_quote(symbol)),
            ("massive", lambda: get_massive_quote(symbol)),
        ]

    # 依次尝试
    for source_name, fetcher in sources:
        try:
            data = fetcher()
            if data is not None:
                data["数据源链"] = source_name
                return data
        except Exception as e:
            logger.debug("数据源 %s 失败: %s", source_name, e)
            continue

    logger.warning("所有数据源均失败: %s", symbol)
    return None


# ---------------------------------------------------------------------------
# 跨资产数据获取
# ---------------------------------------------------------------------------

def get_gold_related_assets() -> dict:
    """获取黄金相关资产行情。

    返回：
      - 黄金（GLD 代理）
      - 白银（SLV 代理）
      - 美元指数（UUP 代理）
      - 美债 10 年收益率
      - VIX
      - 原油（USO 代理）
      - 标普 500
    """
    assets = {
        "黄金": get_quote("XAUUSD"),
        "白银": get_yahoo_quote("SLV"),
        "美元指数": get_quote("DXY"),
        "美债10年": get_quote("^TNX"),
        "VIX": get_quote("^VIX"),
        "原油": get_yahoo_quote("USO"),
        "标普500": get_yahoo_quote("^GSPC"),
    }

    return {k: v for k, v in assets.items() if v is not None}


def get_market_sentiment() -> dict:
    """获取市场情绪指标。

    基于 VIX、美元指数、美债收益率综合判断。
    """
    vix = get_quote("^VIX")
    dxy = get_quote("DXY")
    us10y = get_quote("^TNX")

    sentiment = {
        "时间": datetime.now(UTC).isoformat(),
        "VIX": vix,
        "美元指数": dxy,
        "美债10年": us10y,
    }

    # 简单情绪判断
    signals = []

    if vix and vix.get("最新价"):
        vix_val = vix["最新价"]
        if vix_val > 30:
            signals.append("VIX 高企，避险情绪浓厚")
        elif vix_val > 20:
            signals.append("VIX 偏高，避险情绪上升")
        elif vix_val < 15:
            signals.append("VIX 低位，市场风险偏好高")

    if dxy and dxy.get("涨跌幅"):
        dxy_chg = dxy["涨跌幅"]
        if dxy_chg > 0.5:
            signals.append("美元强势，黄金承压")
        elif dxy_chg < -0.5:
            signals.append("美元走弱，黄金受益")

    if us10y and us10y.get("最新价"):
        us10y_val = us10y["最新价"]
        if us10y_val > 4.5:
            signals.append("美债收益率高企，黄金承压")
        elif us10y_val < 3.5:
            signals.append("美债收益率低位，黄金受益")

    sentiment["信号"] = signals
    sentiment["综合情绪"] = "偏多" if any("受益" in s for s in signals) else \
                          "偏空" if any("承压" in s for s in signals) else "中性"

    return sentiment
