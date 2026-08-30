"""FRED（美联储经济数据）数据源。

提供美债实际收益率（TIPS）、通胀预期等关键宏观数据。
FRED API 免费，但需要 API key（https://fred.stlouisfed.org/docs/api/api_key.html）。

关键指标：
- DFII10: 10 年期美债实际收益率（TIPS）— 黄金的"定价锚"
- T10YIE: 10 年期盈亏平衡通胀率（通胀预期）
- DGS10: 10 年期美债名义收益率
- DFF: 联邦基金利率
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gold_model import config

logger = logging.getLogger("gold_model.fred")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "fred_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FRED API Key（从 https://fred.stlouisfed.org/docs/api/api_key.html 免费申请）
FRED_API_KEY = "79204a148b7810723a8ba5fd21fdaf33"  # 在这里填入你的 API key

# FRED 指标映射
FRED_SERIES = {
    "DFII10": {"名称": "10年期美债实际收益率（TIPS）", "单位": "%", "说明": "黄金的定价锚——实际收益率涨则黄金跌"},
    "T10YIE": {"名称": "10年期盈亏平衡通胀率", "单位": "%", "说明": "市场预期的未来 10 年通胀率"},
    "DGS10": {"名称": "10年期美债名义收益率", "单位": "%", "说明": "名义利率"},
    "DGS2": {"名称": "2年期美债收益率", "单位": "%", "说明": "短期利率，反映加息预期"},
    "DFF": {"名称": "联邦基金利率", "单位": "%", "说明": "美联储基准利率"},
    "DTWEXBGS": {"名称": "美元指数（广义）", "单位": "", "说明": "美联储编制的广义美元指数"},
    "VIXCLS": {"名称": "VIX 恐慌指数", "单位": "", "说明": "市场恐慌情绪"},
}


def _get_cache_path(series_id: str) -> Path:
    """获取缓存文件路径。"""
    return CACHE_DIR / f"{series_id}.json"


def _fetch_fred(series_id: str, days: int = 30) -> pd.DataFrame | None:
    """从 FRED API 获取数据。

    参数：
      - series_id: FRED 指标代码，如 "DFII10"
      - days: 获取最近几天的数据
    """
    if not FRED_API_KEY:
        logger.warning("未配置 FRED API key，跳过。请在 fred.py 中设置 FRED_API_KEY")
        return None

    try:
        import requests

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": days,
        }

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "observations" in data:
                observations = data["observations"]
                df = pd.DataFrame(observations)
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                return df[["date", "value"]].dropna()
        else:
            logger.warning("FRED API 请求失败: %d - %s", resp.status_code, resp.text[:200])

    except Exception as e:
        logger.error("FRED 获取失败 %s: %s", series_id, e)

    return None


def get_fred_series(series_id: str, days: int = 30, use_cache: bool = True) -> dict | None:
    """获取 FRED 指标数据。

    返回：
      - 指标代码
      - 名称
      - 最新值
      - 单位
      - 时间
      - 近期变化
    """
    if series_id not in FRED_SERIES:
        logger.warning("未知的 FRED 指标: %s", series_id)
        return None

    cache_path = _get_cache_path(series_id)

    # 尝试读取缓存
    if use_cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        age_hours = (datetime.now(UTC) - mtime).total_seconds() / 3600
        if age_hours < 24:  # 缓存 24 小时
            try:
                import json
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return data
            except Exception as e:
                logger.warning("FRED 缓存读取失败 %s: %s", cache_path, e)

    # 从 API 获取
    df = _fetch_fred(series_id, days)

    if df is None or df.empty:
        return None

    info = FRED_SERIES[series_id]
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    result = {
        "指标代码": series_id,
        "名称": info["名称"],
        "最新值": round(float(latest["value"]), 2),
        "单位": info["单位"],
        "时间": latest["date"].isoformat(),
        "日变化": round(float(latest["value"] - prev["value"]), 2),
        "说明": info["说明"],
        "数据来源": "FRED",
        "近期数据": df.tail(10).to_dict("records"),
    }

    # 保存缓存
    if use_cache:
        import json
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def get_real_yield() -> dict | None:
    """获取 10 年期美债实际收益率（TIPS）。"""
    return get_fred_series("DFII10")


def get_inflation_expectation() -> dict | None:
    """获取 10 年期盈亏平衡通胀率。"""
    return get_fred_series("T10YIE")


def get_nominal_yield() -> dict | None:
    """获取 10 年期美债名义收益率。"""
    return get_fred_series("DGS10")


def get_fed_funds_rate() -> dict | None:
    """获取联邦基金利率。"""
    return get_fred_series("DFF")


def get_gold_pricing_model() -> dict:
    """黄金定价模型关键指标。

    黄金与实际收益率呈强负相关（约 -0.8），是判断黄金中长期走势的最重要指标。
    """
    real_yield = get_real_yield()
    inflation_exp = get_inflation_expectation()
    nominal_yield = get_nominal_yield()

    result = {
        "时间": datetime.now(UTC).isoformat(),
        "实际收益率": real_yield,
        "通胀预期": inflation_exp,
        "名义收益率": nominal_yield,
    }

    # 分析
    if real_yield:
        ry_value = real_yield["最新值"]
        if ry_value > 2.0:
            result["实际收益率分析"] = "实际收益率处于高位（>2%），对黄金形成压制"
        elif ry_value > 1.0:
            result["实际收益率分析"] = "实际收益率偏高（1-2%），黄金承压"
        elif ry_value > 0:
            result["实际收益率分析"] = "实际收益率温和（0-1%），对黄金影响中性"
        elif ry_value > -1.0:
            result["实际收益率分析"] = "实际收益率为负，利好黄金"
        else:
            result["实际收益率分析"] = "实际收益率深度负值（<-1%），强烈利好黄金"

    return result
