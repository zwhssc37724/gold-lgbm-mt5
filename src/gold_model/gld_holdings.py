"""SPDR GLD 黄金 ETF 持仓量数据。

数据源（2026-08-30 修复）：
  主源：SPDR 官方 API https://api.spdrgoldshares.com/api/v1/data
        ?product=GLD&exchange=NYSE&lang=en
        ——返回 total_tonnes / total_ounces / shares_outstanding / total_nav_usd 等，
        与官网 /usa/gld/ 页面显示同源（从页面 JS chunk 逆向得到端点）。
  备源：HSBC 金库清单 PDF（spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv
        实际返回 PDF，Total Allocated Fine Weight）——仅伦敦金库部分，仅作参考。

旧版抓官网 /usa/ 首页正则解析——该站 2025 改版为 Next.js CSR，SSR HTML 不再包含
持仓数字，正则永远匹配不到（"GLD 网页解析失败，未找到持仓量数据"的根因）。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from gold_model import config

logger = logging.getLogger("gold_model.gld_holdings")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "gld_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SPDR 官方数据 API（与官网页面同源）
API_URL = "https://api.spdrgoldshares.com/api/v1/data"
OZ_PER_TONNE = 32150.7466


def _parse_number(s: str) -> float | None:
    """'1,042.357' / '33,512,833.02' / 'US$ 408.89' → float。"""
    m = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", s or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def fetch_gld_holdings() -> dict | None:
    """获取 GLD 最新持仓量（官方 API，24h 缓存）。

    返回：
      - 持仓量（吨）/ 持仓量（盎司）/ 日期
      - 总资产净值、流通股数、每盎司 NAV 等附加数据（可用时）
    """
    cache_path = CACHE_DIR / "gld_latest.json"

    # 检查缓存（每天只更新一次）
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        age_hours = (datetime.now(UTC) - mtime).total_seconds() / 3600
        if age_hours < 24:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("GLD 缓存读取失败: %s", e)

    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Origin": "https://www.spdrgoldshares.com",
            "Referer": "https://www.spdrgoldshares.com/",
        }
        resp = requests.get(
            API_URL,
            params={"product": "GLD", "exchange": "NYSE", "lang": "en"},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("GLD API 请求失败: %d", resp.status_code)
            return None

        data = (resp.json() or {}).get("data", {})
        tonnes = _parse_number(str(data.get("total_tonnes", {}).get("value", "")))
        ounces = _parse_number(str(data.get("total_ounces", {}).get("value", "")))
        if tonnes is None and ounces is None:
            logger.warning("GLD API 返回无持仓数据: %s", list(data.keys())[:10])
            return None
        if tonnes is None:
            tonnes = ounces / OZ_PER_TONNE  # type: ignore[operator]
        if ounces is None:
            ounces = tonnes * OZ_PER_TONNE

        result = {
            "持仓量（吨）": round(tonnes, 2),
            "持仓量（盎司）": round(ounces, 0),
            "日期": data.get("total_tonnes", {}).get("date") or datetime.now(UTC).strftime("%Y-%m-%d"),
            "总资产净值": data.get("total_nav_usd", {}).get("value"),
            "流通股数": data.get("shares_outstanding", {}).get("value"),
            "每股含金量（盎司）": data.get("metal_entitlement", {}).get("value"),
            "现货金价": data.get("spot_mid_usd", {}).get("value"),
            "数据来源": "SPDR 官方 API（api.spdrgoldshares.com）",
            "更新时间": datetime.now(UTC).isoformat(),
        }

        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info("GLD 持仓量获取成功: %.2f 吨", tonnes)
        return result

    except Exception as e:
        logger.error("GLD API 抓取失败: %s", e)

    return None


def get_gld_summary() -> dict:
    """获取 GLD 持仓量摘要。

    如果无法获取实时数据，返回说明信息。
    """
    holdings = fetch_gld_holdings()

    if holdings is None:
        return {
            "状态": "数据缺失",
            "说明": "GLD 持仓量数据获取失败，请访问 https://www.spdrgoldshares.com/usa/gld/ 手动查看",
            "数据源": "https://api.spdrgoldshares.com/api/v1/data",
        }

    return holdings
