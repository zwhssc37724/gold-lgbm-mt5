"""SPDR GLD 黄金 ETF 持仓量数据。

SPDR Gold Shares (GLD) 是全球最大的黄金 ETF，其持仓量变化反映机构资金对黄金的态度。
数据来源：https://www.spdrgoldshares.com/usa/ （每日更新）

注意：GLD 官网没有公开 API，本模块通过网页抓取获取数据。
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


def fetch_gld_holdings() -> dict | None:
    """获取 GLD 最新持仓量。

    返回：
      - 持仓量（吨）
      - 持仓量（盎司）
      - 日期
      - 日变化
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

    # 从 GLD 官网抓取
    try:
        import requests

        # GLD 官方数据页面
        url = "https://www.spdrgoldshares.com/usa/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            # 尝试从 HTML 中提取持仓量
            # GLD 页面通常包含 "Total Net Asset Value" 和 "Gold Holdings"
            text = resp.text

            # 尝试多种匹配模式
            # 模式1: "Gold Holdings: 1,234.56 tonnes"
            pattern1 = r"Gold Holdings[^\n]*?([\d,]+\.?\d*)\s*(?:tonnes|tons)"
            tonnes_match = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)
            if not tonnes_match:
                # 模式2: "1,234.56 tonnes"
                tonnes_match = re.search(r"([\d,]+\.?\d*)\s*(?:tonnes|tons)", text, re.IGNORECASE)

            if tonnes_match:
                tonnes = float(tonnes_match.group(1).replace(",", ""))
                ounces = tonnes * 32150.7  # 1 吨 = 32150.7 盎司

                result = {
                    "持仓量（吨）": round(tonnes, 2),
                    "持仓量（盎司）": round(ounces, 2),
                    "日期": datetime.now(UTC).strftime("%Y-%m-%d"),
                    "数据来源": "SPDR 官网",
                    "更新时间": datetime.now(UTC).isoformat(),
                }

                # 保存缓存
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                logger.info("GLD 持仓量获取成功: %.2f 吨", tonnes)
                return result

        logger.warning("GLD 网页解析失败，未找到持仓量数据")

    except Exception as e:
        logger.error("GLD 抓取失败: %s", e)

    return None


def get_gld_summary() -> dict:
    """获取 GLD 持仓量摘要。

    如果无法获取实时数据，返回说明信息。
    """
    holdings = fetch_gld_holdings()

    if holdings is None:
        return {
            "状态": "数据缺失",
            "说明": "GLD 持仓量数据获取失败，请访问 https://www.spdrgoldshares.com/usa/ 手动查看",
            "数据源": "https://www.spdrgoldshares.com/usa/",
        }

    return holdings


def get_gld_historical_trend() -> dict:
    """获取 GLD 持仓量历史趋势（简要分析）。

    基于最近几次抓取的数据判断趋势。
    """
    # 读取所有缓存文件
    cache_files = sorted(CACHE_DIR.glob("gld_*.json"))

    if len(cache_files) < 2:
        return {
            "状态": "数据不足",
            "说明": "需要至少 2 天的数据才能判断趋势",
        }

    # 读取最近两次数据
    latest = json.loads(cache_files[-1].read_text(encoding="utf-8"))
    previous = json.loads(cache_files[-2].read_text(encoding="utf-8"))

    change = latest["持仓量（吨）"] - previous["持仓量（吨）"]

    return {
        "最新持仓": latest["持仓量（吨）"],
        "上次持仓": previous["持仓量（吨）"],
        "日变化": round(change, 2),
        "趋势": "增持" if change > 0 else "减持" if change < 0 else "持平",
        "分析": "GLD 增持通常意味着机构资金流入黄金，利好金价；减持则相反。",
    }
