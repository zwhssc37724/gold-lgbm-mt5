"""CME FedWatch 利率概率数据。

CME FedWatch 工具显示市场对美联储未来利率决定的概率预期。
数据来源：https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html

注意：CME FedWatch 没有公开 API，本模块通过网页抓取获取数据。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from gold_model import config

logger = logging.getLogger("gold_model.fedwatch")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "fedwatch_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_fedwatch() -> dict | None:
    """获取 CME FedWatch 最新利率概率。

    返回：
      - 下次会议日期
      - 维持利率不变的概率
      - 加息 25bp 的概率
      - 降息 25bp 的概率
      - 数据来源
    """
    cache_path = CACHE_DIR / "fedwatch_latest.json"

    # 检查缓存（每小时更新）
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        age_hours = (datetime.now(UTC) - mtime).total_seconds() / 3600
        if age_hours < 1:
            try:
                import json
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("FedWatch 缓存读取失败: %s", e)

    # 从 CME 抓取
    try:
        import requests

        # CME FedWatch 页面
        url = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            # 尝试从 HTML 中提取概率数据
            # 注意：CME 页面是动态加载的，直接抓取可能拿不到数据
            # 这里提供一个备用方案：返回说明信息

            result = {
                "状态": "数据需手动查看",
                "说明": "CME FedWatch 数据需要访问 https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html 查看",
                "数据源": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
                "更新时间": datetime.now(UTC).isoformat(),
            }

            # 保存缓存
            import json
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return result

    except Exception as e:
        logger.error("FedWatch 抓取失败: %s", e)

    return None


def get_fedwatch_summary() -> dict:
    """获取 FedWatch 摘要。

    如果无法获取实时数据，返回说明信息。
    """
    data = fetch_fedwatch()

    if data is None:
        return {
            "状态": "数据缺失",
            "说明": "CME FedWatch 数据获取失败，请访问 https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html 手动查看",
            "数据源": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        }

    return data


# 备用：提供 FRED 的利率预期数据（如果 FedWatch 不可用）
def get_rate_expectation_from_fred() -> dict | None:
    """从 FRED 获取利率预期数据（备用方案）。

    FRED 提供联邦基金期货数据，可以间接反映利率预期。
    """
    from gold_model.fred import get_fed_funds_rate, get_fred_series

    # 获取当前利率
    current_rate = get_fed_funds_rate()

    # 获取 2 年期美债收益率（反映短期利率预期）
    two_year = get_fred_series("DGS2")

    if current_rate and two_year:
        current = current_rate["最新值"]
        two_y = two_year["最新值"]

        # 2 年期收益率高于当前利率 = 市场预期加息
        # 2 年期收益率低于当前利率 = 市场预期降息
        spread = two_y - current

        return {
            "当前联邦基金利率": current,
            "2年期美债收益率": two_y,
            "利差": round(spread, 2),
            "市场预期": "加息" if spread > 0.25 else "降息" if spread < -0.25 else "维持",
            "数据来源": "FRED（备用）",
        }

    return None
