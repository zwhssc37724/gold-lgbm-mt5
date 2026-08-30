"""CFTC 持仓报告数据获取。

提供 COMEX 黄金期货的非商业持仓数据（对冲基金等大投机者的持仓），
是判断黄金中长期趋势的重要领先指标。

数据源：
- CFTC 官网：https://www.cftc.gov/dea/futures/deacmxsf.htm
- 每周五下午 3:30 ET 更新（北京时间周六凌晨）

字段说明：
- 非商业多头：对冲基金等大投机者的多头持仓
- 非商业空头：对冲基金等大投机者的空头持仓
- 商业多头/空头：生产商、贸易商的对冲持仓
- 非商业净多头 = 非商业多头 - 非商业空头（关键指标）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from gold_model import config

logger = logging.getLogger("gold_model.cftc")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "cftc_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CFTC 数据缓存有效期（7 天，因为每周只更新一次）
CFTC_CACHE_TTL_DAYS = 7


@dataclass
class CFTCPosition:
    """CFTC 持仓数据。"""
    symbol: str                    # 品种，如 GOLD
    report_date: str               # 报告日期，如 08/25/26
    open_interest: int             # 总持仓量
    non_commercial_long: int       # 非商业多头
    non_commercial_short: int      # 非商业空头
    non_commercial_spreads: int    # 非商业套利
    commercial_long: int           # 商业多头
    commercial_short: int          # 商业空头
    nonreportable_long: int        # 非报告持仓多头
    nonreportable_short: int       # 非报告持仓空头
    change_non_commercial_long: int      # 非商业多头变化
    change_non_commercial_short: int     # 非商业空头变化
    change_open_interest: int            # 总持仓变化

    @property
    def non_commercial_net(self) -> int:
        """非商业净多头（关键指标）。"""
        return self.non_commercial_long - self.non_commercial_short

    @property
    def non_commercial_net_change(self) -> int:
        """非商业净多头变化。"""
        return self.change_non_commercial_long - self.change_non_commercial_short

    @property
    def sentiment(self) -> str:
        """基于净多头的情绪判断。"""
        net = self.non_commercial_net
        if net > 200000:
            return "极度看多（历史高位，警惕反转）"
        elif net > 100000:
            return "强烈看多"
        elif net > 50000:
            return "偏多"
        elif net > 0:
            return "轻微偏多"
        elif net > -50000:
            return "轻微偏空"
        elif net > -100000:
            return "偏空"
        else:
            return "强烈看空（历史低位，警惕反弹）"


def _parse_cftc_text(text: str, symbol: str = "GOLD") -> CFTCPosition | None:
    """解析 CFTC 文本报告。

    CFTC 报告是固定格式的文本，需要正则提取。
    """
    # 找到目标品种部分
    pattern = rf"{symbol}.*?Code-\d+.*?FUTURES ONLY POSITIONS AS OF (\d{{2}}/\d{{2}}/\d{{2}}).*?OPEN INTEREST:\s+([\d,]+).*?COMMITMENTS\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+).*?CHANGES FROM \d{{2}}/\d{{2}}/\d{{2}}.*?([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)\s+([\d,+-]+)"

    match = re.search(pattern, text, re.DOTALL)
    if not match:
        logger.warning("未找到 %s 的 CFTC 数据", symbol)
        return None

    groups = match.groups()

    def parse_int(s: str) -> int:
        return int(s.replace(",", "").replace("+", ""))

    return CFTCPosition(
        symbol=symbol,
        report_date=groups[0],
        open_interest=parse_int(groups[1]),
        non_commercial_long=parse_int(groups[2]),
        non_commercial_short=parse_int(groups[3]),
        non_commercial_spreads=parse_int(groups[4]),
        commercial_long=parse_int(groups[5]),
        commercial_short=parse_int(groups[6]),
        nonreportable_long=parse_int(groups[7]),
        nonreportable_short=parse_int(groups[8]),
        change_non_commercial_long=parse_int(groups[9]),
        change_non_commercial_short=parse_int(groups[10]),
        change_open_interest=parse_int(groups[11]),
    )


def fetch_cftc_gold() -> CFTCPosition | None:
    """获取 COMEX 黄金的 CFTC 持仓数据。

    返回：CFTCPosition 对象，或 None（获取失败）
    """
    cache_path = CACHE_DIR / "cftc_gold_latest.json"

    # 检查缓存
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
        age_days = (datetime.now(UTC) - mtime).days
        if age_days < CFTC_CACHE_TTL_DAYS:
            try:
                import json
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return CFTCPosition(**data)
            except Exception as e:
                logger.warning("CFTC 缓存读取失败: %s", e)

    # 从 CFTC 官网抓取
    try:
        import requests

        url = "https://www.cftc.gov/dea/futures/deacmxsf.htm"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.cftc.gov/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            position = _parse_cftc_text(resp.text, "GOLD")
            if position:
                # 保存缓存
                import json
                cache_path.write_text(json.dumps(vars(position), ensure_ascii=False, indent=2))
                logger.info("CFTC 黄金持仓获取成功: 报告日期 %s, 净多头 %d", position.report_date, position.non_commercial_net)
                return position
        else:
            logger.warning("CFTC 请求失败: %d", resp.status_code)

    except Exception as e:
        logger.error("CFTC 抓取失败: %s", e)

    return None


def get_cftc_summary() -> dict:
    """获取 CFTC 黄金持仓摘要。

    返回：
      - 报告日期
      - 非商业净多头（关键指标）
      - 净多头变化
      - 总持仓量
      - 情绪判断
      - 数据来源
    """
    position = fetch_cftc_gold()

    if position is None:
        return {
            "状态": "数据缺失",
            "说明": "CFTC 数据获取失败，请检查网络或稍后重试",
            "数据源": "https://www.cftc.gov/dea/futures/deacmxsf.htm",
        }

    return {
        "品种": position.symbol,
        "报告日期": position.report_date,
        "总持仓量": position.open_interest,
        "非商业多头": position.non_commercial_long,
        "非商业空头": position.non_commercial_short,
        "非商业净多头": position.non_commercial_net,
        "净多头变化": position.non_commercial_net_change,
        "商业多头": position.commercial_long,
        "商业空头": position.commercial_short,
        "情绪判断": position.sentiment,
        "数据来源": "CFTC 官网",
        "更新时间": datetime.now(UTC).isoformat(),
    }
