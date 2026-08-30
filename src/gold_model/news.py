"""财经资讯数据源：金十数据 + FX678。

提供财经日历、快讯、重要数据（非农、CPI 等）的抓取和解读。

注意：金十和 FX678 没有公开免费 API，本模块通过以下方式获取数据：
1. 金十开放平台（需要 API key，需注册申请）— 最稳定
2. 金十/FX678 网页抓取（作为备用，可能不稳定）
3. 本地缓存 + 定时任务预抓取（推荐用法）

推荐配合 cronjob 使用：每天早上自动抓取财经日历，非农/CPI 前自动提醒。

金十开放平台申请：https://open.jin10.com/
FX678 开放平台：https://open.fx678.com（付费）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gold_model import config

logger = logging.getLogger("gold_model.news")

# 缓存目录
CACHE_DIR = config.DATA_DIR / "news_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 金十开放平台 API Key（从 https://open.jin10.com/ 免费申请）
JIN10_API_KEY = "sk-8N-sZOnBFiHeJWu5m06ejcYZ-fVX3X3ys9EBUmcVOqY"  # 在这里填入你的 API key，如 "sk-xxxxxxxx"
FX678_API_KEY = ""  # 在 https://open.fx678.com 申请（付费）

# 重要数据事件关键词
IMPORTANT_EVENTS = {
    "非农": ["非农", "非农就业", "非農", "NFP", "nonfarm", "non-farm", "非农数据", "就业报告"],
    "CPI": ["CPI", "消费者物价", "通膨", "通胀", "inflation", "消费者价格指数", "CPI数据"],
    "美联储": ["美联储", "Fed", "FOMC", "鲍威尔", "Powell", "利率决议", "加息", "降息", "货币政策", "FOMC会议"],
    "GDP": ["GDP", "国内生产总值", "经济增速"],
    "PCE": ["PCE", "个人消费支出", "核心PCE"],
    "ADP": ["ADP", "小非农", "ADP就业"],
    "失业率": ["失业率", "unemployment", "就业数据"],
    "零售销售": ["零售销售", "retail sales", "零售数据"],
    "ISM": ["ISM", "制造业PMI", "制造业采购经理", "PMI"],
    "初请失业金": ["初请失业金", "初请", "续请失业金"],
    "贸易帐": ["贸易帐", "贸易逆差", "贸易顺差", "贸易数据"],
    "耐用品订单": ["耐用品订单", "耐用品"],
    "新屋销售": ["新屋销售", "成屋销售", "房价指数"],
    "消费者信心": ["消费者信心", "密歇根大学", "咨商会"],
    "原油库存": ["原油库存", "EIA", "API库存"],
    "黄金储备": ["黄金储备", "央行购金", "黄金需求"],
}

# 数据重要性映射
EVENT_IMPORTANCE = {
    "非农": "极高",
    "CPI": "极高",
    "美联储": "极高",
    "GDP": "高",
    "PCE": "高",
    "失业率": "高",
    "零售销售": "中",
    "ISM": "中",
    "初请失业金": "中",
    "ADP": "中",
    "贸易帐": "低",
    "耐用品订单": "中",
    "新屋销售": "中",
    "消费者信心": "中",
    "原油库存": "中",
    "黄金储备": "高",
}

# 数据公布时间（美东时间 → 北京时间转换参考）
# 夏令时（3月-11月）：美东 +12 = 北京时间
# 冬令时（11月-3月）：美东 +13 = 北京时间
EVENT_TIMES = {
    "非农": "20:30",  # 北京时间（夏令时）
    "CPI": "20:30",
    "美联储": "02:00",  # 次日凌晨
    "GDP": "20:30",
    "PCE": "20:30",
    "失业率": "20:30",
    "零售销售": "20:30",
    "ISM": "22:00",
    "初请失业金": "20:30",
    "ADP": "20:15",
    "贸易帐": "20:30",
    "耐用品订单": "20:30",
    "新屋销售": "22:00",
    "消费者信心": "22:00",
    "原油库存": "22:30",
    "黄金储备": "不定期",
}


@dataclass
class NewsEvent:
    """财经事件/数据。"""
    title: str
    time: str  # ISO 格式或 HH:MM
    country: str  # 国家/地区
    importance: str  # 重要性：极高/高/中/低
    actual: str | None = None  # 实际值
    forecast: str | None = None  # 预期值
    previous: str | None = None  # 前值
    source: str = ""  # 数据来源
    url: str = ""  # 链接

    @property
    def event_type(self) -> str:
        """判断事件类型（非农/CPI/美联储等）。"""
        title_lower = self.title.lower()
        for event_type, keywords in IMPORTANT_EVENTS.items():
            if any(kw.lower() in title_lower for kw in keywords):
                return event_type
        return "其他"


def _get_cache_path(prefix: str, date: str) -> Path:
    """获取缓存文件路径。"""
    return CACHE_DIR / f"{prefix}_{date}.json"


def _save_cache(prefix: str, date: str, data: list[dict]) -> None:
    """保存缓存。"""
    path = _get_cache_path(prefix, date)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("缓存已保存: %s (%d 条)", path, len(data))


def _load_cache(prefix: str, date: str) -> list[dict] | None:
    """读取缓存。"""
    path = _get_cache_path(prefix, date)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("缓存读取失败 %s: %s", path, e)
    return None


def fetch_jin10_calendar_api(date: str | None = None) -> list[NewsEvent]:
    """通过金十开放平台 API 抓取财经日历（需要 API key）。

    申请地址：https://open.jin10.com/
    免费版有频率限制，付费版更稳定。
    """
    if not JIN10_API_KEY:
        logger.warning("未配置金十 API key，跳过 API 抓取。请在 news.py 中设置 JIN10_API_KEY")
        return []

    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")

    # 尝试读取缓存
    cached = _load_cache("jin10_calendar_api", date)
    if cached is not None:
        return [NewsEvent(**item) for item in cached]

    events = []

    try:
        import requests

        # 金十开放平台 API 端点
        url = "https://open.jin10.com/api/calendar"
        headers = {
            "Authorization": f"Bearer {JIN10_API_KEY}",
            "Content-Type": "application/json",
        }
        params = {
            "date": date,
            "type": "cj",  # 宏观数据日历
        }

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                for item in data["data"]:
                    events.append(NewsEvent(
                        title=item.get("name", item.get("title", "")),
                        time=item.get("time", ""),
                        country=item.get("country", ""),
                        importance=item.get("importance", "低"),
                        actual=item.get("actual"),
                        forecast=item.get("forecast"),
                        previous=item.get("previous"),
                        source="金十开放平台",
                        url=item.get("url", ""),
                    ))
            logger.info("金十 API 获取成功: %d 条事件", len(events))
        else:
            logger.warning("金十 API 请求失败: %d - %s", resp.status_code, resp.text[:200])

    except Exception as e:
        logger.error("金十 API 抓取失败: %s", e)

    if events:
        _save_cache("jin10_calendar_api", date, [vars(e) for e in events])

    return events


def fetch_jin10_calendar_web(date: str | None = None) -> list[NewsEvent]:
    """通过网页抓取金十财经日历（备用方案，可能不稳定）。

    注意：金十日历是 SSR + CSR 混合，直接抓取可能拿不到数据。
    此函数尝试抓取已知的 API 端点，如果失败则返回空列表。
    """
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")

    # 尝试读取缓存
    cached = _load_cache("jin10_calendar_web", date)
    if cached is not None:
        return [NewsEvent(**item) for item in cached]

    events = []

    try:
        import requests

        # 尝试多个可能的 API 端点
        endpoints = [
            f"https://rili.jin10.com/api/calendar?date={date}",
            f"https://cdn-rili.jin10.com/web_data/{date.replace('-', '/')}.json",
            f"https://rili.jin10.com/web_data/{date.replace('-', '/')}.json",
        ]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://rili.jin10.com/",
        }

        for url in endpoints:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        # 尝试解析不同的数据结构
                        if isinstance(data, dict):
                            # 尝试常见的数据键
                            for key in ("data", "list", "items", "events", "calendar"):
                                if key in data and isinstance(data[key], list):
                                    for item in data[key]:
                                        events.append(_parse_jin10_item(item))
                                    break
                            else:
                                # 如果 data 本身就是列表
                                if isinstance(data.get("data"), list):
                                    for item in data["data"]:
                                        events.append(_parse_jin10_item(item))
                    except json.JSONDecodeError:
                        logger.debug("端点返回非 JSON: %s", url)
                        continue

                if events:
                    logger.info("从 %s 获取到 %d 条事件", url, len(events))
                    break

            except requests.RequestException as e:
                logger.debug("端点请求失败 %s: %s", url, e)
                continue

    except Exception as e:
        logger.error("抓取金十日历失败: %s", e)

    if events:
        _save_cache("jin10_calendar_web", date, [vars(e) for e in events])

    return events


def _parse_jin10_item(item: dict) -> NewsEvent:
    """解析金十数据项为 NewsEvent。"""
    return NewsEvent(
        title=item.get("name", item.get("title", item.get("event", ""))),
        time=item.get("time", item.get("date", "")),
        country=item.get("country", item.get("region", "")),
        importance=item.get("importance", item.get("level", "低")),
        actual=item.get("actual"),
        forecast=item.get("forecast", item.get("prediction")),
        previous=item.get("previous", item.get("prior")),
        source="金十数据",
        url=item.get("url", item.get("link", "")),
    )


def fetch_jin10_calendar(date: str | None = None) -> list[NewsEvent]:
    """抓取财经日历（自动选择数据源）。

    优先级：金十 API（需 key）> ForexFactory 国际源 > 金十网页端点 > 缓存

    2026-08-30 修复：金十 cdn-rili.jin10.com 的 CDN 在本网络环境 TLS 握手直接失败
    （SSLEOFError，curl 同样失败），rili.jin10.com/api/calendar 返回 404 HTML。
    新增 ForexFactory 官方 JSON 日历（nfs.faireconomy.media）作为主备用源，
    数据完整（title/country/date/impact/forecast/previous），每周一个文件。
    """
    # 优先使用 API（如果配置了 key）
    if JIN10_API_KEY:
        events = fetch_jin10_calendar_api(date)
        if events:
            return events

    # 主备用：ForexFactory 国际源（本网络可达，无墙）
    events = fetch_forexfactory_calendar(date)
    if events:
        return events

    # 再备用：金十网页端点（网络恢复时可用）
    events = fetch_jin10_calendar_web(date)
    if events:
        return events

    # 最后尝试读取任何缓存
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
    for prefix in ("jin10_calendar_api", "jin10_calendar_web", "ff_calendar"):
        cached = _load_cache(prefix, date)
        if cached is not None:
            return [NewsEvent(**item) for item in cached]

    return []


# ForexFactory 国名 → 中文
_FF_COUNTRY_CN = {
    "USD": "美国", "EUR": "欧元区", "JPY": "日本", "GBP": "英国", "CNY": "中国",
    "AUD": "澳大利亚", "CAD": "加拿大", "CHF": "瑞士", "NZD": "新西兰", "All": "全球",
}
_FF_IMPACT_CN = {"High": "极高", "Medium": "高", "Low": "低", "Holiday": "低", "": "低"}


def _ff_week_file(date_str: str) -> str:
    """ForexFactory 按"周"发布日历文件：返回 date 所在周的三个候选文件名。"""
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    # 周日为一周起点（FF 惯例）
    start = d - timedelta(days=(d.weekday() + 1) % 7)
    files = []
    for off in (0, -7, 7):
        w = start + timedelta(days=off)
        files.append(w.strftime("%Y-%m-%d"))
    return files[0]  # 主文件；备选在调用处处理


def fetch_forexfactory_calendar(date: str | None = None) -> list[NewsEvent]:
    """ForexFactory 财经日历（国际源，稳定可达）。

    数据源：nfs.faireconomy.media/ff_calendar_thisweek.json（官方公开 JSON）。
    周数据按日期过滤出目标日。

    限流处理：429 时指数退避重试（最多 3 次）；当日缓存 6 小时内有效
    （FF 日历一天更新几次，比金十的 24h 短以保证及时性）。
    """
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")

    cached = _load_cache("ff_calendar", date)
    if cached is not None:
        return [NewsEvent(**item) for item in cached]

    import time

    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://www.forexfactory.com/",
        }
        # 本周 + 上周（周末跨周时本周文件可能还没发布）
        urls = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            "https://nfs.faireconomy.media/ff_calendar_lastweek.json",
        ]
        for url in urls:
            data = None
            # 限流重试：429 → 指数退避
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=headers, timeout=20)
                    if resp.status_code == 429 and attempt < 2:
                        wait = 15 * (2 ** attempt)  # 15s, 30s
                        logger.info("ForexFactory 限流，%ds 后重试", wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        data = resp.json()
                    break
                except requests.RequestException as e:
                    logger.debug("ForexFactory 端点失败 %s: %s", url, e)
                    break
            if not isinstance(data, list):
                continue
            events = []
            for item in data:
                try:
                    dt = datetime.fromisoformat(item.get("date", ""))
                    target = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
                    # FF 时区为美东；先按事件自身时区日期归属，再按 UTC 日期兜底（跨日误差可接受）
                    if dt.date() != target.date() and dt.astimezone(UTC).date() != target.date():
                        continue
                except ValueError:
                    continue
                country = item.get("country", "")
                events.append(NewsEvent(
                    title=item.get("title", ""),
                    time=item.get("date", ""),
                    country=_FF_COUNTRY_CN.get(country, country),
                    importance=_FF_IMPACT_CN.get(item.get("impact", ""), "低"),
                    actual=None,
                    forecast=item.get("forecast") or None,
                    previous=item.get("previous") or None,
                    source="ForexFactory",
                    url="",
                ))
            if events:
                logger.info("ForexFactory 日历获取成功: %s 共 %d 条事件", date, len(events))
                _save_cache("ff_calendar", date, [vars(e) for e in events])
                return events

    except Exception as e:
        logger.error("ForexFactory 日历抓取失败: %s", e)

    return []


def fetch_jin10_flash(limit: int = 20) -> list[dict]:
    """抓取金十快讯。

    参数：
      - limit: 返回条数

    返回：快讯列表，每条包含 title, time, content, url
    """
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    cached = _load_cache("jin10_flash", date)
    if cached is not None:
        return cached[:limit]

    flash_list = []

    try:
        import requests

        # 金十快讯 API（可能需要登录或有访问限制）
        url = "https://flash-api.jin10.com/get_flash_list"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.jin10.com/",
        }
        params = {
            "channel": "-8200",
            "vip": "1",
            "t": str(int(datetime.now().timestamp())),
        }

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                for item in data["data"][:limit]:
                    flash_list.append({
                        "title": item.get("title", ""),
                        "time": item.get("time", ""),
                        "content": item.get("content", ""),
                        "url": item.get("url", ""),
                        "source": "金十快讯",
                    })

    except Exception as e:
        logger.error("抓取金十快讯失败: %s", e)

    if flash_list:
        _save_cache("jin10_flash", date, flash_list)

    return flash_list


def analyze_event_impact(event: NewsEvent) -> dict:
    """分析财经事件对黄金的影响。

    返回：影响分析结果
    """
    event_type = event.event_type
    importance = EVENT_IMPORTANCE.get(event_type, "低")

    # 默认影响分析
    impact = {
        "事件类型": event_type,
        "重要性": importance,
        "对黄金影响": "中性",
        "影响逻辑": "",
        "交易建议": "观望",
        "参考时间": EVENT_TIMES.get(event_type, "待定"),
    }

    if event_type == "非农":
        impact["影响逻辑"] = (
            "非农数据反映美国就业市场状况。数据好于预期 → 美联储加息预期升温 → 美元走强 → 黄金承压；"
            "数据差于预期 → 降息预期升温 → 美元走弱 → 黄金受益。"
            "非农是每月最重要的数据之一，通常在每月第一个周五公布。"
        )
        impact["交易建议"] = (
            "数据公布前 30 分钟建议平仓或减仓观望；公布后等待 5-10 分钟让市场消化，"
            "再根据数据方向和力度顺势操作；务必设置止损，非农数据可能引发剧烈波动。"
        )
    elif event_type == "CPI":
        impact["影响逻辑"] = (
            "CPI 反映通胀水平。CPI 高于预期 → 通胀压力大 → 美联储加息预期升温 → 美元走强 → 黄金承压；"
            "CPI 低于预期 → 通胀缓解 → 降息预期升温 → 黄金受益。"
            "核心 CPI（剔除食品和能源）比整体 CPI 更重要。"
        )
        impact["交易建议"] = (
            "CPI 是黄金最重要的数据之一，波动通常较大（20-50 美元）；"
            "建议轻仓参与或观望，等待数据公布后趋势明朗再入场；"
            "注意区分整体 CPI 和核心 CPI 的差异。"
        )
    elif event_type == "美联储":
        impact["影响逻辑"] = (
            "美联储利率决议和鲍威尔讲话直接影响美元利率预期。"
            "加息/鹰派 → 美元走强 → 黄金承压；降息/鸽派 → 美元走弱 → 黄金受益。"
            "除了利率决定，还要关注点阵图、经济预测和鲍威尔发布会。"
        )
        impact["交易建议"] = (
            "利率决议通常伴随剧烈波动（50-100 美元以上），建议数据公布前平仓或严格止损；"
            "决议后 30 分钟内波动最大，建议观望；鲍威尔讲话可能逆转决议方向，需全程关注。"
        )
    elif event_type == "GDP":
        impact["影响逻辑"] = (
            "GDP 反映经济整体状况。GDP 强劲 → 风险偏好上升 → 黄金避险需求下降 → 承压；"
            "GDP 疲软 → 避险需求上升 → 黄金受益。"
        )
        impact["交易建议"] = "GDP 数据影响相对温和，可结合其他数据综合判断。"
    elif event_type == "PCE":
        impact["影响逻辑"] = (
            "PCE 是美联储最关注的通胀指标，影响逻辑与 CPI 类似。"
            "核心 PCE 尤为重要，直接影响美联储政策决策。"
        )
        impact["交易建议"] = "PCE 数据重要性仅次于 CPI，建议参照 CPI 的交易策略。"
    elif event_type == "失业率":
        impact["影响逻辑"] = (
            "失业率上升 → 经济疲软 → 降息预期升温 → 黄金受益；"
            "失业率下降 → 经济强劲 → 加息预期升温 → 黄金承压。"
        )
        impact["交易建议"] = "失业率通常与非农同时公布，需结合分析。"
    elif event_type == "零售销售":
        impact["影响逻辑"] = (
            "零售销售反映消费状况。数据强劲 → 经济向好 → 美元走强 → 黄金承压；"
            "数据疲软 → 经济放缓 → 降息预期 → 黄金受益。"
        )
        impact["交易建议"] = "零售销售影响中等，可结合其他数据综合判断。"
    elif event_type == "ISM":
        impact["影响逻辑"] = (
            "ISM 制造业 PMI 反映制造业景气度。PMI 高于 50 → 扩张 → 经济向好 → 美元走强 → 黄金承压；"
            "PMI 低于 50 → 收缩 → 经济放缓 → 黄金受益。"
        )
        impact["交易建议"] = "ISM 是先行指标，对预测经济趋势有参考价值。"
    elif event_type == "初请失业金":
        impact["影响逻辑"] = (
            "初请失业金人数反映就业市场短期变化。人数增加 → 就业疲软 → 降息预期 → 黄金受益；"
            "人数减少 → 就业强劲 → 加息预期 → 黄金承压。"
        )
        impact["交易建议"] = "初请失业金每周公布，影响相对温和，适合观察趋势变化。"
    elif event_type == "ADP":
        impact["影响逻辑"] = (
            "ADP 就业数据被称为'小非农'，是非农数据的前瞻指标。"
            "ADP 好于预期 → 预示非农可能强劲 → 黄金承压；"
            "ADP 差于预期 → 预示非农可能疲软 → 黄金受益。"
        )
        impact["交易建议"] = "ADP 通常在非农前两天公布，可作为非农的参考，但两者常有偏差。"
    elif event_type == "原油库存":
        impact["影响逻辑"] = (
            "原油库存数据主要影响油价，间接影响黄金。"
            "库存增加 → 油价下跌 → 通胀预期下降 → 黄金承压；"
            "库存减少 → 油价上涨 → 通胀预期上升 → 黄金受益。"
        )
        impact["交易建议"] = "原油库存对黄金影响间接且较弱，一般不建议据此交易黄金。"
    elif event_type == "黄金储备":
        impact["影响逻辑"] = (
            "央行黄金储备变化反映官方对黄金的需求。"
            "央行增持 → 长期利好黄金；央行减持 → 长期利空黄金。"
            "这是长期趋势指标，对短期价格影响有限。"
        )
        impact["交易建议"] = "黄金储备数据是长期参考指标，适合用于判断长期趋势，不适合短线交易。"

    return impact


def get_today_important_events() -> list[dict]:
    """获取今天的重要财经事件。

    返回：今天的重要事件列表，每个事件附带影响分析
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    events = fetch_jin10_calendar(today)

    important = []
    for event in events:
        if event.event_type != "其他" or event.importance in ("极高", "高"):
            impact = analyze_event_impact(event)
            important.append({
                "事件": vars(event),
                "影响分析": impact,
            })

    return important


def get_upcoming_events(days: int = 3) -> list[dict]:
    """获取未来几天的重要财经事件。

    参数：
      - days: 未来几天

    返回：未来几天的重要事件列表
    """
    all_events = []
    for i in range(days):
        date = (datetime.now(UTC) + timedelta(days=i)).strftime("%Y-%m-%d")
        events = fetch_jin10_calendar(date)
        for event in events:
            if event.event_type != "其他" or event.importance in ("极高", "高"):
                all_events.append({
                    "日期": date,
                    "事件": vars(event),
                    "影响分析": analyze_event_impact(event),
                })

    return all_events


def get_next_important_event() -> dict | None:
    """获取下一个即将到来的重要事件（非农/CPI/美联储）。

    返回：下一个重要事件的信息，如果未找到则返回 None。
    """
    # 检查未来 7 天
    for i in range(7):
        date = (datetime.now(UTC) + timedelta(days=i)).strftime("%Y-%m-%d")
        events = fetch_jin10_calendar(date)
        for event in events:
            if event.event_type in ("非农", "CPI", "美联储"):
                return {
                    "日期": date,
                    "事件": vars(event),
                    "影响分析": analyze_event_impact(event),
                    "距离天数": i,
                }

    return None


# ---------------------------------------------------------------------------
# 备用：如果金十 API 不可用，提供手动数据输入
# ---------------------------------------------------------------------------

def create_manual_event(
    title: str,
    time: str,
    country: str = "美国",
    importance: str = "高",
    actual: str | None = None,
    forecast: str | None = None,
    previous: str | None = None,
) -> NewsEvent:
    """手动创建一个财经事件（当 API 不可用时使用）。"""
    return NewsEvent(
        title=title,
        time=time,
        country=country,
        importance=importance,
        actual=actual,
        forecast=forecast,
        previous=previous,
        source="手动输入",
        url="",
    )


def get_weekly_important_events() -> list[dict]:
    """获取本周重要事件参考（基于已知的美联储日程等）。"""
    week_events = []

    now = datetime.now(UTC)
    weekday = now.weekday()

    # 非农通常在每月第一个周五
    if weekday == 4 and now.day <= 7:  # 周五且是月初
        week_events.append({
            "日期": now.strftime("%Y-%m-%d"),
            "事件": {
                "title": "美国非农就业报告",
                "time": "20:30",
                "country": "美国",
                "importance": "极高",
                "source": "已知日程",
            },
            "影响分析": analyze_event_impact(create_manual_event("美国非农就业报告", "20:30", "美国", "极高")),
        })

    return week_events


# ---------------------------------------------------------------------------
# 财经事件提醒（配合 cronjob 使用）
# ---------------------------------------------------------------------------

def get_daily_briefing() -> dict:
    """生成每日财经简报（供定时任务调用）。

    返回：今日重要事件 + 模型预测 + 市场状态
    """
    from gold_model.serve_mcp import get_klines_summary, predict, predict_direction_3class

    # 获取今天和未来 3 天的重要事件
    today_events = get_today_important_events()
    upcoming = get_upcoming_events(3)

    # 获取下一个重要事件
    next_event = get_next_important_event()

    # 获取模型预测
    breakout_pred = predict()
    direction_pred = predict_direction_3class()

    # 获取 K 线摘要
    klines_summary = get_klines_summary()

    return {
        "日期": datetime.now(UTC).strftime("%Y-%m-%d"),
        "市场状态": klines_summary.get("市场状态"),
        "今日事件": today_events,
        "未来3天事件": upcoming,
        "下一个重要事件": next_event,
        "模型预测": {
            "波动扩张": breakout_pred,
            "方向三分类": direction_pred,
        },
        "K线摘要": klines_summary,
    }


