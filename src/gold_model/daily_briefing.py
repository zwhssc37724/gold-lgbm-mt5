"""每日黄金早报生成器。

汇总 MT5 模型预测、Yahoo Finance 宏观数据、金十财经日历，生成每日黄金分析报告。

用法：
    uv run python -m gold_model.daily_briefing

输出：每日黄金早报（Markdown 格式，可保存到文件或发送）
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from gold_model import config, mt5_client, news, yahoo_finance

logger = logging.getLogger("gold_model.daily_briefing")

# 报告输出目录
REPORT_DIR = config.DATA_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_daily_briefing() -> dict:
    """生成每日黄金早报。

    返回包含以下内容的字典：
      - 日期
      - 市场状态
      - MT5 模型预测（波动扩张 + 方向三分类）
      - Yahoo Finance 宏观数据（美元指数、美债、VIX、金银比等）
      - 黄金情绪分析
      - 今日财经事件
      - 未来几天重要事件
      - 交易建议
    """
    now = datetime.now(UTC)

    # 1. 市场状态
    market_status = mt5_client.is_market_open()

    # 2. MT5 模型预测（如果 MT5 可用）
    mt5_predictions = {}
    try:
        from gold_model.serve_mcp import predict, predict_direction_3class
        mt5_predictions["波动扩张"] = predict()
        mt5_predictions["方向三分类"] = predict_direction_3class()
    except Exception as e:
        logger.warning("MT5 预测失败: %s", e)
        mt5_predictions["错误"] = str(e)

    # 3. Yahoo Finance 宏观数据
    macro = yahoo_finance.get_macro_indicators()
    sentiment = yahoo_finance.get_gold_sentiment()

    # 4. 财经事件
    today_events = news.get_today_important_events()
    upcoming_events = news.get_upcoming_events(3)
    next_event = news.get_next_important_event()

    # 5. 生成交易建议
    trading_advice = _generate_trading_advice(
        market_status=market_status,
        mt5_predictions=mt5_predictions,
        macro=macro,
        sentiment=sentiment,
        next_event=next_event,
    )

    # 6. 汇总报告
    report = {
        "日期": now.strftime("%Y-%m-%d"),
        "时间": now.isoformat(),
        "市场状态": market_status,
        "MT5模型预测": mt5_predictions,
        "Yahoo宏观数据": macro,
        "黄金情绪": sentiment,
        "今日事件": today_events,
        "未来3天事件": upcoming_events,
        "下一个重要事件": next_event,
        "交易建议": trading_advice,
    }

    return report


def _generate_trading_advice(
    market_status: dict,
    mt5_predictions: dict,
    macro: dict,
    sentiment: dict,
    next_event: dict | None,
) -> dict:
    """生成交易建议。

    基于多维度数据综合判断：
      - 市场状态（是否开市）
      - MT5 模型信号
      - 宏观数据（美元指数、VIX）
      - 黄金情绪
      - 即将到来的重要事件
    """
    advice = {
        "总体建议": "观望",
        "仓位建议": "空仓或轻仓",
        "风险提示": [],
        "机会提示": [],
        "关键价位": {},
    }

    # 市场状态
    if market_status.get("状态") in ("weekend", "closed"):
        advice["总体建议"] = "休市，不交易"
        advice["风险提示"].append("市场休市，避免隔夜持仓风险")
        return advice

    # MT5 模型信号
    direction_pred = mt5_predictions.get("方向三分类", {})
    breakout_pred = mt5_predictions.get("波动扩张", {})

    direction_signal = direction_pred.get("预测类别", "观望")
    breakout_signal = breakout_pred.get("信号", "中性")

    # 宏观情绪
    sentiment_score = sentiment.get("综合情绪", "中性")

    # 综合判断
    bullish_signals = 0
    bearish_signals = 0

    if direction_signal == "看多":
        bullish_signals += 1
    elif direction_signal == "看空":
        bearish_signals += 1

    if sentiment_score == "偏多":
        bullish_signals += 1
    elif sentiment_score == "偏空":
        bearish_signals += 1

    # 波动扩张信号
    if breakout_signal == "预期扩张":
        advice["机会提示"].append("模型预测波动扩张，可能有大行情")
    elif breakout_signal == "预期收敛":
        advice["风险提示"].append("模型预测波动收敛，行情可能平淡")

    # 重要事件提醒
    if next_event:
        days_until = next_event.get("距离天数", 0)
        event_name = next_event.get("事件", {}).get("title", "未知事件")
        if days_until == 0:
            advice["风险提示"].append(f"今天有重要事件：{event_name}，建议数据公布前观望")
        elif days_until == 1:
            advice["风险提示"].append(f"明天有重要事件：{event_name}，建议提前减仓")

    # 综合建议
    if bullish_signals > bearish_signals:
        advice["总体建议"] = "偏多"
        advice["仓位建议"] = "轻仓试多（≤10%）"
    elif bearish_signals > bullish_signals:
        advice["总体建议"] = "偏空"
        advice["仓位建议"] = "轻仓试空（≤10%）"
    else:
        advice["总体建议"] = "观望"
        advice["仓位建议"] = "空仓或极轻仓（≤5%）"

    # 关键价位
    risk_mgmt = direction_pred.get("风险管理", {})
    if risk_mgmt:
        advice["关键价位"] = {
            "当前价格": risk_mgmt.get("当前价格"),
            "支撑": direction_pred.get("支撑阻力", {}).get("近20根支撑"),
            "阻力": direction_pred.get("支撑阻力", {}).get("近20根阻力"),
            "止损": risk_mgmt.get("建议止损"),
            "止盈1": risk_mgmt.get("建议止盈1（1:1）"),
            "止盈2": risk_mgmt.get("建议止盈2（2:1）"),
        }

    return advice


def format_report_markdown(report: dict) -> str:
    """将报告格式化为 Markdown。"""
    lines = [
        f"# 每日黄金早报 - {report['日期']}",
        "",
        f"**生成时间**: {report['时间']}",
        "",
        "---",
        "",
        "## 市场状态",
        "",
        f"- **状态**: {report['市场状态'].get('状态', '未知')}",
        f"- **说明**: {report['市场状态'].get('说明', '无')}",
    ]

    if report['市场状态'].get('下次开市'):
        lines.append(f"- **下次开市**: {report['市场状态']['下次开市']}")

    lines.extend([
        "",
        "---",
        "",
        "## MT5 模型预测",
        "",
    ])

    # 波动扩张预测
    breakout = report['MT5模型预测'].get('波动扩张', {})
    if breakout and '错误' not in breakout:
        lines.extend([
            "### 波动扩张预测",
            "",
            f"- **信号**: {breakout.get('信号', '未知')}",
            f"- **扩张概率**: {breakout.get('扩张概率', 0):.2%}",
            f"- **最新收盘价**: {breakout.get('最新收盘价', '未知')}",
            "",
        ])

    # 方向三分类预测
    direction = report['MT5模型预测'].get('方向三分类', {})
    if direction and '错误' not in direction:
        lines.extend([
            "### 方向三分类预测",
            "",
            f"- **信号**: {direction.get('信号', '未知')}",
            f"- **看空概率**: {direction.get('看空概率', 0):.2%}",
            f"- **观望概率**: {direction.get('观望概率', 0):.2%}",
            f"- **看多概率**: {direction.get('看多概率', 0):.2%}",
            "",
            "#### 风险管理",
            "",
        ])
        risk = direction.get('风险管理', {})
        if risk:
            lines.extend([
                f"- **当前价格**: {risk.get('当前价格', '未知')}",
                f"- **ATR**: {risk.get('ATR（平均波幅）', '未知')}",
                f"- **建议止损**: {risk.get('建议止损', '未知')}",
                f"- **建议止盈1**: {risk.get('建议止盈1（1:1）', '未知')}",
                f"- **建议止盈2**: {risk.get('建议止盈2（2:1）', '未知')}",
                f"- **建议仓位**: {risk.get('建议仓位', '未知')}",
                "",
            ])

    # Yahoo 宏观数据
    lines.extend([
        "---",
        "",
        "## 宏观数据",
        "",
    ])

    macro = report.get('Yahoo宏观数据', {})
    if macro.get('美元指数'):
        dxy = macro['美元指数']
        lines.extend([
            "### 美元指数",
            "",
            f"- **最新价**: {dxy.get('最新价', '未知')}",
            f"- **涨跌**: {dxy.get('涨跌', '未知')}",
            f"- **涨跌幅**: {dxy.get('涨跌幅', '未知')}%",
            "",
        ])

    if macro.get('美债10年收益率'):
        us10y = macro['美债10年收益率']
        lines.extend([
            "### 美债 10 年收益率",
            "",
            f"- **最新价**: {us10y.get('最新价', '未知')}%",
            "",
        ])

    if macro.get('VIX恐慌指数'):
        vix = macro['VIX恐慌指数']
        lines.extend([
            "### VIX 恐慌指数",
            "",
            f"- **最新价**: {vix.get('最新价', '未知')}",
            "",
        ])

    if macro.get('金银比'):
        lines.extend([
            "### 金银比",
            "",
            f"- **当前比率**: {macro['金银比']}",
            "",
        ])

    # 黄金情绪
    sentiment = report.get('黄金情绪', {})
    lines.extend([
        "### 黄金情绪分析",
        "",
        f"- **综合情绪**: {sentiment.get('综合情绪', '中性')}",
        "",
        "各维度分析：",
        "",
    ])
    for key, value in sentiment.get('各维度', {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    # 今日事件
    lines.extend([
        "---",
        "",
        "## 今日财经事件",
        "",
    ])
    today_events = report.get('今日事件', [])
    if today_events:
        for event in today_events:
            evt = event.get('事件', {})
            impact = event.get('影响分析', {})
            lines.extend([
                f"### {evt.get('title', '未知事件')}",
                "",
                f"- **时间**: {evt.get('time', '未知')}",
                f"- **重要性**: {evt.get('importance', '未知')}",
                f"- **影响**: {impact.get('影响逻辑', '无')}",
                f"- **建议**: {impact.get('交易建议', '观望')}",
                "",
            ])
    else:
        lines.append("今天无重要财经事件。")
        lines.append("")

    # 未来事件
    lines.extend([
        "---",
        "",
        "## 未来 3 天重要事件",
        "",
    ])
    upcoming = report.get('未来3天事件', [])
    if upcoming:
        for event in upcoming[:5]:  # 最多显示 5 个
            evt = event.get('事件', {})
            lines.append(f"- **{event.get('日期', '未知')}**: {evt.get('title', '未知')} ({evt.get('importance', '未知')})")
        lines.append("")
    else:
        lines.append("未来 3 天无重要财经事件。")
        lines.append("")

    # 下一个重要事件
    next_event = report.get('下一个重要事件')
    if next_event:
        lines.extend([
            "### 下一个重要事件提醒",
            "",
            f"- **事件**: {next_event.get('事件', {}).get('title', '未知')}",
            f"- **日期**: {next_event.get('日期', '未知')}",
            f"- **距离**: {next_event.get('距离天数', 0)} 天",
            "",
        ])

    # 交易建议
    advice = report.get('交易建议', {})
    lines.extend([
        "---",
        "",
        "## 交易建议",
        "",
        f"- **总体建议**: {advice.get('总体建议', '观望')}",
        f"- **仓位建议**: {advice.get('仓位建议', '空仓')}",
        "",
    ])

    if advice.get('风险提示'):
        lines.append("### 风险提示")
        lines.append("")
        for risk in advice['风险提示']:
            lines.append(f"- {risk}")
        lines.append("")

    if advice.get('机会提示'):
        lines.append("### 机会提示")
        lines.append("")
        for opp in advice['机会提示']:
            lines.append(f"- {opp}")
        lines.append("")

    if advice.get('关键价位'):
        lines.extend([
            "### 关键价位",
            "",
        ])
        for key, value in advice['关键价位'].items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*本报告由 AI 自动生成，仅供参考，不构成投资建议。*",
        "*数据来源：MT5、Yahoo Finance、金十数据*",
    ])

    return "\n".join(lines)


def save_report(report: dict, format: str = "markdown") -> Path:
    """保存报告到文件。

    参数：
      - report: 报告字典
      - format: 格式，"markdown" 或 "json"

    返回：保存的文件路径
    """
    date_str = report["日期"].replace("-", "")

    if format == "markdown":
        content = format_report_markdown(report)
        path = REPORT_DIR / f"黄金早报_{date_str}.md"
        path.write_text(content, encoding="utf-8")
    else:
        path = REPORT_DIR / f"黄金早报_{date_str}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("报告已保存: %s", path)
    return path


def main() -> None:
    """主函数：生成并保存每日早报。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("开始生成每日黄金早报...")
    report = generate_daily_briefing()

    # 保存 Markdown 和 JSON 两种格式
    md_path = save_report(report, "markdown")
    json_path = save_report(report, "json")

    logger.info("早报生成完成: %s, %s", md_path, json_path)

    # 打印摘要
    print("\n" + "=" * 50)
    print("每日黄金早报摘要")
    print("=" * 50)
    print(f"日期: {report['日期']}")
    print(f"市场状态: {report['市场状态'].get('状态', '未知')}")
    print(f"总体建议: {report['交易建议'].get('总体建议', '观望')}")
    print(f"仓位建议: {report['交易建议'].get('仓位建议', '空仓')}")
    print(f"今日事件数: {len(report['今日事件'])}")
    print(f"未来3天事件数: {len(report['未来3天事件'])}")
    print("=" * 50)


if __name__ == "__main__":
    main()
