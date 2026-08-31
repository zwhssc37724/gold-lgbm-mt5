"""MT5 交易记录拉取：账户历史订单与成交明细（只取数，不做分析）。

数据来源（MetaTrader5 官方接口）：
- history_deals_get:  成交明细（每笔成交一行：开仓/平仓/手续费/掉期）
- history_orders_get: 历史订单（含已撤销/过期/已拒绝的挂单）

时间窗注意（2026-08-31 修复）：MT5 API 按「终端服务器时区」解释 naive
datetime（TradeMax 服务器 = UTC+2/3），所以用本地 naive 时间构造窗口即可
覆盖目标区间，不要用 tz-aware（会被 API 拒绝或错位）。为跨服务器稳妥起见，
查询窗口整体向前后各扩 48 小时，再在 Python 侧按真实时间戳过滤。

设计约定：
- 直接 import MetaTrader5 调用（与 mt5_client 的懒初始化无关，独立连接）
- MT5 不可用 → 抛 RuntimeError（MCP 层转成 {"错误": ...}）
- 输出中文键，原始语义保留：盈亏 = profit + commission + swap（已实现）
- 输出的时间列已是北京时间（mt5_client.to_beijing 转换，服务器钟 EET/EEST）；
  配对/持仓时长计算不受影响（同一时区内做差）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from gold_model import config
from gold_model.mt5_client import to_beijing, to_beijing_ts

logger = logging.getLogger("gold_model.trade_history")

# deal.type: 0=买 1=卖 2=买卖成交 3=红利 4=信用 5=余额 6=权益 7=手续费 8=税费
DEAL_TYPE_CN = {0: "买入", 1: "卖出", 2: "买卖成交", 3: "红利", 4: "信用",
                5: "出入金", 6: "权益", 7: "手续费", 8: "税费", 9: "掉期"}
# deal.entry: 0=入场 1=出场 2=反向 3=开平
DEAL_ENTRY_CN = {0: "开仓", 1: "平仓", 2: "反转", 3: "开平"}
# order.state: 1=已下单 2=已取消 3=已拒绝 4=过期 5=已成交 6=部分成交
ORDER_STATE_CN = {1: "已下单", 2: "已撤销", 3: "已拒绝", 4: "过期", 5: "已成交", 6: "部分成交"}
# order.type: 0=市价买 1=市价卖 2=限价买 3=限价卖 4=止损买 5=止损卖
ORDER_TYPE_CN = {0: "市价买", 1: "市价卖", 2: "限价买", 3: "限价卖", 4: "止损买", 5: "止损卖"}

# 服务器时区与本地时区的最大可能偏差（扩窗用）
_WINDOW_PAD = timedelta(hours=48)


def _resolve_range(days: int | None, period: str | None) -> tuple[datetime, datetime]:
    """把 (days, period) 解析成本地 naive 时间窗（MT5 按服务器时区解释）。"""
    now = datetime.now()
    if period == "今天":
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    if period == "本周":
        start = now - timedelta(days=now.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0), now
    if period == "本月":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now
    n = int(days) if days else 30
    return now - timedelta(days=n), now


def _fetch(callback, start: datetime, end: datetime):
    """带扩窗的 MT5 历史查询（naive datetime，服务器时区语义）。"""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 初始化失败：{mt5.last_error()}")
    try:
        raw = callback(start - _WINDOW_PAD, end + _WINDOW_PAD)
        return [] if raw is None else list(raw)
    finally:
        mt5.shutdown()


def get_deals(days: int | None = None, period: str | None = None,
              symbol: str | None = None) -> pd.DataFrame:
    """拉取成交明细，返回中文列 DataFrame。

    列：时间, 订单号, 品种, 类型, 方向(开/平仓), 手数, 价格, 止损, 止盈,
        手续费, 掉期, 盈亏(已实现,含费), 注释, 账户
    """
    start, end = _resolve_range(days, period)
    import MetaTrader5 as mt5

    raw = _fetch(mt5.history_deals_get, start, end)
    rows = []
    for d in raw:
        t = pd.Timestamp(d.time, unit="s")
        # 扩窗后的真实过滤（按本地时间语义，与请求窗一致）
        if not (start <= t.to_pydatetime() <= end + timedelta(days=1)):
            continue
        if symbol and symbol not in d.symbol:
            continue
        rows.append({
            "时间": to_beijing_ts(t),
            "订单号": d.order,
            "品种": d.symbol,
            "类型": DEAL_TYPE_CN.get(d.type, str(d.type)),
            "方向": DEAL_ENTRY_CN.get(d.entry, str(d.entry)),
            "手数": float(d.volume),
            "价格": float(d.price),
            "手续费": float(d.commission),
            "掉期": float(d.swap),
            "盈亏": round(float(d.profit) + float(d.commission) + float(d.swap), 2),
            "注释": d.comment,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("时间").reset_index(drop=True)


def get_orders(days: int | None = None, period: str | None = None,
               symbol: str | None = None) -> pd.DataFrame:
    """拉取历史订单（含已撤销/过期挂单），返回中文列 DataFrame。

    列：下单时间, 订单号, 品种, 订单类型, 手数, 价格, 止损, 止盈,
        状态, 完成时间, 注释
    """
    start, end = _resolve_range(days, period)
    import MetaTrader5 as mt5

    raw = _fetch(mt5.history_orders_get, start, end)
    rows = []
    for o in raw:
        t = pd.Timestamp(o.time_setup, unit="s")
        if not (start <= t.to_pydatetime() <= end + timedelta(days=1)):
            continue
        if symbol and symbol not in o.symbol:
            continue
        rows.append({
            "下单时间": to_beijing_ts(t),
            "订单号": o.ticket,
            "品种": o.symbol,
            "订单类型": ORDER_TYPE_CN.get(o.type, str(o.type)),
            "手数": float(o.volume_current),
            "价格": float(o.price_open) if o.price_open else None,
            "止损": float(o.sl) if o.sl else None,
            "止盈": float(o.tp) if o.tp else None,
            "状态": ORDER_STATE_CN.get(o.state, str(o.state)),
            "完成时间": to_beijing_ts(pd.Timestamp(o.time_done, unit="s")) if o.time_done else None,
            "注释": o.comment,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("下单时间").reset_index(drop=True)


def summarize(deals: pd.DataFrame) -> dict:
    """成交记录基础汇总（描述性统计，不含行为分析）。"""
    if deals.empty:
        return {"记录数": 0}
    closed = deals[deals["方向"] == "平仓"]
    entries = deals[deals["方向"] == "开仓"]
    profit = float(deals["盈亏"].sum())
    fees = float(deals["手续费"].sum() + deals["掉期"].sum())
    out = {
        "记录数": len(deals),
        "开仓笔数": len(entries),
        "平仓笔数": len(closed),
        "总盈亏": round(profit, 2),
        "总成本(手续费+掉期)": round(fees, 2),
        "时间范围": [str(deals["时间"].iloc[0]), str(deals["时间"].iloc[-1])],
        "品种分布": deals["品种"].value_counts().to_dict(),
    }
    if not closed.empty:
        wins = closed[closed["盈亏"] > 0]
        losses = closed[closed["盈亏"] <= 0]
        out["平仓盈亏分布"] = {
            "盈利笔数": len(wins),
            "亏损笔数": len(losses),
            "平均盈利": round(float(wins["盈亏"].mean()), 2) if not wins.empty else 0.0,
            "平均亏损": round(float(losses["盈亏"].mean()), 2) if not losses.empty else 0.0,
        }
    return out


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="拉取 MT5 交易记录（只取数）")
    parser.add_argument("--days", type=int, default=30, help="近 N 天（默认 30）")
    parser.add_argument("--period", choices=["今天", "本周", "本月"], help="快捷时间范围（优先于 days）")
    parser.add_argument("--symbol", default=None, help="过滤品种（默认全部）")
    parser.add_argument("--format", choices=["summary", "csv", "json"], default="summary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        deals = get_deals(days=args.days, period=args.period, symbol=args.symbol)
        orders = get_orders(days=args.days, period=args.period, symbol=args.symbol)
    except RuntimeError as e:
        print(json.dumps({"错误": str(e)}, ensure_ascii=False))
        return

    if args.format == "csv":
        deals.to_csv(config.DATA_DIR / "trade_deals.csv", index=False, encoding="utf-8-sig")
        orders.to_csv(config.DATA_DIR / "trade_orders.csv", index=False, encoding="utf-8-sig")
        print(f"已导出 {len(deals)} 笔成交 / {len(orders)} 笔订单 -> data/trade_*.csv")
    elif args.format == "json":
        payload = {"成交": deals.to_dict("records"), "订单": orders.to_dict("records")}
        (config.DATA_DIR / "trade_history.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print("已导出 -> data/trade_history.json")
    else:
        print(json.dumps(summarize(deals), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
