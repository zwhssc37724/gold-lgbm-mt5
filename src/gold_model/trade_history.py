"""MT5 交易记录拉取：账户历史订单与成交明细（只取数，不做分析）。

数据来源（MetaTrader5 官方接口）：
- history_deals_get:  成交明细（每笔成交一行：开仓/平仓/挂单成交/出入金/手续费）
- history_orders_get: 历史订单（含挂单撤销等未成交单）

时间范围支持：今天 / 本周 / 本月 / 近 N 天 / 全部（MT5 接口按时间窗查询）。

设计约定：
- MT5 不可用 → 返回 {"错误": ...}，绝不用合成数据冒充交易记录
- 输出中文键（与项目其他工具一致）
- 原始字段保留英文（deal/order 的官方字段名），附中文摘要字段
- 涉及金额的币种取账户币种（通常 USD）
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from gold_model import config
from gold_model import mt5_client as _mt5mod

logger = logging.getLogger("gold_model.trade_history")

# deal.type: 0=买 1=卖 2=买卖成交 3=红利 4=信用 5=余额(出入金) 6=权益 7=手续费 8=税费
DEAL_TYPE_CN = {
    0: "买入",
    1: "卖出",
    2: "买卖成交",
    3: "红利",
    4: "信用",
    5: "出入金",
    6: "权益",
    7: "手续费",
    8: "税费",
    9: "掉期",
}
# deal.entry: 0=入场 1=出场 2=反向 3=入出场(平仓)
DEAL_ENTRY_CN = {0: "开仓", 1: "平仓", 2: "反转", 3: "开平"}
# order.state: 1=已下单 2=已取消 3=已拒绝 5=已成交
ORDER_STATE_CN = {1: "已下单", 2: "已撤销", 3: "已拒绝", 4: "过期", 5: "已成交"}
# order.type: 0=市价买 1=市价卖 2=限价买 3=限价卖 4=止损买 5=止损卖
ORDER_TYPE_CN = {
    0: "市价买",
    1: "市价卖",
    2: "限价买",
    3: "限价卖",
    4: "止损买",
    5: "止损卖",
}


def _resolve_range(days: int | None, period: str | None) -> tuple[datetime, datetime]:
    """把 (days, period) 解析成 UTC 时间窗。默认近 30 天。"""
    now = datetime.now(UTC)
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


def _with_lock(fn):
    """MT5 调用必须在全局锁内（MetaTrader5 包非线程安全）。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _mt5mod._MT5_LOCK:
            mt5 = _mt5mod._mt5()
            if mt5 is None:
                raise RuntimeError("MT5 不可用：无法拉取交易记录")
            return fn(mt5, *args, **kwargs)

    return wrapper


@_with_lock
def get_deals(mt5, days: int | None = None, period: str | None = None,
              symbol: str | None = None) -> pd.DataFrame:
    """拉取成交明细（含开仓/平仓/手续费/掉期），返回 DataFrame。

    列（中文）：时间, 订单号, 品种, 类型, 方向(开/平仓), 手数, 价格, 止损, 止盈,
              手续费, 掉期, 盈亏, 注释, 账户
    """
    start, end = _resolve_range(days, period)
    kwargs = {"from": start, "to": end}
    if symbol:
        kwargs["group"] = f"*{symbol}*"
    raw = mt5.history_deals_get(**kwargs)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    rows = []
    for d in raw:
        rows.append({
            "时间": pd.Timestamp(d.time, unit="s", tz="UTC"),
            "订单号": int(d.order),
            "品种": d.symbol,
            "类型": DEAL_TYPE_CN.get(d.type, str(d.type)),
            "方向": DEAL_ENTRY_CN.get(d.entry, str(d.entry)),
            "手数": float(d.volume),
            "价格": float(d.price),
            "止损": float(d.sl) if d.sl else None,
            "止盈": float(d.tp) if d.tp else None,
            "手续费": float(d.commission),
            "掉期": float(d.swap),
            "盈亏": float(d.profit) + float(d.commission) + float(d.swap),
            "注释": d.comment,
            "账户": d.login,
        })
    return pd.DataFrame(rows).sort_values("时间").reset_index(drop=True)


@_with_lock
def get_orders(mt5, days: int | None = None, period: str | None = None,
               symbol: str | None = None) -> pd.DataFrame:
    """拉取历史订单（含已撤销/已拒绝的挂单），返回 DataFrame。

    列（中文）：下单时间, 订单号, 品种, 订单类型, 手数, 价格, 止损, 止盈,
              状态, 成交时间, 成交价, 注释
    """
    start, end = _resolve_range(days, period)
    kwargs = {"from": start, "to": end}
    if symbol:
        kwargs["group"] = f"*{symbol}*"
    raw = mt5.history_orders_get(**kwargs)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    rows = []
    for o in raw:
        rows.append({
            "下单时间": pd.Timestamp(o.time_setup, unit="s", tz="UTC"),
            "订单号": int(o.ticket),
            "品种": o.symbol,
            "订单类型": ORDER_TYPE_CN.get(o.type, str(o.type)),
            "手数": float(o.volume_current),
            "价格": float(o.price_open),
            "止损": float(o.sl) if o.sl else None,
            "止盈": float(o.tp) if o.tp else None,
            "状态": ORDER_STATE_CN.get(o.state, str(o.state)),
            "成交时间": pd.Timestamp(o.time_done, unit="s", tz="UTC") if o.time_done else None,
            "成交价": float(o.price_current) if o.price_current else None,
            "注释": o.comment,
        })
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
        "成交笔数": len(deals),
        "开仓笔数": len(entries),
        "平仓笔数": len(closed),
        "总盈亏": round(profit, 2),
        "总成本(手续费+掉期)": round(fees, 2),
        "时间范围": [str(deals["时间"].iloc[0]), str(deals["时间"].iloc[-1])],
        "品种分布": deals["品种"].value_counts().to_dict(),
    }
    if not closed.empty:
        out["平仓盈亏分布"] = {
            "盈利笔数": int((closed["盈亏"] > 0).sum()),
            "亏损笔数": int((closed["盈亏"] <= 0).sum()),
            "平均盈利": round(float(closed.loc[closed["盈亏"] > 0, "盈亏"].mean() or 0), 2),
            "平均亏损": round(float(closed.loc[closed["盈亏"] <= 0, "盈亏"].mean() or 0), 2),
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
