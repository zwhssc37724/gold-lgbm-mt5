"""HTTP MCP 服务，对外提供黄金交易模型预测能力。

启动方式（在工程根目录）：
    uv run gold-mcp

服务地址（streamable-http）：http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import logging
import math

import joblib
import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from gold_model import (
    calibration,
    cftc,
    config,
    gld_holdings,
    mt5_client,
    trade_history,
)
from gold_model import drift as drift_mod
from gold_model.features import DIRECTION3_NAMES_CN, build_features
from gold_model.ledger import record_prediction

logger = logging.getLogger("gold_model.mcp")

mcp = MCPServer("gold-trading-model")

# 输出标签中英文映射（纯语义翻译，不含操作建议——分析由上层 agent 负责）
BREAKOUT_SIGNAL_CN = {
    "EXPECT_EXPANSION": "预期扩张",
    "EXPECT_COMPRESSION": "预期收敛",
    "NEUTRAL": "中性",
}
KLINE_FIELD_CN = {
    "time": "时间",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "tick_volume": "成交量",
    "spread": "点差",
}

# 时间范围 → 交易日数（现货黄金每周约 5 个交易日，每月约 22 个交易日）
RANGE_TRADING_DAYS = {
    "一天": 1,
    "一周": 5,
    "一个月": 22,
    "三个月": 66,
    "半年": 132,
    "一年": 264,
}


def _bars_for_range(timeframe: str, 时间范围: str) -> int:
    """按时间范围与 K 线周期换算需要拉取的 K 线根数。"""
    days = RANGE_TRADING_DAYS[时间范围]
    minutes = mt5_client.TIMEFRAMES.get(timeframe, ("H1", 60))[1]
    return int(max(10, min(math.ceil(days * 1440 / minutes), config.BARS)))


_MODEL_CACHE: dict[str, tuple[float, object]] = {}


def _load_model(path_key: str):
    """加载模型并按文件 mtime 自动失效（重新训练后无需重启服务）。"""
    paths = {
        "breakout": config.BREAKOUT_MODEL_PATH,
        "breakout_m15": config.BREAKOUT_M15_MODEL_PATH,
        "direction3": config.DIRECTION3_MODEL_PATH,
        "direction_d1": config.DIRECTION_D1_MODEL_PATH,
    }
    path = paths[path_key]
    if not path.exists():
        raise FileNotFoundError(
            f"模型未找到：{path}。请先运行 `uv run gold-train --target {path_key}`"
        )
    mtime = path.stat().st_mtime
    cached = _MODEL_CACHE.get(path_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    bundle = joblib.load(path)
    _MODEL_CACHE[path_key] = (mtime, bundle)
    logger.info("model loaded (%s, mtime=%.0f)", path_key, mtime)
    return bundle


def _build_input_row(symbol: str, timeframe: str, bundle: dict) -> pd.DataFrame:
    needed = 300  # 滚动特征需要的历史长度
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=needed)
    if mt5_client.is_synthetic(df):
        raise RuntimeError(
            "MT5 数据不可用，当前为合成数据——拒绝预测。请检查 MT5 终端连接后再试。"
        )
    X_price = build_features(df)
    # 宏观特征（与训练一致；拉取失败时退化为 0 值列）
    from gold_model import macro_features

    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
    feats = [c for c in bundle["features"] if c in X.columns]
    row = X[feats].iloc[-1:].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return row, df


def _add_market_status(result: dict) -> dict:
    """给结果添加市场状态字段。"""
    result["市场状态"] = mt5_client.is_market_open()
    return result


def _get_atr(df: pd.DataFrame, period: int = 14) -> float:
    """计算 ATR（平均真实波幅）。"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def _get_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """计算近期支撑阻力位。"""
    recent = df.tail(window)
    return {
        f"近{window}根支撑": round(float(recent["low"].min()), 2),
        f"近{window}根阻力": round(float(recent["high"].max()), 2),
        f"近{window}根中轴": round(float((recent["high"].max() + recent["low"].min()) / 2), 2),
    }


@mcp.tool()
def get_quote(symbol: str = config.SYMBOL) -> dict:
    """获取某品种的 MT5 实时报价（买价/卖价/最新价）。"""
    return _add_market_status(mt5_client.get_quote(symbol))


@mcp.tool()
def get_market_status() -> dict:
    """获取当前市场状态（开市/休市/周末）。"""
    return mt5_client.is_market_open()


@mcp.tool()
def get_klines(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    bars: int = 500,
    范围: str = "",
    限制: int = 500,
) -> dict:
    """获取某品种的 MT5 K 线数据（OHLCV）。

    参数：
      - timeframe：K 线周期，M1/M5/M15/M30/H1/H4/D1
      - bars：直接指定返回的 K 线根数（10..50000）
      - 范围：按时间范围获取，可选 一天/一周/一个月/三个月/半年/一年。
        指定后忽略 bars，自动按周期换算 K 线根数
        （一周 = 近 5 个交易日；例如 H1 一周 ≈ 120 根，M15 一周 ≈ 480 根）。
      - 限制：最多返回的 K 线根数（默认 500，防止数据量过大）。
        如需完整数据，请分批次调用或使用更大的限制值。

    范围 与 bars 都不传时，默认返回最近 500 根。
    """
    if 范围:
        if 范围 not in RANGE_TRADING_DAYS:
            raise ValueError(f"不支持的范围「{范围}」，可选：{'、'.join(RANGE_TRADING_DAYS)}")
        bars = _bars_for_range(timeframe, 范围)
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=bars)

    # 限制返回数量，防止 token 爆炸
    total = len(df)
    if total > 限制:
        df = df.tail(限制)
        truncated = True
    else:
        truncated = False

    records = _json_records(df)
    return {
        "数据": records,
        "统计": {
            "总根数": total,
            "返回根数": len(records),
            "是否截断": truncated,
            "最新时间": records[-1]["时间"] if records else None,
            "最旧时间": records[0]["时间"] if records else None,
        },
        "市场状态": mt5_client.is_market_open(),
    }


@mcp.tool()
def get_klines_summary(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """获取 K 线数据摘要（不返回原始数据，只返回统计信息）。

    适合快速了解市场状态而不消耗大量 token。
    """
    df = mt5_client.get_klines(symbol=symbol, timeframe=timeframe, bars=100)
    if df.empty:
        return {"错误": "无数据"}

    latest = df.iloc[-1]
    prev_24 = df.tail(24)

    return {
        "品种": symbol,
        "周期": timeframe,
        "最新价格": round(float(latest["close"]), 2),
        "最新时间": latest["time"].isoformat(),
        "24小时最高": round(float(prev_24["high"].max()), 2),
        "24小时最低": round(float(prev_24["low"].min()), 2),
        "24小时涨跌": round(float((latest["close"] / prev_24.iloc[0]["open"] - 1) * 100), 2),
        "24小时成交量": int(prev_24["tick_volume"].sum()),
        "ATR14": round(_get_atr(df), 2),
        "支撑阻力": _get_support_resistance(df),
        "市场状态": mt5_client.is_market_open(),
    }


@mcp.tool()
def predict(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """波动扩张预测（二分类）：预测下一根 K 线振幅是否突破近 100 根中位数。

    纯数据输出（不含操作建议，分析由上层 agent 负责）：
      - 扩张概率：模型输出 0~1
      - 信号类别：预期扩张 / 预期收敛 / 中性
      - 最新收盘价、实时报价、K线时间、模型、ATR14、支撑阻力
      - 市场状态
    """
    bundle = _load_model("breakout")
    row, df = _build_input_row(symbol, timeframe, bundle)
    proba = float(bundle["model"].predict(row)[0])
    raw_signal = "EXPECT_EXPANSION" if proba >= 0.6 else "EXPECT_COMPRESSION" if proba <= 0.4 else "NEUTRAL"
    quote = mt5_client.get_quote(symbol)
    current_price = float(df["close"].iloc[-1])
    atr = _get_atr(df)

    result = {
        "品种": symbol,
        "周期": timeframe,
        "目标": "波动扩张",
        "扩张概率": round(proba, 4),
        "信号": BREAKOUT_SIGNAL_CN[raw_signal],
        "最新收盘价": round(current_price, 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（二分类）",
        "ATR14": round(atr, 2),
        "支撑阻力": _get_support_resistance(df),
    }

    _add_market_status(result)
    record_prediction("breakout", result)
    return result


@mcp.tool()
def predict_m15(symbol: str = config.SYMBOL) -> dict:
    """M15 波动扩张预测（二分类）：预测下一根 15 分钟 K 线振幅是否突破近 100 根中位数。

    与 H1 版 predict 同任务，但周期是 M15——**预警提前约 45 分钟**。
    需要 `uv run gold-train --target breakout_m15` 先训练。
    纯数据输出（同 predict，分析由上层 agent 负责）。
    """
    bundle = _load_model("breakout_m15")
    row, df = _build_input_row(symbol, "M15", bundle)
    proba = float(bundle["model"].predict(row)[0])
    raw_signal = "EXPECT_EXPANSION" if proba >= 0.6 else "EXPECT_COMPRESSION" if proba <= 0.4 else "NEUTRAL"
    quote = mt5_client.get_quote(symbol)
    current_price = float(df["close"].iloc[-1])
    atr = _get_atr(df)

    result = {
        "品种": symbol,
        "周期": "M15",
        "目标": "波动扩张（提前预警版）",
        "扩张概率": round(proba, 4),
        "信号": BREAKOUT_SIGNAL_CN[raw_signal],
        "最新收盘价": round(current_price, 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（二分类，M15）",
        "ATR14": round(atr, 2),
        "支撑阻力": _get_support_resistance(df),
    }

    _add_market_status(result)
    record_prediction("breakout_m15", result)
    return result


@mcp.tool()
def predict_direction_3class(symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME) -> dict:
    """方向三分类预测：未来 24 根 H1 K 线（约 1 个交易日）的方向 — 看空 / 观望 / 看多。

    标签定义（按未来 24 根对数收益率，自适应阈值）：
      - 0 看空 / 1 观望 / 2 看多

    纯数据输出（不含操作建议，分析由上层 agent 负责）：
      - 看空概率、观望概率、看多概率（三类分布）
      - 预测类别、看多减看空置信度
      - 校准概率（isotonic，历史同置信度下的真实方向频率）
      - 最新收盘价、实时报价、K线时间、模型、ATR14、支撑阻力
      - 市场状态
    """
    bundle = _load_model("direction3")
    row, df = _build_input_row(symbol, timeframe, bundle)
    proba_vec = bundle["model"].predict(row)[0]  # shape (3,)
    proba_vec = np.asarray(proba_vec, dtype=float)
    pred_class = int(np.argmax(proba_vec))
    quote = mt5_client.get_quote(symbol)
    confidence_long_short = float(proba_vec[2] - proba_vec[0])  # 看多减看空
    current_price = float(df["close"].iloc[-1])
    atr = _get_atr(df)

    result = {
        "品种": symbol,
        "周期": timeframe,
        "目标": "方向三分类（看空/观望/看多）",
        "看空概率": round(float(proba_vec[0]), 4),
        "观望概率": round(float(proba_vec[1]), 4),
        "看多概率": round(float(proba_vec[2]), 4),
        "预测类别": DIRECTION3_NAMES_CN[pred_class],
        "看多减看空置信度": round(confidence_long_short, 4),
        "最新收盘价": round(current_price, 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（三分类）",
        "预测周期": f"未来 {bundle.get('horizon', 24)} 根 K 线（{bundle.get('horizon', 24) * 60} 分钟）",
        "分类阈值": (
            "自适应（ATR24 中位数×1.0）"
            if bundle.get("adaptive_threshold")
            else f"±{float(bundle.get('threshold') or 0.003) * 100:.2f}%"
        ),
        "ATR14": round(atr, 2),
        "支撑阻力": _get_support_resistance(df),
    }

    # 校准概率（isotonic 校准器存在时）：把原始置信度翻译成真实频率
    try:
        calib = calibration.load_calibrators(config.DIRECTION3_MODEL_PATH)
        if calib is not None:
            calibrated = calibration.calibrate_confidence(calib, confidence_long_short)
            result["校准概率"] = calibrated
            result["校准方法"] = "isotonic，WF 样本外拟合"
    except Exception as exc:  # 校准失败不影响预测
        logger.warning("calibration lookup failed: %s", exc)

    _add_market_status(result)
    record_prediction("direction3", result)
    return result


@mcp.tool()
def predict_direction_d1(symbol: str = config.SYMBOL) -> dict:
    """日线方向三分类预测：未来 5 个交易日的方向 — 看空 / 观望 / 看多。

    模型定位（实验结论，供上层 agent 参考的事实）：
      - WF 宏 AUC 0.559（5 窗口全 >0.5），但独立交易无效（≈黄金多头 beta）
      - 实验支持其作为 H1 高置信信号的过滤器使用

    纯数据输出（不含操作建议）：
      - 看空/观望/看多概率、预测类别、看多减看空置信度
      - 最新收盘价、实时报价、K线时间、模型、ATR14、支撑阻力（近 20 日）
      - 市场状态
    """
    bundle = _load_model("direction_d1")
    row, df = _build_input_row(symbol, "D1", bundle)
    proba_vec = np.asarray(bundle["model"].predict(row)[0], dtype=float)
    pred_class = int(np.argmax(proba_vec))
    quote = mt5_client.get_quote(symbol)
    confidence = float(proba_vec[2] - proba_vec[0])
    current_price = float(df["close"].iloc[-1])
    atr = _get_atr(df)

    result = {
        "品种": symbol,
        "周期": "D1",
        "目标": "方向三分类（未来 5 个交易日）",
        "看空概率": round(float(proba_vec[0]), 4),
        "观望概率": round(float(proba_vec[1]), 4),
        "看多概率": round(float(proba_vec[2]), 4),
        "预测类别": DIRECTION3_NAMES_CN[pred_class],
        "看多减看空置信度": round(confidence, 4),
        "最新收盘价": round(current_price, 2),
        "实时报价": quote,
        "K线时间": str(df["time"].iloc[-1]),
        "模型": "LightGBM + Optuna（D1 三分类）",
        "预测周期": f"未来 {bundle.get('horizon', 5)} 个交易日",
        "分类阈值": (
            "自适应（ATR24 中位数×1.0）"
            if bundle.get("adaptive_threshold")
            else f"±{float(bundle.get('threshold') or 0.003) * 100:.2f}%"
        ),
        "ATR14": round(atr, 2),
        "支撑阻力": _get_support_resistance(df, window=20),
    }

    _add_market_status(result)
    record_prediction("direction_d1", result)
    return result


def _json_records(df: pd.DataFrame) -> list[dict]:
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return v

    return [{KLINE_FIELD_CN.get(k, k): _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]


# ---------------------------------------------------------------------------
# 独家数据工具（无外部 MCP 等价物）
# ---------------------------------------------------------------------------

@mcp.tool()
def get_cftc_position() -> dict:
    """获取 CFTC 黄金持仓报告。

    CFTC 每周五公布对冲基金等大投机者的黄金期货持仓，
    是判断黄金中长期趋势的重要领先指标。

    返回：
      - 报告日期
      - 非商业净多头（关键指标）
      - 净多头变化
      - 情绪判断
    """
    return cftc.get_cftc_summary()


@mcp.tool()
def get_gld_holdings() -> dict:
    """获取 SPDR GLD 黄金 ETF 持仓量。

    GLD 是全球最大黄金 ETF，持仓量变化反映机构资金对黄金的态度。

    返回：
      - 持仓量（吨）
      - 持仓量（盎司）
      - 日期
      - 数据来源
    """
    return gld_holdings.get_gld_summary()


@mcp.tool()
def get_trade_history(days: int = 30, period: str = "", symbol: str = "", limit: int = 200) -> dict:
    """获取 MT5 账户交易记录（成交明细 + 订单，只取数不含分析）。

    参数：
      - days: 近 N 天（默认 30）
      - period: 快捷范围「今天/本周/本月」（指定后忽略 days）
      - symbol: 过滤品种（空 = 全部品种）
      - limit: 返回明细条数上限（默认 200，防止 token 爆炸）

    返回：
      - 汇总: 笔数/盈亏/成本/品种分布/盈亏分布
      - 成交明细: 时间/订单号/品种/类型/开平仓方向/手数/价格/手续费/掉期/盈亏/注释
      - 订单明细: 下单时间/类型/状态（含已撤销/过期挂单）/止损止盈/完成时间
    """
    try:
        deals = trade_history.get_deals(
            days=days, period=period or None, symbol=symbol or None
        )
        orders = trade_history.get_orders(
            days=days, period=period or None, symbol=symbol or None
        )
    except RuntimeError as e:
        return {"错误": str(e)}

    def _records(df: pd.DataFrame, n: int) -> list[dict]:
        if df.empty:
            return []
        recent = df.tail(n)
        return [
            {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
             for k, v in rec.items()}
            for rec in recent.to_dict("records")
        ]

    return {
        "汇总": trade_history.summarize(deals),
        "成交明细": _records(deals, limit),
        "订单明细": _records(orders, limit),
    }


@mcp.tool()
def check_drift(target: str = "direction3", bars: int = 500) -> dict:
    """特征漂移检查（PSI）：近期真实行情 vs 训练时的特征分布。

    模型悄悄失效不会报警——这个工具就是报警器。每次重训会自动保存训练特征
    的分位数参考（models/drift_reference_<target>.json），此工具对比近期数据。

    参数：
      - target: direction3 / direction_d1 / breakout
      - bars: 近期 K 线根数（默认 500）

    返回：结论（稳定/轻度漂移/显著漂移）、Top10 漂移特征、PSI 阈值。
    PSI > 0.25 的特征多时建议重训模型。
    """
    if target not in ("breakout", "breakout_m15", "direction3", "direction_d1"):
        raise ValueError("target 仅支持 breakout / breakout_m15 / direction3 / direction_d1")
    return drift_mod.check_drift(target=target, bars=bars)


@mcp.tool()
def get_comprehensive_analysis() -> dict:
    """获取黄金综合分析（本项目独家数据汇总）。

    返回：
      - MT5 模型预测
      - CFTC 持仓
      - GLD 持仓
      - 市场状态

    宏观数据/快讯/事件等已由外部 MCP 提供（jin10、Alpha Vantage 等），
    此工具只聚合本项目独家的数据源。
    """
    # MT5 模型预测
    mt5_predictions = {}
    try:
        mt5_predictions["波动扩张"] = predict()
        mt5_predictions["方向三分类"] = predict_direction_3class()
    except Exception as e:
        mt5_predictions["错误"] = str(e)

    return {
        "MT5模型预测": mt5_predictions,
        "CFTC持仓": cftc.get_cftc_summary(),
        "GLD持仓": gld_holdings.get_gld_summary(),
        "市场状态": mt5_client.is_market_open(),
        "时间": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("MCP 服务已启动，地址 http://%s:%d%s", config.MCP_HOST, config.MCP_PORT, config.MCP_PATH)
    mcp.run(
        transport="streamable-http",
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        streamable_http_path=config.MCP_PATH,
    )


if __name__ == "__main__":
    main()

