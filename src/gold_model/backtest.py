"""向量化回测器：验证「信号 → 开仓 → ATR 止损/止盈」在历史上的真实表现。

设计要点：
- 逐 K 线模拟持仓（非完全向量化，但逻辑简单可靠，2 万根 H1 秒级跑完）。
- 成本建模：点差（spread 点，XAUUSD 1 点 = 0.01 美元）+ 滑点。金价 ~4000 美元
  时代点差普遍 20-35 点（0.20-0.35 美元），默认 25 点偏保守。
- 出场规则：ATR 止损 / ATR 止盈 / 最大持仓时长（K 线数）。
- 同时跑基线策略（波动率动量、简单均线、买入持有）做对照，模型必须赢基线。

用法：
    uv run gold-backtest --model breakout
    uv run gold-backtest --model direction3
    uv run gold-backtest --model none        # 只跑基线
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from gold_model import config, macro_features, mt5_client
from gold_model.features import build_features

logger = logging.getLogger("gold_model.backtest")

# 回测参数（保守默认）
SPREAD_POINTS = 25          # 点差（点），1 点 = 0.01 USD
SLIPPAGE_POINTS = 5         # 每次成交滑点（点）
STOP_ATR_MULT = 2.0         # 止损 = 2×ATR(14)
TP_ATR_MULT = 3.0           # 止盈 = 3×ATR(14)
MAX_HOLD_BARS = 24          # 最长持仓（H1 根数，direction3 对应其预测视野）
INITIAL_CAPITAL = 10_000.0
POSITION_SIZE_PCT = 0.10    # 每次开仓动用资金比例（10%，无杠杆）

# 隔夜利息（swap）：XAUUSD 多头典型 -2% ~ -4% 年化（借美元买金），取保守 -3.5%；
# 空头收窄到 -1.0%（经纪商点差后多数也为负）。按持仓跨自然日数计。
SWAP_ANNUAL_LONG = -0.035
SWAP_ANNUAL_SHORT = -0.010

POINT = 0.01  # XAUUSD 1 点


def _cost_per_trade(price: float) -> float:
    """单边成本（点差+滑点）折算成价格单位。"""
    return (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT


def _swap_cost(direction: str, entry_px: float, size: float, days_held: int) -> float:
    """隔夜利息成本（负=扣钱）。按自然日数计，价格单位。"""
    if days_held <= 0:
        return 0.0
    rate = SWAP_ANNUAL_LONG if direction == "long" else SWAP_ANNUAL_SHORT
    return rate / 365.0 * days_held * entry_px * size


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def load_price_data(use_snapshot: bool = True) -> pd.DataFrame:
    """加载干净 H1 数据（优先快照）。"""
    if use_snapshot and config.DATA_SNAPSHOT.exists():
        df = pd.read_parquet(config.DATA_SNAPSHOT)
    else:
        df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, bars=config.BARS)
        if mt5_client.is_synthetic(df):
            raise RuntimeError("MT5 不可用且无快照：拒绝用合成数据回测。")
        df = mt5_client.filter_dense_history(df, timeframe=config.TIMEFRAME).reset_index(drop=True)
    return df.reset_index(drop=True)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    direction: str = "long",
    stop_atr: float = STOP_ATR_MULT,
    tp_atr: float = TP_ATR_MULT,
    max_hold: int = MAX_HOLD_BARS,
) -> dict:
    """在 H1 数据上模拟一个方向性策略。

    signals: 布尔序列（True=该 K 线收盘时开仓）。
    direction: "long" / "short"。
    返回统计 dict + 逐笔交易 DataFrame。
    """
    if len(signals) != len(df):
        raise ValueError("signals 长度与 df 不一致")
    atr = _atr(df).to_numpy()
    o, h, l, c = (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    cost = _cost_per_trade(c)
    sig = signals.fillna(False).to_numpy(dtype=bool)

    trades: list[dict] = []
    equity = INITIAL_CAPITAL
    equity_curve = np.full(len(df), np.nan)
    in_pos = False
    entry_i = -1
    entry_px = stop_px = tp_px = 0.0
    size = 0.0

    def _close(i: int, px: float, reason: str):
        nonlocal equity, in_pos, size
        if direction == "long":
            pnl = (px - entry_px) * size
        else:
            pnl = (entry_px - px) * size
        # 隔夜利息：按持仓跨自然日数计
        days_held = int((df["time"].iloc[i] - df["time"].iloc[entry_i]).total_seconds() // 86400)
        swap = _swap_cost(direction, entry_px, size, days_held)
        pnl += swap
        equity += pnl
        trades.append(
            {
                "开仓K线": entry_i,
                "平仓K线": i,
                "持仓根数": i - entry_i,
                "持仓天数": days_held,
                "开仓价": round(entry_px, 2),
                "平仓价": round(px, 2),
                "盈亏": round(pnl, 2),
                "其中隔夜利息": round(swap, 2),
                "收益率": round(pnl / equity if equity else 0.0, 6),
                "原因": reason,
            }
        )
        in_pos = False

    for i in range(len(df)):
        if in_pos:
            # 先查止损/止盈（用当根高低点，保守：同根先触发止损）
            if direction == "long":
                if l[i] <= stop_px:
                    _close(i, stop_px, "止损")
                elif h[i] >= tp_px:
                    _close(i, tp_px, "止盈")
            else:
                if h[i] >= stop_px:
                    _close(i, stop_px, "止损")
                elif l[i] <= tp_px:
                    _close(i, tp_px, "止盈")
            if in_pos and i - entry_i >= max_hold:
                _close(i, c[i], "到期平仓")
            if in_pos:
                equity_curve[i] = equity + _mark_to_market(i, direction, entry_px, size, c)
                continue

        # 空仓且触发信号：下一根开盘价成交（避免未来函数）
        if sig[i] and i + 1 < len(df) and np.isfinite(atr[i]) and atr[i] > 0:
            entry_px = o[i + 1]
            size = equity * POSITION_SIZE_PCT / entry_px
            if direction == "long":
                stop_px = entry_px - stop_atr * atr[i]
                tp_px = entry_px + tp_atr * atr[i]
            else:
                stop_px = entry_px + stop_atr * atr[i]
                tp_px = entry_px - tp_atr * atr[i]
            entry_i = i + 1
            in_pos = True
            equity -= cost * size  # 开仓成本
            # 开仓当根也检查止损止盈
            j = i + 1
            if direction == "long":
                if l[j] <= stop_px:
                    _close(j, stop_px, "止损")
                elif h[j] >= tp_px:
                    _close(j, tp_px, "止盈")
            else:
                if h[j] >= stop_px:
                    _close(j, stop_px, "止损")
                elif l[j] <= tp_px:
                    _close(j, tp_px, "止盈")
            if in_pos:
                equity_curve[i + 1] = equity + _mark_to_market(j, direction, entry_px, size, c)
        else:
            equity_curve[i] = equity

    trades_df = pd.DataFrame(trades)
    curve = pd.Series(equity_curve, index=df["time"]).ffill().fillna(INITIAL_CAPITAL)

    stats = _summarize(trades_df, curve)
    return {"统计": stats, "交易": trades_df, "净值曲线": curve}


def _mark_to_market(i: int, direction: str, entry_px: float, size: float, c) -> float:
    px = c[i]
    return (px - entry_px) * size if direction == "long" else (entry_px - px) * size


def _summarize(trades: pd.DataFrame, curve: pd.Series) -> dict:
    if trades.empty:
        return {"交易笔数": 0, "总收益率%": 0.0, "年化收益%": 0.0, "最大回撤%": 0.0, "夏普": 0.0,
                "胜率": 0.0, "盈亏比": 0.0}
    pnl = trades["盈亏"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    total_ret = float(curve.iloc[-1] / INITIAL_CAPITAL - 1)
    days = max((curve.index[-1] - curve.index[0]).total_seconds() / 86400, 1)
    ann = (1 + total_ret) ** (365 / days) - 1 if total_ret > -1 else -1.0
    dd = float((curve / curve.cummax() - 1).min())
    ret = curve.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(365 * 24)) if ret.std() > 0 else 0.0
    return {
        "交易笔数": len(trades),
        "总收益率%": round(total_ret * 100, 2),
        "年化收益%": round(ann * 100, 2),
        "最大回撤%": round(dd * 100, 2),
        "夏普(小时→年化)": round(sharpe, 2),
        "胜率": round(float((pnl > 0).mean()), 4),
        "平均盈亏比": round(float(wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else 0.0, 2),
        "平均持仓根数": round(float(trades["持仓根数"].mean()), 1),
    }


# ---------------------------------------------------------------------------
# 信号生成：模型 vs 基线
# ---------------------------------------------------------------------------

def random_control_signals(signals: pd.Series, seed: int) -> pd.Series:
    """随机信号对照：与输入信号同频率、随机位置（真信号必须跑赢它的分布）。"""
    rng = np.random.default_rng(seed)
    freq = float(signals.fillna(False).mean())
    return pd.Series(rng.random(len(signals)) < freq, index=signals.index)


def model_signal(df: pd.DataFrame, model_key: str, threshold: float | None = None) -> tuple[pd.Series, str]:
    """用已训练模型对整段历史生成信号（样本内，供管道验证；样本外结论以 walk-forward 为准）。"""
    import joblib

    path = config.BREAKOUT_MODEL_PATH if model_key == "breakout" else config.DIRECTION3_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"模型不存在：{path}")
    bundle = joblib.load(path)
    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
    feats = [c for c in bundle["features"] if c in X.columns]
    X = X[feats].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if model_key == "breakout":
        proba = bundle["model"].predict(X)
        thr = threshold if threshold is not None else 0.6
        sig = pd.Series(proba >= thr, index=df.index)
        return sig, f"breakout 扩张概率≥{thr}（方向中性，用作择时过滤）"

    proba = np.asarray(bundle["model"].predict(X))
    if proba.ndim == 2:
        pred = proba.argmax(axis=1)
        long_sig = pd.Series(pred == 2, index=df.index)
        short_sig = pd.Series(pred == 0, index=df.index)
        return (long_sig, short_sig), "direction3 预测类别"
    raise ValueError("unexpected model output")


def baseline_signals(df: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    """基线策略集合。"""
    c = df["close"]
    sig = {}
    # 1. 均线交叉（20/50）
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    cross_up = (ma20 > ma50) & (ma20.shift() <= ma50.shift())
    sig["均线交叉多头"] = (cross_up.fillna(False), "MA20 上穿 MA50 开多")
    # 2. 波动率动量（hl_range > 近100中位数 → 波动扩张，同 breakout 基线）
    rng = df["high"] - df["low"]
    expand = rng > rng.rolling(100).median()
    sig["波动扩张"] = (expand.fillna(False), "当根振幅>近100根中位数")
    return sig


def buy_and_hold(df: pd.DataFrame) -> dict:
    """买入持有基准（含点差成本）。"""
    c = df["close"].to_numpy()
    entry = c[0] + _cost_per_trade(c[0]) / POINT * POINT
    size = INITIAL_CAPITAL / entry
    curve = pd.Series(c * size, index=df["time"])
    return {"统计": _summarize(pd.DataFrame([{"盈亏": 0.0, "持仓根数": len(df)}]), curve), "净值曲线": curve}


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------

def compare_strategies(use_snapshot: bool = True, models: list[str] | None = None) -> dict:
    df = load_price_data(use_snapshot=use_snapshot)
    logger.info("回测数据：%d 根 H1，%s ~ %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
    results: dict[str, dict] = {}

    # 买入持有
    bh = buy_and_hold(df)
    results["买入持有"] = bh["统计"]

    # 基线策略
    for name, (sig, desc) in baseline_signals(df).items():
        r = run_backtest(df, sig, direction="long")
        results[name] = r["统计"]

    # 模型策略
    for mk in models or []:
        try:
            if mk == "breakout":
                sig, desc = model_signal(df, "breakout")
                r = run_backtest(df, sig, direction="long")
                results[f"模型:{desc}"] = r["统计"]
                # 随机对照（同频率）：模型夏普必须显著高于随机对照分布才算择时有效
                rand_sharpes = []
                for seed in (1, 2, 3):
                    rc = run_backtest(df, random_control_signals(sig, seed), direction="long")
                    rand_sharpes.append(rc["统计"]["夏普(小时→年化)"])
                results["模型:随机对照(同频率)"] = {
                    "夏普(小时→年化)": round(float(np.mean(rand_sharpes)), 2),
                    "夏普范围": [round(float(min(rand_sharpes)), 2), round(float(max(rand_sharpes)), 2)],
                    "说明": "同频率随机信号均值（3种子）；模型须显著高于此",
                }
            elif mk == "direction3":
                (long_sig, short_sig), desc = model_signal(df, "direction3")
                r_long = run_backtest(df, long_sig, direction="long")
                results["模型:direction3 做多"] = r_long["统计"]
                r_short = run_backtest(df, short_sig, direction="short")
                results["模型:direction3 做空"] = r_short["统计"]
                rand_sharpes = []
                for seed in (1, 2, 3):
                    rc = run_backtest(df, random_control_signals(long_sig, seed), direction="long")
                    rand_sharpes.append(rc["统计"]["夏普(小时→年化)"])
                results["模型:随机对照(做多,同频率)"] = {
                    "夏普(小时→年化)": round(float(np.mean(rand_sharpes)), 2),
                    "夏普范围": [round(float(min(rand_sharpes)), 2), round(float(max(rand_sharpes)), 2)],
                    "说明": "同频率随机信号均值（3种子）；模型须显著高于此",
                }
        except FileNotFoundError as e:
            logger.warning("跳过 %s：%s", mk, e)

    return {"数据范围": [str(df["time"].iloc[0]), str(df["time"].iloc[-1])],
            "根数": len(df),
            "成本假设": f"点差 {SPREAD_POINTS} 点 + 滑点 {SLIPPAGE_POINTS} 点/边",
            "结果": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金策略向量化回测")
    parser.add_argument("--model", default="none", help="none / breakout / direction3 / all")
    parser.add_argument("--no-snapshot", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models = {"breakout": ["breakout"], "direction3": ["direction3"], "all": ["breakout", "direction3"]}.get(args.model, [])
    report = compare_strategies(use_snapshot=not args.no_snapshot, models=models)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = config.DATA_DIR / "reports" / "backtest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"报告已保存 -> {out}")


if __name__ == "__main__":
    main()
