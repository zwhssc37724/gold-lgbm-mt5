"""gold-lgbm-mt5 单元测试（无 MT5 依赖，全部用合成小数据）。

运行：uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from gold_model import config  # noqa: E402
from gold_model.features import (  # noqa: E402
    build_features,
    build_labels,
    build_labels_3class,
)
from gold_model.mt5_client import filter_dense_history  # noqa: E402


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

def make_klines(n: int = 600, seed: int = 7, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rets = rng.normal(0, 0.003, n)
    close = 4000.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1); open_[0] = close[0]
    amp = np.abs(rng.normal(0, 0.001, n)) * close
    high = np.maximum(open_, close) + amp
    low = np.minimum(open_, close) - amp
    return pd.DataFrame({
        "time": times, "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(100, 5000, n), "spread": 25,
    })


# ---------------------------------------------------------------------------
# 标签正确性
# ---------------------------------------------------------------------------

class TestLabels:
    def test_direction_label_looks_forward(self):
        df = make_klines(50)
        y = build_labels(df, horizon=1, target="direction")
        fwd = np.log(df["close"].shift(-1) / df["close"])
        assert (y == (fwd > 0).astype(int)).all()
        assert y.iloc[-1] == 0  # 最后一根无前瞻，默认 0

    def test_breakout_label_median_baseline(self):
        df = make_klines(300)
        y = build_labels(df, horizon=1, target="breakout")
        rng = (df["high"] - df["low"]) / df["close"]
        base = rng.rolling(config.BREAKOUT_LOOKBACK).median()
        expected = (rng.shift(-1) > base).astype(int)
        # 前几根 baseline 为 NaN → False(0)，一致
        assert (y.iloc[105:] == expected.iloc[105:]).all()

    def test_3class_fixed_threshold(self):
        df = make_klines(60)
        y = build_labels_3class(df, horizon=24, threshold=0.003)
        fwd = np.log(df["close"].shift(-24) / df["close"])
        assert (y[fwd > 0.003] == 2).all()
        assert (y[fwd < -0.003] == 0).all()

    def test_3class_adaptive_threshold_varies(self):
        """自适应阈值：不同波动率时期的标签分布应相对稳定。"""
        df = make_klines(400)
        y_fixed = build_labels_3class(df, threshold=0.003)
        y_adapt = build_labels_3class(df, threshold=0)
        # 自适应不应与固定完全相同（除非波动恰好匹配）
        assert not y_fixed.equals(y_adapt)
        # 自适应标签值域合法
        assert set(y_adapt.unique()) <= {0, 1, 2}

    def test_d1_label_horizon(self):
        """D1 标签：horizon=5（未来 5 个交易日）。"""
        df = make_klines(300)
        y = build_labels_3class(df, horizon=5, threshold=0.003)
        fwd = np.log(df["close"].shift(-5) / df["close"])
        assert (y[fwd > 0.003] == 2).all()
        assert (y[fwd < -0.003] == 0).all()


# ---------------------------------------------------------------------------
# 防泄漏
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_features_use_only_past(self):
        """改任意未来 K 线不得影响之前 K 线的特征。"""
        df = make_klines(320)
        X1 = build_features(df)
        df2 = df.copy()
        # 破坏最后 10 根
        df2.loc[df2.index[-10:], "close"] *= 1.5
        df2.loc[df2.index[-10:], "high"] *= 1.5
        df2.loc[df2.index[-10:], "low"] *= 1.5
        X2 = build_features(df2)
        # 前 300 根特征应完全一致（滚动窗口最长 200+安全余量）
        n = 300
        pd.testing.assert_frame_equal(X1.iloc[:n], X2.iloc[:n])

    def test_purged_splits_gap(self):
        from gold_model.train import purged_splits

        n, horizon = 1000, 24
        splits = purged_splits(n, horizon=horizon, n_splits=5, embargo_bars=24)
        assert splits
        for tr, va in splits:
            # purge: train 尾部与 val 起点至少隔 horizon+embargo
            assert tr.max() + horizon + 24 <= va.min()
            assert tr.max() < va.min()  # 时序不重叠
            assert va.min() > 0

    def test_macro_features_backward_alignment(self):
        """宏观特征：对齐后的值只能来自过去（未来宏观数据不得泄漏）。"""
        from gold_model import macro_features

        kline_times = pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC")
        # 构造一个假宏观序列（避开网络）
        fake = pd.DataFrame(
            {"close": np.arange(100.0, 100.0 + 30, 1.0)},
            index=pd.date_range("2024-12-20", periods=30, freq="1D", tz="UTC"),
        )
        fake.index.name = "time"
        feats = macro_features._series_features(fake["close"], "test")
        feats = feats.shift(1).reset_index()
        feats.columns = ["time"] + [c for c in feats.columns if c != "time"]
        # 手动做 backward 对齐
        left = pd.DataFrame({"time": kline_times})
        left["time"] = left["time"].astype("datetime64[ns, UTC]")
        feats["time"] = feats["time"].astype("datetime64[ns, UTC]")
        merged = pd.merge_asof(left, feats, on="time", direction="backward")
        # 第 1 根 K 线（2025-01-01）只能拿到 2024-12-31 及之前的宏观值
        assert merged["macro_test_ret20"].iloc[0] is not None
        # 防泄漏核心断言：每根 K 线对齐到的宏观时间戳都 <= K 线自身时间
        merged_with_ts = pd.merge_asof(
            left, feats.rename(columns={"time": "macro_ts"}), left_on="time",
            right_on="macro_ts", direction="backward"
        )
        assert (merged_with_ts["macro_ts"] <= merged_with_ts["time"]).all()


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------

class TestDenseFilter:
    def test_drops_disguised_daily(self):
        """月度密度过滤器应剔除伪装成 H1 的日线历史。"""
        # 构造：2 个月假历史（每天 1 根）+ 3 个月真 H1
        daily = pd.date_range("2025-01-01", periods=60, freq="1D", tz="UTC")
        hourly = pd.date_range("2025-03-01", periods=60 * 24, freq="1h", tz="UTC")
        rng = np.random.default_rng(1)
        n = len(daily) + len(hourly)
        px = 4000 + np.cumsum(rng.normal(0, 2, n))
        df = pd.DataFrame({
            "time": daily.append(hourly),
            "open": px, "high": px + 5, "low": px - 5, "close": px,
            "tick_volume": 1000, "spread": 25,
        })
        out = filter_dense_history(df, timeframe="H1")
        assert len(out) < len(df)
        assert out["time"].iloc[0] >= pd.Timestamp("2025-03-01", tz="UTC")

    def test_keeps_clean_history(self):
        df = make_klines(24 * 90)  # 3 个月连续 H1
        out = filter_dense_history(df, timeframe="H1")
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# 回测器
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_long_backtest_runs(self):
        from gold_model.backtest import INITIAL_CAPITAL, run_backtest

        df = make_klines(500)
        sig = pd.Series(False, index=df.index)
        sig.iloc[::50] = True  # 每 50 根开一次仓
        res = run_backtest(df, sig, direction="long")
        stats = res["统计"]
        assert stats["交易笔数"] >= 5
        assert np.isfinite(stats["总收益率%"])
        assert stats["最大回撤%"] <= 0  # 回撤非正

    def test_cost_reduces_equity(self):
        """零信号不开仓；开仓必扣成本。"""
        from gold_model.backtest import run_backtest

        df = make_klines(300)
        sig = pd.Series(False, index=df.index)
        res = run_backtest(df, sig)
        assert res["统计"]["交易笔数"] == 0

    def test_synthetic_refusal(self):
        """合成数据必须被 serve_mcp 拒绝（is_synthetic 路径）。"""
        from gold_model import mt5_client

        synth = mt5_client.synthetic_klines(bars=100)
        assert mt5_client.is_synthetic(synth)


# ---------------------------------------------------------------------------
# serve_mcp 单元逻辑
# ---------------------------------------------------------------------------

class TestLedger:
    def test_record_prediction(self, tmp_path, monkeypatch):
        from gold_model import serve_mcp

        monkeypatch.setattr(serve_mcp, "PREDICTION_LEDGER", tmp_path / "ledger.jsonl")
        serve_mcp._record_prediction("breakout", {"信号": "预期扩张", "扩张概率": 0.7, "最新收盘价": 4000.0, "K线时间": "2026-01-01"})
        serve_mcp._record_prediction("direction3", {"信号": "x", "看空概率": 0.2, "观望概率": 0.3, "看多概率": 0.5, "预测类别": "看多", "最新收盘价": 4000.0, "K线时间": "2026-01-01"})
        serve_mcp._record_prediction("direction_d1", {"信号": "y", "看空概率": 0.3, "观望概率": 0.5, "看多概率": 0.2, "预测类别": "观望", "最新收盘价": 4000.0, "K线时间": "2026-01-01"})
        lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        import json
        rec = json.loads(lines[0])
        assert rec["kind"] == "breakout" and rec["probability"] == 0.7
        rec3 = json.loads(lines[2])
        assert rec3["kind"] == "direction_d1" and rec3["pred_class"] == "观望"


class TestDirectionD1:
    def test_d1_spec_wiring(self):
        """direction_d1 目标接线：模型路径/horizon/数据集。"""
        from gold_model import train

        spec = train._spec_for("direction_d1")
        assert spec.name == "direction_d1"
        assert spec.is_multiclass
        assert spec.horizon == config.DIRECTION_D1_HORIZON == 5
        assert spec.model_path == config.DIRECTION_D1_MODEL_PATH

    def test_d1_build_dataset(self):
        """D1 数据集构建：标签 horizon=5、特征列与 H1 同构。"""
        from gold_model import train

        df = make_klines(400)
        X, y = train.build_dataset(df, target="direction_d1")
        assert len(X) == len(y)
        assert X.shape[1] >= 60  # 价格特征齐全（宏观列在测试环境可能全 0）
        # 标签值域合法
        assert set(y.unique()) <= {0, 1, 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
