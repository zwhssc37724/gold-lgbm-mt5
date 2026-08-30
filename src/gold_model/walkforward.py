"""共享的 walk-forward 评估与 OOS 概率收集工具。

动机（2026-08-30 重构）：此前 walk-forward 逻辑散落在 train.py 和 4 个实验脚本里，
各复制一份，口径漂移（H1 用 4 窗口、D1 用 5 窗口、purge/embargo 各异）。
本模块提供唯一的实现：
  - oos_probabilities(): 训练→预测→收集真样本外概率（信号验证/校准/门控共用）
  - oos_frame(): 同上，但返回带时间戳与前瞻收益的 DataFrame（交叉实验用）

防泄漏保证：
  - 扩展窗训练（train 永远在 test 之前）
  - train 尾部 purge horizon 根（标签前瞻窗不得跨 train/test 边界）
  - 返回的每行带原始 df 行号（oos_pos），供映射回原始 K 线做回测对齐
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class WFConfig:
    """Walk-forward 配置（按任务选择合理默认）。"""

    n_windows: int = 4
    horizon: int = 24          # purge 根数（标签前瞻窗）
    min_train: int = 500       # 低于此跳过该窗口
    min_test: int = 100        # 低于此跳过该窗口
    num_boost_round: int = 600
    early_stopping: bool = False  # 实验口径默认关（无验证集）；正式训练开


def _window_bounds(n: int, n_windows: int, train_frac: float = 0.75) -> list[tuple[int, int]]:
    """返回 [(train_end, test_end), ...]，窗口覆盖最后 (1-train_frac) 的数据。"""
    span = n - int(n * train_frac)
    win = max(span // n_windows, 200)
    start = n - n_windows * win
    out = []
    for w in range(n_windows):
        tr_end = start + w * win
        te_end = min(tr_end + win, n)
        if te_end - tr_end < 100:
            continue
        out.append((tr_end, te_end))
    return out


def oos_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cfg: WFConfig | None = None,
    valid_frac: float = 0.0,
) -> dict:
    """滚动训练→滚动预测，收集真样本外概率。

    参数：
      valid_frac: >0 时在 train 尾部再切 valid 段做早停（口径与 gating_backtest_fixed 一致）
    返回：
      {
        "proba": np.ndarray (n_oos, n_class) 或 (n_oos,)，
        "oos_pos": np.ndarray，每行对应的 X 行号（0-based），
        "windows": [(train_end, test_end), ...]，
      }
    """
    cfg = cfg or WFConfig()
    n = len(X)
    bounds = _window_bounds(n, cfg.n_windows)
    probs: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for tr_end, te_end in bounds:
        tr_idx = np.arange(0, max(0, tr_end - cfg.horizon))  # purge
        te_idx = np.arange(tr_end, te_end)
        if len(tr_idx) < cfg.min_train or len(te_idx) < cfg.min_test:
            continue
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        if valid_frac > 0 and len(tr_idx) > 1000:
            cut = int(len(tr_idx) * (1 - valid_frac))
            dtr = lgb.Dataset(X.iloc[tr_idx[:cut]], label=y.iloc[tr_idx[:cut]])
            dval = lgb.Dataset(X.iloc[tr_idx[cut:]], label=y.iloc[tr_idx[cut:]], reference=dtr)
            model = lgb.train(
                params, dtr, num_boost_round=2000, valid_sets=[dval],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
        else:
            model = lgb.train(params, dtrain, num_boost_round=cfg.num_boost_round)
        p = np.asarray(model.predict(X.iloc[te_idx]))
        probs.append(p)
        positions.append(te_idx)
    if not probs:
        return {"proba": np.zeros((0,)), "oos_pos": np.zeros(0, dtype=int), "windows": []}
    # LightGBM: 二分类 predict → (n,)；多分类 → (n, k)。二分类一维拼接，多分类 vstack。
    if probs[0].ndim == 1:
        proba = np.concatenate([np.asarray(p).ravel() for p in probs])
    elif len(probs) == 1:
        proba = probs[0]
    else:
        proba = np.vstack(probs)
    return {"proba": proba, "oos_pos": np.concatenate(positions), "windows": bounds}


def oos_frame(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    fwd: pd.Series,
    params: dict,
    cfg: WFConfig | None = None,
    valid_frac: float = 0.0,
) -> pd.DataFrame:
    """同 oos_probabilities，但返回带时间戳/前瞻收益的 DataFrame。

    fwd: 与 X 对齐的前瞻对数收益（标签的连续版），用于命中率/平均收益统计。
    返回列：oos_pos, time, proba（每类一列 proba_0/1/2 或单列）, fwd
    """
    r = oos_probabilities(X, y, params, cfg, valid_frac)
    pos = r["oos_pos"]
    proba = r["proba"]
    out = pd.DataFrame({"oos_pos": pos})
    out["time"] = pd.to_datetime(df["time"].to_numpy()[pos])
    out["fwd"] = fwd.to_numpy()[pos]
    if proba.ndim == 2:
        for c in range(proba.shape[1]):
            out[f"proba_{c}"] = proba[:, c]
    else:
        out["proba"] = proba
    return out
