"""D1 per-window 细看 + 对照（随机信号 / 零特征基线 / 参数敏感性）。

目的：判断 δ=0.15 做多（夏普 3.74、165 笔、胜率 60.6%）是真信号还是
对 D1 多头漂移的 beta。关键对照：
1. 全信号做多（cf > -1，即所有 OOS 日都开多）：纯 beta
2. 随机信号（同频率随机开多）：噪声底座
3. 单窗口拆解：是否靠某一个窗口撑起来的
4. conf 分位阈值敏感性：δ = P75/P80/P85/P90/P95 分位数
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

from gold_model import config
from gold_model.backtest import run_backtest
from gold_model.features import build_features, build_labels_3class
from gold_model import macro_features
from d1_direction_test import PARAMS, N_WINDOWS, D1_HORIZON, load_d1

df = load_d1()
n_raw = len(df)

X_price = build_features(df)
X_macro = macro_features.build_macro_features(df["time"])
X_full = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
y_full = build_labels_3class(df, horizon=D1_HORIZON, threshold=0)
valid = X_full.notna().all(axis=1) & y_full.notna()
valid_idx = np.where(valid.to_numpy())[0]

mask = valid & y_full.notna()
X = X_full[mask].reset_index(drop=True)
y = y_full[mask].reset_index(drop=True)
X = X.iloc[:-D1_HORIZON]
y = y.iloc[:-D1_HORIZON]
valid_idx = valid_idx[: len(X)]

n = len(X)
win = n // (N_WINDOWS + 2)

# 每窗口训练一次，存模型和概率
win_models = []
probas, oos_rows = [], []
per_win = []
for w in range(N_WINDOWS):
    tr_end = n - (N_WINDOWS - w) * win
    te_end = min(tr_end + win, n)
    tr_idx = np.arange(0, tr_end - D1_HORIZON)
    te_idx = np.arange(tr_end, te_end)
    if len(tr_idx) < 500 or len(te_idx) < 100:
        continue
    m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx]), num_boost_round=400)
    p = np.asarray(m.predict(X.iloc[te_idx]))
    probas.append(p)
    oos_rows.append(valid_idx[te_idx])
    conf = p[:, 2] - p[:, 0]
    rows = valid_idx[te_idx]
    fwd_w = np.log(df["close"].to_numpy()[rows + D1_HORIZON] / df["close"].to_numpy()[rows])
    per_win.append({
        "窗口": w,
        "起点": str(df["time"].iloc[rows[0]].date()),
        "终点": str(df["time"].iloc[rows[-1]].date()),
        "n": int(len(rows)),
        "全窗口涨占比": round(float((fwd_w > 0).mean()), 4),
        "δ=0.15多头 n": int((conf > 0.15).sum()),
        "δ=0.15多头 涨占比": round(float((fwd_w[conf > 0.15] > 0).mean()), 4) if (conf > 0.15).sum() > 0 else None,
    })
print(json.dumps(per_win, ensure_ascii=False, indent=2))

proba = np.vstack(probas)
conf_all = proba[:, 2] - proba[:, 0]
oos_pos = np.concatenate(oos_rows)

conf_full = np.full(n_raw, np.nan)
conf_full[oos_pos] = conf_all
cf = pd.Series(conf_full)

results = {}
# 1) 全信号做多（纯 beta 底座）
sig_all = (cf > -1).fillna(False)
results["对照:全部OOS日开多"] = run_backtest(df, sig_all, direction="long", max_hold=D1_HORIZON)["统计"]

# 2) 随机信号（同 δ=0.15 频率，3 个种子）
rng_freq = float((cf > 0.15).mean())
oos_mask = np.isfinite(conf_full)
for seed in [1, 2, 3]:
    rng = np.random.default_rng(seed)
    sig_rand = np.full(n_raw, False)
    rand_pick = rng.random(n_raw) < rng_freq
    sig_rand[oos_mask & rand_pick] = True
    results[f"对照:随机开多(seed{seed})频率={rng_freq:.3f}"] = run_backtest(
        df, pd.Series(sig_rand), direction="long", max_hold=D1_HORIZON
    )["统计"]

# 3) 分位数阈值敏感性
for q in [75, 80, 85, 90, 95]:
    delta = float(np.nanquantile(conf_full, q / 100))
    sig = (cf > delta).fillna(False)
    results[f"δ=P{q}({delta:.3f}) 做多"] = run_backtest(df, sig, direction="long", max_hold=D1_HORIZON)["统计"]

print(json.dumps(results, indent=2, ensure_ascii=False))
out = config.DATA_DIR / "reports" / "d1_gating_controls.json"
out.write_text(json.dumps({"per_window": per_win, "结果": results}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"报告 -> {out}")
