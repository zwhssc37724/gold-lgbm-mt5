"""D1 含成本回测：WF 样本外置信度 → 原始 df 行号 → 复用 H1 回测引擎（同一成本口径）。

与 H1 gating_backtest_fixed.py 同构：
- WF 5 窗口样本外概率，conf = p_long - p_short
- 信号映射回原始 D1 行号（对齐）
- run_backtest：点差 25 点 + 滑点 5 点，止损 2×ATR / 止盈 3×ATR / 最长持仓 5 根 D1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import logging
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

from gold_model import config
from gold_model.backtest import run_backtest, buy_and_hold
from d1_direction_test import build_d1_dataset, load_d1, PARAMS, N_WINDOWS, D1_HORIZON

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

df = load_d1()
n_raw = len(df)

X, y, fwd = build_d1_dataset(df)
# 对齐：build_d1_dataset 剔除了 NaN 行（前 200 根 MA200 缺失）+ 末尾 horizon 根
# 重建同样的 valid mask 拿到原始行号
from gold_model.features import build_features, build_labels_3class
from gold_model import macro_features

X_price = build_features(df)
X_macro = macro_features.build_macro_features(df["time"])
X_full = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
y_full = build_labels_3class(df, horizon=D1_HORIZON, threshold=0)
valid = X_full.notna().all(axis=1) & y_full.notna()
valid_idx = np.where(valid.to_numpy())[0]
# build_d1_dataset 还做了 .iloc[:-HORIZON]，即去掉 valid_idx 的最后 HORIZON 个
valid_idx = valid_idx[: len(X)]
assert len(valid_idx) == len(X) == len(y) == len(fwd), (len(valid_idx), len(X), len(y), len(fwd))

n = len(X)
win = n // (N_WINDOWS + 2)
start = n - N_WINDOWS * win
probas = []
oos_rows = []
for w in range(N_WINDOWS):
    tr_end = n - (N_WINDOWS - w) * win
    te_end = min(tr_end + win, n)
    tr_idx = np.arange(0, tr_end - D1_HORIZON)
    te_idx = np.arange(tr_end, te_end)
    if len(tr_idx) < 500 or len(te_idx) < 100:
        continue
    m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx]), num_boost_round=400)
    probas.append(np.asarray(m.predict(X.iloc[te_idx])))
    oos_rows.append(valid_idx[te_idx])
proba = np.vstack(probas)
conf = proba[:, 2] - proba[:, 0]
oos_pos = np.concatenate(oos_rows)
assert len(oos_pos) == len(conf)

conf_full = np.full(n_raw, np.nan)
conf_full[oos_pos] = conf
print("OOS 覆盖 D1 行数:", int(np.isfinite(conf_full).sum()), "/", n_raw)
print("OOS 时间范围:", df["time"].iloc[oos_pos[0]], "->", df["time"].iloc[oos_pos[-1]])

cf = pd.Series(conf_full)
results = {}
bh = buy_and_hold(df)
results["买入持有"] = bh["统计"]

for delta in [0.15, 0.20, 0.25]:
    sig_long = (cf > delta).fillna(False)
    sig_short = (cf < -delta).fillna(False)
    r_l = run_backtest(df, sig_long, direction="long", max_hold=D1_HORIZON)
    r_s = run_backtest(df, sig_short, direction="short", max_hold=D1_HORIZON)
    results[f"δ={delta} 做多"] = r_l["统计"]
    results[f"δ={delta} 做空"] = r_s["统计"]

print(json.dumps(results, indent=2, ensure_ascii=False))
out = config.DATA_DIR / "reports" / "d1_gating_backtest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "OOS范围": [str(df["time"].iloc[oos_pos[0]]), str(df["time"].iloc[oos_pos[-1]])],
    "成本口径": "点差25点+滑点5点，止损2×ATR/止盈3×ATR/最长持仓5根D1",
    "结果": results,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"报告 -> {out}")
