"""修正对齐后的门控回测：OOS 置信度映射回原始 df 行号，再跑交易回测。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import warnings

warnings.filterwarnings("ignore")

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from gold_model import config, macro_features
from gold_model.backtest import run_backtest
from gold_model.features import build_features, build_labels_3class
from gold_model.train import load_raw_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

df = load_raw_bars(23343, use_snapshot=True)
X_price = build_features(df)
X_macro = macro_features.build_macro_features(df["time"])
X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
y3 = build_labels_3class(df, threshold=0)
valid = X.notna().all(axis=1) & y3.notna()
Xv, yv = X[valid].reset_index(drop=True), y3[valid].reset_index(drop=True)
Xv, yv = Xv.iloc[:-24], yv.iloc[:-24]
n = len(Xv)

d3 = joblib.load(config.DIRECTION3_MODEL_PATH)
params = dict(d3["params"]) | {"verbosity": -1}

n_win = 4
win = int(n * 0.25 / n_win)
start = n - n_win * win
probas = []
for w in range(n_win):
    t = start + w * win
    te = t + win
    tr_end = t - 24
    cut = int(tr_end * 0.85)
    dtr = lgb.Dataset(Xv.iloc[:cut], label=yv.iloc[:cut])
    dval = lgb.Dataset(Xv.iloc[cut:tr_end], label=yv.iloc[cut:tr_end], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    probas.append(np.asarray(m.predict(Xv.iloc[t:te])))
proba = np.vstack(probas)
conf = proba[:, 2] - proba[:, 0]

# 对齐：Xv 行 j -> df 行 valid_idx[j]；OOS 窗口 [start:n)
valid_idx = np.where(valid.to_numpy())[0][:n]
conf_full = np.full(len(df), np.nan)
oos_pos = valid_idx[start:]
assert len(oos_pos) == len(conf), (len(oos_pos), len(conf))
conf_full[oos_pos] = conf
print("信号覆盖 df 行:", int(np.isfinite(conf_full).sum()), "/", len(df), flush=True)

cf = pd.Series(conf_full)
results = {}
for delta in [0.20, 0.25, 0.30]:
    sig_long = (cf > delta).fillna(False)
    sig_short = (cf < -delta).fillna(False)
    r_l = run_backtest(df, sig_long, direction="long")
    r_s = run_backtest(df, sig_short, direction="short")
    results[f"δ={delta} 做多"] = r_l["统计"]
    results[f"δ={delta} 做空"] = r_s["统计"]
print(json.dumps(results, indent=2, ensure_ascii=False))

out = config.DATA_DIR / "reports" / "gating_backtest_fixed.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"报告 -> {out}")
