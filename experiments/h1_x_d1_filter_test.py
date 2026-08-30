import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings

warnings.filterwarnings("ignore")

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from gold_model import config, macro_features
from gold_model.features import build_features, build_labels_3class
from gold_model.train import load_raw_bars
from d1_direction_test import PARAMS, N_WINDOWS, D1_HORIZON, load_d1

# ---- H1 全 OOS 置信度（复现 gating_backtest_fixed.py 的 4 窗口 WF） ----
df = load_raw_bars(23343, use_snapshot=True)
X_price = build_features(df)
X_macro = macro_features.build_macro_features(df["time"])
X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
y3 = build_labels_3class(df, threshold=0)
valid = X.notna().all(axis=1) & y3.notna()
yv = y3[valid].reset_index(drop=True).iloc[:-24]
Xv = X[valid].reset_index(drop=True).iloc[:-24]
n = len(Xv)
d3 = joblib.load(config.DIRECTION3_MODEL_PATH)
params = dict(d3["params"]) | {"verbosity": -1}
win = int(n * 0.25 / 4)
start = n - 4 * win
valid_idx = np.where(valid.to_numpy())[0][:n]
oos_pos = valid_idx[start:]
probas = []
for w in range(4):
    t = start + w * win
    te = t + win
    tr_end = t - 24
    cut = int(tr_end * 0.85)
    dtr = lgb.Dataset(Xv.iloc[:cut], label=yv.iloc[:cut])
    dval = lgb.Dataset(Xv.iloc[cut:tr_end], label=yv.iloc[cut:tr_end], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
    probas.append(np.asarray(m.predict(Xv.iloc[t:te])))
proba = np.vstack(probas)
conf_h1 = proba[:, 2] - proba[:, 0]

# ---- D1 OOS 置信度（同 d1_gating_backtest.py 的 5 窗口 WF） ----
d1 = load_d1()
X1p = build_features(d1)
X1m = macro_features.build_macro_features(d1["time"])
X1f = pd.concat([X1p.reset_index(drop=True), X1m], axis=1)
y1f = build_labels_3class(d1, horizon=D1_HORIZON, threshold=0)
valid1 = X1f.notna().all(axis=1) & y1f.notna()
vi1 = np.where(valid1.to_numpy())[0]
mask1 = valid1 & y1f.notna()
X1 = X1f[mask1].reset_index(drop=True)
y1 = y1f[mask1].reset_index(drop=True)
X1, y1 = X1.iloc[:-D1_HORIZON], y1.iloc[:-D1_HORIZON]
vi1 = vi1[: len(X1)]
n1 = len(X1)
win1 = n1 // (N_WINDOWS + 2)
probs1, rows1 = [], []
for w in range(N_WINDOWS):
    tr_end = n1 - (N_WINDOWS - w) * win1
    te_end = min(tr_end + win1, n1)
    tr_idx = np.arange(0, tr_end - D1_HORIZON)
    te_idx = np.arange(tr_end, te_end)
    if len(tr_idx) < 500 or len(te_idx) < 100:
        continue
    m = lgb.train(PARAMS, lgb.Dataset(X1.iloc[tr_idx], label=y1.iloc[tr_idx]), num_boost_round=400)
    probs1.append(np.asarray(m.predict(X1.iloc[te_idx])))
    rows1.append(vi1[te_idx])
proba1 = np.vstack(probs1)
conf_d1 = proba1[:, 2] - proba1[:, 0]
d1_rows = np.concatenate(rows1)

# D1 bar 完结时刻 = 时间戳 + 1 天（bar 覆盖当天 00:00~23:59，次日 00:00 才可读）
d1_conf = pd.DataFrame({
    "available_from": pd.to_datetime(d1["time"].to_numpy()[d1_rows]) + pd.Timedelta(days=1),
    "conf": conf_d1,
}).sort_values("available_from").reset_index(drop=True)
d1_conf["available_from"] = d1_conf["available_from"].astype("datetime64[us, UTC]")

# ---- 交叉 ----
h1_times = pd.to_datetime(df["time"].to_numpy()[oos_pos])
h1_close = df["close"].to_numpy()
fwd = np.log(h1_close[oos_pos + 24] / h1_close[oos_pos])

sig = pd.DataFrame({"h1_time": h1_times, "conf_h1": conf_h1, "fwd": fwd}).sort_values("h1_time").reset_index(drop=True)
sig["h1_time"] = sig["h1_time"].astype("datetime64[us, UTC]")
merged = pd.merge_asof(sig, d1_conf, left_on="h1_time", right_on="available_from", direction="backward")
merged = merged.dropna(subset=["conf"]).reset_index(drop=True)
print("可交叉 OOS H1 信号:", len(merged), "/", len(sig), " 时间:", merged["h1_time"].min().date(), "->", merged["h1_time"].max().date())

res = {}
for h1_thr in [0.15, 0.20, 0.25]:
    for d1_thr in [None, -0.25, 0.0, 0.15, 0.25]:
        base = merged[merged["conf_h1"] > h1_thr]
        if d1_thr is None:
            sub = base
            tag = f"H1>{h1_thr}（无D1过滤）"
        else:
            sub = base[base["conf"] > d1_thr]
            tag = f"H1>{h1_thr} & D1>{d1_thr}"
        if len(sub) < 10:
            res[tag] = {"n": len(sub), "样本不足": True}
            continue
        res[tag] = {"n": len(sub), "命中率": round(float((sub["fwd"] > 0).mean()), 4),
                    "平均24h收益%": round(float(sub["fwd"].mean() * 100), 3)}
print(json.dumps(res, ensure_ascii=False, indent=2))
out = config.DATA_DIR / "reports" / "h1_x_d1_filter_test.json"
out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("报告 ->", out)
