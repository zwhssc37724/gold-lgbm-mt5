"""实验 B：置信度门控 —— 不改模型，只在高置信窗口出手，能否跑赢基线/全量策略？

对每个 walk-forward 窗口，用当窗模型概率，测试不同 δ 门控下的：
  - 方向命中率（p_long-p_short > δ 时做多，事后收益>阈值记赢）
  - 回测口径（ATR 止损止盈 + 成本）下的策略表现
对照组：argmax 全量开仓。
"""
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
from sklearn.metrics import accuracy_score

from gold_model import config
from gold_model import macro_features
from gold_model.features import build_features, build_labels_3class
from gold_model.train import load_raw_bars, walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gating_test")


def main():
    df = load_raw_bars(23343, use_snapshot=True)
    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
    y3 = build_labels_3class(df, threshold=0)
    valid = X.notna().all(axis=1) & y3.notna()
    X, y3 = X[valid].reset_index(drop=True), y3[valid].reset_index(drop=True)
    X, y3 = X.iloc[:-24], y3.iloc[:-24]
    n = len(X)

    d3 = joblib.load(config.DIRECTION3_MODEL_PATH)
    params = dict(d3["params"]) | {"verbosity": -1}

    # walk-forward 逐窗重训，收集 pooled 概率（真样本外）
    n_win = 4
    win = int(n * 0.25 / n_win)
    start = n - n_win * win
    probas, ys = [], []
    for w in range(n_win):
        t = start + w * win
        te = t + win
        tr_end = t - 24
        cut = int(len(X.iloc[:tr_end]) * 0.85)
        dtr = lgb.Dataset(X.iloc[:cut], label=y3.iloc[:cut])
        dval = lgb.Dataset(X.iloc[cut:tr_end], label=y3.iloc[cut:tr_end], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        probas.append(np.asarray(m.predict(X.iloc[t:te])))
        ys.append(y3.iloc[t:te].reset_index(drop=True))
    proba = np.vstack(probas)
    y = pd.concat(ys, ignore_index=True)
    log.info("pooled OOS: %d samples", len(y))

    # ---- 1) 置信度门控的方向命中率 ----
    conf = proba[:, 2] - proba[:, 0]
    long_mask = conf > 0
    rows = []
    for delta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        m_long = conf > delta
        m_short = conf < -delta
        row = {"δ": delta, "多单数": int(m_long.sum()), "空单数": int(m_short.sum())}
        if m_long.sum() >= 30:
            hit_long = float((y[m_long] == 2).mean())
            row["多单看多命中率"] = round(hit_long, 4)
            row["多单基线(实际看多占比)"] = round(float((y == 2).mean()), 4)
        if m_short.sum() >= 30:
            row["空单看空命中率"] = round(float((y[m_short] == 0).mean()), 4)
            row["空单基线(实际看空占比)"] = round(float((y == 0).mean()), 4)
        # 综合方向命中（不含观望目标）
        dm = m_long | m_short
        if dm.sum() >= 30:
            pred = np.where(conf[dm] > 0, 2, 0)
            row["综合方向命中率"] = round(float((y.to_numpy()[dm] == pred).mean()), 4)
        rows.append(row)
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # ---- 2) 回测口径：门控 vs argmax ----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gold_model.backtest import run_backtest

    df_bt = df.reset_index(drop=True).iloc[24:][valid.iloc[:-24].to_numpy()].reset_index(drop=True)
    # 信号序列对齐回测数据（预测发生在 K 线 i，实际窗从 i+1 开始——回测器已是 next-bar 成交）
    sig_all_long = pd.Series(False, index=df_bt.index[:0])
    n_bt = len(df_bt)
    res = {}
    idx_series = pd.Series(np.arange(n_bt))
    conf_full = np.full(n_bt, np.nan)
    conf_full[: len(conf)] = conf
    conf_s = pd.Series(conf_full)

    for delta in [0.0, 0.10, 0.20]:
        sig_long = (conf_s > delta).fillna(False)
        sig_short = (conf_s < -delta).fillna(False)
        r_l = run_backtest(df_bt, sig_long, direction="long")
        r_s = run_backtest(df_bt, sig_short, direction="short")
        res[f"δ={delta} 做多"] = r_l["统计"]
        res[f"δ={delta} 做空"] = r_s["统计"]
    print(json.dumps(res, indent=2, ensure_ascii=False))

    out = config.DATA_DIR / "reports" / "gating_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"门控命中率": rows, "回测": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"报告 -> {out}")


if __name__ == "__main__":
    main()
