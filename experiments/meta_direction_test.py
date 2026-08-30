"""实验 A：meta-labeling —— 方向在「预测波动扩张」的窗口里是否更可学？

设计（4 窗 walk-forward，每窗重训，无泄漏）：
  对每个窗口 [t, t+w)：
    1. 在 [0, t-1) 训练 breakout（purge 1，超参取自已保存 Optuna 最优）
    2. p_exp = 测试窗扩张概率；子集 S = {p_exp >= thr}
    3. 全局方向模型：[0, t-24) 全量训练 → 在 S 上评估
    4. 专用方向模型：[0, t-25) ∩ {p_exp_train >= thr} 训练 → 在 S 上评估
    5. 基线：S 上的多数类
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
from sklearn.metrics import accuracy_score, roc_auc_score

from gold_model import config
from gold_model import macro_features
from gold_model.features import build_features, build_labels, build_labels_3class
from gold_model.train import load_raw_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("meta_test")


def train_with_es(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series, num_round: int = 3000):
    n = len(X_tr)
    cut = max(int(n * 0.85), 10)
    dtr = lgb.Dataset(X_tr.iloc[:cut], label=y_tr.iloc[:cut])
    dval = lgb.Dataset(X_tr.iloc[cut:], label=y_tr.iloc[cut:], reference=dtr)
    return lgb.train(
        params, dtr, num_boost_round=num_round, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def score(y_true, proba):
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y_true, pred))
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return acc, auc


def main():
    df = load_raw_bars(23343, use_snapshot=True)
    X_price = build_features(df)
    X_macro = macro_features.build_macro_features(df["time"])
    X = pd.concat([X_price.reset_index(drop=True), X_macro], axis=1)
    y3 = build_labels_3class(df, threshold=0)
    yb = build_labels(df, horizon=1, target="breakout")
    valid = X.notna().all(axis=1) & y3.notna() & yb.notna()
    X, y3, yb = X[valid].reset_index(drop=True), y3[valid].reset_index(drop=True), yb[valid].reset_index(drop=True)
    X, y3, yb = X.iloc[:-24], y3.iloc[:-24], yb.iloc[:-24]
    n = len(X)
    log.info("dataset: %d rows", n)

    bb = joblib.load(config.BREAKOUT_MODEL_PATH)
    d3 = joblib.load(config.DIRECTION3_MODEL_PATH)
    p_bo = dict(bb["params"]) | {"verbosity": -1}
    p_d3 = dict(d3["params"]) | {"verbosity": -1}

    thresholds = [0.50, 0.55, 0.60, 0.70]
    n_win = 4
    win = int(n * 0.25 / n_win)
    start = n - n_win * win

    pooled = {thr: {"global": [], "meta": []} for thr in thresholds}  # (y, proba)
    rows = []

    for w in range(n_win):
        t = start + w * win
        te = t + win
        log.info("window %d: train=[0,%d) test=[%d,%d)", w, t, t, te)

        bo = train_with_es(p_bo, X.iloc[: t - 1], yb.iloc[: t - 1])
        p_exp_te = bo.predict(X.iloc[t:te])
        tr_end = t - 25
        p_exp_tr = bo.predict(X.iloc[:tr_end])

        gm = train_with_es(p_d3, X.iloc[: t - 24], y3.iloc[: t - 24])
        proba_g = np.asarray(gm.predict(X.iloc[t:te]))
        y_te = y3.iloc[t:te].reset_index(drop=True)

        for thr in thresholds:
            m_te = p_exp_te >= thr
            m_tr = p_exp_tr >= thr
            n_sub = int(m_te.sum())
            row = {"窗口": w, "阈值": thr, "子集大小": n_sub, "训练子集": int(m_tr.sum())}
            if n_sub < 40 or int(m_tr.sum()) < 300:
                rows.append(row | {"跳过": "子集过小"})
                continue
            X_te_sub = X.iloc[t:te][m_te]
            y_sub = y_te[m_te].reset_index(drop=True)
            proba_g_sub = proba_g[m_te]

            mm = train_with_es(p_d3, X.iloc[:tr_end][m_tr], y3.iloc[:tr_end][m_tr])
            proba_m_sub = np.asarray(mm.predict(X_te_sub))

            acc_g, auc_g = score(y_sub, proba_g_sub)
            acc_m, auc_m = score(y_sub, proba_m_sub)
            major = int(y_sub.mode().iloc[0])
            acc_b = float((y_sub == major).mean())
            cls_dist = y_sub.value_counts(normalize=True).sort_index().round(3).to_dict()

            row |= {
                "全局_acc": round(acc_g, 3), "全局_auc": round(auc_g, 3),
                "专用_acc": round(acc_m, 3), "专用_auc": round(auc_m, 3),
                "基线_acc": round(acc_b, 3),
                "子集三类分布": {int(k): v for k, v in cls_dist.items()},
            }
            rows.append(row)
            pooled[thr]["global"].append((y_sub, proba_g_sub))
            pooled[thr]["meta"].append((y_sub, proba_m_sub))
            log.info("thr=%.2f n=%d 全局AUC=%.3f 专用AUC=%.3f 基线acc=%.3f",
                     thr, n_sub, auc_g, auc_m, acc_b)

    # 汇总：pooled
    summary = []
    for thr in thresholds:
        entry = {"扩张阈值": thr}
        for mode in ("global", "meta"):
            parts = pooled[thr][mode]
            if not parts:
                entry[mode] = "无有效子集"
                continue
            ys = pd.concat([p[0] for p in parts], ignore_index=True)
            ps = np.vstack([p[1] for p in parts])
            acc, auc = score(ys, ps)
            entry[mode] = {"pooled_acc": round(acc, 4), "pooled_auc": round(auc, 4), "n": int(len(ys))}
        if isinstance(entry.get("global"), dict):
            ys = pd.concat([p[0] for p in pooled[thr]["global"]], ignore_index=True)
            major = int(ys.mode().iloc[0])
            entry["基线_pooled_acc"] = round(float((ys == major).mean()), 4)
        summary.append(entry)

    report = {"逐窗口": rows, "汇总": summary}
    out = config.DATA_DIR / "reports" / "meta_direction_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
