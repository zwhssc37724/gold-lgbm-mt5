"""概率校准（isotonic regression，按任务在 WF 样本外概率上拟合）。

为什么需要：LightGBM 的多分类输出是未校准的"排序分数"。置信度门控阈值
（如 |p_看多−p_看空|>0.25）在原始概率上无频率含义——0.25 不代表任何真实
事件率。校准后 p=0.7 才真正意味着"历史同样情形下 70% 涨"。

方法：
  - 对每个二分类问题（看涨 vs 非看涨；看跌 vs 非看跌）单独拟合 isotonic
  - 训练数据 = WF 样本外概率（真 OOS，避免用模型见过的样本自欺）
  - 校准器随模型 bundle 一起 joblib 落盘

用法（训练后）：
    cal = fit_calibrators(oos_proba, y_oos)          # dict: {"long": iso, "short": iso}
    save_calibrators(cal, config.DIRECTION3_MODEL_PATH)
    cal = load_calibrators(config.DIRECTION3_MODEL_PATH)
    p_long_cal = cal["long"].predict_proba(conf)    # conf = p2 - p0（或任意分数）
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


def _fit_one(scores: np.ndarray, events: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, events)
    return iso


def fit_calibrators(proba_oos: np.ndarray, y_oos: np.ndarray) -> dict:
    """在 WF 样本外概率上拟合 isotonic 校准器。

    proba_oos: (n, 3) 三分类概率（0=看空, 1=观望, 2=看多）
    y_oos: (n,) 真实标签
    返回 {"long": iso, "short": iso}：
      - long: score = p2 - p0 → 校准后的 P(未来看涨 | score)
      - short: score = p0 - p2 → 校准后的 P(未来看跌 | score)
    """
    proba_oos = np.asarray(proba_oos, dtype=float)
    y_oos = np.asarray(y_oos)
    if proba_oos.ndim != 2 or proba_oos.shape[1] != 3:
        raise ValueError(f"expected (n,3) proba, got {proba_oos.shape}")
    conf_long = proba_oos[:, 2] - proba_oos[:, 0]
    conf_short = proba_oos[:, 0] - proba_oos[:, 2]
    ev_long = (y_oos == 2).astype(int)
    ev_short = (y_oos == 0).astype(int)
    return {
        "long": _fit_one(conf_long, ev_long),
        "short": _fit_one(conf_short, ev_short),
    }


def calibrate_confidence(calibrators: dict, confidence: float) -> dict:
    """把原始置信度（p多−p空，可正可负）翻译成校准后的方向概率。

    返回 {"看涨概率(校准)": float, "看跌概率(校准)": float}
    """
    c = float(confidence)
    p_up = float(calibrators["long"].predict([max(c, -c) if c >= 0 else c])[0])
    # long 校准器在 conf>0 段有效；负置信度喂给 long 校准器没有频率含义，
    # 用 short 校准器（score=p0-p2=|c|）给看跌概率
    p_down = float(calibrators["short"].predict([abs(c) if c < 0 else -c])[0])
    return {"看涨概率(校准)": round(p_up, 4), "看跌概率(校准)": round(p_down, 4)}


def reliability_curve(scores: np.ndarray, events: np.ndarray, n_bins: int = 10) -> list[dict]:
    """可靠性曲线数据（分箱命中率 vs 平均分数），供诊断输出。"""
    scores = np.asarray(scores, dtype=float)
    events = np.asarray(events)
    qs = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (scores >= lo) & (scores <= hi if i == n_bins - 1 else scores < hi)
        if m.sum() < 5:
            continue
        out.append({
            "分数区间": [round(float(lo), 3), round(float(hi), 3)],
            "样本数": int(m.sum()),
            "平均分数": round(float(scores[m].mean()), 3),
            "实际命中率": round(float(events[m].mean()), 4),
        })
    return out


def save_calibrators(calibrators: dict, model_path: Path) -> Path:
    """校准器存为 <model>.calib.pkl，与模型同目录同基名。"""
    path = Path(str(model_path).replace(".pkl", ".calib.pkl"))
    joblib.dump(calibrators, path)
    logger.info("calibrators saved -> %s", path)
    return path


def load_calibrators(model_path: Path) -> dict | None:
    """加载校准器；不存在返回 None（调用方退回原始概率）。"""
    path = Path(str(model_path).replace(".pkl", ".calib.pkl"))
    if not path.exists():
        return None
    return joblib.load(path)
