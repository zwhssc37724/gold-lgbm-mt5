"""训练黄金交易模型（LightGBM + Optuna 调参）。

支持两类任务：
  - 突破/扩张（二分类）：下一根 K 线振幅是否突破近 100 根中位数
  - 方向三分类：未来 24 根 H1 收益按 ±0.3% 阈值划分为看空/观望/看多

用法（在工程根目录）：
    uv run gold-train
    uv run gold-train --target breakout      # 默认
    uv run gold-train --target direction3    # 三分类
    uv run gold-train --bars 50000 --trials 60
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from gold_model import config, mt5_client
from gold_model.features import build_features, build_labels, build_labels_3class

logger = logging.getLogger("gold_model.train")


@dataclass
class TargetSpec:
    """每种预测任务的 LightGBM 训练配置。"""

    name: str
    is_multiclass: bool
    num_class: int
    objective: str
    metric: str
    optuna_direction: str
    model_path: Path


def _spec_for(target: str) -> TargetSpec:
    if target == "breakout":
        return TargetSpec(
            name="breakout",
            is_multiclass=False,
            num_class=1,
            objective="binary",
            metric="auc",
            optuna_direction="maximize",
            model_path=config.BREAKOUT_MODEL_PATH,
        )
    if target == "direction3":
        return TargetSpec(
            name="direction3",
            is_multiclass=True,
            num_class=3,
            objective="multiclass",
            metric="multi_logloss",
            optuna_direction="minimize",  # 越低越好
            model_path=config.DIRECTION3_MODEL_PATH,
        )
    raise ValueError(f"unknown target: {target}")


def load_dataset(bars: int, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, bars=bars)
    logger.info("klines loaded: %d rows (source includes fallback)", len(df))
    X = build_features(df)
    if target == "direction3":
        y = build_labels_3class(df)
        horizon = config.DIRECTION3_HORIZON
    else:
        y = build_labels(df, horizon=config.BREAKOUT_HORIZON, target=target)
        horizon = config.BREAKOUT_HORIZON
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].astype(float), y[mask].astype(int)
    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]
    if target == "direction3":
        dist = y.value_counts(normalize=True).sort_index().to_dict()
        logger.info(
            "dataset ready: target=%s, %d samples, %d features, 三类分布=%s",
            target,
            len(X),
            X.shape[1],
            {int(k): round(v, 4) for k, v in dist.items()},
        )
    else:
        logger.info(
            "dataset ready: target=%s, %d samples, %d features, 正样本率=%.4f",
            target,
            len(X),
            X.shape[1],
            y.mean(),
        )
    return X, y, df


def time_series_splits(n: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """扩展窗时序切分，训练窗口永远早于验证窗口。"""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    first_train = int(n * 0.6)
    step = (n - first_train) // n_splits
    for k in range(n_splits):
        train_end = first_train + k * step
        val_end = train_end + step
        splits.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return splits


def make_objective(X: pd.DataFrame, y: pd.Series, splits, spec: TargetSpec):
    def _run(trial: optuna.Trial) -> float:
        params: dict = {
            "objective": spec.objective,
            "metric": spec.metric,
            "verbosity": -1,
            "seed": config.RANDOM_STATE,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        if spec.is_multiclass:
            params["num_class"] = spec.num_class

        scores: list[float] = []
        for train_idx, val_idx in splits:
            dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            scores.append(booster.best_score["valid_0"][spec.metric])
        return float(np.mean(scores))

    return _run


def _best_sign(value: float, spec: TargetSpec) -> str:
    """根据任务类型格式化最优指标字符串。"""
    if spec.metric == "auc":
        return f"CV AUC={value:.5f}"
    if spec.metric == "multi_logloss":
        return f"CV 多分类对数损失={value:.5f}"
    return f"CV {spec.metric}={value:.5f}"


def _format_test_metrics(spec: TargetSpec, y_te: pd.Series, proba: np.ndarray) -> dict:
    """根据任务类型计算并格式化测试集指标。"""
    result: dict = {"测试样本数": len(y_te)}
    if spec.is_multiclass:
        pred = proba.argmax(axis=1)
        result["测试准确率"] = float(accuracy_score(y_te, pred))
        # 一对多宏平均 AUC（macro）
        try:
            macro_auc = float(
                roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            macro_auc = float("nan")
        result["测试宏平均AUC（OvR）"] = macro_auc
        dist = y_te.value_counts(normalize=True).sort_index().to_dict()
        result["测试集三类分布"] = {int(k): round(v, 4) for k, v in dist.items()}
    else:
        result["测试AUC"] = float(roc_auc_score(y_te, proba))
        result["测试正样本率"] = round(float(y_te.mean()), 4)
    return result


def train(bars: int, trials: int, target: str) -> dict:
    spec = _spec_for(target)
    X, y, _raw = load_dataset(bars, target=target)

    # 时序切分：末 20% 作为不打扰的测试集
    test_start = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]
    logger.info("train=%d  test=%d (time-ordered split)", len(X_tr), len(X_te))

    splits = time_series_splits(len(X_tr))
    study = optuna.create_study(
        direction=spec.optuna_direction,
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        make_objective(X_tr, y_tr, splits, spec),
        n_trials=trials,
        show_progress_bar=False,
    )
    best = dict(study.best_params)
    best.update(
        {
            "objective": spec.objective,
            "metric": spec.metric,
            "verbosity": -1,
            "seed": config.RANDOM_STATE,
        }
    )
    if spec.is_multiclass:
        best["num_class"] = spec.num_class
    logger.info("best %s  params=%s", _best_sign(study.best_value, spec), best)

    # 在训练窗口上用最后一折做早停重训
    train_idx, val_idx = splits[-1]
    dtrain = lgb.Dataset(X_tr.iloc[train_idx], label=y_tr.iloc[train_idx])
    dval = lgb.Dataset(X_tr.iloc[val_idx], label=y_tr.iloc[val_idx], reference=dtrain)
    model = lgb.train(
        best,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    proba = model.predict(X_te)
    test_metrics = _format_test_metrics(spec, y_te, proba)
    logger.info("test metrics: %s", test_metrics)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": list(X.columns),
        "params": best,
        "target": target,
        "is_multiclass": spec.is_multiclass,
        "num_class": spec.num_class,
        "symbol": config.SYMBOL,
        "timeframe": config.TIMEFRAME,
        "horizon": config.DIRECTION3_HORIZON if target == "direction3" else config.BREAKOUT_HORIZON,
        "threshold": config.DIRECTION3_THRESHOLD if target == "direction3" else None,
    }
    joblib.dump(bundle, spec.model_path)
    config.FEATURES_PATH.write_text(json.dumps(list(X.columns), ensure_ascii=False, indent=2))

    # Top-15 特征重要性
    imp = sorted(zip(X.columns, model.feature_importance("gain")), key=lambda kv: -kv[1])[:15]
    for name, gain in imp:
        logger.info("  特征 %-16s 增益=%.1f", name, gain)

    result = {
        "目标": target,
        "最佳CV": {spec.metric: float(study.best_value)},
        "测试集指标": test_metrics,
        "模型路径": str(spec.model_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练黄金交易模型")
    parser.add_argument("--bars", type=int, default=config.BARS)
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument(
        "--target",
        choices=["breakout", "direction3"],
        default="breakout",
        help="breakout：二分类，预测下一根 K 线波动扩张；direction3：三分类，看空/观望/看多",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    result = train(bars=args.bars, trials=args.trials, target=args.target)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"模型已保存 -> {_spec_for(args.target).model_path}")


if __name__ == "__main__":
    main()
