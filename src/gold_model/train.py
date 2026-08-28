"""Train the LightGBM gold direction model with Optuna hyperparameter search.

Usage (from project root):
    uv run gold-train                  # pull data (MT5 or synthetic) and train
    uv run gold-train --bars 50000 --trials 60
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from gold_model import config, mt5_client
from gold_model.features import build_features, build_labels

logger = logging.getLogger("gold_model.train")


def load_dataset(bars: int, target: str = "direction") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = mt5_client.get_klines(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, bars=bars)
    logger.info("klines loaded: %d rows (source includes fallback)", len(df))
    X = build_features(df)
    y = build_labels(df, horizon=config.LABEL_HORIZON, target=target)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].astype(float), y[mask].astype(int)
    X = X.iloc[:-config.LABEL_HORIZON]
    y = y.iloc[:-config.LABEL_HORIZON]
    logger.info("dataset ready: target=%s, %d samples, %d features, positive rate %.4f", target, len(X), X.shape[1], y.mean())
    return X, y, df


def time_series_splits(n: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV splits, train always before validation."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    first_train = int(n * 0.6)
    step = (n - first_train) // n_splits
    for k in range(n_splits):
        train_end = first_train + k * step
        val_end = train_end + step
        splits.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return splits


def objective(X: pd.DataFrame, y: pd.Series, splits) -> optuna.trial.Trial:
    def _run(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
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
            scores.append(booster.best_score["valid_0"]["auc"])
        return float(np.mean(scores))

    return _run  # type: ignore[return-value]


def train(bars: int, trials: int, target: str = "direction") -> dict:
    X, y, _raw = load_dataset(bars, target=target)

    # Hold out the most recent 20% as the untouched test set.
    test_start = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]
    logger.info("train=%d  test=%d (time-ordered split)", len(X_tr), len(X_te))

    splits = time_series_splits(len(X_tr))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective(X_tr, y_tr, splits), n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": config.RANDOM_STATE,
        }
    )
    logger.info("best CV AUC=%.5f  params=%s", study.best_value, best)

    # Retrain on full training window with early stopping on the last CV fold.
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
    test_auc = float(roc_auc_score(y_te, proba))
    logger.info("TEST AUC = %.5f  (n=%d, positive rate %.4f)", test_auc, len(y_te), y_te.mean())

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "params": best,
            "target": target,
            "horizon": config.LABEL_HORIZON,
            "symbol": config.SYMBOL,
            "timeframe": config.TIMEFRAME,
        },
        config.MODEL_PATH,
    )
    config.FEATURES_PATH.write_text(json.dumps(list(X.columns), ensure_ascii=False, indent=2))

    # Feature importance (top 15)
    imp = sorted(zip(X.columns, model.feature_importance("gain")), key=lambda kv: -kv[1])[:15]
    for name, gain in imp:
        logger.info("  feat %-16s gain=%.1f", name, gain)

    return {"target": target, "test_auc": test_auc, "cv_auc": float(study.best_value), "n_test": len(y_te)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gold LightGBM model")
    parser.add_argument("--bars", type=int, default=config.BARS)
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument(
        "--target",
        choices=["breakout", "direction"],
        default="breakout",
        help="breakout: next bar range expansion (learnable); direction: next bar up/down",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    result = train(bars=args.bars, trials=args.trials, target=args.target)
    print(json.dumps(result, indent=2))
    print(f"model saved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
