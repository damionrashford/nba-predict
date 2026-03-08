"""Optuna-based hyperparameter tuning for XGBoost models.

Provides tuning functions for both classification (game winner) and
regression (point spread, player performance, win totals) models.
Uses temporal cross-validation to avoid leakage.
"""

import optuna
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from nba_predict.config import RANDOM_SEED

# Suppress Optuna info logs (keep warnings/errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgb_search_space(trial: optuna.Trial) -> dict:
    """Define the XGBoost hyperparameter search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }


def tune_classifier(X_train, y_train, X_val, y_val,
                     n_trials: int = 100, metric: str = "log_loss") -> dict:
    """Tune XGBClassifier hyperparameters using Optuna.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data for early stopping and evaluation.
        n_trials: Number of Optuna trials to run.
        metric: Optimization metric — 'log_loss' (default) or 'accuracy'.

    Returns:
        Dict with best_params and best_score.
    """
    def objective(trial):
        params = _xgb_search_space(trial)
        params["eval_metric"] = "logloss"

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]

        if metric == "log_loss":
            return log_loss(y_val, y_prob)  # minimize
        else:
            y_pred = model.predict(X_val)
            return -accuracy_score(y_val, y_pred)  # minimize negative accuracy

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best.update({"n_estimators": 500, "early_stopping_rounds": 50,
                 "random_state": RANDOM_SEED, "n_jobs": -1, "eval_metric": "logloss"})

    return {"best_params": best, "best_score": study.best_value}


def tune_regressor(X_train, y_train, X_val, y_val,
                    n_trials: int = 100) -> dict:
    """Tune XGBRegressor hyperparameters using Optuna.

    Minimizes MAE on the validation set.

    Returns:
        Dict with best_params and best_score.
    """
    def objective(trial):
        params = _xgb_search_space(trial)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best.update({"n_estimators": 500, "early_stopping_rounds": 50,
                 "random_state": RANDOM_SEED, "n_jobs": -1})

    return {"best_params": best, "best_score": study.best_value}
