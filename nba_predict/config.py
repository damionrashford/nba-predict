"""Central configuration for the NBA prediction system."""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ─── Reproducibility ─────────────────────────────────────────────────────────

RANDOM_SEED = 42

# ─── Temporal Splits ─────────────────────────────────────────────────────────
# Season 2000 is dropped because there's no prior-season data for features.
# Season years refer to the year the season ENDS (e.g., 2025 = 2024-25 season).

FIRST_USABLE_SEASON = 2001
TRAIN_SEASONS = list(range(2001, 2022))  # 21 seasons
VAL_SEASONS = [2022, 2023]                # hyperparameter tuning
TEST_SEASONS = [2024, 2025]               # final held-out evaluation
LIVE_SEASON = 2026                        # partial — predict remaining games

# ─── Feature Engineering ─────────────────────────────────────────────────────

ROLLING_WINDOWS = [5, 10, 20]   # game windows for rolling stats
MIN_ROLLING_PERIODS = 3          # minimum games before computing rolling features

# ─── XGBoost Defaults ────────────────────────────────────────────────────────

XGBOOST_CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

XGBOOST_REGRESSOR_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ─── Sentiment Configuration ────────────────────────────────────────────────

SENTIMENT_ROLLING_WINDOWS = [5, 10, 20]  # game windows (matches ROLLING_WINDOWS)

SENTIMENT_BASE_COLS = [
    "sentiment_score",
    "sentiment_volume",
    "sentiment_std",
    "sentiment_engagement",
    "sentiment_pos_ratio",
]
