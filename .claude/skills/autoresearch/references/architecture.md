# AutoResearch System Architecture

## File Map

```
autoresearch/                     THE EXPERIMENT ARENA
  constants.py                    SACRED — frozen baselines, NBA_CORE weights, timeouts
  evaluate.py                     SACRED — computes NBA_CORE, enforces timeout
  experiment.py                   ** THE ONLY EDITABLE FILE **
  program.md                      SACRED — full agent protocol
  results.tsv                     Experiment log (append only)
  outputs/models/                 Experiment .joblib artifacts (isolated from production)

nba_predict/                      SACRED — production code (DO NOT MODIFY)
  config.py                       Temporal splits, XGB defaults, paths, seed=42
  pipeline.py                     Training orchestrator (registry + dynamic import)
  tuning.py                       Optuna hyperparameter search
  models/
    game_winner.py                Model 1: XGBClassifier + isotonic calibration
    point_spread.py               Model 2: XGBRegressor for margin
    player_performance.py         Model 3: 3x XGBRegressor (PTS, AST, REB)
    season_outcomes.py            Model 4: XGB+Ridge blend (wins) + XGB (MVP)
  features/
    game_features.py              Rolling stats with shift(1) anti-leakage
    team_features.py              Prior-season team quality joins
    player_features.py            Roster quality aggregation (top5 BPM, total VORP)
    matchup_features.py           Home-away pairing + ~150 differential features
    selection.py                  Feature selection utilities
  data/
    loader.py                     CSV loaders with team name normalization
    cleaning.py                   Date parsing, safe float, multi-team stripping
    team_mapping.py               30-team canonical 3-letter codes
    sentiment.py                  Social sentiment data handling
  evaluation/
    metrics.py                    accuracy, MAE, RMSE, AUC, Brier, ECE, Spearman
    baselines.py                  Always-home, better-record, prior-SRS, constant-spread, last-season
    report.py                     Markdown report generator

data/                             SACRED — 17 CSV datasets (Basketball Reference)
outputs/models/                   SACRED — production .joblib artifacts
```

## Temporal Splits (IMMUTABLE)

- Train: 2001-2021 (21 seasons, ~13,000 games)
- Validation: 2022-2023 (2 seasons, ~2,000 games)
- Test: 2024-2025 (2 seasons, ~2,000 games)
- Live: 2026 (partial season)

## Current Performance (Baseline NBA_CORE: 0.2254)

| Model | Metric | Current | Baseline | Component Score |
|-------|--------|---------|----------|-----------------|
| Game Winner | Accuracy | 0.6586 | 0.5441 | 0.2511 |
| Point Spread | MAE | 11.05 | 12.56 | 0.1200 |
| Player PTS | MAE | 2.29 | 2.48 | 0.0779 |
| Player AST | MAE | 0.66 | 0.73 | 0.0916 |
| Player REB | MAE | 0.80 | 0.80 | 0.0047 |
| Win Totals | MAE | 9.60 | 8.80 | **0.0000** (WORSE) |
| MVP Race | Spearman | 0.8935 | — | 0.8935 |

## Sacred Imports Available in experiment.py

```python
# Config
from nba_predict.config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, RANDOM_SEED,
    XGBOOST_CLASSIFIER_PARAMS, XGBOOST_REGRESSOR_PARAMS,
)

# Features (read-only pipelines)
from nba_predict.features.matchup_features import build_matchup_dataset, get_feature_columns
from nba_predict.features.game_features import build_game_features
from nba_predict.features.team_features import build_prior_season_features
from nba_predict.features.player_features import build_roster_quality

# Data loaders
from nba_predict.data.loader import (
    load_schedules, load_teams_advanced, load_teams_per_game,
    load_players_advanced, load_players_per_game,
    load_awards, load_rosters, load_standings,
)

# Metrics
from nba_predict.evaluation.metrics import classification_metrics, regression_metrics
from nba_predict.evaluation.baselines import (
    always_home_baseline, better_record_baseline, srs_baseline,
    constant_spread_baseline, last_season_baseline,
)
```

## Available sklearn Algorithms (no install needed)

- `sklearn.ensemble.HistGradientBoostingClassifier` / `Regressor`
- `sklearn.ensemble.RandomForestClassifier` / `Regressor`
- `sklearn.ensemble.GradientBoostingClassifier` / `Regressor`
- `sklearn.ensemble.BaggingClassifier` / `Regressor`
- `sklearn.ensemble.StackingClassifier` / `Regressor`
- `sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.svm.SVR`, `SVC`
- `sklearn.calibration.CalibratedClassifierCV`
- `sklearn.feature_selection.SelectFromModel`, `mutual_info_classif`
