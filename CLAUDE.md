# NBA Prediction System

## Project Overview

NBA prediction system trained on 26 seasons (2000-2026) of Basketball Reference data. 4 XGBoost models, 200 features, strict temporal anti-leakage, FastMCP agent interface.

## Tech Stack

- Python 3.13, XGBoost, scikit-learn, pandas, numpy, scipy, joblib
- Optuna for hyperparameter tuning
- pytest for unit testing (51 tests)
- FastMCP for agent interface (optional)
- No deep learning — pure gradient-boosted trees

## Directory Structure

```
nba-ref/
├── data/                    # 23 CSV datasets from Basketball Reference
├── nba_predict/             # Core prediction library (sacred — read-only)
│   ├── config.py            # Central config: splits, seeds, paths, XGB defaults
│   ├── pipeline.py          # Training orchestrator (registry + dynamic import)
│   ├── tuning.py            # Optuna hyperparameter search
│   ├── models/
│   │   ├── game_winner.py       # Model 1: Binary classification (home win?)
│   │   ├── point_spread.py      # Model 2: Margin regression
│   │   ├── player_performance.py # Model 3: Multi-target PTS/AST/REB
│   │   └── season_outcomes.py   # Model 4: Win totals + championship + MVP
│   ├── features/
│   │   ├── game_features.py     # Rolling stats with shift(1) anti-leakage
│   │   ├── team_features.py     # Prior-season team quality joins
│   │   ├── player_features.py   # Roster quality aggregation
│   │   ├── matchup_features.py  # Home-away pairing + differentials
│   │   └── selection.py         # Feature selection utilities
│   ├── data/
│   │   ├── loader.py            # 12 CSV loaders with normalization
│   │   ├── cleaning.py          # Data cleaning (dates, team names, floats)
│   │   ├── team_mapping.py      # 30-team canonical name mapping
│   │   ├── sentiment.py         # Social sentiment data handling
│   │   └── social_team_mapping.py
│   └── evaluation/
│       ├── metrics.py           # Accuracy, MAE, RMSE, AUC, Brier, ECE
│       ├── baselines.py         # Naive baselines (always home, last season, etc.)
│       └── report.py            # Markdown report generator
├── scripts/
│   ├── train.py             # Train all or specific models
│   ├── predict.py           # Make predictions on live season
│   ├── evaluate.py          # Train all + generate report
│   ├── generate_predictions.py # Batch-generate all prediction CSVs
│   ├── collect_data.py      # Scrape Basketball Reference (908 lines)
│   ├── collect_sentiment.py # Scrape social sentiment
│   ├── collect_injuries.py  # Scrape injury reports
│   ├── collect_odds.py      # Scrape betting odds
│   └── collect_tracking.py  # Scrape player tracking data
├── agent/
│   ├── agent.py             # FastMCP agent entry point
│   ├── config.yaml          # MCP server config
│   ├── instruction.md       # Agent system prompt
│   └── mcp/
│       ├── nba_mcp.py       # Data query MCP server (7 tools)
│       └── predict_mcp.py   # Prediction MCP server (5 tools)
├── autoresearch/            # Autonomous experiment framework (Karpathy-style)
│   ├── constants.py         # Sacred: baselines, NBA_CORE weights, timeout
│   ├── evaluate.py          # Sacred: computes NBA_CORE composite score
│   ├── experiment.py        # THE ONLY EDITABLE FILE for the agent
│   ├── program.md           # Agent instructions for autonomous loop
│   ├── results.tsv          # Experiment log (33 experiments)
│   ├── injury_features.py   # Injury report feature engineering
│   ├── tracking_features.py # Player tracking feature engineering
│   └── outputs/models/      # Experiment model artifacts (isolated)
├── tests/                   # Unit test suite (51 tests)
│   ├── test_cleaning.py     # Data parsing tests
│   ├── test_team_mapping.py # Team normalization tests
│   ├── test_metrics.py      # Evaluation metric tests
│   └── test_baselines.py    # Baseline model tests
├── outputs/
│   ├── models/              # Trained .joblib artifacts
│   ├── predictions/         # Sample prediction CSVs (6 files)
│   └── reports/             # Evaluation markdown reports
└── notebooks/               # EDA + analysis notebooks (4 notebooks)
```

## Temporal Splits (SACRED — never change)

- Train: 2001-2021 (21 seasons)
- Validation: 2022-2023 (hyperparameter tuning + early stopping)
- Test: 2024-2025 (final held-out evaluation)
- Live: 2026 (partial season for active predictions)

Season 2000 is dropped (no prior-season data for features).

## Commands

```bash
# Train all models
python scripts/train.py

# Train a specific model
python scripts/train.py --model game_winner

# Generate full evaluation report
python scripts/evaluate.py

# Make predictions
python scripts/predict.py --model game_winner --date 2026-03-10
python scripts/predict.py --model player_performance --player "LeBron James"
python scripts/predict.py --model season_outcomes --season 2026

# Generate all sample prediction CSVs
python scripts/generate_predictions.py

# Run unit tests
python -m pytest tests/ -v

# Run autoresearch evaluator (tests experiment.py)
python autoresearch/evaluate.py
```

## Current Performance (exp032 — latest)

| Model | Metric | Value | Baseline | vs Baseline |
|-------|--------|-------|----------|-------------|
| Game Winner | Accuracy | 0.6606 | 0.5441 | +11.7pp |
| Game Winner | AUC-ROC | 0.7189 | — | — |
| Point Spread | MAE | 10.94 pts | 12.56 | +12.9% |
| Player PTS | MAE | 2.25 | 2.48 | +9.3% |
| Player AST | MAE | 0.66 | 0.73 | +9.6% |
| Player REB | MAE | 0.77 | 0.80 | +3.8% |
| Win Totals | MAE | 8.61 | 8.80 | +2.2% |
| MVP Race | Spearman | 0.9111 | — | Strong |
| **NBA_CORE** | Composite | **0.2380** | 0.0 | — |

## AutoResearch System

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) and [nanochat](https://github.com/karpathy/nanochat). An autonomous experiment loop where an LLM agent modifies `autoresearch/experiment.py`, runs evaluations, and keeps/discards results based on NBA_CORE improvement.

**Key rules:**
- `autoresearch/experiment.py` is the ONLY editable file
- `autoresearch/evaluate.py` and `autoresearch/constants.py` are sacred
- All core code in `nba_predict/` is sacred — never modified by the agent
- Experiment models save to `autoresearch/outputs/models/` (isolated from core)
- Results logged to `autoresearch/results.tsv`
- Agent follows instructions in `autoresearch/program.md`

**NBA_CORE** is a weighted composite score (0 = naive baselines, 1 = perfect):
- Game winner accuracy: 20% weight
- Point spread MAE: 20% weight
- Win totals MAE: 20% weight
- MVP Spearman: 15% weight
- Player PTS/AST MAE: 10% each
- Player REB MAE: 5% weight

## Anti-Leakage Design

1. All rolling features use `.shift(1)` — current game's result excluded from its own features
2. Team quality stats joined from prior season only (season N-1 -> season N games)
3. Roster quality aggregated from prior season only
4. Temporal split prevents future season information in training data

## Key Patterns

- Team names: always use 3-letter codes (LAL, BOS, GSW). `team_mapping.py` normalizes all variants.
- Missing values: XGBoost handles natively (learns optimal split direction)
- Game winner: raw XGBoost logistic output (isotonic calibration was dropped — it overfits val)
- Win totals: regression-to-mean shrinkage (s=0.33) with optional Ridge residual correction
- MVP: XGB+Ridge blend (w=0.65, alpha=0.1) for Spearman 0.9111
- Point spread: Huber loss XGB + Ridge blend (65/35)
- Player models: per-target hyperparams and per-target feature exclusions (e.g., REB excludes scoring interactions)
- Model artifacts: `.joblib` dicts containing model + feature_cols + any calibrators
- Predict scripts use `model.get_booster().feature_names` for per-target feature alignment
