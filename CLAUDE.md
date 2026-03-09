# NBA Prediction System

## Project Overview

NBA prediction system trained on 26 seasons (2000-2026) of Basketball Reference data. 4 XGBoost models, 200 features, strict temporal anti-leakage, FastMCP agent interface.

## Tech Stack

- Python 3.13, XGBoost, scikit-learn, pandas, numpy, scipy, joblib
- Optuna for hyperparameter tuning
- pytest for unit testing (51 tests)
- FastMCP for agent interface (optional)
- No deep learning — pure gradient-boosted trees

## Temporal Splits (SACRED — never change)

- Train: 2001-2021 (21 seasons)
- Validation: 2022-2023 (hyperparameter tuning + early stopping)
- Test: 2024-2025 (final held-out evaluation)
- Live: 2026 (partial season for active predictions)

Season 2000 is dropped (no prior-season data for features).

## Commands

```bash
# Train all models (uses latest experiment improvements)
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

# Promote autoresearch models → production + regenerate outputs
python scripts/promote.py

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

## Workflows

### Training (standard)

`python scripts/train.py` trains all 4 models using `nba_predict/models/`. Models save to `outputs/models/`. This always produces the latest best configuration (all autoresearch improvements are ported back into the core model code).

### AutoResearch (experiment loop)

Autonomous experiment loop where an LLM agent modifies `autoresearch/experiment.py`, runs evaluations, and keeps/discards results based on NBA_CORE improvement. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

**The loop:**
1. Agent modifies `autoresearch/experiment.py` (ONE focused change)
2. Runs `python autoresearch/evaluate.py` → computes NBA_CORE
3. If improved: commit + append `keep` to `results.tsv`
4. If worse: revert + append `discard` to `results.tsv`
5. **Promote** (on keep): `python scripts/promote.py`
   - Copies winning artifacts to `outputs/models/`
   - Regenerates prediction CSVs
   - Syncs data to `docs/data/` for the site
6. Repeat

**Sacred files (never modified by the agent):**
- `autoresearch/evaluate.py` — immutable evaluation harness
- `autoresearch/constants.py` — baselines, NBA_CORE weights, timeout
- `nba_predict/config.py` — splits, seeds, paths
- `nba_predict/features/**` — feature engineering
- `nba_predict/data/**` — data loading
- `nba_predict/evaluation/**` — metrics and baselines

**Editable by the agent:**
- `autoresearch/experiment.py` — the experiment sandbox
- `nba_predict/models/**` — core model code (updated after winning experiments)
- `scripts/**` — training and prediction scripts

### Promotion Pipeline

When an autoresearch experiment wins, improvements flow back to production:

```
experiment.py wins → python scripts/promote.py
    → copies .joblib artifacts to outputs/models/
    → regenerates outputs/predictions/*.csv
    → syncs docs/data/ for GitHub Pages site
```

After promotion, the agent should also port code improvements from `experiment.py` back into the corresponding `nba_predict/models/` files so `scripts/train.py` always produces the best models.

**NBA_CORE** is a weighted composite score (0 = naive baselines, 1 = perfect):
- Game winner accuracy: 20% weight
- Point spread MAE: 20% weight
- Win totals MAE: 20% weight
- MVP Spearman: 15% weight
- Player PTS/AST MAE: 10% each
- Player REB MAE: 5% weight

## Anti-Leakage Design

1. All rolling features use `.shift(1)` — current game's result excluded from its own features
2. Team quality stats joined from prior season only (season N-1 → season N games)
3. Roster quality aggregated from prior season only
4. Temporal split prevents future season information in training data

## Key Patterns

- Team names: always use 3-letter codes (LAL, BOS, GSW). `team_mapping.py` normalizes all variants.
- Missing values: XGBoost handles natively (learns optimal split direction)
- Game winner: raw XGBoost logistic output (isotonic calibration dropped — overfits val)
- Win totals: regression-to-mean shrinkage (s=0.33) with optional Ridge residual correction
- MVP: XGB+Ridge blend (w=0.65, alpha=0.1) for Spearman 0.9111
- Point spread: Huber loss XGB + Ridge blend (65/35)
- Player models: per-target hyperparams and per-target feature exclusions (REB excludes scoring interactions)
- Model artifacts: `.joblib` dicts containing model + feature_cols + any calibrators/scalers
- Predict scripts use `model.get_booster().feature_names` for per-target feature alignment
