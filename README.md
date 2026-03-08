# NBA Prediction System

Production-grade NBA prediction system trained on **26 seasons** (2000-2026) of Basketball Reference data. 4 XGBoost models, 200 engineered features, strict temporal anti-leakage, and an autonomous experiment loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Key Results

Evaluated on a **held-out test set** (2024-2025 seasons) never seen during training or tuning:

| Model | Metric | Score | Baseline | Improvement |
|-------|--------|-------|----------|-------------|
| Game Winner | Accuracy | **66.1%** | 54.4% (always home) | +11.7 pp |
| Game Winner | AUC-ROC | **0.719** | — | — |
| Point Spread | MAE | **10.94 pts** | 12.56 pts (constant +3.5) | +12.9% |
| Player PTS | MAE | **2.25** | 2.48 (last season) | +9.3% |
| Player AST | MAE | **0.66** | 0.73 | +9.6% |
| Player REB | MAE | **0.77** | 0.80 | +3.8% |
| Win Totals | MAE | **8.61 wins** | 8.80 (last season) | +2.2% |
| MVP Race | Spearman ρ | **0.911** | — | Strong |
| **NBA_CORE** | **Composite** | **0.2380** | 0.0 | — |

> **NBA_CORE** is a weighted composite score where 0 = naive baselines and 1 = perfect prediction. After 33 autonomous experiments, improved from 0.2254 → 0.2380 (+5.6%).

---

## Architecture

```
Basketball Reference (26 seasons)
        │
        ▼
┌───────────────────┐
│   Data Collection  │  908-line scraper with proxy rotation,
│   (5 scrapers)     │  rate limiting, team code normalization
└───────┬───────────┘
        │  23 CSV datasets (~39 MB, 335K+ rows)
        ▼
┌───────────────────┐
│ Feature Engineering│  Rolling stats (.shift(1) anti-leakage)
│   (200 features)   │  Prior-season team quality (N-1 joins)
│                    │  Roster quality aggregation
│                    │  Home-away matchup differentials
└───────┬───────────┘
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│  4 XGBoost Models  │────▶│  Evaluation Suite    │
│  Game Winner       │     │  5 naive baselines   │
│  Point Spread      │     │  Accuracy/MAE/AUC    │
│  Player Performance│     │  Calibration (ECE)   │
│  Season Outcomes   │     │  Markdown reports    │
└───────┬───────────┘     └─────────────────────┘
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│  Predictions       │     │  AutoResearch Loop   │
│  CLI interface     │     │  Autonomous tuning   │
│  FastMCP agent     │     │  33 experiments      │
│  (12 tools)        │     │  NBA_CORE tracking   │
└───────────────────┘     └─────────────────────┘
```

---

## How It Works

### The 4 Models

| Model | Task | Algorithm | Key Insight |
|-------|------|-----------|-------------|
| **Game Winner** | Predict home team win (binary) | XGBClassifier | Differential features (home - away) dominate; `diff_win_pct_season` is #1 |
| **Point Spread** | Predict margin of victory | XGBRegressor + Ridge blend | Huber loss + 65/35 XGB-Ridge ensemble reduces outlier sensitivity |
| **Player Performance** | Predict next-season PTS/AST/REB | 3 × XGBRegressor | Interaction features (usage × pts, minutes × pts) capture role changes; per-target feature selection |
| **Season Outcomes** | Win totals + MVP award share | Shrinkage + Ridge; XGB+Ridge | Win totals use regression-to-mean shrinkage; MVP uses XGB+Ridge blend (Spearman 0.911) |

### Anti-Leakage Design

This system enforces **three layers of temporal anti-leakage** to prevent overly optimistic results:

1. **Rolling features use `.shift(1)`** — The current game's outcome is never included in its own features. A 10-game rolling average at game N uses games 1 through N-1.

2. **Prior-season-only joins** — Team quality stats (SRS, Off Rtg, Def Rtg) are joined from season N-1 to predict season N games. No in-season team stats leak.

3. **Strict temporal split** — Train (2001-2021), Validation (2022-2023), Test (2024-2025), Live (2026). No future data ever touches training.

### Feature Engineering Pipeline

| Category | Count | Source | Method |
|----------|-------|--------|--------|
| Rolling game stats | ~30 per side | schedules.csv | 5/10/20-game windows, `.shift(1)` |
| Prior-season team quality | ~25 per side | teams_advanced, teams_per_game | Season N-1 join |
| Roster quality | ~7 per side | players_advanced, rosters | Top-5/8 player BPM, total VORP, depth, continuity |
| Matchup differentials | ~30 | Computed | home - away for all key features |
| Sentiment | ~10 per side | sentiment.csv | Rolling sentiment score/volume/engagement |
| Context | ~10 | schedules.csv | Rest days, B2B, streak, game progress |
| **Total** | **~200** | | |

The strongest predictors are **differential features** — absolute stats matter less than the relative matchup quality between teams.

---

## AutoResearch: Autonomous Experiment Loop

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), an LLM agent autonomously modifies `experiment.py`, runs evaluations, and keeps/discards results based on NBA_CORE improvement.

**Rules:**
- Only `autoresearch/experiment.py` is editable — all production code is sacred
- Each experiment must complete in <5 minutes
- Keep if NBA_CORE improves; discard if it doesn't
- Equal score with simpler code = keep (Occam's razor)

**33 experiments** completed, tracking every change in `results.tsv`:

| Milestone | NBA_CORE | Change |
|-----------|----------|--------|
| Baseline (production) | 0.2254 | — |
| Win totals shrinkage | 0.2281 | +0.0027 |
| Point spread Huber+Ridge | 0.2312 | +0.0031 |
| Player interaction features | 0.2322 | +0.0010 |
| Per-season win total shrinkage | 0.2337 | +0.0015 |
| MVP XGB+Ridge blend | 0.2371 | +0.0034 |
| Drop isotonic calibration | **0.2380** | +0.0009 |

---

## Project Structure

```
nba-ref/
├── nba_predict/                  # Core ML library (sacred — read-only)
│   ├── config.py                 # Splits, seeds, paths, XGB defaults
│   ├── pipeline.py               # Training orchestrator (registry pattern)
│   ├── tuning.py                 # Optuna hyperparameter search
│   ├── models/
│   │   ├── game_winner.py        # Binary classification (home win?)
│   │   ├── point_spread.py       # Margin regression
│   │   ├── player_performance.py # Multi-target PTS/AST/REB
│   │   └── season_outcomes.py    # Win totals + MVP + championships
│   ├── features/
│   │   ├── game_features.py      # Rolling stats with shift(1)
│   │   ├── team_features.py      # Prior-season team quality
│   │   ├── player_features.py    # Roster quality aggregation
│   │   ├── matchup_features.py   # Home-away pairing + differentials
│   │   └── selection.py          # Feature selection utilities
│   ├── data/
│   │   ├── loader.py             # 12 CSV loaders with normalization
│   │   ├── cleaning.py           # Date/team/float parsing
│   │   ├── team_mapping.py       # 30-team canonical mapping
│   │   ├── sentiment.py          # Social sentiment handling
│   │   └── social_team_mapping.py
│   └── evaluation/
│       ├── metrics.py            # Accuracy, MAE, RMSE, AUC, Brier, ECE
│       ├── baselines.py          # 5 naive baselines
│       └── report.py             # Markdown report generator
├── scripts/
│   ├── train.py                  # Train all or specific models
│   ├── predict.py                # Make predictions (live 2026 season)
│   ├── evaluate.py               # Train + generate evaluation report
│   ├── generate_predictions.py   # Batch-generate all prediction CSVs
│   ├── collect_data.py           # Scrape Basketball Reference (908 lines)
│   ├── collect_sentiment.py      # Scrape Reddit + Mastodon sentiment
│   ├── collect_injuries.py       # Scrape injury reports
│   ├── collect_odds.py           # Scrape betting odds
│   └── collect_tracking.py       # Scrape player tracking data
├── autoresearch/                 # Autonomous experiment loop
│   ├── experiment.py             # THE ONLY EDITABLE FILE for agent
│   ├── evaluate.py               # Sacred evaluation harness
│   ├── constants.py              # Sacred baselines + NBA_CORE weights
│   ├── program.md                # Agent protocol instructions
│   ├── results.tsv               # 33-experiment log
│   ├── injury_features.py        # Injury report feature engineering
│   ├── tracking_features.py      # Player tracking feature engineering
│   └── outputs/models/           # Experiment model artifacts (isolated)
├── agent/                        # FastMCP agent interface
│   ├── agent.py                  # Entry point
│   ├── instruction.md            # Agent system prompt
│   ├── config.yaml               # MCP server config
│   └── mcp/
│       ├── nba_mcp.py            # Data query tools (7 tools)
│       └── predict_mcp.py        # Prediction tools (5 tools)
├── tests/                        # Unit test suite (51 tests)
│   ├── test_cleaning.py          # Data parsing tests
│   ├── test_team_mapping.py      # Team normalization tests
│   ├── test_metrics.py           # Evaluation metric tests
│   └── test_baselines.py         # Baseline model tests
├── notebooks/                    # Analysis & visualization
│   ├── 01_eda_data_quality.ipynb # Data audit & distributions
│   ├── 02_feature_engineering.ipynb # Feature construction & validation
│   ├── 03_game_predictions.ipynb # ROC curves, calibration, baselines
│   └── 04_player_season_models.ipynb # Player projections, MVP, win totals
├── data/                         # 23 CSV datasets (~39 MB)
├── outputs/
│   ├── models/                   # Trained .joblib artifacts
│   ├── predictions/              # Sample prediction CSVs (6 files)
│   └── reports/                  # Evaluation markdown reports
└── requirements.txt
```

**7,200+ lines of Python** | **23 datasets** | **26 seasons** | **4 models** | **200 features** | **51 tests**

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/damionrashford/nba-predict.git
cd nba-predict
pip install -r requirements.txt

# Train all models (~3-5 minutes)
python scripts/train.py

# Generate evaluation report
python scripts/evaluate.py

# Make predictions
python scripts/predict.py --model game_winner --date 2026-03-10
python scripts/predict.py --model player_performance --player "LeBron James"
python scripts/predict.py --model season_outcomes --season 2026

# Generate all sample prediction CSVs
python scripts/generate_predictions.py

# Run unit tests
python -m pytest tests/ -v

# Run autonomous experiment
python autoresearch/evaluate.py
```

---

## Data

23 CSV datasets scraped from Basketball Reference covering 2000-2026:

| Category | Datasets | Key Stats |
|----------|----------|-----------|
| **Games** | schedules (65K rows), standings | Scores, margins, streaks, rest days |
| **Teams** | teams_advanced, teams_per_game, teams_opponent, teams_shooting | Off/Def Rtg, SRS, pace, Four Factors |
| **Players** | per_game, advanced, per_36, per_100, totals, shooting, adj_shooting, play_by_play | PER, BPM, VORP, WS, usage, all counting stats |
| **Personnel** | rosters (15K rows), draft, awards | Height, weight, experience, MVP votes |
| **External** | sentiment, injuries (94K rows), tracking, clutch, arenas | Social sentiment, injury reports, movement data |

Data collection uses a **908-line scraper** with proxy rotation (Proxifly + TheSpeedX), parallel execution (10 workers), 3.5s rate limiting, and automatic team code normalization for historical franchise relocations.

---

## Agent Interface

Optional FastMCP agent with 12 tools for natural language NBA queries:

**Data Tools (7):** `query_team_stats`, `query_player_stats`, `query_head_to_head`, `query_standings`, `query_schedule`, `list_teams`, `list_seasons`

**Prediction Tools (5):** `predict_game_winner`, `predict_point_spread`, `predict_player_stats`, `predict_season_wins`, `predict_mvp_race`

```bash
# Interactive agent
cd agent
fast-agent go -i instruction.md -c config.yaml --servers nba_data,nba_predict

# Example: "Who wins tonight: Lakers vs Celtics?"
```

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| [01_eda_data_quality](notebooks/01_eda_data_quality.ipynb) | Dataset inventory, games per season, home win rate trends, margin distributions, rest day analysis, COVID bubble anomaly, missing data audit |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | Rolling feature construction, shift(1) anti-leakage verification, SRS predictive power (R²), roster quality distributions, top 20 correlated features |
| [03_game_predictions](notebooks/03_game_predictions.ipynb) | ROC curve (AUC=0.719), calibration plot, confusion matrix, baseline comparisons, feature importance, predicted vs actual scatter, error by season phase |
| [04_player_season_models](notebooks/04_player_season_models.ipynb) | PTS/AST/REB predicted vs actual, error by age group, win totals with team labels, championship contenders, MVP predicted vs actual award share |

---

## Tech Stack

- **Python 3.13** — Core language
- **XGBoost** — Gradient-boosted trees for all models (no deep learning)
- **scikit-learn** — Ridge regression, StandardScaler, calibration
- **pandas / numpy / scipy** — Data manipulation and statistics
- **Optuna** — Bayesian hyperparameter optimization
- **joblib** — Model serialization
- **matplotlib / seaborn** — Visualization
- **pytest** — Unit testing (51 tests)
- **FastMCP** — Agent tool interface
- **BeautifulSoup / cloudscraper** — Web scraping with Cloudflare bypass

---

## Known Limitations & Future Work

- **Win Totals** (20% of NBA_CORE) — Now beats baseline (8.61 vs 8.80 MAE) via regression-to-mean shrinkage, but still the weakest model due to small sample size (~600 team-seasons)
- **Injury/tracking integration** — Feature modules built (`autoresearch/injury_features.py`, `autoresearch/tracking_features.py`) and wired into experiment pipeline; initial tests show noise from sparse coverage (injuries 2022+ only, tracking 2014+) — needs selective feature gating per model
- **Betting odds** — Collected but not used; could serve as calibration targets or strong baseline features
- **Sentiment sparsity** — Social sentiment only available from 2024+; historical seasons have NaN (XGBoost handles gracefully)
- **No live API** — Predictions via CLI only; no REST endpoint or deployment

---

## License

This project is for educational and portfolio purposes. NBA data sourced from [Basketball Reference](https://www.basketball-reference.com/).
