# NBA Prediction System

Predict NBA game winners, point spreads, player stats, MVP races, and season win totals using XGBoost trained on 26 seasons of Basketball Reference data.

4 models | 200 features | 66.1% game winner accuracy | strict temporal anti-leakage | autonomous experiment loop ([Karpathy-style](https://github.com/karpathy/autoresearch))

**[Live Demo](https://damionrashford.github.io/nba-predict/)** — interactive charts, model metrics, and browsable predictions

---

## Results

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
| MVP Race | Spearman ρ | **0.911** | — | — |
| **NBA_CORE** | **Composite** | **0.2380** | 0.0 | — |

> **NBA_CORE** is a weighted composite score where 0 = naive baselines and 1 = perfect prediction. After 33 autonomous experiments, improved from 0.2254 → 0.2380 (+5.6%).

---

## Models

| Model | Task | Algorithm | Key Insight |
|-------|------|-----------|-------------|
| **Game Winner** | Predict home team win (binary) | XGBClassifier | Differential features (home − away) dominate |
| **Point Spread** | Predict margin of victory | XGBRegressor + Ridge blend | Huber loss + 65/35 ensemble reduces outlier sensitivity |
| **Player Performance** | Predict next-season PTS/AST/REB | 3 × XGBRegressor | Interaction features (usage × pts) capture role changes |
| **Season Outcomes** | Win totals + MVP award share | Shrinkage + Ridge | Regression-to-mean shrinkage; XGB+Ridge MVP blend |

---

## Anti-Leakage

1. **Rolling features use `.shift(1)`** — current game outcome never included in its own features
2. **Prior-season-only joins** — team quality stats joined from season N-1 to predict season N
3. **Strict temporal split** — Train (2001-2021), Validation (2022-2023), Test (2024-2025), Live (2026)

---

## Features (200 total)

| Category | Examples |
|----------|----------|
| Rolling game stats | 5/10/20-game windows for points, rebounds, assists, win% |
| Prior-season team quality | Off/Def Rtg, SRS, pace, Four Factors from season N-1 |
| Roster quality | Top-5/8 player BPM, total VORP, depth, roster continuity |
| Matchup differentials | home − away for all key features |
| Context | Rest days, back-to-backs, streak, game progress |

Strongest predictors are **differential features** — relative matchup quality matters more than absolute stats.

---

## AutoResearch

An LLM agent autonomously modifies `experiment.py`, runs evaluations, and keeps or discards changes based on NBA_CORE improvement. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

| Milestone | NBA_CORE | Change |
|-----------|----------|--------|
| Baseline | 0.2254 | — |
| Win totals shrinkage | 0.2281 | +0.0027 |
| Point spread Huber+Ridge | 0.2312 | +0.0031 |
| Player interaction features | 0.2322 | +0.0010 |
| Per-season win total shrinkage | 0.2337 | +0.0015 |
| MVP XGB+Ridge blend | 0.2371 | +0.0034 |
| Drop isotonic calibration | **0.2380** | +0.0009 |

---

## Quick Start

```bash
git clone https://github.com/damionrashford/nba-predict.git
cd nba-predict
pip install -r requirements.txt

python scripts/train.py                                          # train all models
python scripts/evaluate.py                                       # generate evaluation report
python scripts/predict.py --model game_winner --date 2026-03-10  # predict a game
python scripts/predict.py --model player_performance --player "LeBron James"
python scripts/predict.py --model season_outcomes --season 2026
python -m pytest tests/ -v                                       # run tests
```

---

## Data

23 CSV datasets scraped from [Basketball Reference](https://www.basketball-reference.com/) (2000-2026, ~39 MB, 335K+ rows):

| Category | Datasets |
|----------|----------|
| Games | `schedules.csv`, `standings.csv` |
| Teams | `teams_advanced.csv`, `teams_per_game.csv`, `teams_opponent.csv`, `teams_shooting.csv` |
| Players | `players_per_game.csv`, `players_advanced.csv`, `players_per_36.csv`, `players_per_100.csv`, `players_totals.csv`, `players_shooting.csv`, `players_adj_shooting.csv`, `players_play_by_play.csv` |
| Personnel | `rosters.csv`, `draft.csv`, `awards.csv` |
| External | `sentiment.csv`, `injuries.csv`, `tracking.csv`, `clutch.csv`, `arenas.csv` |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | [Python 3.13](https://www.python.org/) |
| ML | [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), [Optuna](https://optuna.org/) |
| Data | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Visualization | [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) |
| Scraping | [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/), [cloudscraper](https://github.com/VeNoMouS/cloudscraper) |
| Agent | [FastMCP](https://github.com/jlowin/fastmcp) |
| Testing | [pytest](https://docs.pytest.org/) |
