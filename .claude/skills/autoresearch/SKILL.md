---
name: autoresearch
description: >
  Autonomous ML experiment loop for improving NBA prediction models.
  Modifies autoresearch/experiment.py, runs evaluations, keeps/discards results
  based on NBA_CORE composite score. Use when user says "run autoresearch",
  "improve models", "run experiments", "optimize predictions", or "autoresearch loop".
allowed-tools: Read Edit Write Bash Grep Glob
---

# AutoResearch: Autonomous NBA Model Optimization

Run an autonomous experiment loop that improves the NBA prediction system by
modifying `autoresearch/experiment.py`, evaluating changes via NBA_CORE score,
and keeping only improvements. Inspired by Karpathy's autoresearch pattern.

## When to Use

- "Run autoresearch"
- "Improve the models"
- "Run the experiment loop"
- "Optimize predictions"
- "Fix win totals"
- "Increase NBA_CORE"

## Architecture

Read these files to understand the system before starting:

- [autoresearch/program.md](../../../autoresearch/program.md) — Full agent protocol
- [autoresearch/constants.py](../../../autoresearch/constants.py) — Sacred baselines and NBA_CORE weights
- [autoresearch/evaluate.py](../../../autoresearch/evaluate.py) — Sacred evaluator (DO NOT EDIT)
- [autoresearch/experiment.py](../../../autoresearch/experiment.py) — THE ONLY EDITABLE FILE
- [autoresearch/results.tsv](../../../autoresearch/results.tsv) — Experiment log

See [references/architecture.md](references/architecture.md) for the full system map.

## The Loop

Execute this loop continuously until stopped:

### Step 1: Assess State

```bash
# Check current best and recent experiments
cat autoresearch/results.tsv
git log --oneline -10
```

### Step 2: Formulate Hypothesis

Read `autoresearch/experiment.py` and identify the weakest model. Prioritize:

1. **Win Totals** — currently WORSE than baseline (9.60 MAE vs 8.80). Fix this first.
2. **Player REB** — barely beats baseline (0.80 vs 0.80). Nearly zero improvement.
3. **Game Winner** — at 65.86%, room to push toward 67-68%.

### Step 3: Modify experiment.py

Edit ONLY `autoresearch/experiment.py`. You may change:
- Model algorithms (XGBoost, HistGradientBoosting, Ridge, ElasticNet, etc.)
- Hyperparameters (depth, learning rate, regularization, blend weights)
- Feature selection (drop low-importance features, add interactions)
- Ensemble strategies (stacking, different blend ratios)
- Calibration methods (Platt, isotonic, threshold tuning)

You may NOT change:
- Any file in `nba_predict/`, `data/`, `scripts/`, or `agent/`
- The temporal splits (Train 2001-2021, Val 2022-2023, Test 2024-2025)
- The evaluation metric definitions in `autoresearch/evaluate.py`
- The baseline constants in `autoresearch/constants.py`

### Step 4: Run Experiment

```bash
python autoresearch/evaluate.py 2>&1 | tee autoresearch/run.log
```

If it takes longer than 5 minutes, kill it and treat as timeout.

### Step 5: Read Results

```bash
grep "^nba_core:" autoresearch/run.log
grep "component:" autoresearch/run.log
```

### Step 6: Keep or Discard

Extract the nba_core value. Compare to the best score in `autoresearch/results.tsv`.

**If NBA_CORE improved**: Append a row to results.tsv with status `keep`.
**If NBA_CORE same or worse**: Revert experiment.py to the previous version. Append with status `discard`.

### Step 7: Log Results

Append a tab-separated row to `autoresearch/results.tsv`:

```
commit_hash	nba_core	gw_acc	spread_mae	pts_mae	ast_mae	reb_mae	wt_mae	mvp_spearman	status	description
```

### Step 8: Repeat

Go back to Step 1. NEVER STOP unless the user interrupts.

## Constraints

- ONLY edit `autoresearch/experiment.py`
- Use only packages already installed: xgboost, sklearn, scipy, numpy, pandas, joblib
- Equal NBA_CORE with simpler code = KEEP (prefer simplicity)
- Tiny NBA_CORE gain (<0.001) with ugly complexity = DISCARD
- If experiment crashes with trivial bug, fix and retry once. Log fundamental failures as `crash`.

## Experiment Ideas (ordered by expected impact)

See [references/experiment-ideas.md](references/experiment-ideas.md) for the full prioritized list.

**Quick wins to try first:**
1. Win totals: pure Ridge (drop XGBoost from the blend entirely)
2. Win totals: try blend ratios 20/80, 10/90, 0/100
3. Win totals: tune Ridge alpha (1.0, 5.0, 20.0, 50.0)
4. Game winner: drop features with importance < 0.005
5. Win totals: ElasticNet or BayesianRidge

## NBA_CORE Formula

Weighted composite score (0 = naive baseline, 1 = perfect):

| Component | Weight | Normalization |
|-----------|--------|---------------|
| game_winner | 0.20 | (accuracy - 0.5441) / (1 - 0.5441) |
| point_spread | 0.20 | 1 - (mae / 12.56) |
| win_totals | 0.20 | 1 - (mae / 8.80) |
| mvp_race | 0.15 | raw spearman rho |
| player_pts | 0.10 | 1 - (mae / 2.48) |
| player_ast | 0.10 | 1 - (mae / 0.73) |
| player_reb | 0.05 | 1 - (mae / 0.80) |

Components worse than baseline are clamped to 0. Current baseline NBA_CORE: **0.2254**.
