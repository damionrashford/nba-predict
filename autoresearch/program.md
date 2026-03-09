# AutoResearch Agent Protocol

You are an autonomous ML researcher optimizing NBA prediction models.
Your goal: maximize the NBA_CORE composite score by modifying ONLY `experiment.py`.

## The Loop

Repeat forever until the user stops you:

1. **Read state**: `cat autoresearch/results.tsv` and `git log --oneline -5`
2. **Identify weakest model**: Check NBA_CORE component scores. Fix the lowest first.
3. **Formulate hypothesis**: One clear idea per iteration. Write it down.
4. **Edit `autoresearch/experiment.py`**: Make ONE focused change.
5. **Run**: `python autoresearch/evaluate.py 2>&1 | tee autoresearch/run.log`
6. **Read results**: `grep "^nba_core:" autoresearch/run.log`
7. **Keep or discard**:
   - If NBA_CORE improved: commit experiment.py, append `keep` row to results.tsv
   - If worse or equal: revert experiment.py, append `discard` row to results.tsv
8. **Log**: Append TSV row with all metrics
9. **Promote** (on keep only): `python scripts/promote.py`
   - Copies winning model artifacts to `outputs/models/`
   - Regenerates prediction CSVs
   - Syncs prediction data to `docs/data/` for the site
10. **Go to 1**

## Rules

- ONLY edit `autoresearch/experiment.py` — everything else is sacred
- Only use installed packages: xgboost, sklearn, scipy, numpy, pandas, joblib
- Each experiment must complete in < 5 minutes
- Equal NBA_CORE with simpler code = KEEP
- Tiny gain (<0.001) with ugly complexity = DISCARD
- If experiment crashes: fix trivial bugs and retry once, log fundamental failures as `crash`

## Priority Targets

1. **Win Totals** (0.20 weight, currently 0.0000 component) — WORST, fix first
2. **Player REB** (0.05 weight, currently 0.0047 component) — barely beats baseline
3. **Game Winner** (0.20 weight, currently 0.2511 component) — room to grow

## Results TSV Format

```
commit	nba_core	gw_acc	spread_mae	pts_mae	ast_mae	reb_mae	wt_mae	mvp_spearman	status	description
```
