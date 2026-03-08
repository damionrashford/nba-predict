"""Sacred constants for the autoresearch experiment loop.

DO NOT MODIFY. These define the fixed evaluation arena.
The agent's experiment.py is the ONLY editable file.
"""

# ── Baseline Performance (naive heuristics — the floor) ──────────────────
# From production eval_report_20260305_222559.md

BASELINE_GAME_WINNER_ACC = 0.5441       # Always-home baseline accuracy
BASELINE_POINT_SPREAD_MAE = 12.56       # Constant +3.5 baseline MAE
BASELINE_PLAYER_PTS_MAE = 2.48          # Last-season baseline MAE
BASELINE_PLAYER_AST_MAE = 0.73          # Last-season baseline MAE
BASELINE_PLAYER_REB_MAE = 0.80          # Last-season baseline MAE
BASELINE_WIN_TOTALS_MAE = 8.80          # Last-season baseline MAE

# ── Production Best (current models — the bar to beat) ───────────────────

PRODUCTION_GAME_WINNER_ACC = 0.6586
PRODUCTION_POINT_SPREAD_MAE = 11.05
PRODUCTION_PLAYER_PTS_MAE = 2.29
PRODUCTION_PLAYER_AST_MAE = 0.66
PRODUCTION_PLAYER_REB_MAE = 0.80
PRODUCTION_WIN_TOTALS_MAE = 9.60        # NOTE: worse than baseline!
PRODUCTION_MVP_SPEARMAN = 0.8935

# ── Time Budget ──────────────────────────────────────────────────────────

EXPERIMENT_TIMEOUT_SECONDS = 300        # 5 minutes max per experiment run
HARD_KILL_SECONDS = 600                 # 10 minutes = force kill

# ── NBA_CORE Formula Weights ─────────────────────────────────────────────
# Each component is normalized 0-1 where 0 = naive baseline, 1 = perfect.
# Components worse than baseline get clamped to 0.

NBA_CORE_WEIGHTS = {
    "game_winner":  0.20,   # Highest-value prediction
    "point_spread": 0.20,   # Second highest-value
    "player_pts":   0.10,
    "player_ast":   0.10,
    "player_reb":   0.05,   # Hardest to beat baseline
    "win_totals":   0.20,   # Currently WORST model — incentivize fix
    "mvp_race":     0.15,
}
