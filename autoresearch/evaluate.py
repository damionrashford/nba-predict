#!/usr/bin/env python3
"""Sacred evaluator — computes NBA_CORE from experiment results.

DO NOT MODIFY. This is the immutable evaluation harness.
The agent's experiment.py is the ONLY editable file.

Usage:
    python autoresearch/evaluate.py
    python autoresearch/evaluate.py > autoresearch/run.log 2>&1
"""

import signal
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.constants import (
    BASELINE_GAME_WINNER_ACC,
    BASELINE_PLAYER_AST_MAE,
    BASELINE_PLAYER_PTS_MAE,
    BASELINE_PLAYER_REB_MAE,
    BASELINE_POINT_SPREAD_MAE,
    BASELINE_WIN_TOTALS_MAE,
    EXPERIMENT_TIMEOUT_SECONDS,
    HARD_KILL_SECONDS,
    NBA_CORE_WEIGHTS,
)


class ExperimentTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExperimentTimeout(
        f"Experiment exceeded {EXPERIMENT_TIMEOUT_SECONDS}s timeout"
    )


def compute_nba_core(results: dict) -> tuple[float, dict[str, float]]:
    """Compute the composite NBA_CORE score.

    Each component is normalized to [0, 1]:
      - 0 = naive baseline performance
      - 1 = perfect performance
    Components worse than baseline are clamped to 0.

    Returns:
        (nba_core_score, component_scores_dict)
    """
    components = {}

    # Classification: (accuracy - baseline) / (1 - baseline)
    if "game_winner" in results and "accuracy" in results["game_winner"]:
        acc = results["game_winner"]["accuracy"]
        raw = (acc - BASELINE_GAME_WINNER_ACC) / (1.0 - BASELINE_GAME_WINNER_ACC)
        components["game_winner"] = max(0.0, min(1.0, raw))

    # Regression: 1 - (model_mae / baseline_mae)
    regression_map = {
        "point_spread": BASELINE_POINT_SPREAD_MAE,
        "player_pts": BASELINE_PLAYER_PTS_MAE,
        "player_ast": BASELINE_PLAYER_AST_MAE,
        "player_reb": BASELINE_PLAYER_REB_MAE,
        "win_totals": BASELINE_WIN_TOTALS_MAE,
    }

    for key, baseline_mae in regression_map.items():
        if key in results and "mae" in results[key]:
            raw = 1.0 - results[key]["mae"] / baseline_mae
            components[key] = max(0.0, min(1.0, raw))

    # Correlation: raw Spearman rho
    if "mvp_race" in results and "spearman" in results["mvp_race"]:
        val = results["mvp_race"]["spearman"]
        if val is not None and val == val:  # not NaN
            components["mvp_race"] = max(0.0, min(1.0, val))

    # Weighted sum
    score = sum(
        NBA_CORE_WEIGHTS.get(k, 0) * v for k, v in components.items()
    )

    return score, components


def validate_results(results: dict) -> list[str]:
    """Check that experiment returned the expected format."""
    errors = []
    required_keys = {
        "game_winner": ["accuracy"],
        "point_spread": ["mae"],
        "player_pts": ["mae"],
        "player_ast": ["mae"],
        "player_reb": ["mae"],
        "win_totals": ["mae"],
        "mvp_race": ["spearman"],
    }

    for model_key, metric_keys in required_keys.items():
        if model_key not in results:
            errors.append(f"Missing model key: {model_key}")
            continue
        for mk in metric_keys:
            if mk not in results[model_key]:
                errors.append(f"Missing metric: {model_key}.{mk}")

    return errors


def main():
    print("=" * 60)
    print("  AutoResearch — NBA_CORE Evaluation")
    print("=" * 60)

    # Set timeout (Unix only — graceful degradation on Windows)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(HARD_KILL_SECONDS)
    except (AttributeError, OSError):
        pass  # Windows or no signal support

    start = time.time()

    try:
        from autoresearch.experiment import run_experiment

        print("\nRunning experiment...")
        results = run_experiment()
        elapsed = time.time() - start

        # Cancel alarm
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass

        print(f"\nExperiment completed in {elapsed:.1f}s")

        if elapsed > EXPERIMENT_TIMEOUT_SECONDS:
            print(f"WARNING: Exceeded soft timeout ({EXPERIMENT_TIMEOUT_SECONDS}s)")

        # Validate output format
        errors = validate_results(results)
        if errors:
            print("\nVALIDATION ERRORS:")
            for e in errors:
                print(f"  - {e}")
            print("\nExperiment produced incomplete results.")
            sys.exit(1)

        # Compute NBA_CORE
        nba_core, components = compute_nba_core(results)

        # Print results in grep-friendly format
        print(f"\n{'─' * 60}")
        print("  RESULTS")
        print(f"{'─' * 60}")

        # Per-model raw metrics
        gw = results["game_winner"]
        print(f"  game_winner_acc:    {gw['accuracy']:.4f}"
              f"  (auc: {gw.get('auc_roc', 0):.4f})")

        ps = results["point_spread"]
        print(f"  point_spread_mae:   {ps['mae']:.2f}"
              f"  (derived_acc: {ps.get('derived_accuracy', 0):.4f})")

        print(f"  player_pts_mae:     {results['player_pts']['mae']:.2f}")
        print(f"  player_ast_mae:     {results['player_ast']['mae']:.2f}")
        print(f"  player_reb_mae:     {results['player_reb']['mae']:.2f}")
        print(f"  win_totals_mae:     {results['win_totals']['mae']:.2f}")

        mvp = results["mvp_race"]
        print(f"  mvp_spearman:       {mvp['spearman']:.4f}"
              f"  (mae: {mvp.get('mae', 0):.4f})")

        # Per-component normalized scores
        print(f"\n{'─' * 60}")
        print("  NBA_CORE COMPONENTS (0=baseline, 1=perfect)")
        print(f"{'─' * 60}")
        for key in NBA_CORE_WEIGHTS:
            val = components.get(key, 0.0)
            weight = NBA_CORE_WEIGHTS[key]
            contrib = weight * val
            print(f"  component:{key:20s}  score={val:.4f}  "
                  f"weight={weight:.2f}  contrib={contrib:.4f}")

        # The ONE number
        print(f"\n{'=' * 60}")
        print(f"nba_core: {nba_core:.6f}")
        print(f"{'=' * 60}")
        print(f"\n  elapsed: {elapsed:.1f}s")

    except ExperimentTimeout:
        elapsed = time.time() - start
        print(f"\nEXPERIMENT TIMED OUT after {elapsed:.1f}s")
        print(f"nba_core: TIMEOUT")
        sys.exit(2)

    except Exception:
        elapsed = time.time() - start
        print(f"\nEXPERIMENT CRASHED after {elapsed:.1f}s")
        traceback.print_exc()
        print(f"nba_core: CRASH")
        sys.exit(1)


if __name__ == "__main__":
    main()
