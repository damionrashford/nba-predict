"""
Pipeline orchestrator — runs the full train → evaluate workflow.

Orchestrates data loading, feature engineering, model training,
and evaluation in the correct order. Each model module is self-contained,
so the pipeline simply calls each one's train() function.
"""

import time
from datetime import datetime

from nba_predict.config import REPORTS_DIR


# Registry of available models
MODEL_REGISTRY = {
    "game_winner": "nba_predict.models.game_winner",
    "point_spread": "nba_predict.models.point_spread",
    "player_performance": "nba_predict.models.player_performance",
    "season_outcomes": "nba_predict.models.season_outcomes",
}


def _import_model(model_name: str):
    """Dynamically import a model module."""
    import importlib
    module_path = MODEL_REGISTRY[model_name]
    return importlib.import_module(module_path)


def train_model(model_name: str) -> dict:
    """Train a single model by name. Returns its evaluation results."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    module = _import_model(model_name)
    start = time.time()
    results = module.train()
    elapsed = time.time() - start

    print(f"\n  [{model_name}] Training completed in {elapsed:.1f}s")
    return results


def train_all() -> dict[str, dict]:
    """Train all models in order. Returns dict of all results."""
    print("=" * 60)
    print(f"  NBA PREDICTION SYSTEM — Full Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}
    total_start = time.time()

    for name in MODEL_REGISTRY:
        print(f"\n{'━' * 60}")
        print(f"  MODEL: {name}")
        print(f"{'━' * 60}")

        try:
            results = train_model(name)
            all_results[name] = results
        except Exception as e:
            print(f"  ERROR training {name}: {e}")
            all_results[name] = {"error": str(e)}

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  All models trained in {total_elapsed:.1f}s")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return all_results


def list_models() -> list[str]:
    """List available model names."""
    return list(MODEL_REGISTRY.keys())
