#!/usr/bin/env python3
"""Promote autoresearch models to production.

Copies winning experiment artifacts from autoresearch/outputs/models/ to
outputs/models/, then regenerates predictions, eval report, and docs site data.

Usage:
    python scripts/promote.py              # promote + regenerate everything
    python scripts/promote.py --artifacts  # only copy model artifacts
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
AUTORESEARCH_MODELS = PROJECT_ROOT / "autoresearch" / "outputs" / "models"
PRODUCTION_MODELS = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"


def promote_artifacts() -> list[str]:
    """Copy model artifacts from autoresearch → production."""
    if not AUTORESEARCH_MODELS.exists():
        print("  No autoresearch models found. Run an experiment first.")
        return []

    PRODUCTION_MODELS.mkdir(parents=True, exist_ok=True)
    promoted = []

    for src in sorted(AUTORESEARCH_MODELS.glob("*.joblib")):
        dst = PRODUCTION_MODELS / src.name
        shutil.copy2(src, dst)
        size_kb = src.stat().st_size / 1024
        print(f"  {src.name:35s} → outputs/models/ ({size_kb:.0f} KB)")
        promoted.append(src.name)

    return promoted


def regenerate_predictions():
    """Regenerate all prediction CSVs from promoted models."""
    from scripts.generate_predictions import main as gen_main
    gen_main()


def regenerate_eval_report():
    """Retrain + generate fresh eval report using core pipeline."""
    from nba_predict.pipeline import train_all
    from nba_predict.evaluation.report import generate_report
    results = train_all()
    report_path = generate_report(results)
    print(f"\n  Eval report: {report_path}")
    return report_path


def sync_docs_data():
    """Copy prediction CSVs to docs/data/ for GitHub Pages site."""
    if not DOCS_DATA_DIR.exists():
        print("  docs/data/ not found, skipping site sync.")
        return

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    synced = 0
    for csv in sorted(PREDICTIONS_DIR.glob("*.csv")):
        dst = DOCS_DATA_DIR / csv.name
        shutil.copy2(csv, dst)
        synced += 1

    print(f"  Synced {synced} prediction CSVs → docs/data/")


def main():
    parser = argparse.ArgumentParser(description="Promote autoresearch models to production")
    parser.add_argument("--artifacts", action="store_true",
                        help="Only copy model artifacts (skip regeneration)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Promoting Autoresearch Models")
    print("=" * 60)

    # Step 1: Copy artifacts
    print("\n[1/4] Copying model artifacts...")
    promoted = promote_artifacts()
    if not promoted:
        sys.exit(1)

    if args.artifacts:
        print("\n  Done (artifacts only).")
        return

    # Step 2: Regenerate predictions
    print("\n[2/4] Regenerating predictions...")
    regenerate_predictions()

    # Step 3: Sync to docs
    print("\n[3/4] Syncing to docs/data/...")
    sync_docs_data()

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("  Promotion complete!")
    print("=" * 60)
    print(f"  Models promoted: {', '.join(promoted)}")
    print(f"  Predictions: outputs/predictions/")
    print(f"  Site data:   docs/data/")
    print(f"\n  To also regenerate the eval report:")
    print(f"    python scripts/evaluate.py")


if __name__ == "__main__":
    main()
