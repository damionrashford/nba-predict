#!/usr/bin/env python3
"""
Train all models and generate a comprehensive evaluation report.

Usage:
    python scripts/evaluate.py         # Train all + generate report
    python scripts/evaluate.py --all   # Same as above
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predict.pipeline import train_all
from nba_predict.evaluation.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate all NBA prediction models")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Train and evaluate all models (default)")
    parser.parse_args()

    print("Training all models and generating evaluation report...\n")
    results = train_all()
    report_path = generate_report(results)
    print(f"\n  Full report: {report_path}")


if __name__ == "__main__":
    main()
