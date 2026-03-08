#!/usr/bin/env python3
"""
Train NBA prediction models.

Usage:
    python scripts/train.py                    # Train all models
    python scripts/train.py --model game_winner  # Train one model
    python scripts/train.py --list             # List available models
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predict.pipeline import list_models, train_all, train_model


def main():
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Name of a specific model to train (default: train all)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available model names and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in list_models():
            print(f"  - {name}")
        return

    if args.model:
        train_model(args.model)
    else:
        train_all()


if __name__ == "__main__":
    main()
