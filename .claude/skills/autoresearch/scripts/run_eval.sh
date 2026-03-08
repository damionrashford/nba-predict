#!/usr/bin/env bash
# Run autoresearch evaluation and extract results
# Usage: bash scripts/run_eval.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Running autoresearch evaluation ==="
python autoresearch/evaluate.py > autoresearch/run.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "CRASH (exit code $EXIT_CODE)"
    echo "--- Last 20 lines of run.log ---"
    tail -20 autoresearch/run.log
    exit 1
fi

echo "=== Results ==="
grep "^nba_core:\|^  game_winner\|^  point_spread\|^  player_\|^  win_totals\|^  mvp_" autoresearch/run.log
