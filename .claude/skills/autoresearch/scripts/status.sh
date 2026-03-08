#!/usr/bin/env bash
# Show current autoresearch status: best score, recent experiments, git state
# Usage: bash scripts/status.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Git State ==="
git log --oneline -5

echo ""
echo "=== Results History ==="
column -t -s $'\t' autoresearch/results.tsv

echo ""
echo "=== Best NBA_CORE ==="
tail -n +2 autoresearch/results.tsv | sort -t$'\t' -k2 -rn | head -1 | cut -f1-3,10-11
