#!/usr/bin/env bash
# Discard the last experiment (reset to previous commit)
# Usage: bash scripts/discard.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CURRENT=$(git rev-parse --short HEAD)
echo "Discarding experiment ${CURRENT}..."
git reset --hard HEAD~1
NEW=$(git rev-parse --short HEAD)
echo "Reset to ${NEW}"
