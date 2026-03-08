#!/usr/bin/env bash
# Append experiment result to results.tsv
# Usage: bash scripts/append_result.sh <nba_core> <gw_acc> <spread_mae> <pts_mae> <ast_mae> <reb_mae> <wt_mae> <mvp_spearman> <status> <description>

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "$#" -lt 10 ]; then
    echo "Usage: $0 <nba_core> <gw_acc> <spread_mae> <pts_mae> <ast_mae> <reb_mae> <wt_mae> <mvp_spearman> <status> <description>"
    exit 1
fi

COMMIT=$(git rev-parse --short HEAD)
NBA_CORE=$1; GW=$2; SPREAD=$3; PTS=$4; AST=$5; REB=$6; WT=$7; MVP=$8; STATUS=$9; DESC=${10}

echo -e "${COMMIT}\t${NBA_CORE}\t${GW}\t${SPREAD}\t${PTS}\t${AST}\t${REB}\t${WT}\t${MVP}\t${STATUS}\t${DESC}" >> autoresearch/results.tsv
echo "Appended: ${COMMIT} | NBA_CORE=${NBA_CORE} | ${STATUS} | ${DESC}"
