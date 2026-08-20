#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-all}
PIPELINE_ID=${2:-}
[[ "$MODE" == all || "$MODE" == pipeline || "$MODE" == device || "$MODE" == unified ]] || {
    printf 'usage: %s all|pipeline|device|unified [PIPELINE_ARRAY_ID]\n' "$0" >&2
    exit 2
}
ROOT="$(git rev-parse --show-toplevel)"
MANIFEST="$ROOT/inputs/scaling_real/manifest.tsv"
INPUTS="$ROOT/.runtime/scaling_real/inputs"
python3 "$ROOT/scripts/fetch_scaling_real_inputs.py" "$MANIFEST" "$INPUTS"
COUNT=$(($(wc -l <"$MANIFEST") - 1))
ARRAY="0-$((COUNT - 1))%2"

if [[ "$MODE" == all || "$MODE" == pipeline ]]; then
    PIPELINE_ID=$(sbatch --parsable --clusters=htc --chdir="$ROOT" --array="$ARRAY" \
        "$ROOT/arc/scaling_real/data_pipeline.sh")
    PIPELINE_ID=${PIPELINE_ID%%;*}
    printf 'Pipeline array: %s\n' "$PIPELINE_ID"
fi
[[ "$MODE" == pipeline ]] && exit 0
[[ "$PIPELINE_ID" =~ ^[0-9]+$ ]] || { printf 'error: pipeline array ID required\n' >&2; exit 2; }

MODES=("$MODE")
[[ "$MODE" == all ]] && MODES=(device unified)
for memory_mode in "${MODES[@]}"; do
    SBATCH_ARGS=(--clusters=htc --chdir="$ROOT" --array="$ARRAY"
                 --dependency="aftercorr:$PIPELINE_ID")
    [[ "$memory_mode" == unified ]] && SBATCH_ARGS+=(--mem=320G)
    sbatch "${SBATCH_ARGS[@]}" "$ROOT/arc/scaling_real/a100.sh" "$PIPELINE_ID" "$memory_mode"
done
