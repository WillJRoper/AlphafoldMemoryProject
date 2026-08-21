#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-all}
PIPELINE_ID=""
HARDWARE=all
[[ "$MODE" == all || "$MODE" == pipeline || "$MODE" == device || "$MODE" == preallocated || "$MODE" == unified ]] || {
    printf 'usage: %s all|pipeline|device|preallocated|unified [PIPELINE_ARRAY_ID] [all|a100|gh200]\n' "$0" >&2
    exit 2
}
(( $# )) && shift
for argument in "$@"; do
    if [[ "$argument" =~ ^[0-9]+$ ]]; then
        PIPELINE_ID=$argument
    elif [[ "$argument" == all || "$argument" == a100 || "$argument" == gh200 ]]; then
        HARDWARE=$argument
    else
        printf 'error: invalid argument %s\n' "$argument" >&2
        exit 2
    fi
done
ROOT="$(git rev-parse --show-toplevel)"
MANIFEST="$ROOT/inputs/scaling_real/manifest.tsv"
INPUTS="$ROOT/.runtime/scaling_real/inputs"
python3 "$ROOT/scripts/fetch_scaling_real_inputs.py" "$MANIFEST" "$INPUTS"
COUNT=$(($(wc -l <"$MANIFEST") - 1))
ARRAY="0-$((COUNT - 1))%2"

if [[ "$MODE" == all || "$MODE" == pipeline ]]; then
    PIPELINE_ID=$(sbatch --parsable --chdir="$ROOT" --array="$ARRAY" \
        "$ROOT/bmrc/scaling_real/data_pipeline.sh")
    PIPELINE_ID=${PIPELINE_ID%%;*}
    printf 'Pipeline array: %s\n' "$PIPELINE_ID"
fi
[[ "$MODE" == pipeline ]] && exit 0
[[ "$PIPELINE_ID" =~ ^[0-9]+$ ]] || { printf 'error: pipeline array ID required\n' >&2; exit 2; }

MODES=("$MODE")
[[ "$MODE" == all ]] && MODES=(device preallocated unified)
for memory_mode in "${MODES[@]}"; do
    SBATCH_ARGS=(--chdir="$ROOT" --array="$ARRAY" --dependency="aftercorr:$PIPELINE_ID")
    [[ "$memory_mode" == unified ]] && SBATCH_ARGS+=(--mem=320G)
    HARDWARES=(a100 gh200)
    [[ "$HARDWARE" != all ]] && HARDWARES=("$HARDWARE")
    for hardware in "${HARDWARES[@]}"; do
        sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/scaling_real/$hardware.sh" \
            "$PIPELINE_ID" "$memory_mode"
    done
done
