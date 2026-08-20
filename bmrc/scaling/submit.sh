#!/usr/bin/env bash

set -euo pipefail

TOKENS=(128 256 512 768 1024 1536 2048 2560 3072 4096 5120 6144 7000)
MODE=${1:-device}
FROM=${2:-0}
[[ "$MODE" == device || "$MODE" == unified ]] || {
    printf 'usage: %s device|unified [FROM_TOKENS]\n' "$0" >&2
    exit 2
}

INDICES=()
for index in "${!TOKENS[@]}"; do
    ((TOKENS[index] >= FROM)) && INDICES+=("$index")
done
((${#INDICES[@]})) || { printf 'error: no token counts >= %s\n' "$FROM" >&2; exit 2; }
ARRAY=$(IFS=,; printf '%s' "${INDICES[*]}")
ROOT="$(git rev-parse --show-toplevel)"
SBATCH_ARGS=(--chdir="$ROOT" --array="$ARRAY%2")
[[ "$MODE" == unified ]] && SBATCH_ARGS+=(--mem=320G)

sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/scaling/a100.sh" "$MODE"
sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/scaling/gh200.sh" "$MODE"
