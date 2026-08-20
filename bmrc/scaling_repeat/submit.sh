#!/usr/bin/env bash

set -euo pipefail

TOKENS=(128 256 512 768 1024 1536 2048 2560 3072 4096 5120 6144 7000)
MODE=${1:-device}
FROM=0
HARDWARE=all
[[ "$MODE" == device || "$MODE" == unified ]] || {
    printf 'usage: %s device|unified [FROM_TOKENS] [all|a100|gh200]\n' "$0" >&2
    exit 2
}
(( $# )) && shift
for argument in "$@"; do
    if [[ "$argument" =~ ^[0-9]+$ ]]; then
        FROM=$argument
    elif [[ "$argument" == all || "$argument" == a100 || "$argument" == gh200 ]]; then
        HARDWARE=$argument
    else
        printf 'error: invalid argument %s\n' "$argument" >&2
        exit 2
    fi
done

INDICES=()
for index in "${!TOKENS[@]}"; do
    ((TOKENS[index] >= FROM)) && INDICES+=("$index")
done
((${#INDICES[@]})) || { printf 'error: no token counts >= %s\n' "$FROM" >&2; exit 2; }
ARRAY=$(IFS=,; printf '%s' "${INDICES[*]}")
ROOT="$(git rev-parse --show-toplevel)"
SBATCH_ARGS=(--chdir="$ROOT" --array="$ARRAY%2")
[[ "$MODE" == unified ]] && SBATCH_ARGS+=(--mem=320G)

HARDWARES=(a100 gh200)
[[ "$HARDWARE" != all ]] && HARDWARES=("$HARDWARE")
for hardware in "${HARDWARES[@]}"; do
    sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/scaling_repeat/$hardware.sh" "$MODE"
done
