#!/usr/bin/env bash

set -euo pipefail

[[ $# -ge 1 ]] || {
    printf 'usage: %s [--after PIPELINE_JOB_ID] DATA_JSON\n' "$0" >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
SBATCH_ARGS=(--chdir="$ROOT")
if [[ "$1" == --after ]]; then
    [[ $# -ge 3 && "$2" =~ ^[0-9]+$ ]] || {
        printf 'error: --after requires a job ID and expected data JSON path\n' >&2
        exit 2
    }
    SBATCH_ARGS+=(--dependency="afterok:$2" --kill-on-invalid-dep=yes)
    shift 2
fi
DATA_JSON="$(realpath -m "$1")"

STATUS=0
for gpu in a100 gh200; do
    if ! sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/inference_${gpu}.sh" "$DATA_JSON"; then
        printf 'error: failed to submit %s inference\n' "$gpu" >&2
        STATUS=1
    fi
done
exit "$STATUS"
