#!/usr/bin/env bash

set -euo pipefail

[[ $# -ge 1 ]] || {
    printf 'usage: %s DATA_JSON [AF3_ARGS...]\n' "$0" >&2
    printf '       %s --after PIPELINE_JOB_ID [DATA_JSON] [AF3_ARGS...]\n' "$0" >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
SBATCH_ARGS=(--chdir="$ROOT")
if [[ "$1" == --after ]]; then
    [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        printf 'error: --after requires a numeric pipeline job ID\n' >&2
        exit 2
    }
    PIPELINE_JOB_ID=$2
    SBATCH_ARGS+=(--dependency="afterok:$PIPELINE_JOB_ID" --kill-on-invalid-dep=yes)
    shift 2
    if [[ $# -ge 1 && "$1" != --* ]]; then
        INPUT_JSON="$(realpath -m "$1")"
        shift
    else
        INPUT_JSON="pipeline-job:$PIPELINE_JOB_ID"
    fi
else
    INPUT_JSON="$(realpath "$1")"
    shift
fi

for gpu in a100 v100 gh200; do
    sbatch "${SBATCH_ARGS[@]}" "$ROOT/bmrc/inference/$gpu/profile_alphafold.sh" \
        "$INPUT_JSON" "$@"
done
