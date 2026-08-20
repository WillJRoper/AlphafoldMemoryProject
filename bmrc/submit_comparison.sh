#!/usr/bin/env bash

set -euo pipefail

[[ $# -ge 1 ]] || {
    printf 'usage: %s INPUT_JSON [AF3_ARGS...]\n' "$0" >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
INPUT_JSON="$(realpath "$1")"
shift
PIPELINE_JOB_ID=$(sbatch --parsable --chdir="$ROOT" \
    "$ROOT/bmrc/data_pipeline/profile_alphafold.sh" "$INPUT_JSON")

printf 'Pipeline job: %s\n' "$PIPELINE_JOB_ID"
exec "$ROOT/bmrc/inference/submit_all.sh" --after "$PIPELINE_JOB_ID" "$@"
