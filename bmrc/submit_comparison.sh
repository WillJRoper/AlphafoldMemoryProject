#!/usr/bin/env bash

set -euo pipefail

[[ $# -eq 1 ]] || {
    printf 'usage: %s INPUT_JSON\n' "$0" >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
INPUT_JSON="$(realpath "$1")"
JOB_NAME=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$INPUT_JSON")
PIPELINE_JOB_ID=$(sbatch --parsable --chdir="$ROOT" "$ROOT/bmrc/data_pipeline.sh" "$INPUT_JSON")
DATA_JSON="$ROOT/profiles/data-pipeline-$PIPELINE_JOB_ID/output/$JOB_NAME/${JOB_NAME}_data.json"

printf 'Pipeline job: %s\n' "$PIPELINE_JOB_ID"
exec "$ROOT/bmrc/submit_all.sh" --after "$PIPELINE_JOB_ID" "$DATA_JSON"
