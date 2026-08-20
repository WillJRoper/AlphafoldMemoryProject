#!/usr/bin/env bash

set -euo pipefail

[[ $# -ge 1 ]] || {
    printf 'usage: %s DATA_JSON [AF3_ARGS...]\n' "$0" >&2
    exit 2
}

ROOT="$(git rev-parse --show-toplevel)"
INPUT_JSON="$(realpath "$1")"
shift
for gpu in a100 v100 gh200; do
    sbatch --chdir="$ROOT" "$ROOT/bmrc/inference/$gpu/profile_alphafold.sh" \
        "$INPUT_JSON" "$@"
done
