#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
# Non-monotonic order reduces systematic warm-cache bias across CPU counts.
SETTINGS=(16:4 4:1 32:8 8:2 48:12 24:6)
PREVIOUS=""

for setting in "${SETTINGS[@]}"; do
    CPUS=${setting%%:*}
    THREADS=${setting##*:}
    SBATCH_ARGS=(
        --parsable
        --chdir="$ROOT"
        --array=0-8%1
        --cpus-per-task="$CPUS"
        --job-name="af3-pipe-c${CPUS}"
    )
    [[ -z "$PREVIOUS" ]] || SBATCH_ARGS+=(--dependency="afterany:$PREVIOUS")
    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        "$ROOT/bmrc/scaling_pipeline/job.sh" "$THREADS")
    JOB_ID=${JOB_ID%%;*}
    printf 'Pipeline scaling array: %s cpus=%s jackhmmer_threads=%s\n' \
        "$JOB_ID" "$CPUS" "$THREADS"
    PREVIOUS=$JOB_ID
done
