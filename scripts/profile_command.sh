#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'usage: %s RUN_DIR STAGE INPUT_JSON SIF AF_VERSION GPU(true|false) MEMORY(device|unified) -- COMMAND...\n' "$0" >&2
    exit 2
}

[[ $# -ge 8 ]] || usage
RUN_DIR=$1
STAGE=$2
INPUT_JSON=$3
SIF=$4
AF_VERSION=$5
HAS_GPU=$6
if [[ $# -ge 9 && "$8" == -- ]]; then
    MEMORY_MODE=$7
    shift 8
elif [[ $# -ge 8 && "$7" == -- ]]; then
    # Jobs already queued before memory-mode support are device-only profiles.
    MEMORY_MODE=device
    shift 7
else
    usage
fi
COMMAND=("$@")

[[ "$HAS_GPU" == true || "$HAS_GPU" == false ]] || usage
[[ "$MEMORY_MODE" == device || "$MEMORY_MODE" == unified ]] || usage

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
PROFILE_MODE=memory_characterisation
SAMPLING_INTERVAL_MS=100
mkdir -p "$RUN_DIR/output"

# Explicit profiling defaults. Submission scripts can later expose alternatives
# without changing command capture, sampling, or metadata handling.
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "$MEMORY_MODE" == unified ]]; then
    export APPTAINERENV_TF_FORCE_UNIFIED_MEMORY=true
    export APPTAINERENV_XLA_CLIENT_MEM_FRACTION=3.2
else
    export APPTAINERENV_TF_FORCE_UNIFIED_MEMORY=false
    export APPTAINERENV_XLA_CLIENT_MEM_FRACTION=0.95
fi

GPU_ARGS=()
if [[ "$HAS_GPU" == true ]]; then
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_ARGS+=(--id="$CUDA_VISIBLE_DEVICES")
    nvidia-smi "${GPU_ARGS[@]}" \
        --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits \
        >"$RUN_DIR/gpu_inventory.csv"
else
    : >"$RUN_DIR/gpu_inventory.csv"
fi

/usr/bin/python3 "$PROJECT_ROOT/scripts/profile_metadata.py" create \
    --output "$RUN_DIR/metadata.json" --input "$INPUT_JSON" --sif "$SIF" \
    --mode "$PROFILE_MODE" --stage "$STAGE" --interval "$SAMPLING_INTERVAL_MS" \
    --gpu-inventory "$RUN_DIR/gpu_inventory.csv" \
    --af-version "$AF_VERSION" --af-commit unknown \
    -- "${COMMAND[@]}"

printf '%s\n' 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,utilization_memory_percent,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz' \
    >"$RUN_DIR/gpu.csv"
SAMPLER_PID=""
if [[ "$HAS_GPU" == true ]]; then
    nvidia-smi "${GPU_ARGS[@]}" \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits --loop-ms="$SAMPLING_INTERVAL_MS" \
        >>"$RUN_DIR/gpu.csv" 2>"$RUN_DIR/nvidia-smi.err" &
    SAMPLER_PID=$!
else
    : >"$RUN_DIR/nvidia-smi.err"
fi

cleanup() {
    if [[ -n "$SAMPLER_PID" ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
}

finish_profile() {
    local exit_status=$1
    trap - EXIT INT TERM
    cleanup
    /usr/bin/python3 "$PROJECT_ROOT/scripts/profile_metadata.py" finish \
        --metadata "$RUN_DIR/metadata.json" --time-file "$RUN_DIR/time.txt" \
        --exit-status "$exit_status"
    printf 'Profile directory: %s\n' "$RUN_DIR"
    exit "$exit_status"
}

trap cleanup EXIT
trap 'finish_profile 130' INT
trap 'finish_profile 143' TERM

set +e
LC_ALL=C /usr/bin/time -v -o "$RUN_DIR/time.txt" "${COMMAND[@]}" 2>&1 |
    tee "$RUN_DIR/alphafold.log"
EXIT_STATUS=${PIPESTATUS[0]}
set -e

finish_profile "$EXIT_STATUS"
