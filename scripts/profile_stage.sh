#!/usr/bin/env bash

set -euo pipefail

error() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

: "${AF3_SITE:?AF3_SITE must be set by site wrapper}"
: "${AF3_STAGE:?AF3_STAGE must be set by stage wrapper}"

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
AF3_SIF="${AF3_SIF:?AF3_SIF must be set by site wrapper}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:?AF3_MODEL_DIR must be set by site wrapper}"
AF3_DB_DIR="${AF3_DB_DIR:?AF3_DB_DIR must be set by site wrapper}"
AF3_VERSION="${AF3_VERSION:-unknown}"
AF3_COMMIT="${AF3_COMMIT:-unknown}"
PROFILE_MODE="${PROFILE_MODE:-memory_characterisation}"
SAMPLING_INTERVAL_MS="${SAMPLING_INTERVAL_MS:-100}"
PROFILES_DIR="${PROFILES_DIR:-$PROJECT_ROOT/profiles}"

INPUT_SPEC=$1
shift

if [[ "$INPUT_SPEC" == pipeline-job:* ]]; then
    PIPELINE_JOB_ID=${INPUT_SPEC#pipeline-job:}
    [[ "$PIPELINE_JOB_ID" =~ ^[0-9]+$ ]] || error "invalid pipeline job ID: $PIPELINE_JOB_ID"
    shopt -s nullglob
    PIPELINE_OUTPUTS=(
        "$PROJECT_ROOT"/profiles/*-"$PIPELINE_JOB_ID"/output/*/*_data.json
    )
    shopt -u nullglob
    ((${#PIPELINE_OUTPUTS[@]} == 1)) || \
        error "expected one *_data.json for pipeline job $PIPELINE_JOB_ID, found ${#PIPELINE_OUTPUTS[@]}"
    INPUT_JSON="$(realpath "${PIPELINE_OUTPUTS[0]}")"
else
    INPUT_JSON="$(realpath "$INPUT_SPEC")"
fi

case "$PROFILE_MODE" in
baseline)
    JAX_PREALLOCATE=true
    JAX_MEMORY_FRACTION=0.95
    ;;
memory_characterisation)
    JAX_PREALLOCATE=false
    JAX_MEMORY_FRACTION=0.95
    ;;
*) error "PROFILE_MODE must be baseline or memory_characterisation" ;;
esac

if [[ "${AF3_UNIFIED_MEMORY:-false}" == true ]]; then
    FORCE_UNIFIED_MEMORY=true
    JAX_MEMORY_FRACTION=3.2
else
    FORCE_UNIFIED_MEMORY=false
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID}"
RUN_DIR="$PROFILES_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/output"

export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="$JAX_PREALLOCATE"
export APPTAINERENV_TF_FORCE_UNIFIED_MEMORY="$FORCE_UNIFIED_MEMORY"
export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="$JAX_MEMORY_FRACTION"

BIND_ARGS=()
COMMAND_ARGS=()
if [[ "$AF3_SITE" == arc ]]; then
    BIND_ARGS+=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
    COMMAND_ARGS+=(--json_path="$INPUT_JSON" --output_dir="$RUN_DIR/output")
    if [[ "$AF3_STAGE" == data-pipeline ]]; then
        BIND_ARGS+=(--bind "$AF3_DB_DIR:$AF3_DB_DIR:ro")
        COMMAND_ARGS+=(--db_dir="$AF3_DB_DIR")
    else
        BIND_ARGS+=(--bind "$AF3_MODEL_DIR:$AF3_MODEL_DIR:ro")
        COMMAND_ARGS+=(--model_dir="$AF3_MODEL_DIR")
    fi
else
    BIND_ARGS+=(--bind "$PROJECT_ROOT:/root/af_inout")
    COMMAND_ARGS+=(--json_path="/root/af_inout/${INPUT_JSON#"$PROJECT_ROOT/"}"
                   --output_dir="/root/af_inout/profiles/$RUN_ID/output")
    if [[ "$AF3_STAGE" == data-pipeline ]]; then
        BIND_ARGS+=(--bind "$AF3_DB_DIR:/root/public_databases")
        COMMAND_ARGS+=(--db_dir=/root/public_databases)
    else
        BIND_ARGS+=(--bind "$AF3_MODEL_DIR:/root/models")
        COMMAND_ARGS+=(--model_dir=/root/models)
    fi
fi

NV_ARGS=()
if [[ "$AF3_STAGE" != data-pipeline ]]; then
    NV_ARGS=(--nv)
    AF3_JAX_CACHE_DIR="${AF3_JAX_CACHE_DIR:-$HOME/.cache}"
    mkdir -p "$AF3_JAX_CACHE_DIR"
fi

if [[ "$AF3_SITE" == bmrc ]]; then
    RUNNER=(apptainer exec "${NV_ARGS[@]}" "${BIND_ARGS[@]}" "$AF3_SIF"
            /alphafold3_venv/bin/python3 /app/alphafold/run_alphafold.py)
else
    RUNNER=(apptainer run "${NV_ARGS[@]}" "${BIND_ARGS[@]}" "$AF3_SIF")
fi

if [[ "$AF3_STAGE" == data-pipeline ]]; then
    COMMAND=("${RUNNER[@]}" "${COMMAND_ARGS[@]}"
             --run_data_pipeline=true --run_inference=false "$@")
else
    COMMAND=("${RUNNER[@]}" "${COMMAND_ARGS[@]}"
             --jax_compilation_cache_dir="$AF3_JAX_CACHE_DIR"
             --run_data_pipeline=false --run_inference=true "$@")
fi

AF_VERSION="$AF3_VERSION"
AF_COMMIT="$AF3_COMMIT"
if [[ "$AF3_SITE" == arc ]]; then
    AF_VERSION="$(apptainer exec "$AF3_SIF" cat /opt/alphafold3_version)"
    AF_COMMIT="$(apptainer exec "$AF3_SIF" cat /opt/alphafold3_commit)"
fi

GPU_ARGS=()
if [[ "$AF3_STAGE" != data-pipeline ]]; then
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_ARGS+=(--id="$CUDA_VISIBLE_DEVICES")
    nvidia-smi "${GPU_ARGS[@]}" \
        --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits \
        >"$RUN_DIR/gpu_inventory.csv"
else
    : >"$RUN_DIR/gpu_inventory.csv"
fi

python3 "$PROJECT_ROOT/scripts/profile_metadata.py" create \
    --output "$RUN_DIR/metadata.json" --input "$INPUT_JSON" --sif "$AF3_SIF" \
    --mode "$PROFILE_MODE" --stage "$AF3_STAGE" --interval "$SAMPLING_INTERVAL_MS" \
    --gpu-inventory "$RUN_DIR/gpu_inventory.csv" \
    --af-version "$AF_VERSION" --af-commit "$AF_COMMIT" \
    -- "${COMMAND[@]}"

SAMPLER_PID=""
printf '%s\n' 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,utilization_memory_percent,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz' \
    >"$RUN_DIR/gpu.csv"
if [[ "$AF3_STAGE" != data-pipeline ]]; then
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
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
LC_ALL=C /usr/bin/time -v -o "$RUN_DIR/time.txt" "${COMMAND[@]}" 2>&1 |
    tee "$RUN_DIR/alphafold.log"
AF3_EXIT_STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT
python3 "$PROJECT_ROOT/scripts/profile_metadata.py" finish \
    --metadata "$RUN_DIR/metadata.json" --time-file "$RUN_DIR/time.txt" \
    --exit-status "$AF3_EXIT_STATUS"

printf 'Profile directory: %s\n' "$RUN_DIR"
exit "$AF3_EXIT_STATUS"
