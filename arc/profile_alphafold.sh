#!/usr/bin/env bash

# Defaults below target the GPU (complete/inference) case on ARC's htc
# cluster. `arc-login` submits to the `arc` cluster by default, so htc is
# selected explicitly here rather than relying on the login node in use.
# The data-pipeline stage needs no GPU and a very different resource shape
# (see colleague's proven submit_dp_* script); override every one of these
# via sbatch CLI flags for that case, e.g.:
#   sbatch --partition=medium,long --gpus-per-node=0 --constraint= \
#     --mem=128G --cpus-per-task=32 --time=2-00:00:00 \
#     --export=ALL,AF3_STAGE=data-pipeline \
#     arc/profile_alphafold.sh inputs/lysozyme_1lyz.json
#SBATCH --job-name=af3-profile
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-profile-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ARC's default python3 may be too old for profile_metadata.py (needs 3.11+).
# Check `module spider Python` on an interactive node and adjust if this fails.
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true

set -euo pipefail

# ============================== Configuration ==============================

# Repository path from which sbatch was called. Slurm executes a copy of this
# script from its spool directory, so BASH_SOURCE does not point to the checkout.
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

# Our own submodule-built image (see containers/alphafold3-v3.0.4.def). Build
# an x86_64 image on an ARC interactive/A100 node, or reuse an ARM64 build on
# the single GH200 node (htc-g057).
AF3_SIF="${AF3_SIF:-$PROJECT_ROOT/images/alphafold3-v3.0.4-x86_64.sif}"

# Only these two paths are needed from the shared artefact directory; the SIF
# and AlphaFold source come from our own submodule/image instead.
AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/model_param}"
AF3_DB_DIR="${AF3_DB_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/public_databases}"

# JAX compilation cache, shared across runs to avoid recompiling per job.
AF3_JAX_CACHE_DIR="${AF3_JAX_CACHE_DIR:-$HOME/.cache}"

# AlphaFold stages to run: complete, data-pipeline, or inference.
AF3_STAGE="${AF3_STAGE:-complete}"

# GPU allocation mode. memory_characterisation disables JAX preallocation;
# baseline retains production-like preallocation.
PROFILE_MODE="${PROFILE_MODE:-memory_characterisation}"

# Delay in milliseconds between nvidia-smi samples.
SAMPLING_INTERVAL_MS="${SAMPLING_INTERVAL_MS:-100}"

# Parent directory for generated, per-job profiling results.
PROFILES_DIR="${PROFILES_DIR:-$PROJECT_ROOT/profiles}"

error() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

# First positional argument is the AF3 input JSON. Any remaining arguments are
# appended unchanged to the AlphaFold command.
INPUT_JSON="$(realpath "$1")"
shift

# Convert profiling mode into the JAX settings passed through Apptainer.
# Unified host memory (TF_FORCE_UNIFIED_MEMORY=true) is opt-in via
# AF3_UNIFIED_MEMORY, for memory-constrained hosts where device memory alone
# is insufficient; it is not part of either profiling profile.
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

# Shared installation recommended unified memory with a large host-side
# fraction; keep that as an explicit opt-in rather than a profiling default.
if [[ "${AF3_UNIFIED_MEMORY:-false}" == true ]]; then
    FORCE_UNIFIED_MEMORY=true
    JAX_MEMORY_FRACTION=3.2
else
    FORCE_UNIFIED_MEMORY=false
fi

# Build stage-specific AF3 arguments. Model parameters and databases are bound
# read-only and only exposed to stages that need them. Only the GPU-requiring
# stages need --nv; the data-pipeline stage runs on CPU-only nodes with no
# GPU allocated to the job.
BIND_ARGS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
STAGE_ARGS=()
NV_FLAG=()
case "$AF3_STAGE" in
complete)
    NV_FLAG=(--nv)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:$AF3_MODEL_DIR:ro" --bind "$AF3_DB_DIR:$AF3_DB_DIR:ro")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=true --model_dir="$AF3_MODEL_DIR" --db_dir="$AF3_DB_DIR")
    ;;
data-pipeline)
    BIND_ARGS+=(--bind "$AF3_DB_DIR:$AF3_DB_DIR:ro")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=false --db_dir="$AF3_DB_DIR")
    ;;
inference)
    NV_FLAG=(--nv)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:$AF3_MODEL_DIR:ro")
    STAGE_ARGS=(--run_data_pipeline=false --run_inference=true --model_dir="$AF3_MODEL_DIR")
    ;;
*) error "AF3_STAGE must be complete, data-pipeline, or inference" ;;
esac

# Keep metadata, raw measurements, logs, timing, and AF3 output together. UTC
# timestamp makes runs sortable; Slurm job ID links them to scheduler records.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID}"
RUN_DIR="$PROFILES_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/output"

# APPTAINERENV_ variables override image defaults inside the container.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="$JAX_PREALLOCATE"
export APPTAINERENV_TF_FORCE_UNIFIED_MEMORY="$FORCE_UNIFIED_MEMORY"
export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="$JAX_MEMORY_FRACTION"

# Construct the command as an array so paths and extra AF3 arguments retain
# their exact shell boundaries. Our own image has a working %runscript, so
# this is a plain apptainer run rather than an explicit python invocation.
COMMAND=(
    apptainer run "${NV_FLAG[@]}" "${BIND_ARGS[@]}" "$AF3_SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --jax_compilation_cache_dir="$AF3_JAX_CACHE_DIR"
    "${STAGE_ARGS[@]}"
    "$@"
)

# Read the AF3 version and commit embedded during our own image's build.
AF_VERSION="$(apptainer exec "$AF3_SIF" cat /opt/alphafold3_version)"
AF_COMMIT="$(apptainer exec "$AF3_SIF" cat /opt/alphafold3_commit)"

# Save an inventory of GPUs visible to this Slurm allocation. Skipped for the
# data-pipeline stage, which requests no GPU.
GPU_ARGS=()
if [[ "$AF3_STAGE" != data-pipeline ]]; then
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_ARGS+=(--id="$CUDA_VISIBLE_DEVICES")
    nvidia-smi "${GPU_ARGS[@]}" \
        --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits \
        >"$RUN_DIR/gpu_inventory.csv"
else
    : >"$RUN_DIR/gpu_inventory.csv"
fi

# Create metadata before AF3 starts so failed runs still record their input,
# image, hardware, Slurm allocation, profiling mode, and exact command.
python3 "$PROJECT_ROOT/scripts/profile_metadata.py" create \
    --output "$RUN_DIR/metadata.json" --input "$INPUT_JSON" --sif "$AF3_SIF" \
    --mode "$PROFILE_MODE" --stage "$AF3_STAGE" --interval "$SAMPLING_INTERVAL_MS" \
    --gpu-inventory "$RUN_DIR/gpu_inventory.csv" \
    --af-version "$AF_VERSION" --af-commit "$AF_COMMIT" \
    -- "${COMMAND[@]}"

SAMPLER_PID=""
if [[ "$AF3_STAGE" != data-pipeline ]]; then
    # Start nvidia-smi in the background. Units are included in column names
    # rather than data values so gpu.csv can be read directly by pandas.
    printf '%s\n' 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,utilization_memory_percent,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz' \
        >"$RUN_DIR/gpu.csv"
    nvidia-smi "${GPU_ARGS[@]}" \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits --loop-ms="$SAMPLING_INTERVAL_MS" \
        >>"$RUN_DIR/gpu.csv" 2>"$RUN_DIR/nvidia-smi.err" &
    SAMPLER_PID=$!
fi

# Stop and reap the sampler on successful completion, AF3 failure, or signal.
cleanup() {
    if [[ -n "$SAMPLER_PID" ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# GNU time writes host resource statistics to time.txt. tee preserves combined
# AF3 stdout/stderr while displaying it in the Slurm log. Temporarily disable
# immediate exit so a failed AF3 run can still finalise its metadata.
set +e
LC_ALL=C /usr/bin/time -v -o "$RUN_DIR/time.txt" "${COMMAND[@]}" 2>&1 |
    tee "$RUN_DIR/alphafold.log"
AF3_EXIT_STATUS=${PIPESTATUS[0]}
set -e

# Stop sampling before updating metadata with duration, maximum host RSS, and
# final AF3 status. Return that status so Slurm records application failure.
cleanup
trap - EXIT
python3 "$PROJECT_ROOT/scripts/profile_metadata.py" finish \
    --metadata "$RUN_DIR/metadata.json" --time-file "$RUN_DIR/time.txt" \
    --exit-status "$AF3_EXIT_STATUS"

printf 'Profile directory: %s\n' "$RUN_DIR"
exit "$AF3_EXIT_STATUS"
