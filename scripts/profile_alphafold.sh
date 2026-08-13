#!/usr/bin/env bash

# Slurm resources for the shared AlphaFold installation. Belmont storage is
# mounted on gpu_strubi nodes; submit from a login node where it is visible.
#SBATCH --job-name=af3-profile
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

set -euo pipefail

# ============================== Configuration ==============================

# Absolute repository path, derived from this script so submission can happen
# from any working directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared Apptainer image containing AlphaFold 3.0.1.
AF3_SIF="${AF3_SIF:-/apps/singularity/alphafold3/alphafold-3.0.1-20250210.sif}"

# Shared model parameters and reference databases on Belmont storage.
AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/belmont/alphafold3-parameters}"
AF3_DB_DIR="${AF3_DB_DIR:-/data/belmont/alphafold-3.0.1-20250212}"

# Shared image release. Exact source commit is not published with the path.
AF3_VERSION="${AF3_VERSION:-3.0.1}"
AF3_COMMIT="${AF3_COMMIT:-unknown}"

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

# Convert profiling mode into the JAX setting passed through Apptainer.
case "$PROFILE_MODE" in
baseline) JAX_PREALLOCATE=true ;;
memory_characterisation) JAX_PREALLOCATE=false ;;
*) error "PROFILE_MODE must be baseline or memory_characterisation" ;;
esac

# Build stage-specific AF3 arguments. Shared data is exposed at conventional
# paths inside the container and mounted read-only.
BIND_ARGS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
STAGE_ARGS=()
case "$AF3_STAGE" in
complete)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:/root/models:ro" --bind "$AF3_DB_DIR:/root/public_databases:ro")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=true --model_dir=/root/models --db_dir=/root/public_databases)
    ;;
data-pipeline)
    BIND_ARGS+=(--bind "$AF3_DB_DIR:/root/public_databases:ro")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=false --db_dir=/root/public_databases)
    ;;
inference)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:/root/models:ro")
    STAGE_ARGS=(--run_data_pipeline=false --run_inference=true --model_dir=/root/models)
    ;;
*) error "AF3_STAGE must be complete, data-pipeline, or inference" ;;
esac

# Keep metadata, raw measurements, logs, timing, and AF3 output together. UTC
# timestamp makes runs sortable; Slurm job ID links them to scheduler records.
RUN_DIR="$PROFILES_DIR/$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/output"

# APPTAINERENV_ variables override image defaults inside the container. Preserve
# Slurm's GPU selection and make the JAX allocation policy explicit.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="$JAX_PREALLOCATE"
export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"

# Construct the command as an array so paths and extra AF3 arguments retain
# their exact shell boundaries.
COMMAND=(
    apptainer run --nv "${BIND_ARGS[@]}" "$AF3_SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    "${STAGE_ARGS[@]}"
    "$@"
)

# Save an inventory of GPUs visible to this Slurm allocation.
GPU_ARGS=()
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_ARGS+=(--id="$CUDA_VISIBLE_DEVICES")
nvidia-smi "${GPU_ARGS[@]}" \
    --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits \
    >"$RUN_DIR/gpu_inventory.csv"

# Create metadata before AF3 starts so failed runs still record their input,
# image, hardware, Slurm allocation, profiling mode, and exact command.
python3 "$PROJECT_ROOT/scripts/profile_metadata.py" create \
    --output "$RUN_DIR/metadata.json" --input "$INPUT_JSON" --sif "$AF3_SIF" \
    --mode "$PROFILE_MODE" --stage "$AF3_STAGE" --interval "$SAMPLING_INTERVAL_MS" \
    --gpu-inventory "$RUN_DIR/gpu_inventory.csv" \
    --af-version "$AF3_VERSION" --af-commit "$AF3_COMMIT" \
    -- "${COMMAND[@]}"

# Start nvidia-smi in the background. Units are included in column names rather
# than data values so gpu.csv can be read directly by pandas.
printf '%s\n' 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,utilization_memory_percent,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz' \
    >"$RUN_DIR/gpu.csv"
nvidia-smi "${GPU_ARGS[@]}" \
    --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits --loop-ms="$SAMPLING_INTERVAL_MS" \
    >>"$RUN_DIR/gpu.csv" 2>"$RUN_DIR/nvidia-smi.err" &
SAMPLER_PID=$!

# Stop and reap the sampler on successful completion, AF3 failure, or signal.
cleanup() {
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
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
