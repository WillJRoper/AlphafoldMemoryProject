#!/usr/bin/env bash

# Defaults below target the GPU (complete/inference) case on ARC's htc
# cluster. The data-pipeline stage needs no GPU and a very different resource
# shape (see colleague's proven submit_dp_* script); override every one of
# these via sbatch CLI flags for that case, e.g.:
#   sbatch --partition=medium,long --gpus-per-node=0 --constraint= \
#     --mem=128G --cpus-per-task=32 --time=2-00:00:00 \
#     --export=ALL,AF3_STAGE=data-pipeline \
#     arc/profile_alphafold.sh inputs/lysozyme_1lyz.json
#SBATCH --job-name=af3-profile
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-profile-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --constraint=[gpu_mem:80GB&gpu_sku:A100]
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

# Shared AlphaFold installation: SIF and source checkout used to locate
# run_alphafold.py, both resolved as absolute paths inside the container.
AF3_HOME="${AF3_HOME:-/data/dtce-oxrse/dtce0101/sw-dev/alphafold}"
AF3_SIF="${AF3_SIF:-$AF3_HOME/af3_304.sif}"
AF3_RUN_SCRIPT="${AF3_RUN_SCRIPT:-$AF3_HOME/alphafold3/run_alphafold.py}"

# Shared model parameters and reference databases.
AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/model_param}"
AF3_DB_DIR="${AF3_DB_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/public_databases}"

# Shared image release, matching the submodule pin. Exact source commit is
# not published with the path.
AF3_VERSION="${AF3_VERSION:-3.0.4}"
AF3_COMMIT="${AF3_COMMIT:-unknown}"

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

# Convert profiling mode into the JAX settings passed through Apptainer. The
# shared installation's recommended configuration uses unified host memory.
case "$PROFILE_MODE" in
baseline)
    JAX_PREALLOCATE=true
    FORCE_UNIFIED_MEMORY=false
    JAX_MEMORY_FRACTION=0.95
    ;;
memory_characterisation)
    JAX_PREALLOCATE=false
    FORCE_UNIFIED_MEMORY=true
    JAX_MEMORY_FRACTION=3.2
    ;;
*) error "PROFILE_MODE must be baseline or memory_characterisation" ;;
esac

# Keep metadata, raw measurements, logs, timing, and AF3 output together. UTC
# timestamp makes runs sortable; Slurm job ID links them to scheduler records.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID}"
RUN_DIR="$PROFILES_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/output"

# Mirror the bind layout of the proven submission scripts: AF3_HOME provides
# the SIF and run_alphafold.py, input/output use their own separate binds.
BIND_ARGS=(
    --bind "$AF3_HOME:$AF3_HOME"
    --bind "$(dirname "$INPUT_JSON"):/root/af_input"
    --bind "$RUN_DIR/output:/root/af_output"
)
STAGE_ARGS=()
# Only the GPU-requiring stages need --nv; the data-pipeline stage runs on
# CPU-only nodes with no GPU allocated to the job.
NV_FLAG=()
case "$AF3_STAGE" in
complete)
    NV_FLAG=(--nv)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:/root/models" --bind "$AF3_DB_DIR:/root/public_databases")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=true --model_dir=/root/models --db_dir=/root/public_databases)
    ;;
data-pipeline)
    BIND_ARGS+=(--bind "$AF3_DB_DIR:/root/public_databases")
    STAGE_ARGS=(--run_data_pipeline=true --run_inference=false --db_dir=/root/public_databases)
    ;;
inference)
    NV_FLAG=(--nv)
    BIND_ARGS+=(--bind "$AF3_MODEL_DIR:/root/models")
    STAGE_ARGS=(--run_data_pipeline=false --run_inference=true --model_dir=/root/models)
    ;;
*) error "AF3_STAGE must be complete, data-pipeline, or inference" ;;
esac

# APPTAINERENV_ variables override image defaults inside the container.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="$JAX_PREALLOCATE"
export APPTAINERENV_TF_FORCE_UNIFIED_MEMORY="$FORCE_UNIFIED_MEMORY"
export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="$JAX_MEMORY_FRACTION"
export APPTAINERENV_XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"

# Construct the command as an array so paths and extra AF3 arguments retain
# their exact shell boundaries. This image has no working %runscript, so
# run_alphafold.py is invoked directly via apptainer exec.
COMMAND=(
    apptainer exec "${NV_FLAG[@]}" "${BIND_ARGS[@]}" "$AF3_SIF"
    python "$AF3_RUN_SCRIPT"
    --json_path="/root/af_input/$(basename "$INPUT_JSON")"
    --output_dir="/root/af_output"
    --jax_compilation_cache_dir="$AF3_JAX_CACHE_DIR"
    "${STAGE_ARGS[@]}"
    "$@"
)

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
    --af-version "$AF3_VERSION" --af-commit "$AF3_COMMIT" \
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
