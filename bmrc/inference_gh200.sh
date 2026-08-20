#!/usr/bin/env bash

#SBATCH --job-name=af3-gh200
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_gh200_bmrc
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --output=logs/af3-gh200-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$(realpath "$1")"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3-arm.sif
MODEL=/data/belmont/alphafold3-parameters
VENV="$ROOT/.runtime/venvs/3.0.3/aarch64"
UV_CACHE="$ROOT/.runtime/uv-cache/aarch64"
RUN_DIR="$ROOT/profiles/gh200-$SLURM_JOB_ID"
CONTAINER_RUN_DIR="/root/af_inout/profiles/gh200-$SLURM_JOB_ID"
mkdir -p "$VENV"
[[ -f "$VENV/pyvenv.cfg" ]] || {
    printf 'error: run sbatch bmrc/setup_arm_environment.sh first\n' >&2
    exit 2
}

COMMAND=(
    apptainer run --nv --env UV_CACHE_DIR=/uv-cache --bind "$VENV:/alphafold3_venv"
    --bind "$UV_CACHE:/uv-cache" --bind "$ROOT:/root/af_inout"
    --bind "$MODEL:/root/models" "$SIF"
    --json_path="/root/af_inout/${INPUT_JSON#"$ROOT/"}"
    --output_dir="$CONTAINER_RUN_DIR/output"
    --model_dir=/root/models
    --run_data_pipeline=false
    --run_inference=true
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" 3.0.3 true -- "${COMMAND[@]}"
