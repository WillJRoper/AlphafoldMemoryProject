#!/usr/bin/env bash

#SBATCH --job-name=af3-v100
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --output=logs/af3-v100-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2-16gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$(realpath "$1")"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
MODEL=/data/belmont/alphafold3-parameters
VENV="$HOME/.local/share/alphafold3/3.0.3/x86_64-venv"
RUN_DIR="$ROOT/profiles/v100-$SLURM_JOB_ID"
CONTAINER_RUN_DIR="/root/af_inout/profiles/v100-$SLURM_JOB_ID"
CACHE="$HOME/.cache/alphafold3/v100"
mkdir -p "$CACHE" "$VENV"

COMMAND=(
    apptainer run --nv --bind "$VENV:/alphafold3_venv"
    --bind "$ROOT:/root/af_inout" --bind "$MODEL:/root/models" "$SIF"
    --json_path="/root/af_inout/${INPUT_JSON#"$ROOT/"}"
    --output_dir="$CONTAINER_RUN_DIR/output"
    --model_dir=/root/models
    --jax_compilation_cache_dir="$CACHE"
    --flash_attention_implementation=xla
    --run_data_pipeline=false
    --run_inference=true
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" 3.0.3 true -- "${COMMAND[@]}"
