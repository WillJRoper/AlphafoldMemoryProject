#!/usr/bin/env bash

#SBATCH --job-name=af3-data-pipeline
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --output=logs/af3-data-pipeline-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$(realpath "$1")"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
DB=/data/belmont/alphafold-3.0.1-20250212
VENV="$HOME/.local/share/alphafold3/3.0.3/x86_64-venv"
RUN_DIR="$ROOT/profiles/data-pipeline-$SLURM_JOB_ID"
CONTAINER_RUN_DIR="/root/af_inout/profiles/data-pipeline-$SLURM_JOB_ID"
mkdir -p "$VENV"
[[ -f "$VENV/pyvenv.cfg" ]] || {
    printf 'error: run sbatch bmrc/setup_x86_environment.sh first\n' >&2
    exit 2
}

COMMAND=(
    apptainer run --bind "$VENV:/alphafold3_venv"
    --bind "$ROOT:/root/af_inout" --bind "$DB:/root/public_databases" "$SIF"
    --json_path="/root/af_inout/${INPUT_JSON#"$ROOT/"}"
    --output_dir="$CONTAINER_RUN_DIR/output"
    --db_dir=/root/public_databases
    --run_data_pipeline=true
    --run_inference=false
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" data-pipeline "$INPUT_JSON" \
    "$SIF" 3.0.3 false -- "${COMMAND[@]}"
