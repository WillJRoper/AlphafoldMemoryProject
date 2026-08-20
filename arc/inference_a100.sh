#!/usr/bin/env bash

#SBATCH --job-name=af3-a100
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-a100-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$(realpath "$1")"
SIF="$ROOT/images/alphafold3-v3.0.4-x86_64.sif"
MODEL=/data/dtce-oxrse/dtce0101/af_artefacts/model_param
RUN_DIR="$ROOT/profiles/a100-$SLURM_JOB_ID"
CACHE="$HOME/.cache/alphafold3/a100"
mkdir -p "$CACHE"

COMMAND=(
    apptainer run --nv --bind "$ROOT:$ROOT" --bind "$MODEL:$MODEL:ro" "$SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --model_dir="$MODEL"
    --jax_compilation_cache_dir="$CACHE"
    --run_data_pipeline=false
    --run_inference=true
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" v3.0.4 true -- "${COMMAND[@]}"
