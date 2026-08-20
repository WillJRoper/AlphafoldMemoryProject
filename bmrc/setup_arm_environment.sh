#!/usr/bin/env bash

#SBATCH --job-name=af3-setup-arm
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_gh200_bmrc
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --output=logs/af3-setup-arm-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SIF=/apps/singularity/alphafold3/alphafold-3.0.3-arm.sif
RUNTIME="$SLURM_SUBMIT_DIR/.runtime"
VENV="$RUNTIME/venvs/3.0.3/aarch64"
UV_CACHE="$RUNTIME/uv-cache/aarch64"
mkdir -p "$VENV" "$UV_CACHE"

apptainer exec --bind "$VENV:/alphafold3_venv" --bind "$UV_CACHE:/uv-cache" "$SIF" \
    env UV_PROJECT_ENVIRONMENT=/alphafold3_venv UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy \
    /bin/uv sync --project /app/alphafold --frozen --all-groups --no-editable
apptainer exec --bind "$VENV:/alphafold3_venv" "$SIF" \
    /alphafold3_venv/bin/python3 -c 'from absl import app; import alphafold3; print(alphafold3.__file__)'
