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
VENV="$HOME/.local/share/alphafold3/3.0.3/aarch64-venv"
mkdir -p "$VENV"

apptainer exec --bind "$VENV:/alphafold3_venv" "$SIF" \
    env UV_PROJECT_ENVIRONMENT=/alphafold3_venv UV_LINK_MODE=copy \
    /bin/uv sync --project /app/alphafold --frozen --all-groups --no-editable
apptainer exec --bind "$VENV:/alphafold3_venv" "$SIF" \
    /alphafold3_venv/bin/python3 -c 'from absl import app; import alphafold3; print(alphafold3.__file__)'
