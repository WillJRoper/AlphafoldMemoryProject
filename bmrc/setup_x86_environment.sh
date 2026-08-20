#!/usr/bin/env bash

#SBATCH --job-name=af3-setup-x86
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --output=logs/af3-setup-x86-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
VENV="$HOME/.local/share/alphafold3/3.0.3/x86_64-venv"
mkdir -p "$VENV"

apptainer exec --bind "$VENV:/alphafold3_venv" "$SIF" \
    env UV_PROJECT_ENVIRONMENT=/alphafold3_venv UV_LINK_MODE=copy \
    /bin/uv sync --project /app/alphafold --frozen --all-groups --no-editable
apptainer exec --bind "$VENV:/alphafold3_venv" "$SIF" \
    /alphafold3_venv/bin/python3 -c 'from absl import app; import alphafold3; print(alphafold3.__file__)'
