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
#SBATCH --time=04:00:00

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

export AF3_SITE=bmrc
export AF3_STAGE=inference
export AF3_UNIFIED_MEMORY="${AF3_UNIFIED_MEMORY:-false}"
export AF3_JAX_CACHE_DIR="${AF3_JAX_CACHE_DIR:-$HOME/.cache/alphafold3/gh200}"
export AF3_SIF="${AF3_SIF:-/apps/singularity/alphafold3/alphafold-3.0.3-arm.sif}"
export AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/belmont/alphafold3-parameters}"
export AF3_DB_DIR="${AF3_DB_DIR:-/data/belmont/alphafold-3.0.1-20250212}"
export AF3_VERSION=3.0.3
export AF3_COMMIT=unknown

exec "$SLURM_SUBMIT_DIR/scripts/profile_stage.sh" "$@"
