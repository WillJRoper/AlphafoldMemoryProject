#!/usr/bin/env bash

#SBATCH --job-name=af3-inference
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-inference-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

export AF3_SITE=arc
export AF3_STAGE=inference
export AF3_SIF="${AF3_SIF:-$SLURM_SUBMIT_DIR/images/alphafold3-v3.0.4-x86_64.sif}"
export AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/model_param}"
export AF3_DB_DIR="${AF3_DB_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/public_databases}"

exec "$SLURM_SUBMIT_DIR/scripts/profile_stage.sh" "$@"
