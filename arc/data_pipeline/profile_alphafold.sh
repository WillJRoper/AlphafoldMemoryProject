#!/usr/bin/env bash

# CPU-only data-pipeline profile. Shared databases remain mounted read-only.
#SBATCH --job-name=af3-data-pipeline
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=medium
#SBATCH --output=logs/af3-data-pipeline-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

export AF3_SITE=arc
export AF3_STAGE=data-pipeline
export AF3_SIF="${AF3_SIF:-$SLURM_SUBMIT_DIR/images/alphafold3-v3.0.4-x86_64.sif}"
export AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/model_param}"
export AF3_DB_DIR="${AF3_DB_DIR:-/data/dtce-oxrse/dtce0101/af_artefacts/public_databases}"

exec "$SLURM_SUBMIT_DIR/scripts/profile_stage.sh" "$@"
