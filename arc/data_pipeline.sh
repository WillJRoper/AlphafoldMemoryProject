#!/usr/bin/env bash

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

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$(realpath "$1")"
SIF="$ROOT/images/alphafold3-v3.0.4-x86_64.sif"
DB=/data/dtce-oxrse/dtce0101/af_artefacts/public_databases
RUN_DIR="$ROOT/profiles/data-pipeline-$SLURM_JOB_ID"

COMMAND=(
    apptainer run --bind "$ROOT:$ROOT" --bind "$DB:$DB:ro" "$SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --db_dir="$DB"
    --run_data_pipeline=true
    --run_inference=false
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" data-pipeline "$INPUT_JSON" \
    "$SIF" v3.0.4 false device -- "${COMMAND[@]}"
