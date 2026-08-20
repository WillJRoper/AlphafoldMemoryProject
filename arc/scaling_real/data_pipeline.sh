#!/usr/bin/env bash

#SBATCH --job-name=af3-real-pipeline
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=medium
#SBATCH --output=logs/af3-real-pipeline-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-26%2

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
MANIFEST="$ROOT/inputs/scaling_real/manifest.tsv"
ROW=$(python3 -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))[int(sys.argv[2])]; print(r["target_tokens"], r["accession"], r["expected_length"])' "$MANIFEST" "$SLURM_ARRAY_TASK_ID")
read -r TARGET ACCESSION LENGTH <<<"$ROW"
INPUT_JSON="$ROOT/.runtime/scaling_real/inputs/$ACCESSION.json"
SIF="$ROOT/images/alphafold3-v3.0.4-x86_64.sif"
DB=/data/dtce-oxrse/dtce0101/af_artefacts/public_databases
RUN_DIR="$ROOT/profiles/scaling-real-arc-pipeline-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${ACCESSION}"

COMMAND=(
    apptainer run --bind "$ROOT:$ROOT" --bind "$DB:$DB:ro" "$SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --db_dir="$DB"
    --run_data_pipeline=true
    --run_inference=false
)

printf 'Target=%s accession=%s residues=%s\n' "$TARGET" "$ACCESSION" "$LENGTH"
exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" data-pipeline "$INPUT_JSON" \
    "$SIF" v3.0.4 false device -- "${COMMAND[@]}"
