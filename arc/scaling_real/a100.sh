#!/usr/bin/env bash

#SBATCH --job-name=af3-real-a100
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-real-a100-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-26%2

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

[[ $# -eq 2 ]] || { printf 'usage: %s PIPELINE_ARRAY_ID device|unified\n' "$0" >&2; exit 2; }
PIPELINE_ARRAY_ID=$1
MEMORY_MODE=$2
[[ "$MEMORY_MODE" == device || "$MEMORY_MODE" == unified ]] || exit 2

ROOT="$SLURM_SUBMIT_DIR"
MANIFEST="$ROOT/inputs/scaling_real/manifest.tsv"
ROW=$(python3 -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))[int(sys.argv[2])]; print(r["target_tokens"], r["accession"], r["expected_length"])' "$MANIFEST" "$SLURM_ARRAY_TASK_ID")
read -r TARGET ACCESSION LENGTH <<<"$ROW"
NAME="real_${TARGET}_${ACCESSION,,}"
INPUT_JSON="$ROOT/profiles/scaling-real-arc-pipeline-${PIPELINE_ARRAY_ID}_${SLURM_ARRAY_TASK_ID}-${ACCESSION}/output/$NAME/${NAME}_data.json"
SIF="$ROOT/images/alphafold3-v3.0.4-x86_64.sif"
MODEL=/data/dtce-oxrse/dtce0101/af_artefacts/model_param
RUN_DIR="$ROOT/profiles/scaling-real-arc-a100-${TARGET}-${ACCESSION}-${MEMORY_MODE}-$SLURM_JOB_ID"

COMMAND=(
    apptainer run --nv --bind "$ROOT:$ROOT" --bind "$MODEL:$MODEL:ro" "$SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --model_dir="$MODEL"
    --run_data_pipeline=false
    --run_inference=true
)

printf 'Target=%s accession=%s residues=%s\n' "$TARGET" "$ACCESSION" "$LENGTH"
exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" v3.0.4 true "$MEMORY_MODE" -- "${COMMAND[@]}"
