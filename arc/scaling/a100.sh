#!/usr/bin/env bash

#SBATCH --job-name=af3-scale-a100
#SBATCH --clusters=htc
#SBATCH --account=dtce-oxrse
#SBATCH --partition=short
#SBATCH --output=logs/af3-scale-a100-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%2

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
set -euo pipefail

TOKENS=(128 256 512 768 1024 1536 2048 2560 3072 4096 5120 6144 7000)
MEMORY_MODE=${1:-device}
[[ "$MEMORY_MODE" == device || "$MEMORY_MODE" == unified ]] || exit 2
TOKEN_COUNT=${TOKENS[$SLURM_ARRAY_TASK_ID]}

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$ROOT/.runtime/scaling/inputs/scaling_${TOKEN_COUNT}.json"
SIF="$ROOT/images/alphafold3-v3.0.4-x86_64.sif"
MODEL=/data/dtce-oxrse/dtce0101/af_artefacts/model_param
RUN_DIR="$ROOT/profiles/scaling-arc-a100-${TOKEN_COUNT}-${MEMORY_MODE}-$SLURM_JOB_ID"
python3 "$ROOT/scripts/generate_scaling_input.py" "$TOKEN_COUNT" "$INPUT_JSON"

COMMAND=(
    apptainer run --nv --bind "$ROOT:$ROOT" --bind "$MODEL:$MODEL:ro" "$SIF"
    --json_path="$INPUT_JSON"
    --output_dir="$RUN_DIR/output"
    --model_dir="$MODEL"
    --run_data_pipeline=false
    --run_inference=true
)

exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" v3.0.4 true "$MEMORY_MODE" -- "${COMMAND[@]}"
