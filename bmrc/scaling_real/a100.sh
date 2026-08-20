#!/usr/bin/env bash

#SBATCH --job-name=af3-real-a100
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_24hr
#SBATCH --output=logs/af3-real-a100-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-pcie-80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-26%2

module load Python/3.11.3-GCCcore-12.3.0
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
INPUT_JSON="$ROOT/profiles/scaling-real-bmrc-pipeline-${PIPELINE_ARRAY_ID}_${SLURM_ARRAY_TASK_ID}-${ACCESSION}/output/$NAME/${NAME}_data.json"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
MODEL=/data/belmont/alphafold3-parameters
VENV="$ROOT/.runtime/venvs/3.0.3/x86_64"
UV_CACHE="$ROOT/.runtime/uv-cache/x86_64"
RUN_DIR="$ROOT/profiles/scaling-real-bmrc-a100-${TARGET}-${ACCESSION}-${MEMORY_MODE}-$SLURM_JOB_ID"
CONTAINER_RUN_DIR="/root/af_inout/profiles/${RUN_DIR##*/}"

[[ -r "$MODEL/af3.bin.zst" ]] || { printf 'error: model parameters unavailable at %s\n' "$MODEL" >&2; exit 2; }
COMMAND=(
    apptainer run --nv --env UV_CACHE_DIR=/uv-cache --bind "$VENV:/alphafold3_venv"
    --bind "$UV_CACHE:/uv-cache" --bind "$ROOT:/root/af_inout"
    --bind "$MODEL:/root/models" "$SIF"
    --json_path="/root/af_inout/${INPUT_JSON#"$ROOT/"}"
    --output_dir="$CONTAINER_RUN_DIR/output"
    --model_dir=/root/models
    --run_data_pipeline=false
    --run_inference=true
)

printf 'Target=%s accession=%s residues=%s\n' "$TARGET" "$ACCESSION" "$LENGTH"
exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" inference "$INPUT_JSON" \
    "$SIF" 3.0.3 true "$MEMORY_MODE" -- "${COMMAND[@]}"
