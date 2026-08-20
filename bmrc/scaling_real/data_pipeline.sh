#!/usr/bin/env bash

#SBATCH --job-name=af3-real-pipeline
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_24hr
#SBATCH --output=logs/af3-real-pipeline-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-26%2

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
MANIFEST="$ROOT/inputs/scaling_real/manifest.tsv"
ROW=$(python3 -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))[int(sys.argv[2])]; print(r["target_tokens"], r["accession"], r["expected_length"])' "$MANIFEST" "$SLURM_ARRAY_TASK_ID")
read -r TARGET ACCESSION LENGTH <<<"$ROW"
INPUT_JSON="$ROOT/.runtime/scaling_real/inputs/$ACCESSION.json"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
DB=/data/belmont/alphafold-3.0.1-20250212
VENV="$ROOT/.runtime/venvs/3.0.3/x86_64"
UV_CACHE="$ROOT/.runtime/uv-cache/x86_64"
RUN_DIR="$ROOT/profiles/scaling-real-bmrc-pipeline-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${ACCESSION}"
CONTAINER_RUN_DIR="/root/af_inout/profiles/${RUN_DIR##*/}"

[[ -f "$VENV/pyvenv.cfg" ]] || { printf 'error: run bmrc/setup_x86_environment.sh first\n' >&2; exit 2; }
[[ -f "$INPUT_JSON" ]] || { printf 'error: missing %s\n' "$INPUT_JSON" >&2; exit 2; }

COMMAND=(
    apptainer run --env UV_CACHE_DIR=/uv-cache --bind "$VENV:/alphafold3_venv"
    --bind "$UV_CACHE:/uv-cache" --bind "$ROOT:/root/af_inout"
    --bind "$DB:/root/public_databases" "$SIF"
    --json_path="/root/af_inout/${INPUT_JSON#"$ROOT/"}"
    --output_dir="$CONTAINER_RUN_DIR/output"
    --db_dir=/root/public_databases
    --jackhmmer_n_cpu=8
    --nhmmer_n_cpu=8
    --hmmsearch_n_cpu=8
    --run_data_pipeline=true
    --run_inference=false
)

printf 'Target=%s accession=%s residues=%s cpus=%s\n' \
    "$TARGET" "$ACCESSION" "$LENGTH" "$SLURM_CPUS_PER_TASK"
exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" data-pipeline "$INPUT_JSON" \
    "$SIF" 3.0.3 false device -- "${COMMAND[@]}"
