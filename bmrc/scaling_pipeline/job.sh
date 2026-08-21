#!/usr/bin/env bash

#SBATCH --job-name=af3-pipeline-scale
#SBATCH --account=gpu_stuart.prj
#SBATCH --partition=gpu_strubi
#SBATCH --qos=gpu_bmrc_24hr
#SBATCH --output=logs/af3-pipeline-scale-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%1

module load Python/3.11.3-GCCcore-12.3.0
set -euo pipefail

[[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]] || {
    printf 'usage: %s JACKHMMER_THREADS\n' "$0" >&2
    exit 2
}
JACKHMMER_THREADS=$1
EXPECTED_CPUS=$((4 * JACKHMMER_THREADS))
[[ "$SLURM_CPUS_PER_TASK" -eq "$EXPECTED_CPUS" ]] || {
    printf 'error: %s CPUs required for four %s-thread Jackhmmer searches, got %s\n' \
        "$EXPECTED_CPUS" "$JACKHMMER_THREADS" "$SLURM_CPUS_PER_TASK" >&2
    exit 2
}

ACCESSIONS=(P60174 A2AJK6 Q8I3Z1)
TARGETS=(250 3000 10000)
LENGTHS=(249 2986 10061)
PROTEIN_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
REPEAT=$((SLURM_ARRAY_TASK_ID / 3 + 1))
ACCESSION=${ACCESSIONS[$PROTEIN_INDEX]}
TARGET=${TARGETS[$PROTEIN_INDEX]}
LENGTH=${LENGTHS[$PROTEIN_INDEX]}

ROOT="$SLURM_SUBMIT_DIR"
INPUT_JSON="$ROOT/.runtime/scaling_real/inputs/$ACCESSION.json"
SIF=/apps/singularity/alphafold3/alphafold-3.0.3.sif
DB=/data/belmont/alphafold-3.0.1-20250212
VENV="$ROOT/.runtime/venvs/3.0.3/x86_64"
UV_CACHE="$ROOT/.runtime/uv-cache/x86_64"
RUN_DIR="$ROOT/profiles/scaling-pipeline-bmrc-${ACCESSION}-c${SLURM_CPUS_PER_TASK}-t${JACKHMMER_THREADS}-r${REPEAT}-${SLURM_JOB_ID}"
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
    --jackhmmer_n_cpu="$JACKHMMER_THREADS"
    --run_data_pipeline=true
    --run_inference=false
)

printf 'Target=%s accession=%s residues=%s cpus=%s jackhmmer_threads=%s repeat=%s\n' \
    "$TARGET" "$ACCESSION" "$LENGTH" "$SLURM_CPUS_PER_TASK" \
    "$JACKHMMER_THREADS" "$REPEAT"
exec "$ROOT/scripts/profile_command.sh" "$RUN_DIR" data-pipeline "$INPUT_JSON" \
    "$SIF" 3.0.3 false device -- "${COMMAND[@]}"
