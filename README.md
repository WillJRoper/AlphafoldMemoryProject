# AlphaFold 3 profiling

Repeatable runtime and memory profiling for AlphaFold 3 under Slurm and
Apptainer.

```text
scripts/profile_command.sh  Generic command profiler
scripts/plot_profile.py     Stage-aware plotting

bmrc/data_pipeline.sh       BMRC data pipeline
bmrc/setup_x86_environment.sh  One-time x86 environment setup
bmrc/setup_arm_environment.sh  One-time ARM environment setup
bmrc/inference_a100.sh      BMRC A100 inference
bmrc/inference_gh200.sh     BMRC GH200 inference
bmrc/submit_all.sh          Submit all inference variants
bmrc/submit_comparison.sh   Submit pipeline plus dependent inference jobs

arc/data_pipeline.sh        ARC data pipeline
arc/inference_a100.sh       ARC A100 inference
```

Submission scripts contain explicit Slurm resources, image/data paths, and AF3
commands. `profile_command.sh` only records metadata, timing, logs, and optional
GPU samples around the supplied command.

## BMRC

BMRC comparisons use matching AF3 3.0.3 images for x86_64 and ARM64:

```text
/apps/singularity/alphafold3/alphafold-3.0.3.sif
/apps/singularity/alphafold3/alphafold-3.0.3-arm.sif
```

Initialize both architecture-specific writable environments once:

```bash
sbatch bmrc/setup_x86_environment.sh
sbatch bmrc/setup_arm_environment.sh
```

BMRC inference scripts explicitly read the model file before starting Apptainer.
This triggers the Belmont autofs mount and verifies access before the bind.
GH200 inference is pinned to `compgh023`, the GH200 node with Belmont mounted.

Setup uses the images' locked dependencies and verifies `absl` and `alphafold3`
imports:

```text
.runtime/venvs/3.0.3/x86_64
.runtime/venvs/3.0.3/aarch64
.runtime/uv-cache/x86_64
.runtime/uv-cache/aarch64
```

Submit one complete comparison:

```bash
bash bmrc/submit_comparison.sh inputs/lysozyme_1lyz.json
```

This submits the data pipeline, then A100 and GH200 inference jobs with
`afterok` dependencies. The generated pipeline JSON has a deterministic path
under `profiles/data-pipeline-JOB_ID/`.

Submit stages manually instead:

```bash
PIPELINE_JOB=$(sbatch --parsable bmrc/data_pipeline.sh inputs/lysozyme_1lyz.json)
DATA_JSON="$PWD/profiles/data-pipeline-$PIPELINE_JOB/output/lysozyme_1lyz/lysozyme_1lyz_data.json"
bash bmrc/submit_all.sh --after "$PIPELINE_JOB" "$DATA_JSON"
```

For an already completed pipeline:

```bash
bash bmrc/submit_all.sh profiles/data-pipeline-JOB_ID/output/JOB/JOB_data.json
```

Scaling profiles support on-demand device allocation for clean memory
measurements, official preallocated device mode for capacity tests, and unified
memory with host spill.


## ARC

ARC uses the repository-built x86_64 AF3 v3.0.4 image and shared model/database
paths:

```bash
sbatch arc/data_pipeline.sh inputs/lysozyme_1lyz.json
sbatch arc/inference_a100.sh profiles/data-pipeline-JOB_ID/output/JOB/JOB_data.json
```

## Results

Each run writes a self-contained directory under `profiles/` containing
metadata, GNU time output, AF3 logs/output, and GPU samples when applicable.
Generated figures are written beneath `plots/`; per-run filenames include the
full profile directory name to prevent collisions.

Generate stage-appropriate plots:

```bash
python3 scripts/plot_profile.py profiles/RUN
```

Validate inference output:

```bash
python3 scripts/validate_profile.py profiles/RUN
```

## Token scaling

Scaling jobs use one synthetic single-chain protein with fixed composition,
query-only MSA, no templates, and exact token counts:

```text
128 256 512 768 1024 1536 2048 2560 3072 4096 5120 6144 7000
```

Each array task is an independent profiled inference run. Start with device-only
memory and retain failures as the observed capacity boundary:

```bash
bash bmrc/scaling_repeat/submit.sh device
bash arc/scaling_repeat/submit.sh device
```

Run a second, independent sweep with unified memory enabled at every size:

```bash
bash bmrc/scaling_repeat/submit.sh unified
bash arc/scaling_repeat/submit.sh unified
```

BMRC also supports the official preallocated device configuration, or all three
allocator modes in one submission:

```bash
bash bmrc/scaling_repeat/submit.sh preallocated
bash bmrc/scaling_repeat/submit.sh all
```

Restrict a BMRC repeat sweep to hardware not already completed:

```bash
bash bmrc/scaling_repeat/submit.sh device gh200
bash bmrc/scaling_repeat/submit.sh unified 4096 a100
```

Unified sweeps request 320 GB host memory; device-only sweeps retain their
smaller requests. Generate aggregate runtime and peak-GPU-memory curves with:

```bash
python3 scripts/plot_scaling.py profiles --family repeat
```

Profile names identify on-demand device, preallocated device, or unified mode,
and metadata records effective allocator settings. Runtime and memory panels use
log-log axes. A thin strip above memory shows observed AF3 compilation buckets.
The plot draws a hardware-colored vertical dotted line at the first on-demand
device OOM. Failed profiles are retained for diagnostics, excluded from scaling
curves, and reported on stdout.

## Natural-sequence scaling

`inputs/scaling_real/manifest.tsv` contains three UniProt proteins near each
target from 250 to 10,000 residues. Entries are reviewed except the explicitly
marked third 10k candidate. Preparation fetches canonical FASTA sequences,
checks exact lengths and residue alphabets, and writes generated inputs beneath
`.runtime/scaling_real/inputs/`.

Submit data pipelines plus all three inference allocator modes:

```bash
bash bmrc/scaling_real/submit.sh all
bash arc/scaling_real/submit.sh all
```

Or submit once and add inference sweeps later:

```bash
bash bmrc/scaling_real/submit.sh pipeline
bash bmrc/scaling_real/submit.sh device PIPELINE_ARRAY_ID
bash bmrc/scaling_real/submit.sh unified PIPELINE_ARRAY_ID
bash bmrc/scaling_real/submit.sh preallocated PIPELINE_ARRAY_ID
```

Append `a100` or `gh200` to submit only that hardware, for example:

```bash
bash bmrc/scaling_real/submit.sh device PIPELINE_ARRAY_ID gh200
```

Inference tasks use `aftercorr`, so each array index waits only for its matching
pipeline index. Plot individual proteins, median trends, and min-max variability
bands with:

```bash
python3 scripts/plot_scaling.py profiles --family real
```

## Data-pipeline scaling

Plot fixed-32-CPU runtime and host-memory scaling from completed natural-suite
pipeline profiles:

```bash
python3 scripts/plot_pipeline_scaling.py profiles --kind length
```

The BMRC strong-scaling suite runs `P60174`, `A2AJK6`, and `Q8I3Z1` three times
at 4, 8, 16, 24, 32, and 48 allocated CPUs. Four concurrent Jackhmmer searches
receive 1, 2, 4, 6, 8, and 12 threads respectively. Arrays are serialized to
limit benchmark-induced GPFS contention:

```bash
bash bmrc/scaling_pipeline/submit.sh
python3 scripts/plot_pipeline_scaling.py profiles --kind threads
```
