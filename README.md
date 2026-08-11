# AlphaFold 3 memory profiling on BMRC GH200

This repository provides a reproducible AlphaFold 3 (AF3) v3.0.4 environment
for profiling runtime and memory use across complete, data-pipeline-only, and
inference-only runs. The initial target is an NVIDIA Grace Hopper GH200 node on
the University of Oxford BMRC cluster. Raw measurements are retained so plots
and aggregate statistics can be produced afterwards without rerunning AF3.

## Repository layout

```text
alphafold3/   Upstream Google DeepMind source (Git submodule, v3.0.4)
containers/  Apptainer definitions
images/      Generated SIF images (ignored)
inputs/      Version-controlled AF3 input JSON files
outputs/     Generated outputs outside profiling runs (ignored)
profiles/    Generated profiling run directories (ignored)
scripts/     Profiling and metadata tools
```

Do not commit model parameters, databases, SIF images, AF3 outputs, or profile
data. Benchmark input JSON files belong in `inputs/` and should normally be
committed.

## AlphaFold source

`alphafold3/` is a submodule of
[`google-deepmind/alphafold3`](https://github.com/google-deepmind/alphafold3),
pinned by this repository to commit `85c4d20505fd5cef05eac22b534d4e793971ae69`
(v3.0.4). Initialise the exact pinned revision after cloning:

```bash
git submodule update --init --recursive
git -C alphafold3 describe --tags --always
git -C alphafold3 rev-parse HEAD
```

To restore the revision recorded by this repository after changing the
submodule checkout, run `git submodule update --init --recursive`. Do not use
`git submodule update --remote`: that follows upstream rather than the pinned
revision.

## Container

Apptainer is used because BMRC runs Slurm jobs with shared host filesystems and
NVIDIA GPU passthrough, without requiring a privileged Docker daemon. The
working definition is `containers/alphafold3-v3.0.4.def`, based on
`nvidia/cuda:12.6.3-base-ubuntu24.04`. It installs the pinned source and records
its version and commit inside the image.

The image is ARM64 and must be built natively on an ARM64 GH200 BMRC node. The
definition currently copies the source from
`/well/stuart/users/wuy477/AlphafoldMemoryProject/alphafold3`; clone or place the
repository there before building. Do not build this image on an x86 login node
or local development machine.

```bash
srun --account=gpu_stuart.prj \
  --partition=gpu_gh200_bmrc \
  --qos=gpu_bmrc_4hr \
  --gres=gpu:1 \
  --time=04:00:00 \
  --pty bash

cd /well/stuart/users/wuy477/AlphafoldMemoryProject
git submodule update --init --recursive
mkdir -p images
apptainer build images/alphafold3-v3.0.4-arm64.sif \
  containers/alphafold3-v3.0.4.def
apptainer inspect images/alphafold3-v3.0.4-arm64.sif
apptainer exec images/alphafold3-v3.0.4-arm64.sif \
  sh -c 'cat /opt/alphafold3_version /opt/alphafold3_commit'
```

Known working resources are account `gpu_stuart.prj`, partition
`gpu_gh200_bmrc`, QOS `gpu_bmrc_4hr`, and one `gpu:1` allocation. The initial
image was tested on `compgh023` with Apptainer 1.5.0 and two GH200 144 GB HBM3e
GPUs. Slurm's `CUDA_VISIBLE_DEVICES` is propagated into the container.

## Profiling modes

The image retains production-like JAX allocation defaults:

```text
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_CLIENT_MEM_FRACTION=0.95
```

`PROFILE_MODE=baseline` preserves those settings. In this mode,
`nvidia-smi` usually reports the large JAX reservation, not AF3's instantaneous
memory requirement.

`PROFILE_MODE=memory-characterisation` sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` at runtime. Use this mode for a more
informative GPU-memory time series. Metadata records the mode and effective
setting; do not combine the two modes as equivalent measurements.

## Profiling workflow

Model parameters and databases are intentionally not assigned repository
defaults. Export their real BMRC paths when required:

```bash
export AF3_MODEL_DIR=/path/to/model/parameters
export AF3_DB_DIR=/path/to/public_databases
```

Submit a complete baseline run:

```bash
sbatch --export=ALL,PROFILE_MODE=baseline \
  scripts/profile_alphafold.sh inputs/example.json
```

Submit memory characterisation at a 500 ms sampling interval:

```bash
sbatch --export=ALL,PROFILE_MODE=memory-characterisation,SAMPLING_INTERVAL_MS=500 \
  scripts/profile_alphafold.sh inputs/example.json
```

Select stages without using a separate implementation:

```bash
# Data pipeline only: requires AF3_DB_DIR.
sbatch --export=ALL,AF3_STAGE=data-pipeline \
  scripts/profile_alphafold.sh inputs/example.json

# Inference only: input must already contain pipeline results; requires AF3_MODEL_DIR.
sbatch --export=ALL,AF3_STAGE=inference \
  scripts/profile_alphafold.sh inputs/processed.json
```

Extra arguments after the input are passed unchanged to AF3, for example
`--num_seeds=2`. Important configuration variables are at the top of
`scripts/profile_alphafold.sh` and may be overridden through `sbatch --export`:

| Variable | Default | Purpose |
|---|---|---|
| `AF3_SIF` | `images/alphafold3-v3.0.4-arm64.sif` | Container image |
| `AF3_MODEL_DIR` | unset | Model parameters |
| `AF3_DB_DIR` | unset | Sequence/template databases |
| `AF3_STAGE` | `complete` | `complete`, `data-pipeline`, or `inference` |
| `PROFILE_MODE` | `baseline` | `baseline` or `memory-characterisation` |
| `SAMPLING_INTERVAL_MS` | `500` | `nvidia-smi` interval in milliseconds |
| `PROFILES_DIR` | `profiles` | Generated run root |

Each job creates a unique `profiles/<UTC>-<job>-<suffix>/` directory containing:

```text
metadata.json       Structured run, input, image, hardware, and outcome data
gpu.csv             Raw timestamped GPU memory/utilisation samples
gpu_inventory.csv   GPU UUID, model, and total-memory inventory
nvidia-smi.err      Sampler diagnostics
time.txt            Raw /usr/bin/time -v output
alphafold.log       Combined AF3 stdout and stderr
output/             AF3 output for this run
```

The sampler is stopped by an exit/signal trap, including when AF3 fails.
`metadata.json` is written before AF3 starts and finalised with timestamps,
duration, exit status, and maximum host resident set size afterwards. Abrupt
node loss or `SIGKILL` can leave status `running`; that is evidence of an
incomplete run rather than a successful result.

For reproducibility, preserve the complete run directory and commit the input
JSON. Metadata includes the input and SIF SHA-256 hashes, image-contained AF3
version and commit, executed argument vector, Slurm allocation, hostname,
visible GPUs, JAX settings, stage, mode, and sampling interval. Record any
non-default Slurm resources and AF3 flags in the submission command.
