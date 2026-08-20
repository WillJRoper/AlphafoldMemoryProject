# AlphaFold 3 profiling

Reproducible runtime and memory profiling for AlphaFold 3 on NVIDIA GPUs.
AlphaFold runs in an Apptainer container launched from a Slurm job.

Submission scripts are site-specific because each HPC has a different shared
AlphaFold installation and Slurm configuration (account/partition/GRES syntax,
and in some cases a different container invocation model entirely):

```text
bmrc/data_pipeline/profile_alphafold.sh  BMRC CPU data pipeline
bmrc/inference/a100/profile_alphafold.sh BMRC A100 inference
bmrc/inference/v100/profile_alphafold.sh BMRC V100 inference
bmrc/inference/gh200/profile_alphafold.sh BMRC GH200 inference
arc/inference/profile_alphafold.sh        ARC GPU inference
arc/data_pipeline/profile_alphafold.sh   ARC CPU data pipeline
```

Both share the same reproducibility tooling in `scripts/` (metadata capture,
GPU sampling, and result validation), so neither duplicates the profiling
logic itself.

## Setup

AlphaFold 3 is retained as a Git submodule pinned to v3.0.4, for custom image
builds, source changes, or forks:

```bash
git submodule update --init --recursive
```

The definition builds for the architecture of its build node when custom images
are needed:

```bash
apptainer build images/alphafold3-v3.0.4-x86_64.sif \
  containers/alphafold3-v3.0.4.def
apptainer build images/alphafold3-v3.0.4-arm64.sif \
  containers/alphafold3-v3.0.4.def
```

### BMRC

BMRC comparison wrappers use matching AF3 3.0.3 images supplied by BMRC, plus
shared model parameters and databases:

```text
/apps/singularity/alphafold3/alphafold-3.0.3.sif
/apps/singularity/alphafold3/alphafold-3.0.3-arm.sif
/data/belmont/alphafold3-parameters
/data/belmont/alphafold-3.0.1-20250212
```

Belmont storage is available to these jobs from `gpu_strubi` and
`gpu_gh200_bmrc` nodes.

### ARC

ARC wrappers use our own submodule-built image rather than any
shared AlphaFold installation; only the model parameters and databases are
taken from shared storage:

```text
/data/dtce-oxrse/dtce0101/af_artefacts/model_param
/data/dtce-oxrse/dtce0101/af_artefacts/public_databases
```

Build an image for ARC (see [Setup](#setup) above for the command), matching
the architecture of the node you intend to use. GPUs on ARC are exposed on the
`htc` cluster, whose A100 nodes are `x86_64` with 40GB of device memory; place
the image at `images/alphafold3-v3.0.4-x86_64.sif`, or set `AF3_SIF` to its
actual path.

Data pipeline and inference now have separate submit scripts and resource
requests. Data pipeline requests CPU-only resources; inference requests one
A100 GPU.

## Profiling

Run stages separately when measuring them independently. Submit data pipeline
first using the original input; it writes AF3's feature-enriched `*_data.json`
inside its profile output. Submit inference using that generated JSON:

```bash
sbatch arc/data_pipeline/profile_alphafold.sh inputs/lysozyme_1lyz.json
sbatch arc/inference/profile_alphafold.sh \
  profiles/DATA_RUN/output/lysozyme_1lyz/lysozyme_1lyz_data.json
```

BMRC comparison uses the generated data JSON once for all three GPU jobs:

```bash
sbatch bmrc/data_pipeline/profile_alphafold.sh inputs/lysozyme_1lyz.json
bash bmrc/inference/submit_all.sh \
  profiles/DATA_RUN/output/lysozyme_1lyz/lysozyme_1lyz_data.json
```

To submit the pipeline and all dependent inference jobs in one command:

```bash
bash bmrc/submit_comparison.sh inputs/lysozyme_1lyz.json
```

For a pipeline job that is already queued or running, attach inference jobs by
job ID. Each inference job waits for successful pipeline completion and resolves
its generated `*_data.json` when it starts:

```bash
bash bmrc/inference/submit_all.sh --after PIPELINE_JOB_ID
```

An explicit expected data-JSON path can also be supplied with the dependency:

```bash
bash bmrc/inference/submit_all.sh --after PIPELINE_JOB_ID \
  profiles/PIPELINE_RUN/output/JOB/JOB_data.json
```

`submit_all.sh` submits A100 80GB, V100 16GB, and GH200 jobs. V100 uses XLA
flash attention because AF3's Triton/cuDNN flash-attention paths require Ampere
or newer. Inference binds model parameters only; data pipeline binds databases
only. Each GPU class has a separate persistent JAX compilation cache. The first
run for a new token bucket is a cold-cache measurement; compare second runs for
steady-state inference, or clear all three cache directories when comparing
cold compilation plus inference.

Set `PROFILE_MODE=baseline` explicitly to retain JAX preallocation:

```bash
sbatch --export=ALL,PROFILE_MODE=baseline \
  bmrc/inference/a100/profile_alphafold.sh \
  profiles/DATA_RUN/output/lysozyme_1lyz/lysozyme_1lyz_data.json
```

Unified host memory (`TF_FORCE_UNIFIED_MEMORY=true`, with a larger
`XLA_CLIENT_MEM_FRACTION`) is disabled by default in all comparison scripts.
Leave it disabled when inputs fit device memory: enabling oversubscription can
change allocation behavior and may add migration overhead even when runtime is
often similar. Enable it only for memory-constrained inputs via:

```bash
sbatch --export=ALL,AF3_UNIFIED_MEMORY=true ...
```

Stage is selected by submit-script directory: `data_pipeline/` or `inference/`.
Generated images, outputs, and profiling results are ignored by Git. Input JSON
files belong in `inputs/`; each profiling run is written beneath `profiles/`.

Validate a completed run with `python3 scripts/validate_profile.py profiles/RUN`.
Pass `--reference 1LYZ.cif` to also calculate RMSD and TM-score with US-align.
