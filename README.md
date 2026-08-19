# AlphaFold 3 profiling

Reproducible runtime and memory profiling for AlphaFold 3 on NVIDIA GPUs.
AlphaFold runs in an Apptainer container launched from a Slurm job.

Submission scripts are site-specific because each HPC has a different shared
AlphaFold installation and Slurm configuration (account/partition/GRES syntax,
and in some cases a different container invocation model entirely):

```text
bmrc/inference/profile_alphafold.sh       BMRC GPU inference
bmrc/data_pipeline/profile_alphafold.sh  BMRC CPU data pipeline
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

The optional ARM64 container definition is `containers/alphafold3-v3.0.4.def`:

```bash
apptainer build images/alphafold3-v3.0.4-arm64.sif \
  containers/alphafold3-v3.0.4.def
```

### BMRC

Shared defaults used by BMRC wrappers:

```text
/apps/singularity/alphafold3/alphafold-3.0.1-20250210.sif
/data/belmont/alphafold3-parameters
/data/belmont/alphafold-3.0.1-20250212
```

Belmont storage is available only from relevant login and `gpu_strubi` nodes.

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

Submit a complete run with the default `memory_characterisation` mode, which
disables JAX GPU-memory preallocation to give a more representative device-memory
time series:

```bash
sbatch bmrc/inference/profile_alphafold.sh inputs/lysozyme_1lyz.json
sbatch arc/inference/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

Run stages separately when measuring them independently. Submit data pipeline
first using the original input; it writes AF3's feature-enriched `*_data.json`
inside its profile output. Submit inference using that generated JSON:

```bash
sbatch arc/data_pipeline/profile_alphafold.sh inputs/lysozyme_1lyz.json
sbatch arc/inference/profile_alphafold.sh \
  profiles/DATA_RUN/output/lysozyme_1lyz/lysozyme_1lyz_data.json
```

The BMRC commands use the same pattern under `bmrc/`. Inference binds model
parameters only; data pipeline binds databases only.

Set `PROFILE_MODE=baseline` explicitly to retain JAX preallocation:

```bash
sbatch --export=ALL,PROFILE_MODE=baseline \
  bmrc/inference/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

Unified host memory (`TF_FORCE_UNIFIED_MEMORY=true`, with a larger
`XLA_CLIENT_MEM_FRACTION`) is supported by both shared installations but is not
part of either profiling mode; enable it explicitly for memory-constrained
hosts via:

```bash
sbatch --export=ALL,AF3_UNIFIED_MEMORY=true ...
```

Stage is selected by submit-script directory: `data_pipeline/` or `inference/`.
Generated images, outputs, and profiling results are ignored by Git. Input JSON
files belong in `inputs/`; each profiling run is written beneath `profiles/`.

Validate a completed run with `python3 scripts/validate_profile.py profiles/RUN`.
Pass `--reference 1LYZ.cif` to also calculate RMSD and TM-score with US-align.
