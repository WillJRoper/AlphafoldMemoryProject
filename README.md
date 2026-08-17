# AlphaFold 3 profiling

Reproducible runtime and memory profiling for AlphaFold 3 on NVIDIA GPUs.
AlphaFold runs in an Apptainer container launched from a Slurm job.

Submission scripts are site-specific because each HPC has a different shared
AlphaFold installation and Slurm configuration (account/partition/GRES syntax,
and in some cases a different container invocation model entirely):

```text
bmrc/profile_alphafold.sh   BMRC: apptainer run, gpu_strubi partition
arc/profile_alphafold.sh    ARC:  apptainer exec, htc cluster, dtce-oxrse account
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

Shared defaults used by `bmrc/profile_alphafold.sh`:

```text
/apps/singularity/alphafold3/alphafold-3.0.1-20250210.sif
/data/belmont/alphafold3-parameters
/data/belmont/alphafold-3.0.1-20250212
```

Belmont storage is available only from relevant login and `gpu_strubi` nodes.

### ARC

`arc/profile_alphafold.sh` uses our own submodule-built image rather than any
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

The data-pipeline stage needs no GPU and a very different resource shape from
inference; the script's default `#SBATCH` block targets the GPU case, and
data-pipeline submissions must override partition, GRES, memory, CPUs, and
time via `sbatch` flags (see the comment at the top of `arc/profile_alphafold.sh`).

## Profiling

Submit a complete run with the default `memory_characterisation` mode, which
uses the shared installation's recommended unified-memory settings and disables
JAX GPU-memory preallocation:

```bash
sbatch bmrc/profile_alphafold.sh inputs/lysozyme_1lyz.json
sbatch arc/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

Set `PROFILE_MODE=baseline` explicitly to retain JAX preallocation:

```bash
sbatch --export=ALL,PROFILE_MODE=baseline \
  bmrc/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

`AF3_STAGE` selects `complete`, `data-pipeline`, or `inference`. Generated
images, outputs, and profiling results are ignored by Git. Input JSON files
belong in `inputs/`; each profiling run is written beneath `profiles/`.

Validate a completed run with `python3 scripts/validate_profile.py profiles/RUN`.
Pass `--reference 1LYZ.cif` to also calculate RMSD and TM-score with US-align.
