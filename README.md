# AlphaFold 3 profiling

Reproducible runtime and memory profiling for AlphaFold 3 on NVIDIA GPUs.
AlphaFold runs in an Apptainer container launched from a Slurm job.

## Setup

Profiling uses the shared AlphaFold 3.0.1 Apptainer image, model parameters, and
databases by default. The repository also retains AlphaFold 3 as a Git submodule
pinned to v3.0.4 for custom image builds or future source changes:

```bash
git submodule update --init --recursive
```

The optional ARM64 container definition is `containers/alphafold3-v3.0.4.def`:

```bash
apptainer build images/alphafold3-v3.0.4-arm64.sif \
  containers/alphafold3-v3.0.4.def
```

Shared defaults used by the profiling script are:

```text
/apps/singularity/alphafold3/alphafold-3.0.1-20250210.sif
/data/belmont/alphafold3-parameters
/data/belmont/alphafold-3.0.1-20250212
```

Belmont storage is available only from relevant login and `gpu_strubi` nodes.

## Profiling

Submit a complete run with the default `memory_characterisation` mode, which
disables JAX GPU-memory preallocation:

```bash
sbatch scripts/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

Set `PROFILE_MODE=baseline` explicitly to retain JAX preallocation:

```bash
sbatch --export=ALL,PROFILE_MODE=baseline \
  scripts/profile_alphafold.sh inputs/lysozyme_1lyz.json
```

`AF3_STAGE` selects `complete`, `data-pipeline`, or `inference`. Generated
images, outputs, and profiling results are ignored by Git. Input JSON files
belong in `inputs/`; each profiling run is written beneath `profiles/`.

Validate a completed run with `python3 scripts/validate_profile.py profiles/RUN`.
Pass `--reference 1LYZ.cif` to also calculate RMSD and TM-score with US-align.
