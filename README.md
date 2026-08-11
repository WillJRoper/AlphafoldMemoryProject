# AlphaFold 3 profiling

Reproducible runtime and memory profiling for AlphaFold 3 on NVIDIA GH200
systems. AlphaFold runs in an Apptainer container launched from a Slurm job.

## Setup

AlphaFold 3 is included as a Git submodule pinned to v3.0.4:

```bash
git submodule update --init --recursive
```

The ARM64 container definition is `containers/alphafold3-v3.0.4.def`. Build it
from the repository root on an ARM64 GH200 node:

```bash
apptainer build images/alphafold3-v3.0.4-arm64.sif \
  containers/alphafold3-v3.0.4.def
```

Model parameters and databases are not included. Set their locations before
submitting a profiling job:

```bash
export AF3_MODEL_DIR=/path/to/models
export AF3_DB_DIR=/path/to/databases
```

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
