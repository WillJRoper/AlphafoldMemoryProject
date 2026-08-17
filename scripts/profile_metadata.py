#!/usr/bin/env python3
"""Write metadata for an AlphaFold profiling run."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise_input(data):
    jobs = data if isinstance(data, list) else [data]
    summaries = []
    for job in jobs:
        polymers = []
        ligands = 0
        for entity in job.get("sequences", []):
            polymers.extend(
                entity[kind]["sequence"]
                for kind in ("protein", "rna", "dna")
                if kind in entity
            )
            ligands += "ligand" in entity
        summaries.append(
            {
                "name": job.get("name"),
                "model_seeds": job.get("modelSeeds", job.get("randomSeed")),
                "sequence_entities": len(job.get("sequences", [])),
                "polymer_entities": len(polymers),
                "polymer_residues": sum(map(len, polymers)),
                "ligand_entities": ligands,
            }
        )
    return summaries


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def create(args):
    input_path = Path(args.input)
    sif_path = Path(args.sif)
    input_data = json.loads(input_path.read_text())
    input_hash = sha256(input_path)
    sif_hash = sha256(sif_path)
    gpu_fields = ("index", "uuid", "name", "memory_total_mib")
    with open(args.gpu_inventory, newline="") as file:
        gpus = [dict(zip(gpu_fields, map(str.strip, row))) for row in csv.reader(file)]

    started = time.time()
    metadata = {
        "schema_version": 1,
        "run_id": Path(args.output).parent.name,
        "status": "running",
        "start_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "end_time_utc": None,
        "wall_clock_seconds": None,
        "exit_status": None,
        "profiling": {
            "mode": args.mode,
            "stage": args.stage,
            "gpu_sampling_interval_ms": args.interval,
            "jax_preallocate": os.environ["APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE"],
            "jax_memory_fraction": os.environ["APPTAINERENV_XLA_CLIENT_MEM_FRACTION"],
            "tf_force_unified_memory": os.environ["APPTAINERENV_TF_FORCE_UNIFIED_MEMORY"],
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        },
        "host": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": gpus,
        },
        "alphafold": {"version": args.af_version, "git_commit": args.af_commit},
        "container": {"path": str(sif_path.resolve()), "sha256": sif_hash},
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_hash,
            "jobs": summarise_input(input_data),
        },
        "command": args.command,
        "maximum_host_rss_kb": None,
        "_started": started,
    }
    write(Path(args.output), metadata)


def finish(args):
    path = Path(args.metadata)
    metadata = json.loads(path.read_text())
    metadata["status"] = "completed" if args.exit_status == 0 else "failed"
    metadata["end_time_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata["wall_clock_seconds"] = round(time.time() - metadata.pop("_started"), 3)
    metadata["exit_status"] = args.exit_status
    for line in Path(args.time_file).read_text().splitlines():
        if "Maximum resident set size (kbytes):" in line:
            metadata["maximum_host_rss_kb"] = int(line.rsplit(":", 1)[1])
    write(path, metadata)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)

    start = commands.add_parser("create")
    for name in ("output", "input", "sif", "mode", "stage", "gpu-inventory", "af-version", "af-commit"):
        start.add_argument(f"--{name}", required=True)
    start.add_argument("--interval", type=int, required=True)
    start.add_argument("command", nargs=argparse.REMAINDER)

    end = commands.add_parser("finish")
    end.add_argument("--metadata", required=True)
    end.add_argument("--time-file", required=True)
    end.add_argument("--exit-status", type=int, required=True)

    args = parser.parse_args()
    create(args) if args.action == "create" else finish(args)


if __name__ == "__main__":
    main()
