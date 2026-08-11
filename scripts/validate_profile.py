#!/usr/bin/env python3
"""Validate a profile directory and optionally compare its model with a reference."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def parse_usalign(output):
    aligned = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.]+)", output)
    scores = re.findall(r"TM-score=\s*([\d.]+)", output)
    return {
        "aligned_residues": int(aligned.group(1)),
        "rmsd_angstrom": float(aligned.group(2)),
        "tm_score_prediction": float(scores[0]),
        "tm_score_reference": float(scores[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--usalign", default="USalign")
    args = parser.parse_args()

    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    if metadata["status"] != "completed" or metadata["exit_status"] != 0:
        raise SystemExit("AlphaFold run did not complete successfully")
    if not (args.run_dir / "time.txt").stat().st_size or not (
        args.run_dir / "alphafold.log"
    ).stat().st_size:
        raise SystemExit("timing data or AlphaFold log is empty")

    with (args.run_dir / "gpu.csv").open() as file:
        gpu_samples = sum(1 for _ in csv.reader(file)) - 1
    if gpu_samples < 1:
        raise SystemExit("gpu.csv contains no samples")

    models = list((args.run_dir / "output").rglob("*_model.cif"))
    summaries = list((args.run_dir / "output").rglob("*_summary_confidences.json"))
    if not models or not summaries:
        raise SystemExit("AlphaFold model or confidence summary is missing")

    # Ranked model is in the shallowest output directory; seed samples are nested.
    prediction = min(models, key=lambda path: len(path.parts))
    summary = min(summaries, key=lambda path: len(path.parts))
    result = {
        "run_id": metadata["run_id"],
        "gpu_samples": gpu_samples,
        "prediction": str(prediction),
        "prediction_sha256": sha256(prediction),
        "summary_confidences": json.loads(summary.read_text()),
    }

    if args.reference:
        output = subprocess.run(
            [args.usalign, str(prediction), str(args.reference), "-mol", "prot"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result["reference"] = str(args.reference)
        result["reference_sha256"] = sha256(args.reference)
        result["structure_alignment"] = parse_usalign(output)

    path = args.run_dir / "validation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
