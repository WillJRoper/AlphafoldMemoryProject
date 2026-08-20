#!/usr/bin/env python3
"""Fetch and validate manifest-pinned UniProt sequences for real scaling tests."""

import argparse
import csv
import json
import os
import tempfile
import urllib.request
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def read_manifest(path):
    with path.open() as file:
        return list(csv.DictReader(file, delimiter="\t"))


def parse_fasta(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("invalid FASTA response")
    return "".join(lines[1:]).upper()


def make_input(row, sequence):
    return {
        "name": f"real_{row['target_tokens']}_{row['accession'].lower()}",
        "modelSeeds": [1],
        "sequences": [{"protein": {"id": "A", "sequence": sequence}}],
        "dialect": "alphafold3",
        "version": 1,
    }


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(data, file, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    for row in read_manifest(args.manifest):
        output = args.output_dir / f"{row['accession']}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            sequence = existing["sequences"][0]["protein"]["sequence"]
        else:
            url = f"https://rest.uniprot.org/uniprotkb/{row['accession']}.fasta"
            request = urllib.request.Request(url, headers={"User-Agent": "af3-scaling/1"})
            with urllib.request.urlopen(request) as response:
                sequence = parse_fasta(response.read().decode())
            write_atomic(output, make_input(row, sequence))

        expected = int(row["expected_length"])
        if len(sequence) != expected:
            raise SystemExit(
                f"{row['accession']}: expected {expected} residues, fetched {len(sequence)}"
            )
        invalid = set(sequence) - AMINO_ACIDS
        if invalid:
            raise SystemExit(f"{row['accession']}: unsupported residues {sorted(invalid)}")
        print(f"{row['accession']}\t{len(sequence)}\t{output}")


if __name__ == "__main__":
    main()
