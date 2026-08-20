#!/usr/bin/env python3
"""Fetch and validate manifest-pinned UniProt sequences for real scaling tests."""

import argparse
import csv
import json
import os
import tempfile
import time
import urllib.error
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


def fetch_sequence(accession, attempts=5, opener=urllib.request.urlopen, sleep=time.sleep):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    request = urllib.request.Request(url, headers={"User-Agent": "af3-scaling/1"})
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout=30) as response:
                return parse_fasta(response.read().decode())
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            last_error = error
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            last_error = error
        if attempt < attempts:
            delay = 2 ** (attempt - 1)
            print(f"  attempt {attempt}/{attempts} failed: {last_error}; retrying in {delay}s", flush=True)
            sleep(delay)
    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    cached = 0
    downloaded = 0
    print(f"Preparing {len(rows)} UniProt inputs in {args.output_dir}", flush=True)
    for index, row in enumerate(rows, 1):
        accession = row["accession"]
        output = args.output_dir / f"{row['accession']}.json"
        print(f"[{index}/{len(rows)}] {accession}: ", end="", flush=True)
        if output.exists():
            print("validating cached input", flush=True)
            existing = json.loads(output.read_text())
            sequence = existing["sequences"][0]["protein"]["sequence"]
            cached += 1
        else:
            print("fetching", flush=True)
            try:
                sequence = fetch_sequence(accession)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                raise SystemExit(f"{accession}: fetch failed after 5 attempts: {error}")
            downloaded += 1

        expected = int(row["expected_length"])
        if len(sequence) != expected:
            raise SystemExit(
                f"{row['accession']}: expected {expected} residues, fetched {len(sequence)}"
            )
        invalid = set(sequence) - AMINO_ACIDS
        if invalid:
            raise SystemExit(f"{row['accession']}: unsupported residues {sorted(invalid)}")
        if not output.exists():
            write_atomic(output, make_input(row, sequence))
        print(f"  ready: {len(sequence)} residues -> {output}", flush=True)
    print(f"Ready: {len(rows)} inputs ({cached} cached, {downloaded} downloaded)")


if __name__ == "__main__":
    main()
