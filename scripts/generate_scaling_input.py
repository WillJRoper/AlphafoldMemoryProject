#!/usr/bin/env python3
"""Generate a deterministic, query-only, single-chain AF3 scaling input."""

import argparse
import json
import os
import tempfile
from pathlib import Path

BASE_SEQUENCE = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWW"
    "CNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)


def make_input(tokens):
    sequence = (BASE_SEQUENCE * ((tokens + len(BASE_SEQUENCE) - 1) // len(BASE_SEQUENCE)))[:tokens]
    return {
        "name": f"scaling_{tokens}",
        "modelSeeds": [1],
        "sequences": [{
            "protein": {
                "id": "A",
                "sequence": sequence,
                "unpairedMsa": f">query\n{sequence}\n",
                "pairedMsa": "",
                "templates": [],
            }
        }],
        "dialect": "alphafold3",
        "version": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokens", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.tokens < 1:
        raise SystemExit("tokens must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as file:
        json.dump(make_input(args.tokens), file, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
