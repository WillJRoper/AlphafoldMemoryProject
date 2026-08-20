#!/usr/bin/env python3
"""Aggregate scaling profiles into runtime and peak-memory curves."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def inference_seconds(log, fallback):
    match = re.search(
        r"Running model inference with seed .*? took ([\d.]+) seconds", log
    )
    return float(match.group(1)) if match else float(fallback)


def collect_profiles(root):
    records = []
    failures = []
    pattern = re.compile(r"scaling-(bmrc|arc)-(a100|gh200)-(\d+)-(device|unified)-")
    for metadata_path in root.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        match = pattern.match(metadata.get("run_id", metadata_path.parent.name))
        if not match:
            continue
        site, gpu, tokens, mode = match.groups()
        label = f"{site.upper()} {gpu.upper()}"
        if metadata.get("status") != "completed" or metadata.get("exit_status") != 0:
            log_path = metadata_path.parent / "alphafold.log"
            log = log_path.read_text() if log_path.exists() else ""
            oom = bool(re.search(
                r"out of memory|oom-kill|resource_exhausted|cuda_error_out_of_memory",
                log,
                re.IGNORECASE,
            ))
            failures.append((label, int(tokens), mode, metadata_path.parent.name, oom))
            continue
        gpu_csv = metadata_path.parent / "gpu.csv"
        samples = pd.read_csv(gpu_csv, skipinitialspace=True)
        if samples.empty:
            failures.append((label, int(tokens), mode, metadata_path.parent.name, False))
            continue
        log = (metadata_path.parent / "alphafold.log").read_text()
        records.append({
            "hardware": label,
            "tokens": int(tokens),
            "mode": mode,
            "runtime_s": inference_seconds(log, metadata["wall_clock_seconds"]),
            "peak_memory_gib": samples["memory_used_mib"].max() / 1024,
        })
    return pd.DataFrame(records), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="?", default=Path("profiles"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.profiles / "scaling.png"

    data, failures = collect_profiles(args.profiles)
    if data.empty:
        raise SystemExit("no completed scaling profiles found")
    data = data.groupby(["hardware", "tokens", "mode"], as_index=False).median(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (hardware, group) in zip(colors, data.groupby("hardware")):
        for mode, style, marker in (("device", "-", "o"), ("unified", "--", "s")):
            subset = group[group["mode"] == mode].sort_values("tokens")
            if subset.empty:
                continue
            axes[0].plot(subset["tokens"], subset["runtime_s"], style,
                         marker=marker, color=color, label=f"{hardware} ({mode})")
            axes[1].plot(subset["tokens"], subset["peak_memory_gib"], style,
                         marker=marker, color=color, label=f"{hardware} ({mode})")
        oom_tokens = [tokens for failed_hardware, tokens, mode, _, oom in failures
                      if failed_hardware == hardware and mode == "device" and oom]
        if oom_tokens:
            threshold = min(oom_tokens)
            for ax in axes:
                ax.axvline(threshold, color=color, ls=":", alpha=0.8)
            axes[0].text(threshold, axes[0].get_ylim()[1], f" {hardware} device OOM",
                         color=color, rotation=90, va="top", fontsize=8)

    axes[0].set_title("AF3 inference runtime scaling")
    axes[0].set_ylabel("Inference runtime (s)")
    axes[1].set_title("AF3 peak GPU-memory scaling")
    axes[1].set_ylabel("Peak GPU memory (GiB)")
    for ax in axes:
        ax.set_xlabel("Protein tokens")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(output)
    for hardware, tokens, mode, run, oom in failures:
        reason = "OOM" if oom else "other"
        print(f"failed: {hardware} {tokens} tokens ({mode}, {reason}) {run}")


if __name__ == "__main__":
    main()
