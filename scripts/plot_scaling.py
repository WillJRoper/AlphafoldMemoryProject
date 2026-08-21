#!/usr/bin/env python3
"""Aggregate scaling profiles into runtime and peak-memory curves."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def inference_seconds(log, fallback):
    match = re.search(
        r"Running model inference with seed .*? took ([\d.]+) seconds", log
    )
    return float(match.group(1)) if match else float(fallback)


def collect_profiles(root, family="repeat"):
    records = []
    failures = []
    incomplete = []
    latest = {}
    if family == "real":
        pattern = re.compile(
            r"scaling-real-(bmrc|arc)-(a100|gh200)-(\d+)-([A-Z0-9]+)-(device|preallocated|unified)-"
        )
    else:
        pattern = re.compile(
            r"scaling-(?:repeat-)?(bmrc|arc)-(a100|gh200)-(\d+)-(device|preallocated|unified)-"
        )
    for metadata_path in root.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        match = pattern.match(metadata.get("run_id", metadata_path.parent.name))
        if not match:
            continue
        if family == "real":
            site, gpu, target, accession, mode = match.groups()
            tokens = metadata.get("input", {}).get("jobs", [{}])[0].get(
                "polymer_residues", int(target)
            )
        else:
            site, gpu, target, mode = match.groups()
            accession = "repeat"
            tokens = int(target)
        label = f"{site.upper()} {gpu.upper()}"
        key = (label, int(target), accession, mode)
        candidate = (metadata.get("start_time_utc") or "", metadata_path, metadata,
                     int(tokens), int(target), accession, mode, label)
        if key not in latest or candidate[0] > latest[key][0]:
            latest[key] = candidate

    for _, metadata_path, metadata, tokens, target, accession, mode, label in latest.values():
        status = metadata.get("status")
        if status == "running":
            incomplete.append((label, target, mode, metadata_path.parent.name))
            continue
        if status != "completed" or metadata.get("exit_status") != 0:
            log_path = metadata_path.parent / "alphafold.log"
            log = log_path.read_text() if log_path.exists() else ""
            oom = bool(re.search(
                r"out of memory|oom-kill|resource_exhausted|cuda_error_out_of_memory",
                log,
                re.IGNORECASE,
            ))
            failures.append((label, target, mode, metadata_path.parent.name, oom))
            continue
        gpu_csv = metadata_path.parent / "gpu.csv"
        samples = pd.read_csv(gpu_csv, skipinitialspace=True)
        if samples.empty:
            failures.append((label, target, mode, metadata_path.parent.name, False))
            continue
        log = (metadata_path.parent / "alphafold.log").read_text()
        records.append({
            "hardware": label,
            "tokens": int(tokens),
            "target": int(target),
            "accession": accession,
            "mode": mode,
            "runtime_s": inference_seconds(log, metadata["wall_clock_seconds"]),
            "peak_memory_gib": samples["memory_used_mib"].max() / 1024,
        })
    return pd.DataFrame(records), failures, incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="?", default=Path("profiles"))
    parser.add_argument("--family", choices=("repeat", "real"), default="repeat")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("plots") / f"scaling_{args.family}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    data, failures, incomplete = collect_profiles(args.profiles, args.family)
    if data.empty:
        raise SystemExit("no completed scaling profiles found")
    raw = data.copy()
    data = data.groupby(["hardware", "target", "mode"], as_index=False).agg(
        tokens=("tokens", "median"),
        runtime_s=("runtime_s", "median"),
        runtime_min=("runtime_s", "min"),
        runtime_max=("runtime_s", "max"),
        peak_memory_gib=("peak_memory_gib", "median"),
        memory_min=("peak_memory_gib", "min"),
        memory_max=("peak_memory_gib", "max"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hardware_handles = []
    for color, (hardware, group) in zip(colors, data.groupby("hardware")):
        hardware_handles.append(Line2D([0], [0], color=color, lw=3, label=hardware))
        for mode, style, marker in (
            ("device", "-", "o"),
            ("preallocated", "-.", "^"),
            ("unified", "--", "s"),
        ):
            subset = group[group["mode"] == mode].sort_values("tokens")
            if subset.empty:
                continue
            axes[0].plot(subset["tokens"], subset["runtime_s"], style,
                         marker=marker, color=color, label="_nolegend_")
            axes[1].plot(subset["tokens"], subset["peak_memory_gib"], style,
                         marker=marker, color=color, label="_nolegend_")
            if args.family == "real":
                points = raw[(raw["hardware"] == hardware) & (raw["mode"] == mode)]
                axes[0].scatter(points["tokens"], points["runtime_s"],
                                color=color, marker=marker, alpha=0.3, s=18)
                axes[1].scatter(points["tokens"], points["peak_memory_gib"],
                                color=color, marker=marker, alpha=0.3, s=18)
                axes[0].fill_between(subset["tokens"], subset["runtime_min"],
                                     subset["runtime_max"], color=color, alpha=0.1)
                axes[1].fill_between(subset["tokens"], subset["memory_min"],
                                     subset["memory_max"], color=color, alpha=0.1)
        oom_tokens = [tokens for failed_hardware, tokens, mode, _, oom in failures
                      if failed_hardware == hardware and mode == "device" and oom]
        if oom_tokens:
            threshold = min(oom_tokens)
            for ax in axes:
                ax.axvline(threshold, color=color, ls=":", alpha=0.8)

    trait_handles = [
        Line2D([0], [0], color="black", ls="-", marker="o", label="On-demand device"),
        Line2D([0], [0], color="black", ls="-.", marker="^", label="Preallocated device"),
        Line2D([0], [0], color="black", ls="--", marker="s", label="Unified memory"),
    ]
    if any(mode == "device" and oom for _, _, mode, _, oom in failures):
        trait_handles.append(Line2D([0], [0], color="black", ls=":",
                                    label="First on-demand OOM"))

    axes[0].set_title("AF3 inference runtime scaling")
    axes[0].set_ylabel("Inference runtime (s)")
    axes[1].set_title("AF3 peak GPU-memory scaling")
    axes[1].set_ylabel("Peak GPU memory (GiB)")
    for ax in axes:
        ax.set_xlabel("Protein tokens")
        ax.grid(alpha=0.3)
        hardware_legend = ax.legend(handles=hardware_handles, title="Hardware",
                                    fontsize=8, title_fontsize=8, loc="upper left")
        ax.add_artist(hardware_legend)
        ax.legend(handles=trait_handles, title="Profile",
                  fontsize=8, title_fontsize=8, loc="upper right")
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(output)
    for hardware, tokens, mode, run, oom in failures:
        reason = "OOM" if oom else "other"
        print(f"failed: {hardware} {tokens} tokens ({mode}, {reason}) {run}")
    for hardware, tokens, mode, run in incomplete:
        print(f"incomplete: {hardware} {tokens} tokens ({mode}) {run}")


if __name__ == "__main__":
    main()
