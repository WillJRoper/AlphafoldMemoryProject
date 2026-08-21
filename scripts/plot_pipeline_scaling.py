#!/usr/bin/env python3
"""Plot AF3 data-pipeline length scaling and CPU strong scaling."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_pipeline_phases(text):
    patterns = {
        "pipeline_s": r"Running data pipeline for .*? took ([\d.]+) seconds",
        "msa_s": r"Getting protein MSAs took ([\d.]+) seconds",
        "dedup_s": r"Deduplicating MSAs took ([\d.]+) seconds",
        "templates_s": r"Getting \d+ protein templates took ([\d.]+) seconds",
    }
    return {
        name: float(match.group(1))
        for name, pattern in patterns.items()
        if (match := re.search(pattern, text))
    }


def collect_pipeline_profiles(root):
    records = []
    strong_pattern = re.compile(
        r"scaling-pipeline-bmrc-([A-Z0-9]+)-c(\d+)-t(\d+)-r(\d+)-"
    )
    for metadata_path in root.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("profiling", {}).get("stage") != "data-pipeline":
            continue
        if metadata.get("status") != "completed" or metadata.get("exit_status") != 0:
            continue
        run_id = metadata.get("run_id", metadata_path.parent.name)
        strong_match = strong_pattern.match(run_id)
        is_length = run_id.startswith("scaling-real-bmrc-pipeline-")
        if not strong_match and not is_length:
            continue

        job = metadata.get("input", {}).get("jobs", [{}])[0]
        input_name = job.get("name", "") or ""
        name_match = re.match(r"real_(\d+)_([a-z0-9]+)", input_name)
        target = int(name_match.group(1)) if name_match else job.get("polymer_residues", 0)
        accession = name_match.group(2).upper() if name_match else "unknown"
        cpus = int(metadata.get("slurm", {}).get("cpus_per_task") or 0)
        threads = None
        repeat = None
        family = "length"
        if strong_match:
            accession, cpus_text, threads_text, repeat_text = strong_match.groups()
            cpus = int(cpus_text)
            threads = int(threads_text)
            repeat = int(repeat_text)
            family = "threads"

        log_path = metadata_path.parent / "alphafold.log"
        phases = parse_pipeline_phases(log_path.read_text() if log_path.exists() else "")
        wall = float(metadata.get("wall_clock_seconds") or 0)
        runtime = phases.get("pipeline_s", wall)
        max_rss = metadata.get("maximum_host_rss_kb")
        records.append({
            "family": family,
            "run": run_id,
            "accession": accession,
            "target": int(target),
            "residues": int(job.get("polymer_residues") or target),
            "cpus": cpus,
            "jackhmmer_threads": threads,
            "repeat": repeat,
            "runtime_s": runtime,
            "wall_s": wall,
            "max_rss_gib": float(max_rss) / 1048576 if max_rss else float("nan"),
            **phases,
        })
    return pd.DataFrame(records)


def plot_length(data, output):
    data = data.sort_values("residues")
    medians = data.groupby("target", as_index=False).agg(
        residues=("residues", "median"),
        runtime_s=("runtime_s", "median"),
        max_rss_gib=("max_rss_gib", "median"),
    ).sort_values("residues")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    scatter = axes[0].scatter(data["residues"], data["runtime_s"],
                              c=data["target"], cmap="viridis", alpha=0.65)
    axes[0].plot(medians["residues"], medians["runtime_s"], color="black",
                 marker="o", lw=1, label="Target-bin median")
    axes[0].set_ylabel("Data-pipeline runtime (s)")
    axes[0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label="Target token group")

    memory = data.dropna(subset=["max_rss_gib"])
    axes[1].scatter(memory["residues"], memory["max_rss_gib"],
                    c=memory["target"], cmap="viridis", alpha=0.65)
    memory_medians = medians.dropna(subset=["max_rss_gib"])
    axes[1].plot(memory_medians["residues"], memory_medians["max_rss_gib"],
                 color="black", marker="o", lw=1)
    axes[1].set_ylabel("Maximum host RSS (GiB)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Protein residues")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title("AF3 data-pipeline length scaling")
    axes[1].set_title("AF3 data-pipeline host memory")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_threads(data, output):
    summary = data.groupby(["accession", "cpus"], as_index=False).agg(
        residues=("residues", "median"),
        runtime_s=("runtime_s", "median"),
        runtime_min=("runtime_s", "min"),
        runtime_max=("runtime_s", "max"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for accession, group in summary.groupby("accession"):
        group = group.sort_values("cpus")
        baseline = group.iloc[0]
        label = f"{accession} ({int(baseline['residues']):,} aa)"
        speedup = baseline["runtime_s"] / group["runtime_s"]
        efficiency = speedup / (group["cpus"] / baseline["cpus"])
        axes[0].plot(group["cpus"], group["runtime_s"], marker="o", label=label)
        axes[0].fill_between(group["cpus"], group["runtime_min"],
                             group["runtime_max"], alpha=0.15)
        axes[1].plot(group["cpus"], speedup, marker="o", label=label)
        axes[2].plot(group["cpus"], efficiency, marker="o", label=label)

    cpus = sorted(summary["cpus"].unique())
    axes[1].plot(cpus, [cpu / cpus[0] for cpu in cpus], color="black", ls=":",
                 label="Ideal")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log", base=2)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cpus, labels=[str(cpu) for cpu in cpus])
        ax.set_xlabel("Allocated CPUs")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Data-pipeline runtime (s)")
    axes[1].set_ylabel("Speedup relative to 4 CPUs")
    axes[2].set_ylabel("Parallel efficiency")
    axes[2].set_ylim(0, 1.1)
    axes[0].set_title("CPU strong scaling")
    axes[1].set_title("Strong-scaling speedup")
    axes[2].set_title("Strong-scaling efficiency")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="?", default=Path("profiles"))
    parser.add_argument("--kind", choices=("all", "length", "threads"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = collect_pipeline_profiles(args.profiles)

    made = False
    if args.kind in ("all", "length"):
        length = data[data["family"] == "length"] if not data.empty else data
        if not length.empty:
            output = args.output_dir / "scaling_pipeline_length.png"
            plot_length(length, output)
            print(output)
            made = True
    if args.kind in ("all", "threads"):
        threads = data[data["family"] == "threads"] if not data.empty else data
        if not threads.empty:
            output = args.output_dir / "scaling_pipeline_threads.png"
            plot_threads(threads, output)
            print(output)
            made = True
    if not made:
        raise SystemExit("no completed matching data-pipeline profiles found")


if __name__ == "__main__":
    main()
