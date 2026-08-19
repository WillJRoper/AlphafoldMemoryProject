#!/usr/bin/env python3
"""Plot GPU time series and diagnostics for an AlphaFold profiling run."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
GIB = 1024.0


def load_gpu_csv(path):
    df = pd.read_csv(path, skipinitialspace=True)
    time = pd.to_datetime(df["timestamp"], format=TIME_FORMAT)
    df["elapsed_min"] = (time - time.iloc[0]).dt.total_seconds() / 60.0
    return df


def active_window(df, util_threshold=20):
    """Return (start, end) elapsed minutes of sustained GPU activity, else (None, None)."""
    busy = df["utilization_gpu_percent"] > util_threshold
    if not busy.any():
        return None, None
    return (
        float(df.loc[busy, "elapsed_min"].min()),
        float(df.loc[busy, "elapsed_min"].max()),
    )


def model_load_at(df, threshold_mib=1024):
    """First elapsed minute at which GPU memory exceeds a resident model threshold."""
    loaded = df["memory_used_mib"] > threshold_mib
    return float(df.loc[loaded, "elapsed_min"].min()) if loaded.any() else None


def memory_exceedance(df):
    """Sorted memory values x and P(memory > x), used for an exceedance curve."""
    mem = df["memory_used_mib"].sort_values().to_numpy()
    n = len(mem)
    exceedance = 1.0 - (pd.RangeIndex(n) + 1) / n
    return mem, exceedance


def parse_phase_durations(path):
    """Extract AF3 phase durations in seconds from its human-readable log."""
    text = path.read_text()
    phases = {}
    match = re.search(r"Running data pipeline for .*? took ([\d.]+) seconds", text)
    if match:
        phases["data pipeline"] = float(match.group(1))
    match = re.search(
        r"Running model inference and extracting output structures with .*? took "
        r"([\d.]+) seconds",
        text,
    )
    if match:
        phases["inference"] = float(match.group(1))
    return phases


def time_seconds(value):
    match = re.search(r"([\d.]+)", value or "")
    return float(match.group(1)) if match else None


def inference_slice(df):
    start, end = active_window(df)
    if start is None:
        return df
    return df[(df["elapsed_min"] >= start) & (df["elapsed_min"] <= end)]


def memory_summary(df):
    """Return baseline, peak, and peak-minus-baseline GPU memory in MiB."""
    baseline = df.loc[df["utilization_gpu_percent"] <= 20, "memory_used_mib"].median()
    peak = df["memory_used_mib"].max()
    return baseline, peak, peak - baseline


def plot_timeseries(df, out, title):
    """Full-run memory overview plus zoomed panels on the GPU-active window."""
    t = df["elapsed_min"]
    total = float(t.iloc[-1])
    win = active_window(df)
    load_at = model_load_at(df)
    mem_gib = df["memory_used_mib"] / GIB

    if win is None:
        z0, z1 = 0.0, total
    else:
        start, end = win
        pad = max((end - start) * 0.2, 10.0 / 60.0)
        z0, z1 = max(0.0, start - pad), min(total, end + pad)

    fig = plt.figure(figsize=(11, 15), layout="constrained")
    gs = fig.add_gridspec(6, 1, height_ratios=[1] + [2] * 5)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    ax = axes[0]
    ax.plot(t, mem_gib, lw=0.8, color="#1f77b4")
    if win is not None:
        ax.axvspan(*win, color="#d62728", alpha=0.15, label="inference (GPU active)")
    if load_at is not None:
        ax.axvline(load_at, ls=":", lw=1, color="grey")
        ax.text(load_at, ax.get_ylim()[1], " model load", fontsize=8, va="top")
    ax.set_ylabel("GPU mem\n(GiB)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    zoom = df[(df["elapsed_min"] >= z0) & (df["elapsed_min"] <= z1)]
    tz = zoom["elapsed_min"]
    base_mib = df.loc[df["utilization_gpu_percent"] <= 20, "memory_used_mib"].median()

    panels = [
        ("GPU memory (GiB)", zoom["memory_used_mib"] / GIB, "#1f77b4"),
        ("GPU utilization (%)", zoom["utilization_gpu_percent"], "#2ca02c"),
        ("Power draw (W)", zoom["power_draw_w"], "#d62728"),
        ("Temperature (°C)", zoom["temperature_gpu_c"], "#ff7f0e"),
        (None, zoom["clocks_sm_mhz"], None),
    ]
    for ax, (label, data, color) in zip(axes[1:], panels):
        if label is None:
            ax.plot(tz, zoom["clocks_memory_mhz"], lw=0.8, color="#8c564b", label="Memory")
            ax.plot(tz, data, lw=0.8, color="#9467bd", label="SM")
            ax.set_ylabel("Clocks (MHz)", fontsize=9)
            ax.legend(fontsize=8, ncol=2)
        else:
            ax.plot(tz, data, lw=0.8, color=color)
            ax.set_ylabel(label, fontsize=9)
            if label.startswith("GPU memory"):
                ax.axhline(base_mib / GIB, ls="--", lw=0.8, color="grey")
                peak = zoom["memory_used_mib"].max() / GIB
                ax.axhline(peak, ls=":", lw=0.8, color="grey")
                ax.text(tz.iloc[-1], peak, f" peak {peak:.2f} GiB", fontsize=8,
                        ha="right", va="bottom")
                ax.text(tz.iloc[-1], base_mib / GIB, f" baseline {base_mib / GIB:.2f} GiB",
                        fontsize=8, ha="right", va="bottom")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_xlim(z0, z1)
    axes[-1].set_xlabel("Elapsed time (min)")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_diagnostics(df, out, title):
    util = df["utilization_gpu_percent"]
    inference = inference_slice(df)
    baseline, peak, extra = memory_summary(df)
    phases = parse_phase_durations(Path(df.attrs["run_dir"]) / "alphafold.log") \
        if "run_dir" in df.attrs else {}
    total_seconds = total_min(df) * 60
    known_seconds = sum(phases.values())
    if known_seconds < total_seconds:
        phases["startup / finalization"] = total_seconds - known_seconds

    fig, axes = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")

    # Explicit phase timing avoids inferring CPU/GPU phases from sparse samples.
    ax = axes[0, 0]
    left = 0.0
    colors = {"data pipeline": "#4c78a8", "inference": "#f58518",
              "startup / finalization": "#9d9d9d"}
    for name, seconds in phases.items():
        ax.barh(["wall clock"], [seconds / 60], left=left / 60,
                color=colors.get(name, "#9d9d9d"), label=name)
        if seconds / 60 > 0.15:
            ax.text(left / 60 + seconds / 120, 0, f"{name}\n{seconds / 60:.1f} min",
                    ha="center", va="center", fontsize=9, color="white")
        left += seconds
    ax.set_xlim(0, total_seconds / 60)
    ax.set_xlabel("Elapsed wall time (min)")
    ax.set_title("AF3 phase timing")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    # Occupancy is more readable than a log-scaled histogram for this profile.
    ax = axes[0, 1]
    labels = ["idle\n<5%", "low\n5-20%", "moderate\n20-80%", "busy\n>=80%"]
    masks = [util < 5, (util >= 5) & (util < 20),
             (util >= 20) & (util < 80), util >= 80]
    values = [100 * mask.mean() for mask in masks]
    bars = ax.bar(labels, values, color=["#bdbdbd", "#72b7b2", "#54a24b", "#e45756"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.set_ylabel("Fraction of samples")
    ax.set_title("GPU utilization occupancy")
    ax.grid(axis="y", alpha=0.3)

    # Memory baseline and incremental inference allocation are the key memory result.
    ax = axes[1, 0]
    ax.barh(["GPU memory"], [baseline / GIB], color="#4c78a8", label="resident baseline")
    ax.barh(["GPU memory"], [extra / GIB], left=baseline / GIB,
            color="#f58518", label="peak extra")
    ax.axvline(peak / GIB, color="#333333", ls="--", lw=1)
    ax.text(peak / GIB, 0.38, f"peak {peak / GIB:.2f} GiB", ha="right", fontsize=9)
    ax.set_xlabel("GPU memory (GiB)")
    ax.set_title("GPU memory residency")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Only inference samples are useful for the operating envelope.
    ax = axes[1, 1]
    scatter = ax.scatter(inference["utilization_gpu_percent"], inference["power_draw_w"],
                         c=inference["elapsed_min"], s=8, alpha=0.65, cmap="viridis")
    ax.set_xlabel("GPU utilization (%)")
    ax.set_ylabel("Power draw (W)")
    ax.set_title("Inference operating envelope")
    ax.grid(alpha=0.3)
    if len(inference) > 1:
        fig.colorbar(scatter, ax=ax, label="Elapsed time (min)")

    # Show sensor ranges without putting unlike units on one axis.
    ax = axes[2, 0]
    ax.axis("off")
    rows = [
        ("Metric", "full mean", "inference mean", "inference peak"),
        ("memory (GiB)", f"{df['memory_used_mib'].mean() / GIB:.2f}",
         f"{inference['memory_used_mib'].mean() / GIB:.2f}", f"{inference['memory_used_mib'].max() / GIB:.2f}"),
        ("power (W)", f"{df['power_draw_w'].mean():.0f}",
         f"{inference['power_draw_w'].mean():.0f}", f"{inference['power_draw_w'].max():.0f}"),
        ("temperature (C)", f"{df['temperature_gpu_c'].mean():.1f}",
         f"{inference['temperature_gpu_c'].mean():.1f}", f"{inference['temperature_gpu_c'].max():.1f}"),
        ("SM clock (MHz)", f"{df['clocks_sm_mhz'].mean():.0f}",
         f"{inference['clocks_sm_mhz'].mean():.0f}", f"{inference['clocks_sm_mhz'].max():.0f}"),
        ("memory clock (MHz)", f"{df['clocks_memory_mhz'].mean():.0f}",
         f"{inference['clocks_memory_mhz'].mean():.0f}", f"{inference['clocks_memory_mhz'].max():.0f}"),
    ]
    ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="right")
    ax.set_title("Sensor summary", pad=12)

    ax = axes[2, 1]
    ax.axis("off")
    stats = [f"wall: {total_seconds / 60:.1f} min"]
    if phases:
        for name, seconds in phases.items():
            stats.append(f"{name}: {seconds / 60:.1f} min")
    load_at = model_load_at(df)
    stats += [
        f"model load: {load_at:.1f} min" if load_at is not None else "model load: n/a",
        f"memory baseline: {baseline / GIB:.2f} GiB",
        f"memory peak: {peak / GIB:.2f} GiB (+{extra / GIB:.2f})",
        f"GPU <5%: {100 * (util < 5).mean():.1f}% of samples",
        f"peak power: {df['power_draw_w'].max():.0f} W",
    ]
    if "run_dir" in df.attrs:
        time_stats = parse_time_txt(Path(df.attrs["run_dir"]) / "time.txt")
        if "Maximum resident set size (kbytes)" in time_stats:
            stats.append(
                f"host max RSS: {time_seconds(time_stats['Maximum resident set size (kbytes)']) / 1048576:.2f} GiB"
            )
        for key, label in (
            ("Percent of CPU this job got", "host CPU"),
            ("File system inputs", "FS input"),
            ("Major (requiring I/O) page faults", "major faults"),
        ):
            if key in time_stats:
                stats.append(f"{label}: {time_stats[key]}")
    ax.set_title("Diagnostic summary", pad=12)
    ax.text(0, 1, "\n".join(stats), va="top", ha="left", fontsize=9,
            family="monospace", transform=ax.transAxes)

    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def total_min(df):
    return float(df["elapsed_min"].iloc[-1])


def parse_time_txt(path):
    stats = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(.*?):\s*(.*)", line)
        if match:
            stats[match.group(1)] = match.group(2).strip()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    gpu_csv = args.run_dir / "gpu.csv"
    if not gpu_csv.exists():
        raise SystemExit(f"{gpu_csv} not found (data-pipeline runs record no GPU data)")
    df = load_gpu_csv(gpu_csv)
    if df.empty:
        raise SystemExit("gpu.csv contains no samples; nothing to plot")
    df.attrs["run_dir"] = str(args.run_dir)

    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    host = metadata.get("host", {})
    gpus = ", ".join(f"{g.get('name', '?')} ({g.get('memory_total_mib', '?')} MiB)"
                     for g in host.get("gpus", [])) or "no GPU"
    profiling = metadata.get("profiling", {})
    title = (
        f"{args.run_dir.name}  AF3 {metadata.get('alphafold', {}).get('version', '?')}  "
        f"{profiling.get('mode', '?')}/{profiling.get('stage', '?')}  "
        f"{gpus}  {host.get('hostname', '?')}"
    )

    plot_timeseries(df, args.run_dir / "gpu_timeseries.png", title)
    plot_diagnostics(df, args.run_dir / "gpu_diagnostics.png", title)

    stats = parse_time_txt(args.run_dir / "time.txt")
    win = active_window(df)
    total = total_min(df)
    print(f"Run: {args.run_dir.name}")
    print(f"Wall time: {total:.1f} min")
    if win is not None:
        print(f"Inference (GPU busy): {win[1] - win[0]:.2f} min "
              f"({100 * (win[1] - win[0]) / total:.1f}% of wall)")
    if model_load_at(df) is not None:
        print(f"Model load on GPU: {model_load_at(df):.1f} min")
    base, peak, extra = memory_summary(df)
    print(f"Peak GPU memory: {peak / GIB:.2f} GiB "
          f"(baseline idle {base / GIB:.2f} GiB, extra {extra / GIB:.2f} GiB)")
    print(f"GPU utilization: mean {df['utilization_gpu_percent'].mean():.1f}%, "
          f"{100 * (df['utilization_gpu_percent'] < 5).mean():.1f}% of time < 5%")
    print(f"Power: mean {df['power_draw_w'].mean():.0f} W, "
          f"peak {df['power_draw_w'].max():.0f} W")
    if "Maximum resident set size (kbytes)" in stats:
        print(f"Host max RSS: {int(stats['Maximum resident set size (kbytes)']) / 1048576:.2f} GiB")
    print(f"Plots: {args.run_dir / 'gpu_timeseries.png'}, "
          f"{args.run_dir / 'gpu_diagnostics.png'}")


if __name__ == "__main__":
    main()
