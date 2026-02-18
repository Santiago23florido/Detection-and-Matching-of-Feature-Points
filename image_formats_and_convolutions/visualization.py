# Q1 plot generation — all saved to docs/rappport/imgs/convolutions/

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

_METHOD_COLORS = {
    "Direct (Python loop)": "tab:blue",
    "filter2D (multi-thread)": "tab:orange",
    "filter2D (1-thread)": "tab:green",
}


def _time_label(t):
    # Format time in a readable way
    if t >= 1.0:
        return f"{t:.2f} s"
    if t >= 0.01:
        return f"{t:.4f} s"
    if t >= 0.001:
        return f"{t * 1000:.2f} ms"
    return f"{t * 1e6:.0f} µs"


def plot_timing_comparison(bench_results, image_name, output_path):
    # Bar chart: median time for each of the 3 methods
    methods = list(bench_results.keys())
    medians = [bench_results[m]["median"] for m in methods]
    colors = [_METHOD_COLORS.get(m, "tab:gray") for m in methods]

    t_direct = medians[0]
    t_fastest = min(medians[1:]) if len(medians) > 1 else medians[0]
    speedup = t_direct / t_fastest if t_fastest > 0 else 0

    short_labels = [m.replace("(Python loop)", "\n(Python loop)")
                      .replace("(multi-thread)", "\n(multi-thread)")
                      .replace("(1-thread)", "\n(1-thread)")
                    for m in methods]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(short_labels, medians, color=colors, width=0.5,
                  edgecolor="white")

    ax.set_yscale("log")
    ax.set_ylabel("Median time (seconds, log scale)")
    ax.grid(True, alpha=0.3, axis="y")

    # Need to draw first so we can compute positions in log scale
    fig.canvas.draw()
    y_lo, y_hi = ax.get_ylim()
    log_range = np.log10(y_hi) - np.log10(y_lo)

    # Put label inside if bar is tall enough, otherwise above it
    for bar, t, color in zip(bars, medians, colors):
        label = _time_label(t)
        bar_fraction = (np.log10(t) - np.log10(y_lo)) / log_range

        if bar_fraction > 0.25:
            ax.text(bar.get_x() + bar.get_width() / 2, t * 0.55, label,
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    color="white")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, t * 1.6, label,
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=color)

    run_counts = [len(bench_results[m].get("all_times", [])) for m in methods]
    if len(set(run_counts)) == 1:
        runs_str = f"{run_counts[0]} runs"
    else:
        runs_str = "/".join(str(r) for r in run_counts) + " runs"
    ax.set_title(
        f"Convolution Performance — {image_name}\n"
        f"Median of {runs_str} · Speedup (best OpenCV): {speedup:.0f}×",
        fontsize=12,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_timing_boxplot(bench_results, image_name, output_path):
    # Box plot showing distribution of all runs per method
    methods = list(bench_results.keys())
    data = [bench_results[m]["all_times"] for m in methods]
    colors = [_METHOD_COLORS.get(m, "tab:gray") for m in methods]

    short_labels = [m.replace("(Python loop)", "\n(Python loop)")
                      .replace("(multi-thread)", "\n(multi-thread)")
                      .replace("(1-thread)", "\n(1-thread)")
                    for m in methods]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True,
                    widths=0.45, showfliers=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    run_counts = [len(d) for d in data]
    if len(set(run_counts)) == 1:
        runs_str = f"{run_counts[0]} runs per method"
    else:
        runs_str = " / ".join(
            f"{m.split('(')[0].strip()} {n}"
            for m, n in zip(methods, run_counts)
        ) + " runs"
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title(
        f"Execution Time Distribution — {image_name}\n"
        f"{runs_str}",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_time_vs_image_size(scaling_results, image_name, output_path):
    # Line plot: how time scales with image resolution
    pixel_counts = scaling_results["pixel_counts"]
    resolutions = scaling_results["resolutions"]
    results = scaling_results["results"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for method, medians in results.items():
        color = _METHOD_COLORS.get(method, "tab:gray")
        ax.plot(pixel_counts, medians, "o-", color=color, label=method,
                markersize=6, linewidth=2)

    labels = [f"{h}×{w}" for h, w in resolutions]
    ax.set_xticks(pixel_counts)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_xlabel("Image size (pixels)")
    ax.set_ylabel("Median time (seconds, log scale)")
    ax.set_title(
        f"Execution Time vs Image Size — {image_name}",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_visual_comparison(img_original, img_direct, img_filter2d, output_path):
    # 4-panel comparison: original, direct, filter2D, and difference map
    diff = np.abs(
        np.clip(img_direct, 0, 255).astype(np.float64)
        - np.clip(img_filter2d, 0, 255).astype(np.float64)
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img_original, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_direct, cmap="gray", vmin=0.0, vmax=255.0)
    axes[0, 1].set_title("Direct Convolution (Python loop)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(img_filter2d, cmap="gray", vmin=0.0, vmax=255.0)
    axes[1, 0].set_title("filter2D Convolution (OpenCV)")
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(diff, cmap="hot")
    axes[1, 1].set_title("Absolute Difference |direct − filter2D|")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Q1 — Visual Comparison of Convolution Methods",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")
