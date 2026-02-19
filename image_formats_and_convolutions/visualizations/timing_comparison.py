# Bar chart: median convolution time for each method

import numpy as np
from matplotlib import pyplot as plt

from ._colors import METHOD_COLORS


def _time_label(t):
    """Format time in a readable way."""
    if t >= 1.0:
        return f"{t:.2f} s"
    if t >= 0.01:
        return f"{t:.4f} s"
    if t >= 0.001:
        return f"{t * 1000:.2f} ms"
    return f"{t * 1e6:.0f} µs"


def plot_timing_comparison(bench_results, image_name, output_path):
    """Bar chart: median time for each of the 3 methods."""
    methods = list(bench_results.keys())
    medians = [bench_results[m]["median"] for m in methods]
    colors = [METHOD_COLORS.get(m, "tab:gray") for m in methods]

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
