# Box plot: execution time distribution per convolution method

from matplotlib import pyplot as plt

from ._colors import METHOD_COLORS


def plot_timing_boxplot(bench_results, image_name, output_path):
    """Box plot showing distribution of all runs per method."""
    methods = list(bench_results.keys())
    data = [bench_results[m]["all_times"] for m in methods]
    colors = [METHOD_COLORS.get(m, "tab:gray") for m in methods]

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
