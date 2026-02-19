# Line plot: convolution time vs image resolution

from matplotlib import pyplot as plt

from ._colors import METHOD_COLORS


def plot_time_vs_image_size(scaling_results, image_name, output_path):
    """Line plot: how time scales with image resolution."""
    pixel_counts = scaling_results["pixel_counts"]
    resolutions = scaling_results["resolutions"]
    results = scaling_results["results"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for method, medians in results.items():
        color = METHOD_COLORS.get(method, "tab:gray")
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
