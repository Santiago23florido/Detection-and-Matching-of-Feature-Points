# Histogram comparison: original vs sharpened (clipped)

import numpy as np
from matplotlib import pyplot as plt


def plot_histogram_comparison(hist_data, output_path):
    """Compare histograms of original and sharpened images."""
    bins = hist_data["bins"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Shared Y-axis limit: use the original histogram's max so the
    # sharpened saturation peaks at 0/255 don't crush the rest.
    y_max = float(np.max(hist_data["original"])) * 1.1

    axes[0].bar(bins, hist_data["original"], width=1.0,
                color="tab:blue", alpha=0.7, label="Original")
    axes[0].set_title("(a) Histogram — Original", fontsize=11)
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 255)
    axes[0].set_ylim(0, y_max)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(bins, hist_data["sharpened"], width=1.0,
                color="tab:orange", alpha=0.7, label="Sharpened")
    axes[1].set_title("(b) Histogram — Sharpened (clipped to [0, 255])",
                      fontsize=11)
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 255)
    axes[1].set_ylim(0, y_max)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Histogram Comparison: Sharpening Spreads the Distribution",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
