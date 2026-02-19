# 1D profile analysis: original, Laplacian, and sharpened on a single row

import numpy as np
from matplotlib import pyplot as plt


def plot_1d_profile_analysis(profile_1d, row, output_path):
    """Show 1D profile: original, Laplacian, and sharpened on a single row."""
    x = np.arange(len(profile_1d["original"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # Original signal
    axes[0].plot(x, profile_1d["original"], color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(f"(a) Original profile f  —  row {row}", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Laplacian
    axes[1].plot(x, profile_1d["laplacian"], color="tab:red", linewidth=1.2)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("(b) 1D Laplacian  ∇²f  (kernel [1, −2, 1])", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Overlay: original vs sharpened
    axes[2].plot(x, profile_1d["original"], color="tab:blue",
                 linewidth=1.2, alpha=0.6, label="Original f")
    axes[2].plot(x, profile_1d["sharpened"], color="tab:orange",
                 linewidth=1.2, label="Sharpened f − ∇²f")
    axes[2].set_ylabel("Intensity")
    axes[2].set_xlabel("Pixel position (x)")
    axes[2].set_title("(c) Overlay: original vs sharpened", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "1D Unsharp Masking Analysis\n"
        "Sharpening amplifies intensity transitions at edges",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
