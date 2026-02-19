# 4-panel comparison: original, direct, filter2D, and difference map

import numpy as np
from matplotlib import pyplot as plt


def plot_visual_comparison(img_original, img_direct, img_filter2d, output_path):
    """4-panel comparison: original, direct, filter2D, and difference map."""
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
        "Visual Comparison of Convolution Methods",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")
