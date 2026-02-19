# Unsharp masking components: original, Laplacian, and sharpened

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_component_images(img_original, components, output_path):
    """Show original, Laplacian, and sharpened images side by side."""
    laplacian = components["laplacian"]
    sharpened = components["sharpened"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Original
    axes[0, 0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) Original image f", fontsize=11)
    axes[0, 0].axis("off")

    # Laplacian (centered at 0 — use diverging colormap)
    abs_max = max(abs(laplacian.min()), abs(laplacian.max()))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im_lap = axes[0, 1].imshow(laplacian, cmap="RdBu_r", norm=norm)
    axes[0, 1].set_title("(b) Laplacian ∇²f", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im_lap, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Laplacian (grayscale, rescaled for visibility)
    lap_rescaled = _rescale_for_display(laplacian)
    axes[1, 0].imshow(lap_rescaled, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(c) |∇²f| rescaled to [0, 255]", fontsize=11)
    axes[1, 0].axis("off")

    # Sharpened
    sharp_clip = np.clip(sharpened, 0, 255)
    axes[1, 1].imshow(sharp_clip, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("(d) Sharpened: f − ∇²f", fontsize=11)
    axes[1, 1].axis("off")

    fig.suptitle(
        "Unsharp Masking Components\n"
        "R[x,y] = f[x,y] − ∇²f[x,y]",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _rescale_for_display(img):
    """Rescale a float image to [0, 255] for display."""
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin) * 255.0
