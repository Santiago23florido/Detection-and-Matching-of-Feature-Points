# Gradient components: Ix, Iy (divergent), ||∇I||, direction, edges

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_gradient_components(img_original, gradient, output_path):
    """6-panel view: original, Ix, Iy, ||∇I||, direction θ, thresholded edges."""
    ix = gradient["ix"]
    iy = gradient["iy"]
    norm = gradient["norm"]
    direction = gradient["direction"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # (a) Original
    axes[0, 0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) Original image I", fontsize=11)
    axes[0, 0].axis("off")

    # (b) Ix — divergent colormap centered at 0
    abs_max_x = max(abs(ix.min()), abs(ix.max()))
    norm_x = TwoSlopeNorm(vmin=-abs_max_x, vcenter=0, vmax=abs_max_x)
    im_ix = axes[0, 1].imshow(ix, cmap="RdBu_r", norm=norm_x)
    axes[0, 1].set_title("(b) Ix = ∂I/∂x  (Sobel)", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im_ix, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (c) Iy — divergent colormap centered at 0
    abs_max_y = max(abs(iy.min()), abs(iy.max()))
    norm_y = TwoSlopeNorm(vmin=-abs_max_y, vcenter=0, vmax=abs_max_y)
    im_iy = axes[0, 2].imshow(iy, cmap="RdBu_r", norm=norm_y)
    axes[0, 2].set_title("(c) Iy = ∂I/∂y  (Sobel)", fontsize=11)
    axes[0, 2].axis("off")
    plt.colorbar(im_iy, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (d) ||∇I|| — grayscale
    im_norm = axes[1, 0].imshow(norm, cmap="gray")
    axes[1, 0].set_title(r"(d) $\|\nabla I\| = \sqrt{Ix^2 + Iy^2}$", fontsize=11)
    axes[1, 0].axis("off")
    plt.colorbar(im_norm, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (e) Gradient direction — circular colormap
    im_dir = axes[1, 1].imshow(direction, cmap="hsv",
                                vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(r"(e) Direction $\theta$ = atan2(Iy, Ix)", fontsize=11)
    axes[1, 1].axis("off")
    plt.colorbar(im_dir, ax=axes[1, 1], fraction=0.046, pad=0.04,
                 label="radians")

    # (f) Thresholded edges: ||∇I|| > μ + σ
    threshold = float(np.mean(norm) + np.std(norm))
    edges = (norm > threshold).astype(np.float64) * 255
    axes[1, 2].imshow(edges, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(
        rf"(f) Edges: $\|\nabla I\|$ > $\mu+\sigma$ (thr={threshold:.0f})",
        fontsize=11,
    )
    axes[1, 2].axis("off")

    fig.suptitle(
        "Gradient Components (Sobel Operator)\n"
        r"$I_x = \partial I/\partial x$,  "
        r"$I_y = \partial I/\partial y$,  "
        r"$\|\nabla I\| = \sqrt{I_x^2 + I_y^2}$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
