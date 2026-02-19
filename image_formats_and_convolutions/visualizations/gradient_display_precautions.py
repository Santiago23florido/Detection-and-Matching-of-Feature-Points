# Display precautions: correct vs incorrect gradient visualisation

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_display_precautions(ix, iy, output_path):
    """2×4 grid comparing four display strategies for Ix and Iy.

    Columns:
      (a) Naive clip [0, 255]  — INCORRECT: negative values lost
      (b) Absolute value |·|   — edge strength only, sign lost
      (c) Shift + rescale      — full range, mid-gray = zero
      (d) Divergent colormap   — RECOMMENDED: sign preserved visually
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row_idx, (component, name) in enumerate([(ix, "Ix"), (iy, "Iy")]):
        vmin, vmax = float(component.min()), float(component.max())

        # (a) Naive clip — WRONG
        naive = np.clip(component, 0, 255)
        axes[row_idx, 0].imshow(naive, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(
            f"(a) {name}: clip [0, 255]\nINCORRECT",
            fontsize=10, color="red",
        )
        axes[row_idx, 0].axis("off")

        # (b) Absolute value
        absolute = np.abs(component)
        axes[row_idx, 1].imshow(absolute, cmap="gray")
        axes[row_idx, 1].set_title(
            f"(b) {name}: |{name}|  (abs value)", fontsize=10,
        )
        axes[row_idx, 1].axis("off")

        # (c) Shift + rescale
        rescaled = (component - vmin) / (vmax - vmin) * 255.0
        axes[row_idx, 2].imshow(rescaled, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 2].set_title(
            f"(c) {name}: rescale to [0, 255]\n(mid-gray = 0)", fontsize=10,
        )
        axes[row_idx, 2].axis("off")

        # (d) Divergent colormap — RECOMMENDED
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = axes[row_idx, 3].imshow(component, cmap="RdBu_r", norm=norm)
        axes[row_idx, 3].set_title(
            f"(d) {name}: divergent cmap\nRECOMMENDED",
            fontsize=10, color="green",
        )
        axes[row_idx, 3].axis("off")
        plt.colorbar(im, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

    # Row labels
    for row_idx, label in enumerate(["Ix = ∂I/∂x", "Iy = ∂I/∂y"]):
        axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight="bold")

    fig.suptitle(
        "Display Precautions for Gradient Components\n"
        "Gradient values are signed — naive clipping loses negative information",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
