# 1D gradient profile: original intensity, Ix, and ||∇I|| along a row

import numpy as np
from matplotlib import pyplot as plt


def plot_gradient_1d_profile(profile_data, row, output_path):
    """3-panel 1D profile: original intensity, Ix, and ||∇I|| at a given row."""
    x = np.arange(len(profile_data["original"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # (a) Original intensity
    axes[0].plot(x, profile_data["original"], color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(f"(a) Original intensity profile I  —  row {row}",
                       fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # (b) Horizontal gradient Ix — shade positive / negative
    ix = profile_data["ix"]
    axes[1].plot(x, ix, color="tab:red", linewidth=1.2)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].fill_between(x, ix, 0, where=(ix > 0),
                          alpha=0.15, color="tab:red",
                          label="+Ix (brightness increasing →)")
    axes[1].fill_between(x, ix, 0, where=(ix < 0),
                          alpha=0.15, color="tab:blue",
                          label="−Ix (brightness decreasing →)")
    axes[1].set_ylabel("Gradient value")
    axes[1].set_title("(b) Horizontal gradient Ix = ∂I/∂x  (Sobel)",
                       fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # (c) Gradient norm
    norm = profile_data["norm"]
    axes[2].plot(x, norm, color="tab:green", linewidth=1.2)
    axes[2].fill_between(x, norm, alpha=0.15, color="tab:green")
    axes[2].set_ylabel("Edge strength")
    axes[2].set_xlabel("Pixel position (x)")
    axes[2].set_title(
        r"(c) Gradient norm $\|\nabla I\| = \sqrt{Ix^2 + Iy^2}$",
        fontsize=11,
    )
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "1D Gradient Analysis\n"
        r"Edges produce signed peaks in $I_x$ and positive peaks in "
        r"$\|\nabla I\|$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
