# Gradient kernel visualisation: Sobel X and Sobel Y

from matplotlib import pyplot as plt


def plot_gradient_kernels(kernels, output_path):
    """Show gradient kernels with annotated coefficient values."""
    n = len(kernels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, kern) in zip(axes, kernels.items()):
        im = ax.imshow(kern, cmap="RdBu_r", vmin=-3, vmax=3)
        for i in range(kern.shape[0]):
            for j in range(kern.shape[1]):
                val = kern[i, j]
                color = "white" if abs(val) >= 2 else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Gradient Approximation Kernels (Sobel Operator)\n"
        r"$I_x \approx I * h_x$,   $I_y \approx I * h_y$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
