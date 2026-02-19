# Kernel decomposition visualization: K = Identity - Laplacian

from matplotlib import pyplot as plt


def plot_kernel_decomposition(decomp, output_path):
    """Visualize the kernel decomposition: K = Identity - Laplacian."""
    kernels = [
        ("K\n(sharpening)", decomp["kernel"]),
        ("=", None),
        ("δ\n(identity)", decomp["identity"]),
        ("−", None),
        ("∇²\n(Laplacian)", decomp["laplacian"]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(12, 3),
                             gridspec_kw={"width_ratios": [3, 1, 3, 1, 3]})

    for ax, (label, kern) in zip(axes, kernels):
        if kern is not None:
            ax.imshow(kern, cmap="RdBu_r", vmin=-5, vmax=5)
            for i in range(kern.shape[0]):
                for j in range(kern.shape[1]):
                    val = kern[i, j]
                    color = "white" if abs(val) > 2 else "black"
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            fontsize=16, fontweight="bold", color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=24, fontweight="bold", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle(
        "Kernel Decomposition: K = δ − ∇²",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
