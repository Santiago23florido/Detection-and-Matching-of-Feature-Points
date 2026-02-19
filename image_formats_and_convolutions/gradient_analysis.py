# Q3 analysis: gradient components Ix, Iy and norm ||∇I||

from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "docs" / "rappport" / "imgs" / "convolutions"


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

def compute_gradient(img, kernel_x=None, kernel_y=None):
    # Compute Ix, Iy, ||∇I|| and direction (CV_64F preserves negatives)
    if kernel_x is None:
        kernel_x = SOBEL_X
    if kernel_y is None:
        kernel_y = SOBEL_Y

    # CV_64F preserves negative values — this is the central precaution
    ix = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    iy = cv2.filter2D(img, cv2.CV_64F, kernel_y)

    gradient_norm = np.sqrt(ix ** 2 + iy ** 2)
    gradient_direction = np.arctan2(iy, ix)

    return {
        "ix": ix,
        "iy": iy,
        "norm": gradient_norm,
        "direction": gradient_direction,
    }


def display_strategies(component):
    # Generate naive_clip, absolute, rescaled and raw views of a signed component
    naive_clip = np.clip(component, 0, 255)
    absolute = np.abs(component)

    vmin, vmax = component.min(), component.max()
    if vmax - vmin < 1e-6:
        rescaled = np.zeros_like(component)
    else:
        rescaled = (component - vmin) / (vmax - vmin) * 255.0

    return {
        "naive_clip": naive_clip,
        "absolute": absolute,
        "rescaled": rescaled,
        "raw": component,
    }


def gradient_statistics(gradient):
    # Return min / max / mean / std for ix, iy, norm
    stats = {}
    for name in ["ix", "iy", "norm"]:
        arr = gradient[name]
        stats[name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return stats


def generate_gradient_analysis(img_original):
    # Full Q3 pipeline: compute gradient, print stats, save plots
    from .visualizations import (
        plot_gradient_components,
        plot_display_precautions,
        plot_gradient_kernels,
        plot_gradient_1d_profile,
    )

    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Q3 — GRADIENT COMPUTATION AND DISPLAY — ANALYSIS")
    print("=" * 65)

    # 1. Compute gradient with Sobel kernels
    print("\n--- Computing Gradient (Sobel kernels, cv2.CV_64F) ---\n")
    gradient = compute_gradient(img_original)
    stats = gradient_statistics(gradient)

    print(f"  {'Component':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'─' * 55}")
    for name, s in stats.items():
        label = {"ix": "Ix (∂I/∂x)", "iy": "Iy (∂I/∂y)", "norm": "||∇I||"}[name]
        print(f"  {label:<12} {s['min']:>10.2f} {s['max']:>10.2f} "
              f"{s['mean']:>10.2f} {s['std']:>10.2f}")

    print(f"\n  Key observation: Ix and Iy contain negative values!")
    print(f"  Ix range: [{stats['ix']['min']:.0f}, {stats['ix']['max']:.0f}]")
    print(f"  Iy range: [{stats['iy']['min']:.0f}, {stats['iy']['max']:.0f}]")
    print(f"  ||∇I|| range: [{stats['norm']['min']:.0f}, "
          f"{stats['norm']['max']:.0f}]  (always ≥ 0)")

    # 2. Display precautions summary
    print("\n--- Display Precautions ---\n")
    print("  Gradient components Ix and Iy are signed (negative ↔ positive).")
    print("  Precautions for correct display:")
    print("  1. Use cv2.CV_64F as output depth in filter2D")
    print("     (preserves negative values; uint8 would clip them to 0)")
    print("  2. Choose an appropriate visualisation strategy:")
    print("     a) Divergent colormap (RdBu_r) centered at 0  ← recommended")
    print("     b) Shift + rescale [min,max] → [0,255]  (mid-gray = zero)")
    print("     c) Absolute value |Ix|  (shows strength, loses sign)")
    print("     d) Naive clip to [0,255]  ← WRONG, loses half the info")
    print("  3. The gradient norm ||∇I|| is always ≥ 0, so a standard")
    print("     grayscale display (normalised) works fine for it.")

    # 3. Generate all plots
    print("\n--- Generating Plots ---\n")

    plot_gradient_kernels(
        {"Sobel X  (hx)": SOBEL_X, "Sobel Y  (hy)": SOBEL_Y},
        output_dir / "gradient_kernels.png",
    )

    plot_gradient_components(
        img_original, gradient,
        output_dir / "gradient_components.png",
    )

    plot_display_precautions(
        gradient["ix"], gradient["iy"],
        output_dir / "gradient_display_precautions.png",
    )

    # 4. 1D profile
    print("\n--- 1D Gradient Profile ---\n")
    row = img_original.shape[0] // 2
    print(f"  Row selected: {row} (middle of the image)")

    profile_data = {
        "original": img_original[row, :].astype(np.float64),
        "ix": gradient["ix"][row, :],
        "norm": gradient["norm"][row, :],
    }

    plot_gradient_1d_profile(
        profile_data, row,
        output_dir / "gradient_1d_profile.png",
    )

    print(f"\n  All plots saved to: {output_dir}")
    print("=" * 65)
