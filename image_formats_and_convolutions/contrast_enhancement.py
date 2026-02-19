# Q2 analysis: contrast enhancement via unsharp masking (K = δ − ∇²)

import cv2
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "docs" / "rappport" / "imgs" / "convolutions"


KERNEL_SHARPENING = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float64)

KERNEL_IDENTITY = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float64)

KERNEL_LAPLACIAN = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float64)

def decompose_kernel(kernel):
    identity = KERNEL_IDENTITY.copy()
    laplacian = KERNEL_LAPLACIAN.copy()
    reconstructed = identity - laplacian

    is_equal = np.allclose(kernel, reconstructed)
    return {
        "kernel": kernel,
        "identity": identity,
        "laplacian": laplacian,
        "reconstructed": reconstructed,
        "decomposition_valid": is_equal,
    }


def compute_components(img):
    identity_result = cv2.filter2D(img, cv2.CV_64F, KERNEL_IDENTITY)
    laplacian_result = cv2.filter2D(img, cv2.CV_64F, KERNEL_LAPLACIAN)
    sharpened_result = cv2.filter2D(img, cv2.CV_64F, KERNEL_SHARPENING)

    return {
        "identity": identity_result,
        "laplacian": laplacian_result,
        "sharpened": sharpened_result,
    }


def extract_1d_profile(img, row=None):
    if row is None:
        row = img.shape[0] // 2
    return img[row, :].astype(np.float64)


def compute_1d_unsharp(profile):
    laplacian_1d = np.array([1, -2, 1], dtype=np.float64)
    laplacian = np.convolve(profile, laplacian_1d, mode="same")
    sharpened = profile - laplacian
    return {
        "original": profile,
        "laplacian": laplacian,
        "sharpened": sharpened,
    }


def histogram_comparison(img_original, img_sharpened):
    orig_clip = np.clip(img_original, 0, 255).astype(np.uint8)
    sharp_clip = np.clip(img_sharpened, 0, 255).astype(np.uint8)

    hist_orig = cv2.calcHist([orig_clip], [0], None, [256], [0, 256]).ravel()
    hist_sharp = cv2.calcHist([sharp_clip], [0], None, [256], [0, 256]).ravel()

    return {
        "original": hist_orig,
        "sharpened": hist_sharp,
        "bins": np.arange(256),
    }


def generate_contrast_enhancement_analysis(img_original, kernel):
    from .visualizations import (
        plot_kernel_decomposition,
        plot_component_images,
        plot_1d_profile_analysis,
        plot_histogram_comparison,
    )

    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  CONTRAST ENHANCEMENT BY UNSHARP MASKING — ANALYSIS")
    print("=" * 65)

    # 1. Kernel decomposition
    print("\n--- Kernel Decomposition ---\n")
    decomp = decompose_kernel(kernel)
    _print_kernel("K (sharpening)", decomp["kernel"])
    _print_kernel("δ (identity)", decomp["identity"])
    _print_kernel("∇² (Laplacian)", decomp["laplacian"])
    _print_kernel("δ - ∇² (reconstructed)", decomp["reconstructed"])

    if decomp["decomposition_valid"]:
        print("  ✓ Verified: K = δ - ∇²  (Identity − Laplacian)")
    else:
        print("  ✗ Decomposition mismatch!")

    print("\n  Interpretation:")
    print("  K * f = (δ - ∇²) * f = f - ∇²f")
    print("  This is unsharp masking with gain α = 1.")
    print("  → Edges are amplified, flat regions are preserved.\n")

    # 2. Compute image components
    print("--- Computing Image Components ---\n")
    components = compute_components(img_original)

    plot_component_images(
        img_original, components,
        output_dir / "unsharp_components.png",
    )

    plot_kernel_decomposition(
        decomp,
        output_dir / "kernel_decomposition.png",
    )

    # 3. 1D profile analysis
    print("--- 1D Profile Analysis ---\n")
    row = img_original.shape[0] // 2
    profile = extract_1d_profile(img_original, row)
    profile_1d = compute_1d_unsharp(profile)

    print(f"  Row selected: {row} (middle of the image)")
    print(f"  Profile length: {len(profile)} pixels")

    plot_1d_profile_analysis(
        profile_1d, row,
        output_dir / "unsharp_1d_profile.png",
    )

    # 4. Histogram comparison
    print("\n--- Histogram Comparison ---\n")
    hist_data = histogram_comparison(img_original, components["sharpened"])

    orig_std = float(np.std(img_original))
    sharp_std = float(np.std(np.clip(components["sharpened"], 0, 255)))
    print(f"  Std dev (original):  {orig_std:.2f}")
    print(f"  Std dev (sharpened): {sharp_std:.2f}")
    print(f"  Spread increase:     {((sharp_std / orig_std) - 1) * 100:.1f}%")

    plot_histogram_comparison(
        hist_data,
        output_dir / "histogram_comparison.png",
    )

    print(f"\n  All plots saved to: {output_dir}")
    print("=" * 65)


def _print_kernel(name, k):
    print(f"  {name}:")
    for row in k:
        print(f"    [{' '.join(f'{v:5.0f}' for v in row)}]")
    print()
