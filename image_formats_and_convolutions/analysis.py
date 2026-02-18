# Q1 analysis: convolution comparison (direct loop vs filter2D)

from pathlib import Path

import cv2
import numpy as np

from .benchmark import benchmark_all_methods, benchmark_vs_image_size
from .visualization import (
    plot_timing_comparison,
    plot_timing_boxplot,
    plot_time_vs_image_size,
    plot_visual_comparison,
)

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "docs" / "rappport" / "imgs" / "convolutions"


def find_image_path(filename):
    # Handles both flat and nested Image_Pairs/ layouts
    candidates = [
        _ROOT / "Image_Pairs" / filename,
        _ROOT / "Image_Pairs" / "Image_Pairs" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found in Image_Pairs/")


def direct_convolution(img, kernel):
    # Convolution with Python loops — slow but straightforward
    (h, w) = img.shape
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    result = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

    for y in range(pad_y, h - pad_y):
        for x in range(pad_x, w - pad_x):
            neighbourhood = img[y - pad_y:y + pad_y + 1,
                                x - pad_x:x + pad_x + 1]
            val = np.sum(neighbourhood * kernel)
            result[y, x] = min(max(val, 0), 255)

    return result


def filter2d_convolution(img, kernel):
    # Convolution with OpenCV's optimised filter2D (C++ / SIMD)
    return cv2.filter2D(img, -1, kernel)


def pixel_comparison(result_a, result_b):
    # Stats about pixel-level differences between two results
    a = np.clip(result_a, 0, 255).astype(np.float64)
    b = np.clip(result_b, 0, 255).astype(np.float64)
    diff = np.abs(a - b)
    return {
        "max_diff": float(np.max(diff)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "identical_ratio": float(np.sum(diff < 1e-6) / diff.size),
    }

def generate_q1_analysis(
    img_original,
    kernel,
    result_direct,
    result_filter2d,
    image_name="FlowerGarden2",
    n_runs=20,
    n_runs_direct=5,
    run_scaling=True,
):
    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    (h, w) = img_original.shape
    comp = pixel_comparison(result_direct, result_filter2d)

    print("\n" + "=" * 65)
    print("  Q1 — FORMAT D'IMAGES ET CONVOLUTIONS — ANALYSIS")
    print("=" * 65)

    # Benchmark the 3 methods
    print(f"\n--- Benchmarking ({image_name}, {h}×{w}) ---\n")
    bench = benchmark_all_methods(
        img_original, kernel,
        n_runs=n_runs,
        n_runs_direct=n_runs_direct,
    )

    print(f"\n--- Performance Comparison (median of N runs) ---\n")
    print(f"  {'Method':<28} {'Median':>12}  {'Runs':>5}")
    print(f"  {'─' * 50}")
    for method, stats in bench.items():
        n = len(stats["all_times"])
        t = stats["median"]
        label = f"{t:.4f} s" if t >= 0.01 else f"{t * 1000:.3f} ms"
        print(f"  {method:<28} {label:>12}  {n:>5}")
    print(f"  {'─' * 50}")

    medians = [s["median"] for s in bench.values()]
    speedup = medians[0] / min(medians[1:]) if len(medians) > 1 else 0
    print(f"  Speedup (Direct / best OpenCV): ~{speedup:.0f}×\n")

    # Pixel-level comparison
    print(f"--- Pixel-Level Comparison ---\n")
    print(f"  Max |diff|:        {comp['max_diff']:.2f}")
    print(f"  Mean |diff|:       {comp['mean_diff']:.4f}")
    print(f"  Identical pixels:  {comp['identical_ratio'] * 100:.1f}%\n")

    if comp["max_diff"] < 1.0:
        print("  Both methods produce numerically identical results.")
    else:
        print("  Minor differences at borders due to padding strategy.")

    # Generate all plots
    print(f"\n--- Generating Plots ---\n")

    plot_timing_comparison(
        bench, image_name,
        output_dir / "timing_comparison.png",
    )

    plot_timing_boxplot(
        bench, image_name,
        output_dir / "timing_boxplot.png",
    )

    plot_visual_comparison(
        img_original, result_direct, result_filter2d,
        output_dir / "visual_comparison.png",
    )

    # Scaling benchmark (optional, slow)
    if run_scaling:
        print(f"\n--- Benchmarking vs Image Size ---\n")
        scaling = benchmark_vs_image_size(
            img_original, kernel,
            n_runs=n_runs,
            n_runs_direct=n_runs_direct,
        )
        plot_time_vs_image_size(
            scaling, image_name,
            output_dir / "time_vs_image_size.png",
        )

    print(f"\n  All plots saved to: {output_dir}")
    print("=" * 65)
