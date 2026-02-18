# Benchmarking utilities for convolution methods

from typing import Callable

import cv2
import numpy as np


def _time_once(fn):
    # Time a single execution in seconds
    t1 = cv2.getTickCount()
    fn()
    t2 = cv2.getTickCount()
    return (t2 - t1) / cv2.getTickFrequency()


def benchmark(fn, n_runs=20):
    # Run fn n_runs times and return stats
    times = [_time_once(fn) for _ in range(n_runs)]
    arr = np.array(times)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "all_times": times,
    }


def benchmark_all_methods(img, kernel, n_runs=20, n_runs_direct=None):
    # Benchmark: direct loop, filter2D multi-thread, filter2D 1-thread
    if n_runs_direct is None:
        n_runs_direct = n_runs

    # Direct Python loop
    print(f"  Benchmarking Direct (Python loop) × {n_runs_direct} …", flush=True)
    direct_stats = benchmark(
        lambda: _direct_convolution_inplace(img, kernel),
        n_runs=n_runs_direct,
    )

    # filter2D multi-threaded
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(0)  # 0 = all available threads
    print(f"  Benchmarking filter2D (multi-thread) × {n_runs} …", flush=True)
    filter2d_mt_stats = benchmark(
        lambda: cv2.filter2D(img, -1, kernel),
        n_runs=n_runs,
    )

    # filter2D single-threaded
    cv2.setNumThreads(1)
    print(f"  Benchmarking filter2D (1-thread) × {n_runs} …", flush=True)
    filter2d_st_stats = benchmark(
        lambda: cv2.filter2D(img, -1, kernel),
        n_runs=n_runs,
    )

    cv2.setNumThreads(original_threads)

    return {
        "Direct (Python loop)": direct_stats,
        "filter2D (multi-thread)": filter2d_mt_stats,
        "filter2D (1-thread)": filter2d_st_stats,
    }


def benchmark_vs_image_size(img, kernel, scales=None, n_runs=20, n_runs_direct=None):
    # Benchmark all 3 methods at different image resolutions
    if scales is None:
        scales = [0.25, 0.5, 0.75, 1.0]

    if n_runs_direct is None:
        n_runs_direct = n_runs

    resolutions = []
    pixel_counts = []
    results = {
        "Direct (Python loop)": [],
        "filter2D (multi-thread)": [],
        "filter2D (1-thread)": [],
    }

    original_threads = cv2.getNumThreads()

    for scale in scales:
        h_new = max(1, int(img.shape[0] * scale))
        w_new = max(1, int(img.shape[1] * scale))
        resolutions.append((h_new, w_new))
        pixel_counts.append(h_new * w_new)
        scaled = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)

        print(f"\n  Scale {scale:.0%} ({h_new}×{w_new}):")

        print(f"    Direct × {n_runs_direct} …", flush=True)
        d = benchmark(
            lambda s=scaled: _direct_convolution_inplace(s, kernel),
            n_runs=n_runs_direct,
        )
        results["Direct (Python loop)"].append(d["median"])

        cv2.setNumThreads(0)
        print(f"    filter2D (multi-thread) × {n_runs} …", flush=True)
        mt = benchmark(lambda s=scaled: cv2.filter2D(s, -1, kernel), n_runs=n_runs)
        results["filter2D (multi-thread)"].append(mt["median"])

        cv2.setNumThreads(1)
        print(f"    filter2D (1-thread) × {n_runs} …", flush=True)
        st = benchmark(lambda s=scaled: cv2.filter2D(s, -1, kernel), n_runs=n_runs)
        results["filter2D (1-thread)"].append(st["median"])

    cv2.setNumThreads(original_threads)

    return {
        "scales": scales,
        "resolutions": resolutions,
        "pixel_counts": pixel_counts,
        "results": results,
    }


def _direct_convolution_inplace(img, kernel):
    # Same loop as Convolutions.py (result discarded, just for timing)
    (h, w) = img.shape
    result = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            val = (
                kernel[1, 1] * img[y, x]
                + kernel[0, 1] * img[y - 1, x]
                + kernel[1, 0] * img[y, x - 1]
                + kernel[2, 1] * img[y + 1, x]
                + kernel[1, 2] * img[y, x + 1]
                + kernel[0, 0] * img[y - 1, x - 1]
                + kernel[0, 2] * img[y - 1, x + 1]
                + kernel[2, 0] * img[y + 1, x - 1]
                + kernel[2, 2] * img[y + 1, x + 1]
            )
            result[y, x] = min(max(val, 0), 255)
