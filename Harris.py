import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np

# Avoid Qt/xcb crashes and headless GUI errors.
# Use a stable non-interactive backend unless user explicitly sets MPLBACKEND.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

from matplotlib import pyplot as plt

"""
- Loads a gray image
- Harris calculation
- Local max's extraction with morphological dilation
- Sample original image, response map and interest points

python Harris.py
python Harris.py -stats 50
python Harris.py -stats 50 -plots
"""

# Baseline params
SUM_WINDOW_SIZE = 5
HARRIS_K = 0.04
MAXLOC_NEIGHBORHOOD_SIZE = 3
RELATIVE_THRESHOLD = 0.01


@dataclass(frozen=True)
class HarrisParams:
    sum_window_size: int = SUM_WINDOW_SIZE
    harris_k: float = HARRIS_K
    maxloc_neighborhood_size: int = MAXLOC_NEIGHBORHOOD_SIZE
    relative_threshold: float = RELATIVE_THRESHOLD


DEFAULT_PARAMS = HarrisParams()


def validate_harris_params(params: HarrisParams):
    if params.sum_window_size <= 0 or params.sum_window_size % 2 == 0:
        raise ValueError("sum_window_size doit être un entier impair strictement positif.")
    if params.maxloc_neighborhood_size <= 0 or params.maxloc_neighborhood_size % 2 == 0:
        raise ValueError("maxloc_neighborhood_size doit être un entier impair strictement positif.")
    if params.harris_k <= 0.0:
        raise ValueError("harris_k doit être strictement positif.")
    if params.relative_threshold < 0.0:
        raise ValueError("relative_threshold doit être positif ou nul.")


def resolve_image_path(script_dir: Path) -> Path:
    image_candidates = [
        script_dir / "Image_Pairs" / "Graffiti0.png",
        script_dir / "Image_Pairs" / "Image_Pairs" / "Graffiti0.png",
    ]
    image_path = next((p for p in image_candidates if p.is_file()), None)
    if image_path is None:
        raise FileNotFoundError(
            "Graffiti0.png introuvable. Checked paths:\n- "
            + "\n- ".join(str(p) for p in image_candidates)
        )
    return image_path


def load_gray_image(image_path: Path) -> np.ndarray:
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Impossible de lire l'image en niveaux de gris: {image_path}")
    return np.float64(img_gray)


def compute_harris(img: np.ndarray, params: HarrisParams):
    validate_harris_params(params)
    (h, w) = img.shape

    # Début du calcul
    t1 = cv2.getTickCount()

    # Copy image and add borders as an extended frame
    Theta = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

    # Calcul des gradients par différences finies
    sigma = 1.0
    img_lisse = cv2.GaussianBlur(
        img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    )

    dx = np.array([[-0.5, 0.0, 0.5]], dtype=np.float64)
    dy = dx.T
    Ix = cv2.filter2D(img_lisse, cv2.CV_64F, dx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(img_lisse, cv2.CV_64F, dy, borderType=cv2.BORDER_REPLICATE)

    # Produits des gradients
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Somme dans une fenêtre (filtre moyenneur)
    w_moy = np.ones((params.sum_window_size, params.sum_window_size), dtype=np.float64) / (
        params.sum_window_size * params.sum_window_size
    )

    Sxx = cv2.filter2D(Ix2, cv2.CV_64F, w_moy, borderType=cv2.BORDER_REPLICATE)
    Syy = cv2.filter2D(Iy2, cv2.CV_64F, w_moy, borderType=cv2.BORDER_REPLICATE)
    Sxy = cv2.filter2D(Ixy, cv2.CV_64F, w_moy, borderType=cv2.BORDER_REPLICATE)

    # Calcul de la fonction de Harris, theta
    detM = Sxx * Syy - (Sxy * Sxy)
    traceM = Sxx + Syy
    Theta = detM - params.harris_k * (traceM * traceM)

    # Calcul des maxima locaux et seuillage
    Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    se = np.ones(
        (params.maxloc_neighborhood_size, params.maxloc_neighborhood_size), np.uint8
    )

    Theta_dil = cv2.dilate(Theta, se)
    Theta_maxloc[Theta < Theta_dil] = 0.0
    Theta_maxloc[Theta < params.relative_threshold * Theta.max()] = 0.0

    t2 = cv2.getTickCount()
    elapsed_time = (t2 - t1) / cv2.getTickFrequency()
    cycles_per_pixel = (t2 - t1) / (h * w)

    return Theta, Theta_maxloc, elapsed_time, cycles_per_pixel


def show_results(
    img: np.ndarray,
    Theta: np.ndarray,
    Theta_maxloc: np.ndarray,
    image_path: Path,
    script_dir: Path,
):
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.title("Image originale")

    plt.subplot(132)
    plt.imshow(Theta, cmap="gray")
    plt.title("Fonction de Harris")

    # Kernel 5x5 for dilation, to get a better visualization of the points
    se_croix = np.uint8(
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]
    )
    Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)

    # Reload image for color display
    img_pts = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_pts is None:
        raise FileNotFoundError(f"Impossible de lire l'image couleur: {image_path}")
    (h, w, c) = img_pts.shape
    print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")
    print("Type de l'image :", img_pts.dtype)

    # Display points (crosses) in red
    img_pts[Theta_ml_dil > 0] = [255, 0, 0]
    plt.subplot(133)
    plt.imshow(img_pts)
    plt.title("Points de Harris")

    plt.tight_layout()
    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        output_dir = script_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_figure = output_dir / "harris_points.png"
        plt.savefig(output_figure, dpi=200, bbox_inches="tight")
        print(
            f"Backend non interactif ({plt.get_backend()}). "
            f"Figure sauvegardée : {output_figure}"
        )
    else:
        plt.show()
    plt.close()


def benchmark_harris(img: np.ndarray, n_runs: int, params: HarrisParams):
    if n_runs <= 0:
        raise ValueError("Le nombre d'exécutions doit être strictement positif.")

    times = np.zeros(n_runs, dtype=np.float64)
    cycles = np.zeros(n_runs, dtype=np.float64)

    for i in range(n_runs):
        _, _, elapsed_time, cycles_per_pixel = compute_harris(img, params)
        times[i] = elapsed_time
        cycles[i] = cycles_per_pixel

    time_mean = np.mean(times)
    time_var = np.var(times)
    time_std = np.std(times)
    cycles_mean = np.mean(cycles)
    cycles_var = np.var(cycles)
    cycles_std = np.std(cycles)

    return time_mean, time_var, time_std, cycles_mean, cycles_var, cycles_std


def run_single(image_path: Path, script_dir: Path, params: HarrisParams):
    img = load_gray_image(image_path)
    (h, w) = img.shape
    print("Dimension de l'image :", h, "lignes x", w, "colonnes")
    print("Type de l'image :", img.dtype)

    Theta, Theta_maxloc, elapsed_time, cycles_per_pixel = compute_harris(img, params)
    print("Mon calcul des points de Harris :", elapsed_time, "s")
    print("Nombre de cycles par pixel :", cycles_per_pixel, "cpp")

    show_results(img, Theta, Theta_maxloc, image_path, script_dir)


def run_stats(image_path: Path, n_runs: int, params: HarrisParams):
    img = load_gray_image(image_path)
    (h, w) = img.shape
    print("Dimension de l'image :", h, "lignes x", w, "colonnes")
    print("Type de l'image :", img.dtype)
    print(f"Mode statistiques : {n_runs} exécutions")

    time_mean, time_var, time_std, cycles_mean, cycles_var, cycles_std = benchmark_harris(
        img, n_runs, params
    )

    print(f"Calcul des points de Harris (temps) - moyenne : {time_mean:.9f} s")
    print(f"Calcul des points de Harris (temps) - variance : {time_var:.12f} s^2")
    print(f"Calcul des points de Harris (temps) - écart-type : {time_std:.9f} s")
    print(f"Nombre de cycles par pixel - moyenne : {cycles_mean:.6f} cpp")
    print(f"Nombre de cycles par pixel - variance : {cycles_var:.9f} cpp^2")
    print(f"Nombre de cycles par pixel - écart-type : {cycles_std:.6f} cpp")


def benchmark_series(
    img: np.ndarray,
    base_params: HarrisParams,
    n_runs: int,
    field_name: str,
    values: list[int] | list[float],
):
    mean_times = np.zeros(len(values), dtype=np.float64)
    mean_cpp = np.zeros(len(values), dtype=np.float64)

    for i, value in enumerate(values):
        params = replace(base_params, **{field_name: value})
        time_mean, _, _, cpp_mean, _, _ = benchmark_harris(img, n_runs, params)
        mean_times[i] = time_mean
        mean_cpp[i] = cpp_mean
        print(f"{field_name}={value} -> temps moyen={time_mean:.9f} s, cpp moyen={cpp_mean:.6f}")

    return mean_times, mean_cpp


def save_line_plot(
    x_values: list[int] | list[float],
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, "-o")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")


def run_parameter_plots(image_path: Path, script_dir: Path, n_runs: int, base_params: HarrisParams):
    img = load_gray_image(image_path)
    output_dir = script_dir / "results" / "plots" / "harris"
    output_dir.mkdir(parents=True, exist_ok=True)

    sum_window_values = [3, 5, 7, 9, 11]
    harris_k_values = [0.02, 0.04, 0.06, 0.08, 0.10]
    maxloc_values = [3, 5, 7, 9, 11]

    print(f"Génération des courbes paramétriques (n_runs={n_runs})...")

    sum_time_means, sum_cpp_means = benchmark_series(
        img, base_params, n_runs, "sum_window_size", sum_window_values
    )
    harris_k_time_means, harris_k_cpp_means = benchmark_series(
        img, base_params, n_runs, "harris_k", harris_k_values
    )
    maxloc_time_means, maxloc_cpp_means = benchmark_series(
        img, base_params, n_runs, "maxloc_neighborhood_size", maxloc_values
    )

    save_line_plot(
        sum_window_values,
        sum_cpp_means,
        "SUM_WINDOW_SIZE",
        "Cycles par pixel moyens (cpp)",
        f"Cycles/pixel moyens vs SUM_WINDOW_SIZE (n={n_runs})",
        output_dir / "cpp_vs_sum_window_size.png",
    )
    save_line_plot(
        sum_window_values,
        sum_time_means,
        "SUM_WINDOW_SIZE",
        "Temps moyen (s)",
        f"Temps moyen vs SUM_WINDOW_SIZE (n={n_runs})",
        output_dir / "time_vs_sum_window_size.png",
    )
    save_line_plot(
        harris_k_values,
        harris_k_cpp_means,
        "HARRIS_K",
        "Cycles par pixel moyens (cpp)",
        f"Cycles/pixel moyens vs HARRIS_K (n={n_runs})",
        output_dir / "cpp_vs_harris_k.png",
    )
    save_line_plot(
        harris_k_values,
        harris_k_time_means,
        "HARRIS_K",
        "Temps moyen (s)",
        f"Temps moyen vs HARRIS_K (n={n_runs})",
        output_dir / "time_vs_harris_k.png",
    )
    save_line_plot(
        maxloc_values,
        maxloc_cpp_means,
        "MAXLOC_NEIGHBORHOOD_SIZE",
        "Cycles par pixel moyens (cpp)",
        f"Cycles/pixel moyens vs MAXLOC_NEIGHBORHOOD_SIZE (n={n_runs})",
        output_dir / "cpp_vs_maxloc_neighborhood_size.png",
    )
    save_line_plot(
        maxloc_values,
        maxloc_time_means,
        "MAXLOC_NEIGHBORHOOD_SIZE",
        "Temps moyen (s)",
        f"Temps moyen vs MAXLOC_NEIGHBORHOOD_SIZE (n={n_runs})",
        output_dir / "time_vs_maxloc_neighborhood_size.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Détection de points d'intérêt de Harris")
    parser.add_argument(
        "-stats",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Exécute le calcul N fois et affiche moyenne/variance/écart-type (N=10 par défaut).",
    )
    parser.add_argument(
        "-plots",
        action="store_true",
        help="Génère 6 graphiques paramétriques (nécessite -stats N).",
    )

    args = parser.parse_args()
    if args.plots and args.stats is None:
        parser.error("L'option -plots nécessite -stats N (exemple: python Harris.py -stats 50 -plots).")

    return args


def main():
    script_dir = Path(__file__).resolve().parent
    image_path = resolve_image_path(script_dir)
    args = parse_args()

    if args.stats is not None:
        run_stats(image_path, args.stats, DEFAULT_PARAMS)
        if args.plots:
            run_parameter_plots(image_path, script_dir, args.stats, DEFAULT_PARAMS)
    else:
        run_single(image_path, script_dir, DEFAULT_PARAMS)


if __name__ == "__main__":
    main()
