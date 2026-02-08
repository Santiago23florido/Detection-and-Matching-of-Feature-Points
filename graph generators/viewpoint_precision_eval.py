from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pair_eval import pair_eval


def _find_image_path() -> Path:
    base = Path(__file__).resolve().parent
    root = base.parent
    candidates = [
        root / "Image_Pairs" / "torb_small1.png",
        root / "Image_Pairs" / "Image_Pairs" / "torb_small1.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("torb_small1.png not found in Image_Pairs/")


def _homography_horizontal_tilt(w: int, h: int, t: float) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[t, 0], [w - t, 0], [w + t, h], [-t, h]])
    return cv2.getPerspectiveTransform(src, dst)


def _homography_vertical_tilt(w: int, h: int, t: float) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, t], [w, 0], [w, h], [0, h - t]])
    return cv2.getPerspectiveTransform(src, dst)


def main() -> None:
    img_path = _find_image_path()
    img1 = cv2.imread(str(img_path))
    if img1 is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img1.shape[:2]

    max_shift_x = 0.3 * w
    max_shift_y = 0.3 * h
    shifts_x = np.linspace(0.0, max_shift_x, 11)
    shifts_y = np.linspace(0.0, max_shift_y, 11)

    orb_h = []
    kaze_h = []
    orb_pairs_h = []
    kaze_pairs_h = []
    for t in shifts_x:
        H = _homography_horizontal_tilt(w, h, float(t))
        img2 = cv2.warpPerspective(
            img1,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        orb = pair_eval(img1, img2, detector=1, M=H)
        kaze = pair_eval(img1, img2, detector=2, M=H)
        orb_h.append(orb["precision"])
        kaze_h.append(kaze["precision"])
        orb_pairs_h.append(orb["evaluated_matches"])
        kaze_pairs_h.append(kaze["evaluated_matches"])

    orb_v = []
    kaze_v = []
    orb_pairs_v = []
    kaze_pairs_v = []
    for t in shifts_y:
        H = _homography_vertical_tilt(w, h, float(t))
        img2 = cv2.warpPerspective(
            img1,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        orb = pair_eval(img1, img2, detector=1, M=H)
        kaze = pair_eval(img1, img2, detector=2, M=H)
        orb_v.append(orb["precision"])
        kaze_v.append(kaze["precision"])
        orb_pairs_v.append(orb["evaluated_matches"])
        kaze_pairs_v.append(kaze["evaluated_matches"])

    output_dir = ROOT / "docs" / "rappport" / "imgs" / "descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "viewpoint_precision_orb_kaze.png"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)

    orb_pairs_h_mean = int(round(np.mean(orb_pairs_h))) if orb_pairs_h else 0
    kaze_pairs_h_mean = int(round(np.mean(kaze_pairs_h))) if kaze_pairs_h else 0

    ax = axes[0]
    ax.plot(shifts_x, orb_h, color="tab:blue", label=f"ORB (avg pairs={orb_pairs_h_mean})")
    ax.plot(shifts_x, kaze_h, color="tab:orange", label=f"KAZE (avg pairs={kaze_pairs_h_mean})")
    ax.set_title("Horizontal tilt")
    ax.set_xlabel("Tilt magnitude (px)")
    ax.set_ylabel("Precision")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    orb_pairs_v_mean = int(round(np.mean(orb_pairs_v))) if orb_pairs_v else 0
    kaze_pairs_v_mean = int(round(np.mean(kaze_pairs_v))) if kaze_pairs_v else 0

    ax = axes[1]
    ax.plot(shifts_y, orb_v, color="tab:blue", label=f"ORB (avg pairs={orb_pairs_v_mean})")
    ax.plot(shifts_y, kaze_v, color="tab:orange", label=f"KAZE (avg pairs={kaze_pairs_v_mean})")
    ax.set_title("Vertical tilt")
    ax.set_xlabel("Tilt magnitude (px)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
