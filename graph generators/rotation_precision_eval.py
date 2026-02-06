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


def main() -> None:
    img_path = _find_image_path()
    img1 = cv2.imread(str(img_path))
    if img1 is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img1.shape[:2]
    center = (w / 2.0, h / 2.0)

    angles = np.linspace(0.0, 360.0, 20)
    orb_prec = []
    kaze_prec = []
    orb_pairs = []
    kaze_pairs = []

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        img2 = cv2.warpAffine(
            img1,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        orb = pair_eval(img1, img2, detector=1, M=M)
        kaze = pair_eval(img1, img2, detector=2, M=M)

        orb_prec.append(orb["precision"])
        kaze_prec.append(kaze["precision"])
        orb_pairs.append(orb["evaluated_matches"])
        kaze_pairs.append(kaze["evaluated_matches"])

    output_dir = ROOT / "docs" / "rappport" / "imgs" / "descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rotation_precision_orb_kaze.png"

    orb_pairs_mean = int(round(np.mean(orb_pairs))) if orb_pairs else 0
    kaze_pairs_mean = int(round(np.mean(kaze_pairs))) if kaze_pairs else 0

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(angles, orb_prec, color="tab:blue", label=f"ORB (avg pairs={orb_pairs_mean})")
    plt.plot(angles, kaze_prec, color="tab:orange", label=f"KAZE (avg pairs={kaze_pairs_mean})")
    plt.xlabel("Rotation (deg)")
    plt.ylabel("Precision")
    plt.title("Precision vs Rotation")
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
