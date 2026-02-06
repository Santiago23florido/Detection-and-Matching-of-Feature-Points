import numpy as np
import cv2

def eval_matches(kp1, kp2, matches, M, img2_shape, threshold=3.0):
    h2, w2 = img2_shape[:2]
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError("M must be 2x3 (affine matrix for warpAffine).")

    correct_mask = []
    errors = []
    used = 0

    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt

        xgt = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
        ygt = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]

        if not (0 <= xgt < w2 and 0 <= ygt < h2):
            continue

        e = float(((x2 - xgt) ** 2 + (y2 - ygt) ** 2) ** 0.5)
        errors.append(e)
        correct_mask.append(e < threshold)
        used += 1

    n_correct = int(np.sum(correct_mask)) if used > 0 else 0
    precision = (n_correct / used) if used > 0 else 0.0

    return {
        "evaluated_matches": used,
        "correct_matches": n_correct,
        "precision": precision,
        "mean_error_px": float(np.mean(errors)) if errors else float("nan"),
        "median_error_px": float(np.median(errors)) if errors else float("nan"),
    }


def pair_eval(img1: np.ndarray, img2: np.ndarray, detector: int, M, threshold=3.0, ratio=0.7):
    if img1 is None or img2 is None:
        raise ValueError("img1 or img2 is None. Check cv2.imread(...).")

    if detector == 1:
        det1 = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        det2 = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif detector == 2:
        det1 = cv2.KAZE_create(upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=2)
        det2 = cv2.KAZE_create(upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("detector must be 1 (ORB) or 2 (KAZE).")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    kp1_all, desc1 = det1.detectAndCompute(gray1, None)
    kp2_all, desc2 = det2.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1_all) == 0 or len(kp2_all) == 0:
        return {
            "evaluated_matches": 0,
            "correct_matches": 0,
            "precision": 0.0,
            "mean_error_px": float("nan"),
            "median_error_px": float("nan"),
        }

    if detector == 1:
        if desc1.dtype != np.uint8:
            desc1 = desc1.astype(np.uint8, copy=False)
        if desc2.dtype != np.uint8:
            desc2 = desc2.astype(np.uint8, copy=False)
    else:
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32, copy=False)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32, copy=False)

    knn = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) == 0:
        return {
            "evaluated_matches": 0,
            "correct_matches": 0,
            "precision": 0.0,
            "mean_error_px": float("nan"),
            "median_error_px": float("nan"),
        }

    q_indices = [m.queryIdx for m in good_matches]
    t_indices = [m.trainIdx for m in good_matches]

    q_map = {old: new for new, old in enumerate(q_indices)}
    t_map = {old: new for new, old in enumerate(t_indices)}

    kp1_good = [kp1_all[i] for i in q_indices]
    kp2_good = [kp2_all[i] for i in t_indices]

    matches_good = [
        cv2.DMatch(_queryIdx=q_map[m.queryIdx], _trainIdx=t_map[m.trainIdx], _imgIdx=m.imgIdx, _distance=m.distance)
        for m in good_matches
    ]

    return eval_matches(kp1_good, kp2_good, matches_good, M, gray2.shape, threshold=threshold)
