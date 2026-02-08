import numpy as np
import cv2


def rotate_scale(img: np.ndarray, angle_deg: float, scale: float = 1.0,
                 tx: float = 0.0, ty: float = 0.0):

    if img is None:
        raise ValueError("img is None. ")

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, scale)  # 2x3
    M[0, 2] += tx
    M[1, 2] += ty

    img_out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return img_out, M


def change_viewpoint(img: np.ndarray, src_pts, dst_pts):

    if img is None:
        raise ValueError("img is None")
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise ValueError("src_pts y dst_pts should have exactly 4 points.")

    h, w = img.shape[:2]
    src = np.float32(src_pts)
    dst = np.float32(dst_pts)

    H = cv2.getPerspectiveTransform(src, dst)  # 3x3

    img_out = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return img_out, H
