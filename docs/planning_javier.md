**Q4 Harris - specific tasks**
Complete `Harris.py` to compute the Harris interest function (single scale, fixed window W) and the corresponding interest points.

- Implement gradients: compute `Ix`, `Iy` (e.g., with `cv2.Sobel` or `cv2.filter2D` using derivative kernels).
- Compute products: `Ix2 = Ix*Ix`, `Iy2 = Iy*Iy`, `Ixy = Ix*Iy`.
- Sum over a fixed window W: apply a window (box or Gaussian) with `cv2.filter2D` or `cv2.GaussianBlur` on `Ix2`, `Iy2`, `Ixy` to obtain `Sxx`, `Syy`, `Sxy`.
- Compute the Harris response: `Theta = det(M) - k*(trace(M)**2)` with `det = Sxx*Syy - Sxy**2`, `trace = Sxx + Syy`, choose `k` (e.g., 0.04).
- Check types and ranges (use `float64`) and run the script to validate timing and visualization.
- Explain morphological dilation: `Theta_dil = cv2.dilate(Theta, se)` produces the local maximum in each neighborhood; compare `Theta` vs `Theta_dil` and suppress non-maxima (`Theta < Theta_dil`) to get local maxima.
- Document parameters used: window size, `k`, `d_maxloc`, `seuil_relatif`, and comment on their effect on the number of detected points.

**Q5 Harris - analysis and extensions**
Comment on detector results and parameter effects, and propose extensions.

- Run the detector varying window size W (small, medium, large) and record changes in number/distribution of points and stability.
- Vary `alpha` (or `k`) in a typical range (e.g., 0.04–0.06) and observe sensitivity: more points vs more selectivity.
- Include comparative screenshots/figures and conclusion notes (trade-offs between detail and noise).
- Propose multi-scale: build a Gaussian pyramid and apply Harris at each level, then merge points with scale-aware criteria.
- Extend local maxima with a minimum distance `r`: apply non-maximum suppression with a radius (circular or square neighborhood) or select points sorted by response while enforcing minimum separation.

**Q9 Quantitative evaluation of matches**
Propose a strategy to evaluate match quality using a known deformation.

- Take an image and generate a deformed version with `cv2.warpAffine` (rotation, translation, scale).
- Keep the transformation matrix `A` and apply it to the original keypoints to get expected locations in the deformed image.
- Run matching (ORB/KAZE + CrossCheck/RatioTest/FLANN) and measure:
  - True positives: matches whose distance to the expected point is < threshold (in pixels).
  - False positives: matches outside the threshold.
  - Precision/recall or accuracy percentage.
- Test multiple deformation levels and compare methods.
