**Objective**
Split the TP on feature detection and matching (Q1-Q9) so 3 people can work in parallel and later integrate into the report.

**Division into 3 tracks**


Person A (Jose): Convolutions and gradients
- Q1: Run `Convolutions.py`, compare time and results between 2D scan and `cv2.filter2D`, and explain `cv2.imread`, `cv2.copyMakeBorder`, `cv2.imshow`, `cv2.waitKey`, `plt.imshow`.
- Q2: Explain why the kernel `[ [0,-1,0],[-1,5,-1],[0,-1,0] ]` enhances contrast.
- Q3: Modify `Convolutions.py` to compute `Ix`, `Iy`, and `||∇I||`, and document handling of negative values and normalization for visualization.

Person B (Javier): Harris detector + quantitative evaluation
- Q4: Complete `Harris.py` with the Harris interest function, compute local maxima, and explain the morphological dilation.
- Q5: Analyze parameter effects (window size, alpha), comment on results, and propose multi-scale extension and minimum separation `r`.
- Q9: Propose quantitative evaluation with known deformation using `cv2.warpAffine`.

Person C (Santiago): ORB, KAZE and matching
- Q6: Run `Features_Detect.py`, compare ORB vs KAZE, parameters and visual repeatability.
- Q7: Explain ORB and KAZE descriptors, scale/rotation invariance (separate detector vs descriptor).
- Q8: Compare `Features_Match_CrossCheck.py`, `Features_Match_RatioTest.py`, `Features_Match_FLANN.py` and justify distance metrics used.

**Deliverables per person**
- Code changes ready for review.
- Clear notes for the report (observations, screenshots and conclusions).
- Parameters used and key results (timings, number of points, match rates, etc.).

**Sync and integration**
- Shared repository with branches or commits per person.
- Short integration meeting to unify report style.
- One person integrates the final PDF report with sections Q1-Q9.
