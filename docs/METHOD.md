# Method

## 1. Capture simulation with exact ground truth
Real distorted photos come without corner labels. Instead, a flat document is placed in 3D and viewed by a pinhole camera:
rotation (yaw, pitch, roll), distance, focal length (0.9 to 1.6 times the canvas width) and an offset. The four document corners are projected with the camera matrix, which gives the exact ground-truth quadrilateral. The document is warped onto a random background (solid, gradient, or low-frequency desk texture), then a lighting gradient, an optional shadow band, Gaussian blur, noise and JPEG compression are applied.

| Level | max roll | max tilt | document height | blur sigma | noise | shadow | extra 90/180/270 turn | background |
|---|---|---|---|---|---|---|---|---|
| easy | 10 deg | 15 deg | 70-85% | 0.6 | 4 | 0.10 | never | random |
| medium | 30 deg | 30 deg | 60-80% | 1.2 | 8 | 0.25 | never | random |
| hard | 45 deg | 40 deg | 50-75% | 2.0 | 12 | 0.40 | 50% | similar brightness to the document |

## 2. Document detection
Five strategies propose quadrilaterals (hull of the foreground mask, simplified with `approxPolyDP` at several tolerances, or the minimum-area rectangle when no 4-point polygon exists):
1. Otsu threshold, bright foreground and dark foreground.
2. Adaptive Gaussian threshold (handles uneven lighting).
3. Canny edges, dilated and closed.
4. Background model: the 4% border strip is assumed background; foreground is Lab distance from its median colour, thresholded with Otsu.
5. GrabCut initialised with a frame-inset rectangle.

Each candidate has five features: **edge support** (fraction of points along the border lying on a Canny edge), **area** (capped at 50% of the frame), **contrast** (brightness difference between inside and a surrounding ring), **angle plausibility** (corners within 45 to 135 degrees) and **mask fit** (IoU with the foreground hull it came from). The score is a weighted sum plus a small per-strategy prior. Weights were found by random search on the *validation* split, fitting on half of the documents and checking on the other half.

| | default weights | tuned weights |
|---|---|---|
| fit half (mean IoU) | 0.760 | 0.906 |
| held-out half | 0.753 | 0.872 |

If the best score is under the threshold, the full frame is returned.

## 3. Rectification and aspect ratio
The quadrilateral is warped to an upright rectangle with a homography. Using the longest edges as the output size stretches a document seen in perspective, so the true width/height ratio is recovered with the method of Zhang and He (2007): assuming square pixels and a principal point at the image centre, the focal length and aspect ratio follow from the four corners. On exact ground-truth corners the median aspect error is 0.000. Because the estimate amplifies corner noise, it is only accepted when within 25% of the edge-length ratio.

## 4. Orientation and deskew
A page can arrive rotated by any quarter-turn. Each of the four rotations is OCR'd at reduced size with Tesseract and scored by the number of confident words; a rotation must beat the current orientation by 15% to be chosen. Residual skew is estimated by searching +/-4 degrees for the rotation that maximises the variance of the horizontal projection profile of the binarised page, and applied only when it beats "no rotation" by 5%.

## 5. Metrics
- **Geometry:** IoU between detected and true quadrilateral; mean corner distance as a percentage of the true diagonal; share of captures above IoU 0.9; aspect-ratio error; orientation accuracy against the true required rotation.
- **OCR:** token multiset precision, recall and F1 against the dataset's word annotations (order-free, because receipts have no reliable reading order); **numeric recall** on tokens containing a digit, i.e. prices, quantities and dates.
- **Uncertainty:** 95% bootstrap intervals that resample documents, since several captures come from one document.

## 6. Variants compared
`raw` (no processing), `rect` (detect and rectify), `rect_ori` (plus orientation), `rect_full` (plus deskew), `oracle` (true corners and true orientation, with the same sizing rule), and the original flat document as a reference.
