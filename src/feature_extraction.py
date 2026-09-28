import cv2
import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_NAMES = [
    "ink_density", "aspect_ratio", "contour_area_ratio",
    "contour_perimeter_norm", "centroid_offset",
    "radial_mean", "radial_std", "radial_cv",
    "radial_range", "radial_slope",
    "angular_coverage", "turning_mean", "turning_std",
    "turning_abs_mean", "line_width_mean", "line_width_std",
    "line_width_cv", "horizontal_spread", "vertical_spread",
    "skew_x", "skew_y", "kurtosis_x", "kurtosis_y"
]

def _clean_binary(image):
    if image is None:
        raise ValueError("Image is empty.")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Handles both dark-on-light and light-on-dark drawings.
    _, a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Choose the mask with a plausible amount of foreground.
    candidates = [a, b]
    mask = min(candidates, key=lambda m: abs((np.count_nonzero(m) / m.size) - 0.15))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return gray, mask

def _largest_component(mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out

def _skeletonize(mask):
    # Morphological skeletonization without scikit-image.
    img = mask.copy()
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

def extract_features(image):
    gray, mask = _clean_binary(image)
    mask = _largest_component(mask)

    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if len(xs) < 30:
        raise ValueError("Not enough drawing pixels detected.")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)

    crop = mask[y0:y1+1, x0:x1+1]
    ink_density = float(np.mean(crop > 0))
    aspect_ratio = bw / bh

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    contour_area_ratio = area / float(bw * bh)
    contour_perimeter_norm = perimeter / float(2 * (bw + bh))

    cx, cy = xs.mean(), ys.mean()
    centroid_offset = np.hypot(cx - (x0+x1)/2, cy - (y0+y1)/2) / np.hypot(bw, bh)

    # Skeleton/path geometry
    skel = _skeletonize(crop)
    sy, sx = np.where(skel > 0)
    if len(sx) < 20:
        sx, sy = xs - x0, ys - y0

    sx = sx.astype(float)
    sy = sy.astype(float)
    ccx, ccy = sx.mean(), sy.mean()

    dx, dy = sx - ccx, sy - ccy
    radii = np.sqrt(dx*dx + dy*dy) + 1e-8
    angles = np.unwrap(np.arctan2(dy, dx))
    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]

    # Normalize radial values by drawing size.
    scale = np.hypot(bw, bh) + 1e-8
    r = radii / scale
    radial_mean = float(np.mean(r))
    radial_std = float(np.std(r))
    radial_cv = float(radial_std / (radial_mean + 1e-8))
    radial_range = float(np.ptp(r))

    if len(r) > 2:
        coeff = np.polyfit(angles, r, 1)
        radial_slope = float(coeff[0])
    else:
        radial_slope = 0.0

    angular_coverage = float((angles.max() - angles.min()) / (2*np.pi)) if len(angles) else 0.0

    # Turning-angle statistics along ordered skeleton points.
    order2 = np.argsort(sy * (w+1) + sx)
    px, py = sx[order2], sy[order2]
    if len(px) >= 3:
        v1 = np.column_stack((np.diff(px[:-1]), np.diff(py[:-1])))
        v2 = np.column_stack((np.diff(px[1:]), np.diff(py[1:])))
        n1 = np.linalg.norm(v1, axis=1) + 1e-8
        n2 = np.linalg.norm(v2, axis=1) + 1e-8
        cosang = np.clip(np.sum(v1*v2, axis=1)/(n1*n2), -1, 1)
        turns = np.arccos(cosang)
        turning_mean = float(np.mean(turns))
        turning_std = float(np.std(turns))
        turning_abs_mean = float(np.mean(np.abs(turns)))
    else:
        turning_mean = turning_std = turning_abs_mean = 0.0

    # Approximate local line width from distance transform.
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    widths = 2 * dist[skel > 0]
    widths = widths[widths > 0]
    if len(widths) == 0:
        widths = np.array([1.0])
    line_width_mean = float(np.mean(widths))
    line_width_std = float(np.std(widths))
    line_width_cv = float(line_width_std / (line_width_mean + 1e-8))

    x_norm = (sx - sx.min()) / max(1, sx.max()-sx.min())
    y_norm = (sy - sy.min()) / max(1, sy.max()-sy.min())

    return np.array([
        ink_density, aspect_ratio, contour_area_ratio,
        contour_perimeter_norm, centroid_offset,
        radial_mean, radial_std, radial_cv,
        radial_range, radial_slope,
        angular_coverage, turning_mean, turning_std,
        turning_abs_mean, line_width_mean, line_width_std,
        line_width_cv, float(np.std(x_norm)), float(np.std(y_norm)),
        float(skew(x_norm, bias=False)) if len(x_norm)>2 else 0.0,
        float(skew(y_norm, bias=False)) if len(y_norm)>2 else 0.0,
        float(kurtosis(x_norm, bias=False)) if len(x_norm)>3 else 0.0,
        float(kurtosis(y_norm, bias=False)) if len(y_norm)>3 else 0.0
    ], dtype=np.float32)

def preprocess_for_display(image):
    _, mask = _clean_binary(image)
    return mask
