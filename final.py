#!/usr/bin/env python3
"""
Automatic corner detection for a set of painted road strips (yellow / white).

Input  : a single image
Output : the 4 corner points (original-image pixel coordinates), the polygon
         area, and an annotated image in the same style as the sample in the PDF.

Usage:
    python final.py road.jpg
    python final.py road.jpg --color white
    python final.py road.jpg --out result.png --debug

What is looked for
    A *group* of several (>= 3) roughly parallel, roughly horizontal paint
    strips lying across the road (speed-breaker / rumble markings).  Single
    long lines (lane divider, road edge line, centre line) are ignored, no
    matter what colour they are, because they are (a) steep in the image and
    (b) not a stack of parallel strips.

Pipeline
    1. Downscale large images to a working width.
    2. Colour masks (yellow, white) -> paint pixels. White uses local contrast
       so thin white lines on grey asphalt are caught but bright sky is not.
    3. Orientation filter: morphological opening with line kernels from
       -35..+35 deg keeps only near-horizontal line segments.
    4. Closing merges neighbouring strips into blobs.
    5. Every blob is cross-sectioned; the number of separate strips in it is
       counted. Blobs with >= 3 strips are scored (strips x length); the best
       blob over all colours wins.
    6. Both outer strips are tracked to their ends -> 4 corners, ordered
       clockwise, mapped back to the original resolution.
"""
import argparse
import json
import sys

import cv2
import numpy as np

WORK_WIDTH = 1280
MAX_TILT_DEG = 35          # strips steeper than this are not "horizontal"
MIN_STRIPS = 3             # a strip *set* needs at least this many strips

# HSV ranges (OpenCV: H 0-179, S/V 0-255)
YELLOW_RANGE = ((12, 80, 110), (38, 255, 255))


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def build_mask(bgr, color):
    """Binary mask of the requested paint colour."""
    blur = cv2.GaussianBlur(bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if color == "yellow":
        mask = cv2.inRange(hsv, np.array(YELLOW_RANGE[0]), np.array(YELLOW_RANGE[1]))
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        mask &= (lab[:, :, 2] > 140).astype(np.uint8) * 255
        return mask

    # white: bright, unsaturated AND brighter than the local surroundings
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sigma = max(4, bgr.shape[1] // 80)
    bg = cv2.GaussianBlur(gray.astype(np.uint8), (0, 0), sigma).astype(np.int16)
    contrast = gray - bg
    mask = (contrast > 18) & (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 110)
    return mask.astype(np.uint8) * 255


def road_roi_top(bgr):
    """Rows above this are ignored (sky, trees, far background)."""
    return int(bgr.shape[0] * 0.40)


def line_kernel(length, angle_deg):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx = np.cos(np.radians(angle_deg)) * c
    dy = -np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def keep_horizontal_lines(mask):
    """Keep only pixels that belong to near-horizontal line segments."""
    w = mask.shape[1]
    L = max(21, (w // 20) | 1)
    out = np.zeros_like(mask)
    for ang in range(-MAX_TILT_DEG, MAX_TILT_DEG + 1, 5):
        out |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, line_kernel(L, ang))
    return out


def merge_strips(mask):
    """Close the gaps between neighbouring strips so a set becomes one blob."""
    w = mask.shape[1]
    kx = max(9, w // 60)
    ky = max(9, w // 45)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def principal_axes(pts):
    """Unit vectors (u: along the strips, left->right; v: across, pointing down)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd((pts - c).astype(np.float64), full_matrices=False)
    u = vt[0]
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v
    return c, u, v


def _runs_per_bin(pts, t, s, gap=2.0):
    """For every 1-px bin along the main axis, split pixels into runs across it
    (one run ~ one strip). Returns {bin: [(s_center, mean_x, mean_y), ...]}."""
    tb = np.round(t).astype(int)
    order = np.lexsort((s, tb))
    tb, s, pts = tb[order], s[order], pts[order]
    out = {}
    edges = np.flatnonzero(np.diff(tb)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(tb)]):
        ss, pp = s[lo:hi], pts[lo:hi]
        cuts = np.flatnonzero(np.diff(ss) > gap) + 1
        runs = []
        for a_, b_ in zip(np.r_[0, cuts], np.r_[cuts, len(ss)]):
            runs.append((ss[a_:b_].mean(), pp[a_:b_, 0].mean(), pp[a_:b_, 1].mean()))
        out[tb[lo]] = runs
    return out


def _track_all(bins, b0, centers, direction, max_jump=4.0, max_gap=300):
    """
    Follow ALL strips simultaneously from cross-section b0 in one direction.
    Each run can be claimed by only one strip (so a strip that has ended can
    not 'steal' its neighbour). A gap in the paint (e.g. where a lane line
    crosses the strip) is bridged by extrapolating the strip's recent trend.
    Returns the last seen (x, y) per strip.
    """
    k = len(centers)
    hist = [[(b0, c)] for c in centers]          # matched (bin, centre) history
    miss = [0] * k
    alive = [True] * k
    last = [None] * k
    lo, hi = min(bins), max(bins)
    spacing = float(np.median(np.diff(sorted(centers)))) if k > 1 else 10.0

    def predict(i, b):
        h = hist[i][-100:]
        (b_a, c_a), (b_z, c_z) = h[0], h[-1]
        slope = (c_z - c_a) / (b_z - b_a) if b_z != b_a else 0.0
        return c_z + slope * (b - b_z)

    b = b0
    while any(alive) and lo <= b + direction <= hi:
        b += direction
        runs = list(bins.get(b, []))
        pred = {i: predict(i, b) for i in range(k) if alive[i]}
        pairs = sorted(
            (abs(r[0] - pred[i]), i, j)
            for i in pred for j, r in enumerate(runs)
        )
        taken_i, taken_j = set(), set()
        for d, i, j in pairs:
            if i in taken_i or j in taken_j:
                continue
            off = runs[j][0] - pred[i]                   # <0: above the prediction
            tol_in = min(max_jump + 0.03 * miss[i], 0.5 * spacing)
            tol_out = min(max_jump + 0.15 * miss[i], 3 * spacing)
            outward_up = (i == 0 and off < 0)            # outermost strips have nothing
            outward_dn = (i == k - 1 and off > 0)        # beyond them, so re-acquire loosely
            if abs(off) > (tol_out if (outward_up or outward_dn) else tol_in):
                continue
            taken_i.add(i); taken_j.add(j)
            hist[i].append((b, runs[j][0]))
            miss[i] = 0
            last[i] = (runs[j][1], runs[j][2])
        for i in pred:
            if i not in taken_i:
                miss[i] += 1
                if miss[i] > max_gap:
                    alive[i] = False
    return last


def best_cross_section(bins, t):
    """Cross-section (bin) near the middle of the span that shows most strips."""
    mid = int(np.median(t))
    cand = [b for b in bins if abs(b - mid) <= 0.2 * (t.max() - t.min())]
    if not cand:
        return None, []
    b0 = max(cand, key=lambda b: (len(bins[b]), -abs(b - mid)))
    return b0, sorted(bins[b0], key=lambda r: r[0])


def analyse_blob(comp):
    """Return dict(n_strips, quad, score) for a blob mask, or None."""
    ys, xs = np.nonzero(comp)
    if len(xs) < 50:
        return None
    P = np.column_stack([xs, ys]).astype(np.float64)
    c, u, v = principal_axes(P)
    t, s = (P - c) @ u, (P - c) @ v
    span = t.max() - t.min()
    if span < 0.12 * comp.shape[1]:
        return None
    bins = _runs_per_bin(P, t, s)
    b0, runs = best_cross_section(bins, t)
    if len(runs) < 2:
        return None

    centers = [r[0] for r in runs]
    left = _track_all(bins, b0, centers, -1)
    right = _track_all(bins, b0, centers, +1)

    # end points of every strip, and its tracked length
    strips = []
    for i, r in enumerate(runs):
        pl = left[i] or (r[1], r[2])
        pr = right[i] or (r[1], r[2])
        strips.append((pl, pr, float(np.hypot(pr[0] - pl[0], pr[1] - pl[1]))))
    longest = max(st[2] for st in strips)
    strips = [st for st in strips if st[2] >= 0.4 * longest]   # drop stubs / crossing fragments
    if len(strips) < 2:
        return None
    (tl, tr, _), (bl, br, _) = strips[0], strips[-1]
    quad = np.array([tl, tr, br, bl], np.float32)
    runs = strips

    # a real strip set: strips must be long compared with the gaps between them
    n = len(runs)
    return dict(n_strips=n, quad=quad, score=n * span, span=span)


def order_clockwise(pts):
    """Order 4 points clockwise starting from the top-left-most point."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # clockwise in image coordinates
    start = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -start, axis=0)


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
def candidate_groups(img, color):
    """Yield analysed strip-group candidates for one colour."""
    roi_top = road_roi_top(img)
    raw = build_mask(img, color)
    raw[:roi_top, :] = 0
    lines = keep_horizontal_lines(raw)
    merged = merge_strips(lines)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.0005 * merged.size:
            continue
        comp = (labels == i) & (lines > 0)
        info = analyse_blob(comp)
        if info:
            info["color"], info["mask"] = color, comp.astype(np.uint8) * 255
            yield info


def detect_corners(image_path, color="auto"):
    """
    Returns (corners[4x2 float array in original pixels], area, color_used, mask)
    or (None, 0, None, None) if nothing was found.
    """
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    scale = WORK_WIDTH / orig.shape[1] if orig.shape[1] > WORK_WIDTH else 1.0
    img = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else orig

    colors = ["yellow", "white"] if color == "auto" else [color]
    cands = [c for col in colors for c in candidate_groups(img, col)]

    good = [c for c in cands if c["n_strips"] >= MIN_STRIPS]
    pool = good or [c for c in cands if c["n_strips"] >= 2]
    if not pool:
        return None, 0.0, None, None

    best = max(pool, key=lambda c: c["score"])
    quad = order_clockwise(best["quad"]) / scale
    area = float(cv2.contourArea(quad.astype(np.float32)))
    return quad, area, best["color"], best["mask"]


def annotate(image_path, corners, area):
    img = cv2.imread(image_path)
    pts = corners.astype(int)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{i}:({x},{y})", (x + 6, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Area: {area:.2f} px^2", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Detect the 4 corners of a set of road strips.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--color", choices=["auto", "yellow", "white"], default="auto",
                    help="strip colour (default: auto = try yellow, then white)")
    ap.add_argument("--out", default=None, help="annotated output path")
    ap.add_argument("--debug", action="store_true", help="also save the binary mask")
    args = ap.parse_args()

    corners, area, col, mask = detect_corners(args.image, args.color)
    if corners is None:
        print(json.dumps({"found": False}))
        sys.exit(1)

    out = args.out or args.image.rsplit(".", 1)[0] + "_corners.png"
    cv2.imwrite(out, annotate(args.image, corners, area))
    if args.debug:
        cv2.imwrite(out.rsplit(".", 1)[0] + "_mask.png", mask)

    print(json.dumps({
        "found": True,
        "color": col,
        "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
        "area_px2": round(area, 2),
        "annotated_image": out,
    }, indent=2))


if __name__ == "__main__":
    main()