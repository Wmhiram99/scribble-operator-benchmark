#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v17g Hybrid (FG+BG) Fragment Noise
- rollback candidates selection (v17e style)
- FG & BG cutting (v17d requirement)
- VOC idx correctness: 0 is valid bg scribble, 255 is NO_SCRIB
- NEW (+1 improvement):
  (A) Micro-cut candidates for high ratios to avoid overshoot (ratio 0.8/0.5)
  (B) Soft-Erase fallback when split-candidates are empty to reduce "after_above_target" stuck (ratio 0.1/0.5)
"""

import os
import sys
import csv
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from PIL import Image
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0) Params (edit paths here)
# ============================================================
GT_DIR = os.environ.get("CITYSCAPES_FRAGMENT_GT_DIR", "datasets/cityscapes/GtMask")
SCRIB_DIR = os.environ.get("CITYSCAPES_FRAGMENT_SCRIBBLE_DIR", "datasets/cityscapes/auto_scribbles_idx")
OUT_ROOT = os.environ.get("CITYSCAPES_FRAGMENT_OUT_ROOT", "outputs/cityscapes/fragmentation")

RATIOS = [0.1]#, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
TOL = 0.05
LIMIT_N: Optional[int] = 13000
RANDOM_SEED = 123

MAX_ITERS_FG = 260
MAX_ITERS_BG_FINE = 90
GAP_EPS = 1e-7

# candidate budget
MAX_CANDS_PER_CC = 12

# base radius multipliers (normal)
ALPHAS = [1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
R_MIN = 5
R_MAX = 36

# micro-cut (for high ratio, to reduce overshoot)
MICRO_ALPHAS = [0.7, 0.85, 1.0, 1.15]
MICRO_R_MIN = 2
MICRO_HALFLEN_SCALE = 0.65  # shorter capsule for micro-cut

MIN_CC_PIXELS_BASE = 60
MIN_PART_PIXELS = 15
TOPK_POINTS = 180

GAP_MIN_ABS_PX = 10
GAP_MIN_REL = 1.4

REMOVED_STROKE_PIX_MIN_ABS = 60
REMOVED_STROKE_PIX_MIN_REL = 0.9

# soft-erase fallback
SOFT_ERASE_ENABLE = True
SOFT_ERASE_MIN_REMOVED_ABS = 50
SOFT_ERASE_MIN_REMOVED_REL = 0.45  # relative to (gap_min*w_est)
SOFT_ERASE_MAX_CANDS = 10

CUTLEN_ABS_MIN = 14
CUTLEN_REL_GAP = 1.15
CUTLEN_ETA = 0.07
CUTLEN_MAX_FRAC = 0.42

# BG seed stroke
BG_SEED_LEN = 80
BG_SEED_THICK = 1
BG_SEED_CLASS_ID = 1
BG_SEED_MIN_DIST = 10
BG_SEED_TRIES = 1500

TOP_CC_PER_REGION = 14
TOP_FG_REGIONS_TO_TRY = 28
TOP_BG_REGIONS_TO_TRY = 16

# plots / debug
MAKE_PLOTS = True
KDE_GRID = 512
KDE_BW_MIN = 1e-3

DEBUG_SAVE_DIFF_OVERLAY = False
DEBUG_OVERLAY_MAX_PER_RATIO = 120

DEBUG_SAVE_ZOOM = False
DEBUG_ZOOM_MAX_PER_RATIO = 120
ZOOM_PAD = 40
ZOOM_SCALE = 8
ZOOM_MIN_BOX = 56

NO_SCRIB = 255  # VOC idx


# ============================================================
# 1) Import scoring module (EXACT)
# ============================================================
SCORE_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if SCORE_MODULE_DIR not in sys.path:
    sys.path.append(SCORE_MODULE_DIR)

try:
    import fluency_score_6 as S
except Exception as e:
    raise RuntimeError("Failed to import fluency_score_6.py. Check SCORE_MODULE_DIR.") from e


# ============================================================
# 2) IO helpers
# ============================================================
def list_common_png_files(gt_dir: str, scrib_dir: str) -> List[str]:
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(".png")}
    sc_files = {f for f in os.listdir(scrib_dir) if f.lower().endswith(".png")}
    return sorted(list(gt_files.intersection(sc_files)))


def ensure_dir(p: str):
    if not p:
        return
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_idx_png(path: str, arr: np.ndarray):
    # production: faster than PIL for massive small PNG writes
    ok, buf = cv2.imencode(".png", arr.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ============================================================
# 3) KDE/plots (include Target curve)
# ============================================================
def _kde_1d(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return dens


def _bw_silverman(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.05
    std = float(np.std(v, ddof=1))
    bw = 1.06 * std * (v.size ** (-1.0 / 5.0))
    return max(bw, KDE_BW_MIN)


def plot_kde_before_after_target(before: np.ndarray, after: np.ndarray, target: np.ndarray,
                                 title: str, out_path: str,
                                 xlim: Tuple[float, float] = (0.0, 1.0)):
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    lo, hi = xlim
    grid = np.linspace(lo, hi, KDE_GRID)
    db = _kde_1d(before, grid, _bw_silverman(before))
    da = _kde_1d(after, grid, _bw_silverman(after))
    dt = _kde_1d(target, grid, _bw_silverman(target))

    plt.figure()
    plt.plot(grid, db, label="Before")
    plt.plot(grid, da, label="After")
    plt.plot(grid, dt, label="Target")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter_target_band(before: np.ndarray, after: np.ndarray,
                             ratio: float, tol: float,
                             status: Optional[List[str]],
                             out_path: str):
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    x_min = float(min(before.min(), after.min()))
    x_max = float(max(before.max(), after.max()))
    xs = np.linspace(x_min, x_max, 256)

    plt.figure()
    if status is None:
        plt.scatter(before, after, s=12, alpha=0.8)
    else:
        status = np.asarray(status)
        ok = (status == "OK")
        plt.scatter(before[ok], after[ok], s=12, alpha=0.85, label="OK")
        plt.scatter(before[~ok], after[~ok], s=18, alpha=0.95, label="FAIL")
        plt.legend()

    plt.plot(xs, xs, linestyle="--", linewidth=1)
    plt.plot(xs, ratio * xs, linewidth=1)
    plt.plot(xs, ratio * xs + tol, linestyle=":", linewidth=1)
    plt.plot(xs, ratio * xs - tol, linestyle=":", linewidth=1)

    plt.xlabel("before")
    plt.ylabel("after")
    plt.title(f"score_total before->after (ratio={ratio:.1f}, tol={tol:.2f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_target_error_kde(after: np.ndarray, target: np.ndarray, tol: float, out_path: str):
    after = np.asarray(after, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = after - target

    lo = float(min(err.min(), -tol * 2.5))
    hi = float(max(err.max(), +tol * 2.5))
    grid = np.linspace(lo, hi, KDE_GRID)

    bw = max(_bw_silverman(err), KDE_BW_MIN)
    de = _kde_1d(err, grid, bw)

    plt.figure()
    plt.plot(grid, de)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axvline(+tol, linestyle=":", linewidth=1)
    plt.axvline(-tol, linestyle=":", linewidth=1)
    plt.xlabel("after - target")
    plt.ylabel("Density")
    plt.title("target error KDE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ============================================================
# 4) Internal mapping for exact scoring
# ============================================================
def scribble_to_internal(scrib_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    orig0_mask = (scrib_orig == 0)
    scrib_int = scrib_orig.copy()
    scrib_int[orig0_mask] = 1
    return scrib_int, orig0_mask


def internal_to_scribble(scrib_int: np.ndarray, orig0_mask: np.ndarray) -> np.ndarray:
    out = scrib_int.copy()
    restore0 = orig0_mask & (out != NO_SCRIB)
    out[restore0] = 0
    return out


# ============================================================
# 5) GT regions + exact score wrapper
# ============================================================
def build_gt_regions_cached(gt_path: str):
    gt = S.read_gt_auto(gt_path)
    if gt.kind == "IDX":
        fg_regions, bg_regions = S.build_gt_regions_from_idx(gt.data, S.GT_IDX_BG_VALUE, S.GT_IDX_IGNORE_VALUE)
    else:
        fg_regions, bg_regions = S.build_gt_regions_from_rgb(gt.data, S.GT_BG_COLOR_RGB, S.GT_IGNORE_COLOR_RGB)
    return fg_regions, bg_regions


def exact_score_from_regions(
    fg_regions: Dict[int, np.ndarray],
    bg_regions: Dict[int, np.ndarray],
    scrib_int: np.ndarray,
    name_for_compute: str = "tmp",
) -> Tuple[float, float, float, float, Dict[Tuple[str, int], float]]:
    scrib_cc = S.build_scribble_cc_idx(scrib_int)
    assign_res = S.assign_scribbles(scrib_cc, fg_regions, bg_regions)
    k_map = assign_res.k_map

    ret = S.compute_score(name_for_compute, fg_regions, bg_regions, k_map)
    score_fg = float(ret[0])
    score_bg = float(ret[1])
    score_total = float(ret[2])

    fg_total_area = 0
    fg_cov_area = 0
    for rid, m in fg_regions.items():
        area = int(m.sum())
        fg_total_area += area
        if float(k_map.get(("FG", rid), 0.0)) > 0.0:
            fg_cov_area += area
    fg_cov_ratio = (float(fg_cov_area) / float(fg_total_area)) if fg_total_area > 0 else 0.0
    return score_fg, score_bg, score_total, fg_cov_ratio, k_map


# ============================================================
# 6) CC label + overlap assignment
# ============================================================
def label_scribble_cc(scrib_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (scrib_int != NO_SCRIB).astype(np.uint8)
    num, lab = cv2.connectedComponents(mask, connectivity=8)
    areas = np.bincount(lab.reshape(-1), minlength=num).astype(np.int64)
    return lab.astype(np.int32), areas


def assign_cc_to_regions_by_overlap(
    lab: np.ndarray,
    areas: np.ndarray,
    fg_regions: Dict[int, np.ndarray],
    bg_regions: Dict[int, np.ndarray],
) -> Dict[int, Tuple[str, int, float]]:
    mapping: Dict[int, Tuple[str, int, float]] = {}
    fg_list = [(rid, fg_regions[rid].astype(bool)) for rid in fg_regions.keys()]
    bg_list = [(rid, bg_regions[rid].astype(bool)) for rid in bg_regions.keys()]
    max_id = areas.shape[0] - 1

    for cc_id in range(1, max_id + 1):
        a = int(areas[cc_id])
        if a <= 0:
            continue
        cc_mask = (lab == cc_id)

        best_t, best_rid, best_ratio = None, None, -1.0

        for rid, rm in fg_list:
            inter = int(np.logical_and(cc_mask, rm).sum())
            if inter <= 0:
                continue
            ratio = inter / float(a)
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = "FG"
                best_rid = rid

        for rid, rm in bg_list:
            inter = int(np.logical_and(cc_mask, rm).sum())
            if inter <= 0:
                continue
            ratio = inter / float(a)
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = "BG"
                best_rid = rid

        if best_t is not None:
            mapping[cc_id] = (best_t, int(best_rid), float(best_ratio))

    return mapping


def get_cc_ids_in_region_sorted(
    region_type: str,
    region_id: int,
    cc_map: Dict[int, Tuple[str, int, float]],
    areas: np.ndarray,
    topk: int,
    min_cc_pixels: int
) -> List[int]:
    ids = []
    for cc_id, (t, rid, _r) in cc_map.items():
        if t == region_type and rid == region_id:
            a = int(areas[cc_id])
            if a >= min_cc_pixels:
                ids.append((cc_id, a))
    ids.sort(key=lambda x: -x[1])
    return [int(x[0]) for x in ids[:topk]]


# ============================================================
# 7) split/cut utils + candidate generation
# ============================================================
def topk_points_by_distance(mask01: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    dist = cv2.distanceTransform((mask01 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return []
    flat = dist.reshape(-1)
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    h, w = mask01.shape
    pts = []
    for i in idx:
        d = float(flat[i])
        if d <= 0:
            break
        y = int(i // w)
        x = int(i % w)
        pts.append((y, x, d))
    return pts


def split_two_largest_components(mask01: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    num, lab = cv2.connectedComponents((mask01 > 0).astype(np.uint8), connectivity=8)
    if num <= 2:
        return None
    areas = np.bincount(lab.reshape(-1), minlength=num).astype(np.int64)
    ids = np.argsort(areas[1:])[::-1] + 1
    a_id, b_id = int(ids[0]), int(ids[1])
    A = (lab == a_id)
    B = (lab == b_id)
    if int(A.sum()) < MIN_PART_PIXELS or int(B.sum()) < MIN_PART_PIXELS:
        return None
    return A, B


def min_distance_between_components(A: np.ndarray, B: np.ndarray) -> float:
    img = np.ones(A.shape, dtype=np.uint8)
    img[A] = 0
    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    vals = dt[B]
    if vals.size == 0:
        return float("inf")
    return float(vals.min())


def estimate_tangent_direction(mask01: np.ndarray, y: int, x: int) -> Tuple[float, float]:
    H, W = mask01.shape
    r = 10
    y1, y2 = max(0, y - r), min(H, y + r + 1)
    x1, x2 = max(0, x - r), min(W, x + r + 1)
    win = mask01[y1:y2, x1:x2]
    ys, xs = np.where(win > 0)
    if len(xs) < 10:
        return (1.0, 0.0)
    xs = xs.astype(np.float32) + x1
    ys = ys.astype(np.float32) + y1
    pts = np.stack([xs, ys], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = (pts.T @ pts) / max(1, pts.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    vx, vy = float(v[0]), float(v[1])
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-6:
        return (1.0, 0.0)
    return (vx / n, vy / n)


def erase_capsule_restricted(
    scrib_int: np.ndarray,
    y: int, x: int,
    vx: float, vy: float,
    half_len: int, radius: int,
    cc_mask: np.ndarray,
    region_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
    out = scrib_int.copy()
    H, W = out.shape
    x1 = int(round(x - vx * half_len))
    y1 = int(round(y - vy * half_len))
    x2 = int(round(x + vx * half_len))
    y2 = int(round(y + vy * half_len))

    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))

    cap = np.zeros((H, W), dtype=np.uint8)
    cv2.line(cap, (x1, y1), (x2, y2), color=1, thickness=int(2 * radius + 1))

    m = (cap > 0) & cc_mask & region_mask & (out != NO_SCRIB)
    removed = int(m.sum())
    if removed > 0:
        out[m] = NO_SCRIB
    return out, removed


def adaptive_min_cc_pixels(ratio: float) -> int:
    v = int(round(MIN_CC_PIXELS_BASE * (0.35 + 0.65 * ratio)))
    return int(max(22, min(MIN_CC_PIXELS_BASE, v)))


def propose_candidates_for_cc(
    scrib_int: np.ndarray,
    lab: np.ndarray,
    cc_id: int,
    region_mask: np.ndarray,
    ratio: float
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Return candidate list: include split-candidates (primary),
    plus soft-erase candidates (fallback) when split candidates are scarce.
    """
    cc_mask = (lab == cc_id)
    inside = np.logical_and(cc_mask, region_mask).astype(np.uint8)
    inside_area = int(inside.sum())
    if inside_area < adaptive_min_cc_pixels(ratio):
        return []

    pts = topk_points_by_distance(inside, TOPK_POINTS)
    if not pts:
        return []

    cands: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    # micro mode for high ratios: prevent overshoot
    micro_mode = (ratio >= 0.75)

    # alpha lists
    alpha_list = list(ALPHAS)
    if ratio <= 0.2:
        alpha_list += [4.0, 4.5]  # aggressive for very low ratio
    if micro_mode:
        alpha_list = MICRO_ALPHAS + alpha_list  # micro first

    for (y, x, d) in pts[: max(10, MAX_CANDS_PER_CC)]:
        w_est = max(1.0, 2.0 * float(d))
        gap_min_px = max(float(GAP_MIN_ABS_PX), float(GAP_MIN_REL) * w_est)

        # estimate stroke length
        L_est = float(inside_area) / w_est

        cut_len = max(
            float(CUTLEN_ABS_MIN),
            float(CUTLEN_REL_GAP) * gap_min_px,
            float(CUTLEN_ETA) * L_est
        )
        cut_len = min(cut_len, float(CUTLEN_MAX_FRAC) * L_est)
        half_len = int(np.ceil(cut_len / 2.0))

        if micro_mode:
            half_len = max(6, int(round(half_len * MICRO_HALFLEN_SCALE)))

        vx, vy = estimate_tangent_direction(inside, y, x)

        # removed threshold
        removed_min = max(
            float(REMOVED_STROKE_PIX_MIN_ABS),
            float(REMOVED_STROKE_PIX_MIN_REL) * gap_min_px * w_est
        )
        if micro_mode:
            removed_min = max(18.0, removed_min * 0.45)  # allow smaller changes to avoid overshoot
        if ratio <= 0.2:
            removed_min = max(35.0, removed_min * 0.70)

        # ---------- (1) split candidates ----------
        split_added = 0
        for alpha in alpha_list:
            r = int(np.ceil(alpha * d))
            if alpha in MICRO_ALPHAS:
                r = max(r, MICRO_R_MIN)
            r = max(r, R_MIN)
            r = min(r, R_MAX)

            cand, removed = erase_capsule_restricted(
                scrib_int, y, x, vx, vy,
                half_len=half_len, radius=r,
                cc_mask=cc_mask, region_mask=region_mask
            )
            if removed < removed_min:
                continue

            after_stroke = np.logical_and((cand != NO_SCRIB), cc_mask & region_mask).astype(np.uint8)
            two = split_two_largest_components(after_stroke)
            if two is None:
                continue
            A, B = two
            md = min_distance_between_components(A, B)
            if md < gap_min_px:
                continue

            meta = {
                "kind": "split",
                "y": int(y), "x": int(x),
                "d": float(d),
                "radius": int(r),
                "half_len": int(half_len),
                "removed": int(removed),
                "gap_min_px": float(gap_min_px),
                "md": float(md),
            }
            cands.append((cand, meta))
            split_added += 1
            if len(cands) >= MAX_CANDS_PER_CC:
                return cands

        # ---------- (2) soft erase fallback (only if no split was found at this center) ----------
        if SOFT_ERASE_ENABLE and split_added == 0 and (ratio <= 0.55):
            # allow soft erases especially for low ratios where split constraint blocks candidates
            soft_removed_min = max(
                float(SOFT_ERASE_MIN_REMOVED_ABS),
                float(SOFT_ERASE_MIN_REMOVED_REL) * gap_min_px * w_est
            )
            # for very low ratio, be more aggressive
            if ratio <= 0.2:
                soft_removed_min *= 0.8

            # try a few radii (bigger usually helps to reduce score)
            soft_radii = [min(R_MAX, max(R_MIN, int(np.ceil(a * d)))) for a in ([2.5, 3.0, 3.5, 4.0] if ratio <= 0.2 else [2.0, 2.5, 3.0])]
            soft_radii = sorted(list(set(soft_radii)))

            for r in soft_radii[:SOFT_ERASE_MAX_CANDS]:
                cand, removed = erase_capsule_restricted(
                    scrib_int, y, x, vx, vy,
                    half_len=max(10, half_len), radius=r,
                    cc_mask=cc_mask, region_mask=region_mask
                )
                if removed < soft_removed_min:
                    continue
                meta = {
                    "kind": "soft",
                    "y": int(y), "x": int(x),
                    "d": float(d),
                    "radius": int(r),
                    "half_len": int(max(10, half_len)),
                    "removed": int(removed),
                    "gap_min_px": float(gap_min_px),
                    "md": -1.0,
                }
                cands.append((cand, meta))
                if len(cands) >= MAX_CANDS_PER_CC:
                    return cands

    return cands


def choose_best_candidate(
    cands: List[Tuple[np.ndarray, Dict[str, Any]]],
    score_fn,
    target_total: float,
    tol: float,
    ratio: float,
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float, Any], float, Dict[str, Any]]]:
    """
    Selection strategy:
    - Prefer candidates with st >= target - tol (avoid overshoot)
    - For high ratios (>=0.75): be stricter against overshoot
      * only accept overshoot if within (target - 2*tol) AND improves gap a lot
    - Among same group: minimal gap
    """
    best_safe = None
    best_safe_gap = None
    best_safe_scores = None
    best_safe_meta = None

    best_ov = None
    best_ov_gap = None
    best_ov_scores = None
    best_ov_meta = None

    strict_overshoot = (ratio >= 0.75)

    for cand, meta in cands:
        sfg2, sbg2, st2, cov2, k2 = score_fn(cand)
        gap2 = abs(float(st2) - float(target_total))
        safe = (float(st2) >= float(target_total) - float(tol))

        if safe:
            if (best_safe is None) or (gap2 < best_safe_gap - 1e-12):
                best_safe = cand
                best_safe_gap = gap2
                best_safe_scores = (sfg2, sbg2, st2, cov2, k2)
                best_safe_meta = meta
        else:
            if strict_overshoot:
                # keep only mild overshoot candidates
                if float(st2) < float(target_total) - float(2.0 * tol):
                    continue
            if (best_ov is None) or (gap2 < best_ov_gap - 1e-12):
                best_ov = cand
                best_ov_gap = gap2
                best_ov_scores = (sfg2, sbg2, st2, cov2, k2)
                best_ov_meta = meta

    if best_safe is not None:
        return best_safe, best_safe_scores, float(best_safe_gap), best_safe_meta
    if best_ov is not None:
        return best_ov, best_ov_scores, float(best_ov_gap), best_ov_meta
    return None


# ============================================================
# 8) BG schedule + seed stroke
# ============================================================
def bg_breaks_required(ratio: float) -> int:
    if ratio >= 0.7:
        return 0
    if ratio >= 0.6:
        return 1
    if ratio >= 0.5:
        return 2
    if ratio >= 0.4:
        return 3
    return 4


def find_bg_seed_point(bg_regions: Dict[int, np.ndarray], scrib_int: np.ndarray, rng: np.random.Generator) -> Optional[Tuple[int, int, int]]:
    if not bg_regions:
        return None
    bg_items = [(rid, int(m.sum())) for rid, m in bg_regions.items()]
    bg_items.sort(key=lambda t: -t[1])
    top_rids = [rid for rid, _ in bg_items[: min(6, len(bg_items))]]

    scrib_mask = (scrib_int != NO_SCRIB).astype(np.uint8)
    inv = (scrib_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    for _ in range(BG_SEED_TRIES):
        rid = int(rng.choice(top_rids))
        m = bg_regions[rid]
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        j = int(rng.integers(0, len(ys)))
        y, x = int(ys[j]), int(xs[j])
        if scrib_int[y, x] != NO_SCRIB:
            continue
        if dist[y, x] < float(BG_SEED_MIN_DIST):
            continue
        return rid, y, x
    return None


def draw_bg_seed_stroke(scrib_int: np.ndarray, y: int, x: int, bg_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = scrib_int.copy()
    h, w = out.shape
    ang = float(rng.uniform(0, 2*np.pi))
    dx = int(round(np.cos(ang) * BG_SEED_LEN / 2))
    dy = int(round(np.sin(ang) * BG_SEED_LEN / 2))

    x1, y1 = x - dx, y - dy
    x2, y2 = x + dx, y + dy
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))

    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.line(tmp, (x1, y1), (x2, y2), color=1, thickness=BG_SEED_THICK)

    m = (tmp > 0) & bg_mask & (out == NO_SCRIB)
    out[m] = np.uint8(BG_SEED_CLASS_ID)
    return out


def ensure_bg_has_cuttable_stroke(bg_regions: Dict[int, np.ndarray], scrib_int: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lab, areas = label_scribble_cc(scrib_int)
    cc_map = assign_cc_to_regions_by_overlap(lab, areas, fg_regions={}, bg_regions=bg_regions)
    any_bg = any(t == "BG" for (t, _rid, _r) in cc_map.values())
    if any_bg:
        return scrib_int

    p = find_bg_seed_point(bg_regions, scrib_int, rng)
    if p is None:
        return scrib_int
    rid, y, x = p
    bg_mask = bg_regions[rid].astype(bool)
    return draw_bg_seed_stroke(scrib_int, y, x, bg_mask, rng)


# ============================================================
# 9) FG region priority
# ============================================================
def pick_fg_regions_priority(fg_regions: Dict[int, np.ndarray], k_map: Dict[Tuple[str, int], float]) -> List[int]:
    items = []
    for rid, m in fg_regions.items():
        area = int(m.sum())
        if area < 80:
            continue
        k = float(k_map.get(("FG", rid), 0.0))
        covered = 1 if k > 0 else 0
        items.append((rid, covered, area, k))
    items.sort(key=lambda t: (-t[1], -t[2], t[3]))
    return [int(t[0]) for t in items]


# ============================================================
# 10) Debug overlay / zoom
# ============================================================
def save_diff_overlay_and_mask(prefix_path: str, before_orig: np.ndarray, after_orig: np.ndarray):
    diff = (before_orig != after_orig)
    vis = np.stack([before_orig, before_orig, before_orig], axis=2).astype(np.uint8)
    vis[diff] = (255, 0, 0)
    Image.fromarray(vis).save(prefix_path + "__diff_overlay.png")
    mask = (diff.astype(np.uint8) * 255)
    Image.fromarray(mask).save(prefix_path + "__diff_mask.png")


def save_zoom_crop_white_overlay(prefix_path: str, before_orig: np.ndarray, after_orig: np.ndarray):
    diff = (before_orig != after_orig)
    ys, xs = np.where(diff)
    if ys.size == 0:
        return
    H, W = before_orig.shape
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    y1 = max(0, y1 - ZOOM_PAD); y2 = min(H - 1, y2 + ZOOM_PAD)
    x1 = max(0, x1 - ZOOM_PAD); x2 = min(W - 1, x2 + ZOOM_PAD)

    if (y2 - y1 + 1) < ZOOM_MIN_BOX:
        mid = (y1 + y2) // 2
        half = ZOOM_MIN_BOX // 2
        y1 = max(0, mid - half)
        y2 = min(H - 1, y1 + ZOOM_MIN_BOX - 1)
    if (x2 - x1 + 1) < ZOOM_MIN_BOX:
        mid = (x1 + x2) // 2
        half = ZOOM_MIN_BOX // 2
        x1 = max(0, mid - half)
        x2 = min(W - 1, x1 + ZOOM_MIN_BOX - 1)

    b = before_orig[y1:y2+1, x1:x2+1]
    a = after_orig[y1:y2+1, x1:x2+1]
    d = diff[y1:y2+1, x1:x2+1]

    overlay = np.full((b.shape[0], b.shape[1], 3), 255, dtype=np.uint8)
    b_st = (b != NO_SCRIB)
    a_st = (a != NO_SCRIB)

    overlay[b_st] = (0, 0, 0)
    overlay[a_st] = (60, 60, 60)
    overlay[d] = (255, 0, 0)

    def up_gray(im: np.ndarray) -> Image.Image:
        pil = Image.fromarray(im.astype(np.uint8))
        return pil.resize((pil.size[0]*ZOOM_SCALE, pil.size[1]*ZOOM_SCALE), resample=Image.NEAREST)

    def up_rgb(im: np.ndarray) -> Image.Image:
        pil = Image.fromarray(im.astype(np.uint8))
        return pil.resize((pil.size[0]*ZOOM_SCALE, pil.size[1]*ZOOM_SCALE), resample=Image.NEAREST)

    up_gray(b).save(prefix_path + "__zoom_before.png")
    up_gray(a).save(prefix_path + "__zoom_after.png")
    up_rgb(overlay).save(prefix_path + "__zoom_overlay.png")


# ============================================================
# 11) Core per-image generation
# ============================================================
def generate_noisy_for_ratio_one_image(
    fg_regions: Dict[int, np.ndarray],
    bg_regions: Dict[int, np.ndarray],
    scrib0_int: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    name: str
):
    import time

    # -----------------------------
    # Adaptive budgets (speed)
    # -----------------------------
    r = float(f"{ratio:.1f}")  # stable key

    # Time budget (sec) per image per ratio (tune if needed)
    RATIO_TIME_BUDGET_SEC = {
        0.9: 1.2,
        0.8: 1.6,
        0.7: 2.0,
        0.6: 2.6,
        0.5: 4.3,
        0.4: 3.6,
        0.3: 4.0,
        0.2: 4.4,
        0.1: 4.8,
    }
    t_start = time.time()
    t_budget = RATIO_TIME_BUDGET_SEC.get(r, 3.0)

    def time_exceeded() -> bool:
        return (time.time() - t_start) > t_budget

    # Reduce heavy search for high ratios (local override of globals)
    # These are used by propose_candidates_for_cc / outer loops in this function.
    global TOPK_POINTS, MAX_CANDS_PER_CC, MAX_ITERS_FG, MAX_ITERS_BG_FINE

    old_TOPK = TOPK_POINTS
    old_MAXC = MAX_CANDS_PER_CC
    old_MIFG = MAX_ITERS_FG
    old_MIBG = MAX_ITERS_BG_FINE

    # Ratio-adaptive search strength (safe defaults)
    if r >= 0.8:
        TOPK_POINTS = min(TOPK_POINTS, 50)
        MAX_CANDS_PER_CC = min(MAX_CANDS_PER_CC, 5)
        MAX_ITERS_FG = min(MAX_ITERS_FG, 70)
        MAX_ITERS_BG_FINE = min(MAX_ITERS_BG_FINE, 35)
        no_improve_limit_fg = 6
        no_improve_limit_bg = 4
    elif r >= 0.6:
        TOPK_POINTS = min(TOPK_POINTS, 70)
        MAX_CANDS_PER_CC = min(MAX_CANDS_PER_CC, 7)
        MAX_ITERS_FG = min(MAX_ITERS_FG, 110)
        MAX_ITERS_BG_FINE = min(MAX_ITERS_BG_FINE, 50)
        no_improve_limit_fg = 7
        no_improve_limit_bg = 5
    else:
        TOPK_POINTS = min(TOPK_POINTS, 90)
        MAX_CANDS_PER_CC = min(MAX_CANDS_PER_CC, 9)
        MAX_ITERS_FG = min(MAX_ITERS_FG, 140)
        MAX_ITERS_BG_FINE = min(MAX_ITERS_BG_FINE, 60)
        no_improve_limit_fg = 8
        no_improve_limit_bg = 6

    # Helper: restore globals before return
    def _restore_globals():
        nonlocal old_TOPK, old_MAXC, old_MIFG, old_MIBG
        TOPK_POINTS = old_TOPK
        MAX_CANDS_PER_CC = old_MAXC
        MAX_ITERS_FG = old_MIFG
        MAX_ITERS_BG_FINE = old_MIBG

    try:
        sfg0, sbg0, st0, cov0, k0 = exact_score_from_regions(
            fg_regions, bg_regions, scrib0_int, name_for_compute=name
        )
        target_total = st0 * ratio

        scrib = scrib0_int.copy()
        sfg, sbg, st, cov, k_map = sfg0, sbg0, st0, cov0, k0

        best = scrib.copy()
        best_gap = abs(st - target_total)
        best_scores = (sfg, sbg, st, cov)

        bg_need = bg_breaks_required(ratio)
        bg_done = 0
        fg_done = 0
        bg_fine_done = 0

        if st < target_total - TOL:
            return False, best, (sfg0, sbg0, st0, cov0), best_scores, target_total, bg_need, bg_done, fg_done, bg_fine_done, \
                   "target_unreachable(st<target)"

        def score_fn_local(arr: np.ndarray):
            return exact_score_from_regions(fg_regions, bg_regions, arr, name_for_compute=name)

        # -----------------------------
        # BG coarse schedule
        # -----------------------------
        if bg_need > 0:
            if time_exceeded():
                final_gap = abs(best_scores[2] - target_total)
                ok = (final_gap <= TOL)
                return ok, best, (sfg0, sbg0, st0, cov0), best_scores, target_total, bg_need, bg_done, fg_done, bg_fine_done, "TIME_BUDGET(bg_coarse_init)"

            scrib = ensure_bg_has_cuttable_stroke(bg_regions, scrib, rng)
            sfg, sbg, st, cov, k_map = score_fn_local(scrib)

            for _ in range(bg_need):
                if time_exceeded():
                    break

                lab, areas = label_scribble_cc(scrib)
                cc_map = assign_cc_to_regions_by_overlap(lab, areas, fg_regions={}, bg_regions=bg_regions)

                bg_rids = sorted(list(bg_regions.keys()), key=lambda rid: -int(bg_regions[rid].sum()))
                accepted = False

                for rid in bg_rids[: min(TOP_BG_REGIONS_TO_TRY, len(bg_rids))]:
                    if time_exceeded():
                        break

                    region_mask = bg_regions[rid].astype(bool)
                    cc_ids = get_cc_ids_in_region_sorted(
                        "BG", rid, cc_map, areas,
                        topk=TOP_CC_PER_REGION,
                        min_cc_pixels=adaptive_min_cc_pixels(ratio)
                    )
                    if not cc_ids:
                        continue

                    gap_cur = abs(st - target_total)
                    for cc_id in cc_ids:
                        if time_exceeded():
                            break

                        cands = propose_candidates_for_cc(scrib, lab, cc_id, region_mask, ratio)
                        if not cands:
                            continue

                        chosen = choose_best_candidate(cands, score_fn_local, target_total, TOL, ratio)
                        if chosen is None:
                            continue
                        cand_best, scores_best, gap_best, _meta = chosen

                        # EARLY HIT
                        if gap_best <= TOL:
                            scrib = cand_best
                            sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                            bg_done += 1
                            accepted = True
                            best_gap = gap_best
                            best = scrib.copy()
                            best_scores = (sfg, sbg, st, cov)
                            break

                        if gap_best < gap_cur - GAP_EPS:
                            scrib = cand_best
                            sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                            bg_done += 1
                            accepted = True

                            if gap_best < best_gap:
                                best_gap = gap_best
                                best = scrib.copy()
                                best_scores = (sfg, sbg, st, cov)
                            break

                    if accepted:
                        break

                if not accepted:
                    if time_exceeded():
                        break
                    scrib = ensure_bg_has_cuttable_stroke(bg_regions, scrib, rng)
                    sfg, sbg, st, cov, k_map = score_fn_local(scrib)

        # -----------------------------
        # FG greedy main
        # -----------------------------
        fg_stuck = False
        no_improve_fg = 0

        for _it in range(MAX_ITERS_FG):
            if time_exceeded():
                break

            gap_cur = abs(st - target_total)
            if gap_cur <= TOL:
                best = scrib.copy()
                best_scores = (sfg, sbg, st, cov)
                best_gap = gap_cur
                break
            if st < target_total - TOL:
                break

            lab, areas = label_scribble_cc(scrib)
            cc_map = assign_cc_to_regions_by_overlap(lab, areas, fg_regions=fg_regions, bg_regions=bg_regions)
            fg_rids = pick_fg_regions_priority(fg_regions, k_map)

            accepted = False
            for rid in fg_rids[: min(TOP_FG_REGIONS_TO_TRY, len(fg_rids))]:
                if time_exceeded():
                    break

                region_mask = fg_regions[rid].astype(bool)
                cc_ids = get_cc_ids_in_region_sorted(
                    "FG", rid, cc_map, areas,
                    topk=TOP_CC_PER_REGION,
                    min_cc_pixels=adaptive_min_cc_pixels(ratio)
                )
                if not cc_ids:
                    continue

                for cc_id in cc_ids:
                    if time_exceeded():
                        break

                    cands = propose_candidates_for_cc(scrib, lab, cc_id, region_mask, ratio)
                    if not cands:
                        continue

                    chosen = choose_best_candidate(cands, score_fn_local, target_total, TOL, ratio)
                    if chosen is None:
                        continue
                    cand_best, scores_best, gap_best, _meta = chosen

                    # EARLY HIT
                    if gap_best <= TOL:
                        scrib = cand_best
                        sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                        fg_done += 1
                        accepted = True
                        best_gap = gap_best
                        best = scrib.copy()
                        best_scores = (sfg, sbg, st, cov)
                        break

                    if gap_best < gap_cur - GAP_EPS:
                        scrib = cand_best
                        sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                        fg_done += 1
                        accepted = True

                        if gap_best < best_gap:
                            best_gap = gap_best
                            best = scrib.copy()
                            best_scores = (sfg, sbg, st, cov)
                        break

                if accepted:
                    break

            if not accepted:
                no_improve_fg += 1
                if no_improve_fg >= no_improve_limit_fg:
                    fg_stuck = True
                    break
            else:
                no_improve_fg = 0

        # -----------------------------
        # BG fine tune
        # -----------------------------
        no_improve_bg = 0

        for _it in range(MAX_ITERS_BG_FINE):
            if time_exceeded():
                break

            gap_cur = abs(st - target_total)
            if gap_cur <= TOL:
                best = scrib.copy()
                best_scores = (sfg, sbg, st, cov)
                best_gap = gap_cur
                break
            if st < target_total - TOL:
                break

            scrib = ensure_bg_has_cuttable_stroke(bg_regions, scrib, rng)
            sfg, sbg, st, cov, k_map = score_fn_local(scrib)

            gap_cur = abs(st - target_total)
            if gap_cur <= TOL:
                best = scrib.copy()
                best_scores = (sfg, sbg, st, cov)
                best_gap = gap_cur
                break

            lab, areas = label_scribble_cc(scrib)
            cc_map = assign_cc_to_regions_by_overlap(lab, areas, fg_regions=fg_regions, bg_regions=bg_regions)

            bg_rids = sorted(list(bg_regions.keys()), key=lambda rid: -int(bg_regions[rid].sum()))
            accepted = False

            for rid in bg_rids[: min(TOP_BG_REGIONS_TO_TRY, len(bg_rids))]:
                if time_exceeded():
                    break

                region_mask = bg_regions[rid].astype(bool)
                cc_ids = get_cc_ids_in_region_sorted(
                    "BG", rid, cc_map, areas,
                    topk=TOP_CC_PER_REGION,
                    min_cc_pixels=adaptive_min_cc_pixels(ratio)
                )
                if not cc_ids:
                    continue

                for cc_id in cc_ids:
                    if time_exceeded():
                        break

                    cands = propose_candidates_for_cc(scrib, lab, cc_id, region_mask, ratio)
                    if not cands:
                        continue

                    chosen = choose_best_candidate(cands, score_fn_local, target_total, TOL, ratio)
                    if chosen is None:
                        continue
                    cand_best, scores_best, gap_best, _meta = chosen

                    # EARLY HIT
                    if gap_best <= TOL:
                        scrib = cand_best
                        sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                        bg_fine_done += 1
                        accepted = True
                        best_gap = gap_best
                        best = scrib.copy()
                        best_scores = (sfg, sbg, st, cov)
                        break

                    if gap_best < gap_cur - GAP_EPS:
                        scrib = cand_best
                        sfg, sbg, st, cov = float(scores_best[0]), float(scores_best[1]), float(scores_best[2]), float(scores_best[3])
                        bg_fine_done += 1
                        accepted = True

                        if gap_best < best_gap:
                            best_gap = gap_best
                            best = scrib.copy()
                            best_scores = (sfg, sbg, st, cov)
                        break

                if accepted:
                    break

            if not accepted:
                no_improve_bg += 1
                if no_improve_bg >= no_improve_limit_bg:
                    break
            else:
                no_improve_bg = 0

        # -----------------------------
        # Finalize
        # -----------------------------
        final_gap = abs(best_scores[2] - target_total)
        ok = (final_gap <= TOL)

        if ok:
            reason = "OK"
        else:
            if time_exceeded():
                reason = "TIME_BUDGET"
            else:
                st_best = float(best_scores[2])
                if (fg_done == 0) and (bg_done == 0) and (bg_fine_done == 0):
                    reason = "no_effective_cut_found"
                elif st_best > target_total + TOL:
                    reason = "not_enough_noise(after_above_target)"
                elif st_best < target_total - TOL:
                    reason = "overshoot(after_below_target)"
                elif fg_stuck:
                    reason = "stuck_no_candidate_fg"
                else:
                    reason = "stuck_gap"

        return ok, best, (sfg0, sbg0, st0, cov0), best_scores, target_total, bg_need, bg_done, fg_done, bg_fine_done, reason

    finally:
        _restore_globals()
# ============================================================
# 12) Main
# ============================================================
def main():
    import time
    from datetime import datetime

    # ===== production switches =====
    global MAKE_PLOTS, DEBUG_SAVE_DIFF_OVERLAY, DEBUG_SAVE_ZOOM
    MAKE_PLOTS = False
    DEBUG_SAVE_DIFF_OVERLAY = False
    DEBUG_SAVE_ZOOM = False

    ensure_dir(OUT_ROOT)
    rng = np.random.default_rng(RANDOM_SEED)

    files = list_common_png_files(GT_DIR, SCRIB_DIR)
    if not files:
        raise RuntimeError("[ERROR] No common .png files between GT_DIR and SCRIB_DIR.")
    if LIMIT_N is not None:
        files = files[:int(LIMIT_N)]
    num_files = len(files)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] num_files={num_files} RATIOS={RATIOS}", flush=True)

    # ===== prepare per-ratio outputs =====
    ratio_cfg = {}
    for ratio in RATIOS:
        ratio_tag = f"ratio_{ratio:.1f}"
        out_dir = os.path.join(OUT_ROOT, ratio_tag)
        img_out_dir = os.path.join(out_dir, "noisy_scribbles")
        ensure_dir(img_out_dir)

        ratio_cfg[ratio] = {
            "ratio_tag": ratio_tag,
            "img_out_dir": img_out_dir,
            "report_csv": os.path.join(out_dir, f"report_{ratio_tag}.csv"),
            "rows": [],
            "ok_cnt": 0,
        }

    # ===== speed meter =====
    t0 = time.time()
    last_t = t0
    last_i = 0
    PROGRESS_EVERY = 100 if num_files >= 2000 else 25

    def fmt_eta(seconds: float) -> str:
        if seconds < 0 or not np.isfinite(seconds):
            return "NA"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h{m:02d}m"
        return f"{m:d}m{s:02d}s"

    # ===== main loop (IMAGE-MAJOR) =====
    for i, fn in enumerate(files, 1):
        name = os.path.splitext(fn)[0]
        gt_path = os.path.join(GT_DIR, fn)
        scrib_path = os.path.join(SCRIB_DIR, fn)

        # 1) regions once
        fg_regions, bg_regions = build_gt_regions_cached(gt_path)

        # 2) scribble once
        scrib_orig = np.array(Image.open(scrib_path), dtype=np.uint8)
        scrib0_int, orig0_mask = scribble_to_internal(scrib_orig)

        # 3) before score once (用于 fast-skip)
        sfg0, sbg0, st0, cov0, _k0 = exact_score_from_regions(
            fg_regions, bg_regions, scrib0_int, name_for_compute=name
        )

        # 4) all ratios
        for ratio in RATIOS:
            cfg = ratio_cfg[ratio]
            target = st0 * ratio

            # fast skip（尤其对 ratio=0.9 很省）
            if abs(st0 - target) <= TOL:
                ok = True
                best_int = scrib0_int
                sfg1, sbg1, st1, cov1 = sfg0, sbg0, st0, cov0
                bg_need = bg_done = fg_done = bg_fine = 0
                reason = "OK_fast_skip"
            else:
                ok, best_int, (_sfg0, _sbg0, _st0, _cov0), (sfg1, sbg1, st1, cov1), _target2, \
                    bg_need, bg_done, fg_done, bg_fine, reason = generate_noisy_for_ratio_one_image(
                        fg_regions, bg_regions, scrib0_int, ratio, rng, name=name
                    )

            scrib_out_orig = internal_to_scribble(best_int, orig0_mask)

            out_path = os.path.join(cfg["img_out_dir"], fn)
            save_idx_png(out_path, scrib_out_orig)

            status = "OK" if ok else "FAIL"
            if ok:
                cfg["ok_cnt"] += 1

            cfg["rows"].append([
                name, fn, f"{ratio:.1f}", status, reason,
                f"{target:.6f}",
                f"{sfg0:.6f}", f"{sbg0:.6f}", f"{st0:.6f}", f"{cov0:.6f}",
                f"{sfg1:.6f}", f"{sbg1:.6f}", f"{st1:.6f}", f"{cov1:.6f}",
                int(bg_need), int(bg_done), int(fg_done), int(bg_fine),
                f"{abs(st1 - target):.6f}",
            ])

        # 5) speed print
        if (i % PROGRESS_EVERY == 0) or (i == num_files):
            now = time.time()
            dt = now - last_t
            di = i - last_i
            sp_recent = di / dt if dt > 1e-9 else 0.0
            sp_avg = i / (now - t0) if (now - t0) > 1e-9 else 0.0
            eta = (num_files - i) / sp_avg if sp_avg > 1e-9 else float("inf")

            ok09 = None
            for r in RATIOS:
                if abs(r - 0.9) < 1e-9:
                    ok09 = ratio_cfg[r]["ok_cnt"]
                    break

            stamp = datetime.now().strftime("%H:%M:%S")
            if ok09 is not None:
                print(f"[{stamp}] [SPEED] {i}/{num_files} | recent={sp_recent:.2f} img/s | avg={sp_avg:.2f} img/s | ETA={fmt_eta(eta)} | ratio0.9_OK={ok09} ({ok09/i:.1%})",
                      flush=True)
            else:
                print(f"[{stamp}] [SPEED] {i}/{num_files} | recent={sp_recent:.2f} img/s | avg={sp_avg:.2f} img/s | ETA={fmt_eta(eta)}",
                      flush=True)

            last_t = now
            last_i = i

    # ===== write reports + summary =====
    summary_rows = []
    for ratio in RATIOS:
        cfg = ratio_cfg[ratio]
        write_csv(
            cfg["report_csv"],
            ["name", "file", "ratio", "status", "fail_reason",
             "target_total",
             "score_fg_before", "score_bg_before", "score_total_before", "fg_covered_ratio_before",
             "score_fg_after", "score_bg_after", "score_total_after", "fg_covered_ratio_after",
             "bg_break_required", "bg_break_done", "fg_break_done", "bg_fine_done",
             "abs(after-target)"],
            cfg["rows"]
        )

        success_rate = cfg["ok_cnt"] / float(num_files) if num_files > 0 else 0.0
        summary_rows.append([f"{ratio:.1f}", num_files, cfg["ok_cnt"], f"{success_rate:.4f}", cfg["report_csv"]])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [OK] ratio_{ratio:.1f}: {cfg['ok_cnt']}/{num_files} ({success_rate:.2%})",
              flush=True)

    summary_csv = os.path.join(OUT_ROOT, "summary_all_ratios.csv")
    write_csv(summary_csv, ["ratio", "num_images", "num_success", "success_rate", "report_csv"], summary_rows)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DONE] Summary: {summary_csv}", flush=True)

if __name__ == "__main__":
    print("[BOOT] entering main()")
    main()
    print("[BOOT] main() finished")