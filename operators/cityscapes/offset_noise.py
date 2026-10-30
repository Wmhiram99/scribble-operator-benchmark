from __future__ import annotations

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
try:
    import pandas as pd
except Exception:
    pd = None

# headless plotting on server
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ============================================================
# Paths (SERVER)
# ============================================================
BASE_DIR = os.environ.get("OFFSET_BASE_DIR", "datasets/cityscapes")
GT_DIR = os.path.join(BASE_DIR, "GtMask")
SCRIBBLE_DIR = os.path.join(BASE_DIR, "auto_scribbles_idx")

OUT_ROOT = os.environ.get("OFFSET_OUT_ROOT", os.path.join(os.getcwd(), "offset_auto_log"))
os.makedirs(OUT_ROOT, exist_ok=True)

EXP_NAME = "offset_auto_log"
SAVE_ROOT = os.path.join(OUT_ROOT, EXP_NAME)
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# Run control
# ============================================================
MAX_IMAGES = 0        # 0=full; e.g. 200 for debug
TOL = 0.05
RATIO_LIST = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]    # 先建议从 0.9 跑起，确认后再扩展到 [0.9,0.8,...]

# ------------------------------------------------------------
# Target mapping (keep pure-offset semantics, make low ratios reachable)
# ------------------------------------------------------------
# "legacy"            : target = score_before * ratio
# "reachable_linear"  : map ratio into [TARGET_EFF_RATIO_MIN, 1.0], then multiply by score_before
TARGET_MAPPING_MODE = "reachable_linear"
TARGET_EFF_RATIO_MIN = 0.75
TARGET_EFF_RATIO_GAMMA = 1.0

# ============================================================
# FAST mode knobs (speed up, keep quality)
# ============================================================
FAST_MODE = True
ASSUME_GRAY_RGB = False

# controller iters
MAX_ITERS = 2 if FAST_MODE else 3

ENABLE_HARDCASE_EXTRA_ITER = False if FAST_MODE else True
HARDCASE_EXTRA_DELTA_TH = 0.08
MAX_ITERS_EXTRA = 1

# reduce lever search (big speedup, minor quality impact)
TOPK_REGIONS = 5 if FAST_MODE else 10
PRUNE_LEVER_RATIO = 0.12  # slightly stronger prune

# fewer multipliers
MULTS_MILD = (0.8, 1.1) if FAST_MODE else (0.7, 1.0, 1.3)
MULTS_STRONG = (1.0, 1.6, 2.4) if FAST_MODE else (1.0, 1.35, 1.65, 2.0, 2.4)
STRONG_DELTA_TH = 0.18

GLOBAL_MAX_SHIFT = 320
MIN_COMP_AREA = 20
COMPONENT_SPLIT_MODE = "binary"  # binary | color
GT_COMPONENT_SPLIT_MODE = "binary"  # binary | color
SCORE_MATCH_COLOR = False

# dual-lever & momentum
ENABLE_DUAL_LEVER = True
SECOND_LEVER_SCALE_MILD = 0.60
SECOND_LEVER_SCALE_STRONG = 0.70

ENABLE_MOMENTUM = True
MOMENTUM_MULT_FAR = 1.15
MOMENTUM_MULT_NEAR = 1.05
MOMENTUM_NEAR_TH = 0.10
MOMENTUM_MAX_TRIES_PER_ITER = 1

ENABLE_NO_IMPROVE_GUARD = True
IMPROVE_EPS = 1e-8

# ============================================================
# Balanced diversity mode (Cityscapes-oriented offset control)
# ============================================================
ENABLE_BALANCED_DIVERSITY = False

# 1) priority de-head + random quota
BAL_PRIORITY_TAU = 0.55
BAL_MAX_LEVERS_PER_ITER = 6
BAL_TOPK_KEEP = 1
BAL_RANDOM_QUOTA = 3
BAL_WEIGHT_EPS = 1e-4
BALANCED_RANDOM_SEED = 20260424

# 2) per-scribble step/total cap
BAL_STEP_MIN_PX = 4
BAL_STEP_MAX_PX = 26
BAL_TOTAL_MAX_PX = 52
BAL_STEP_RADIUS_MULT = 0.28
BAL_TOTAL_RADIUS_MULT = 0.65
BAL_STEP_AREA_MULT = 0.22
BAL_TOTAL_AREA_MULT = 0.55
BAL_USE_TOTAL_CAP = True

# 3) coverage target (force more scribbles to participate)
BAL_COVERAGE_RATIO = 0.30
BAL_COVERAGE_MIN = 2
BAL_COVERAGE_MAX = 12
BAL_COVERAGE_BOOST = 2.20
BAL_COVERAGE_PENALTY = 0.45
BAL_NEW_LABEL_OBJ_SCALE = 0.93

# 4) cooldown on recently moved scribbles
BAL_COOLDOWN_ITERS = 2
BAL_COOLDOWN_FACTOR = 0.22

_BALANCED_RNG = None

# ============================================================
# Score settings (Power-calibrated ONS)
# ============================================================
EPS = 1e-6
OVERLAP_LOW = 0.05
OVERLAP_HIGH = 0.30

GAMMA = 2.2
RADIUS_SCALE = 1.25
SIM_FLOOR = 0.02

BG_RADIUS_K = 1.4
BG_RADIUS_MIN = 20.0
BG_RADIUS_MAX = 90.0

LAMBDA_MIN = 0.10
LAMBDA_MAX = 0.25

WHITE = np.array([255, 255, 255], np.uint8)
BLACK = np.array([0, 0, 0], np.uint8)
_COORD_CACHE = {}


# ============================================================
# Masks / helpers
# ============================================================
def _is_gray_rgb(img: np.ndarray) -> bool:
    """Fast path for Cityscapes trainId maps expanded to RGB."""
    if ASSUME_GRAY_RGB:
        return img.ndim == 3 and img.shape[-1] == 3
    return (
        img.ndim == 3
        and img.shape[-1] == 3
        and np.array_equal(img[..., 0], img[..., 1])
        and np.array_equal(img[..., 0], img[..., 2])
    )

def _fg_mask_against_color(img: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img != int(bg_color[0])
    if _is_gray_rgb(img):
        return img[..., 0] != int(bg_color[0])
    return np.any(img != bg_color.reshape(1, 1, 3), axis=-1)

def gt_fg_mask(gt_rgb: np.ndarray) -> np.ndarray:
    return _fg_mask_against_color(gt_rgb, BLACK)

def scribble_fg_mask(scribble_rgb: np.ndarray) -> np.ndarray:
    return _fg_mask_against_color(scribble_rgb, WHITE)

def smoothstep01(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)

def overlap_soft_weight(overlap: np.ndarray, low=OVERLAP_LOW, high=OVERLAP_HIGH) -> np.ndarray:
    t = (overlap - low) / (high - low + EPS)
    t = np.clip(t, 0.0, 1.0)
    return smoothstep01(t)

def connected_components(bin_mask: np.ndarray):
    bin_mask = (bin_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    return num, labels

def centers_and_areas_from_labels(labels: np.ndarray, num_labels: int):
    h, w = labels.shape
    flat = labels.reshape(-1)

    areas = np.bincount(flat, minlength=num_labels).astype(np.float64)
    key = (int(h), int(w))
    cached = _COORD_CACHE.get(key)
    if cached is None:
        ys = np.repeat(np.arange(h, dtype=np.float64), w)
        xs = np.tile(np.arange(w, dtype=np.float64), h)
        _COORD_CACHE[key] = (ys, xs)
    else:
        ys, xs = cached

    sum_y = np.bincount(flat, weights=ys, minlength=num_labels).astype(np.float64)
    sum_x = np.bincount(flat, weights=xs, minlength=num_labels).astype(np.float64)

    centers = np.zeros((num_labels, 2), np.float64)
    valid = areas > 0
    centers[valid, 0] = sum_y[valid] / (areas[valid] + EPS)
    centers[valid, 1] = sum_x[valid] / (areas[valid] + EPS)
    return centers, areas

def radius_from_center(mask: np.ndarray, center_yx: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 1.0
    dy = ys.astype(np.float64) - float(center_yx[0])
    dx = xs.astype(np.float64) - float(center_yx[1])
    r = np.sqrt(dy * dy + dx * dx).max()
    return float(max(r, 1.0))

def sim_from_dist_power(dist: float, radius: float) -> float:
    denom = (radius * RADIUS_SCALE) + EPS
    d = float(dist / denom)
    d = float(np.clip(d, 0.0, 1.0))
    sim = (1.0 - d) ** GAMMA
    return float(max(SIM_FLOOR, sim))

def clamp_shift(dy: int, dx: int, max_shift: int):
    dy = int(np.clip(dy, -max_shift, max_shift))
    dx = int(np.clip(dx, -max_shift, max_shift))
    if dy == 0 and dx == 0:
        dy = 1
    return dy, dx


# ============================================================
# Scribble components
# ============================================================
def component_stats(labels: np.ndarray, num_labels: int, label_start: int = 1, color_key=None):
    comps = []
    next_label = int(label_start)
    for lb in range(1, num_labels):
        ys, xs = np.where(labels == lb)
        if xs.size == 0:
            continue
        area = xs.size
        if area < MIN_COMP_AREA:
            continue
        cy, cx = ys.mean(), xs.mean()
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        comp = {
            "label": int(next_label),
            "area": int(area),
            "center": (float(cy), float(cx)),
            "bbox": (int(y0), int(x0), int(y1), int(x1)),
            "coords": (ys, xs),
        }
        if color_key is not None:
            comp["color_key"] = int(color_key)
        comps.append(comp)
        next_label += 1
    return comps

def labels_by_color(img_rgb: np.ndarray, bg_color: np.ndarray, min_area: int = 1):
    fg = _fg_mask_against_color(img_rgb, bg_color)
    h, w = img_rgb.shape[:2]
    out_labels = np.zeros((h, w), dtype=np.int32)
    color_by_label = {}
    next_label = 1
    if not np.any(fg):
        return 1, out_labels, color_by_label

    if img_rgb.ndim == 2:
        plane = img_rgb.astype(np.uint32)
        unique_keys = sorted(np.unique(plane[fg]).tolist())
        def _mask_for_key(color_key):
            return plane == np.uint32(color_key)
        def _store_key(color_key):
            v = int(color_key)
            return int((v << 16) | (v << 8) | v)
    elif _is_gray_rgb(img_rgb):
        plane = img_rgb[..., 0].astype(np.uint32)
        unique_keys = sorted(np.unique(plane[fg]).tolist())
        def _mask_for_key(color_key):
            return plane == np.uint32(color_key)
        def _store_key(color_key):
            v = int(color_key)
            return int((v << 16) | (v << 8) | v)
    else:
        arr = img_rgb.astype(np.uint32)
        packed = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
        unique_keys = sorted(np.unique(packed[fg]).tolist())
        def _mask_for_key(color_key):
            return packed == np.uint32(color_key)
        def _store_key(color_key):
            return int(color_key)

    for color_key in unique_keys:
        cmask = _mask_for_key(color_key).astype(np.uint8)
        num_c, labels_c = connected_components(cmask)
        for lb in range(1, num_c):
            region = labels_c == lb
            if int(region.sum()) < int(min_area):
                continue
            out_labels[region] = next_label
            color_by_label[next_label] = _store_key(color_key)
            next_label += 1
    return next_label, out_labels, color_by_label

def build_components_map(scribble_rgb):
    mode = str(COMPONENT_SPLIT_MODE).strip().lower()
    if mode in ("color", "rgb", "class"):
        comps = []
        num_s, s_labels, color_by_label = labels_by_color(scribble_rgb, WHITE, min_area=MIN_COMP_AREA)
        for lb in range(1, num_s):
            ys, xs = np.where(s_labels == lb)
            if xs.size == 0:
                continue
            area = xs.size
            cy, cx = ys.mean(), xs.mean()
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            comps.append({
                "label": int(lb),
                "area": int(area),
                "center": (float(cy), float(cx)),
                "bbox": (int(y0), int(x0), int(y1), int(x1)),
                "coords": (ys, xs),
                "color_key": int(color_by_label.get(lb, -1)),
            })
        return {c["label"]: c for c in comps}, comps

    smask = scribble_fg_mask(scribble_rgb).astype(np.uint8)
    num_s, s_labels = connected_components(smask)
    comps = component_stats(s_labels, num_s)
    return {c["label"]: c for c in comps}, comps


# ============================================================
# Boundary-safe shift (no-loss)
# ============================================================
def feasible_ranges(comp, h, w):
    ys, xs = comp["coords"]
    dy_min = int(-ys.min())
    dy_max = int((h - 1) - ys.max())
    dx_min = int(-xs.min())
    dx_max = int((w - 1) - xs.max())
    return dy_min, dy_max, dx_min, dx_max

def in_range(dy, dx, ranges):
    dy_min, dy_max, dx_min, dx_max = ranges
    return (dy_min <= dy <= dy_max) and (dx_min <= dx <= dx_max)

def safe_shift_for_component(comp, dy, dx, h, w):
    ranges = feasible_ranges(comp, h, w)

    if in_range(dy, dx, ranges):
        return dy, dx, True

    dy2, dx2 = dy, dx
    dy_min, dy_max, dx_min, dx_max = ranges

    # axis-wise bounce
    if dy2 < dy_min or dy2 > dy_max:
        dy2 = -dy2
    if dx2 < dx_min or dx2 > dx_max:
        dx2 = -dx2
    if in_range(dy2, dx2, ranges) and not (dy2 == 0 and dx2 == 0):
        return dy2, dx2, True

    # full flip
    if in_range(-dy, -dx, ranges) and not ((-dy) == 0 and (-dx) == 0):
        return -dy, -dx, True

    # clamp
    dy3 = int(np.clip(dy, dy_min, dy_max))
    dx3 = int(np.clip(dx, dx_min, dx_max))
    if in_range(dy3, dx3, ranges) and not (dy3 == 0 and dx3 == 0):
        return dy3, dx3, True

    return 0, 0, False

def apply_shift_to_component_no_loss(img_rgb, comp, dy, dx, out_rgb=None):
    h, w = img_rgb.shape[:2]
    dy, dx, ok = safe_shift_for_component(comp, dy, dx, h, w)
    if not ok:
        return img_rgb if out_rgb is None else out_rgb, False, 0, 0

    if out_rgb is None:
        out_rgb = img_rgb.copy()

    ys, xs = comp["coords"]
    color = img_rgb[ys[0], xs[0]].copy()

    out_rgb[ys, xs] = WHITE
    out_rgb[ys + dy, xs + dx] = color
    return out_rgb, True, int(dy), int(dx)


# ============================================================
# GT Precompute Cache (BIG SPEEDUP)
# ============================================================
def precompute_gt(gt_rgb: np.ndarray):
    infos = []
    areas = []
    radii = []
    bbs = {}
    gt_color_by_label = {}
    fg_mask = gt_fg_mask(gt_rgb)
    bg_mask = ~fg_mask

    if str(GT_COMPONENT_SPLIT_MODE).strip().lower() in ("color", "rgb", "class"):
        num_gt, gt_labels, gt_color_by_label = labels_by_color(gt_rgb, BLACK, min_area=1)
    else:
        gmask = fg_mask.astype(np.uint8)
        num_gt, gt_labels = connected_components(gmask)

    for g_id in range(1, num_gt):
        g_region = (gt_labels == g_id)
        g_area = int(g_region.sum())
        if g_area == 0:
            continue
        ys, xs = np.where(g_region)
        cy, cx = ys.mean(), xs.mean()
        center = np.array([cy, cx], dtype=np.float64)
        rad = radius_from_center(g_region, center)
        bbs[g_id] = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        info = {
            "rid": int(g_id),
            "area": float(g_area),
            "center": (float(cy), float(cx)),
            "radius": float(rad),
        }
        if g_id in gt_color_by_label:
            info["color_key"] = int(gt_color_by_label[g_id])
        infos.append(info)
        areas.append(float(g_area))
        radii.append(float(rad))

    if len(infos) == 0:
        # no fg
        bg_center = None
        if gt_rgb.size > 0:
            ys, xs = np.where(bg_mask)
            if xs.size > 0:
                bg_center = (float(ys.mean()), float(xs.mean()))
        return {
            "num_gt": num_gt,
            "gt_labels": gt_labels,
            "gt_bbs": bbs,
            "fg_infos": [],
            "w_fg": [],
            "mean_radius": 1.0,
            "bg_center": bg_center,
            "gt_color_by_label": gt_color_by_label,
            "fg_mask": fg_mask,
            "bg_mask": bg_mask,
        }

    areas = np.asarray(areas, np.float64)
    w_fg = areas / (areas.sum() + EPS)
    for i in range(len(infos)):
        infos[i]["w_fg"] = float(w_fg[i])
    mean_radius = float(np.mean(radii)) if len(radii) else 1.0

    ys, xs = np.where(bg_mask)
    bg_center = (float(ys.mean()), float(xs.mean())) if xs.size > 0 else None

    return {
        "num_gt": num_gt,
        "gt_labels": gt_labels,
        "gt_bbs": bbs,
        "fg_infos": infos,
        "w_fg": w_fg,
        "mean_radius": mean_radius,
        "bg_center": bg_center,
        "gt_color_by_label": gt_color_by_label,
        "fg_mask": fg_mask,
        "bg_mask": bg_mask,
    }


# ============================================================
# Score (uses GT cache)
# ============================================================
def compute_lambda_adaptive(gt_rgb: np.ndarray, scribble_rgb: np.ndarray):
    s_mask = scribble_fg_mask(scribble_rgb)
    if s_mask.sum() == 0:
        return LAMBDA_MIN
    g_fg = gt_fg_mask(gt_rgb)
    s_fg = np.logical_and(s_mask, g_fg).sum()
    s_bg = np.logical_and(s_mask, ~g_fg).sum()
    lam = float(s_bg / (s_bg + s_fg + EPS))
    return float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))

def compute_lambda_adaptive_cached(gt_rgb: np.ndarray, gt_cache: dict, scribble_rgb: np.ndarray):
    s_mask = scribble_fg_mask(scribble_rgb)
    if s_mask.sum() == 0:
        return LAMBDA_MIN
    g_fg = gt_cache.get("fg_mask")
    if g_fg is None:
        return compute_lambda_adaptive(gt_rgb, scribble_rgb)
    s_fg = np.logical_and(s_mask, g_fg).sum()
    s_bg = np.logical_and(s_mask, ~g_fg).sum()
    lam = float(s_bg / (s_bg + s_fg + EPS))
    return float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))

def compute_score_bgfg_cached(gt_rgb: np.ndarray, scribble_rgb: np.ndarray, gt_cache: dict):
    if str(COMPONENT_SPLIT_MODE).strip().lower() in ("color", "rgb", "class"):
        num_s, s_labels, s_color_by_label = labels_by_color(scribble_rgb, WHITE, min_area=1)
    else:
        smask = scribble_fg_mask(scribble_rgb).astype(np.uint8)
        num_s, s_labels = connected_components(smask)
        s_color_by_label = {}
    if num_s <= 1:
        # no scribble fg lines at all
        fg_score = SIM_FLOOR
        bg_score = SIM_FLOOR
        lam = compute_lambda_adaptive_cached(gt_rgb, gt_cache, scribble_rgb)
        total = float((1.0 - lam) * fg_score + lam * bg_score)
        return total, fg_score, bg_score, lam, [], None

    s_centers, s_areas = centers_and_areas_from_labels(s_labels, num_s)
    s_areas[0] = 0.0

    fg_infos_out = []
    if len(gt_cache["fg_infos"]) == 0:
        fg_score = 0.0
        fg_mean_radius = 1.0
    else:
        sims = []
        radii = []
        for gi in gt_cache["fg_infos"]:
            g_id = gi["rid"]
            y0, x0, y1, x1 = gt_cache["gt_bbs"].get(g_id, (None, None, None, None))
            if y0 is None:
                continue

            # bbox crop
            reg_bbox = (gt_cache["gt_labels"][y0:y1+1, x0:x1+1] == g_id)
            # s ids inside GT region
            s_ids_in = s_labels[y0:y1+1, x0:x1+1][reg_bbox]
            inter = np.bincount(s_ids_in.reshape(-1), minlength=num_s).astype(np.float64)
            inter[0] = 0.0
            if SCORE_MATCH_COLOR and "color_key" in gi and s_color_by_label:
                allowed = np.zeros((num_s,), dtype=bool)
                for lb, ck in s_color_by_label.items():
                    if int(ck) == int(gi["color_key"]):
                        allowed[int(lb)] = True
                inter[~allowed] = 0.0

            overlap = inter / (s_areas + EPS)
            w = overlap_soft_weight(overlap)
            w_area = w * s_areas

            if w_area.sum() < EPS:
                sim = SIM_FLOOR
                dist = float(gi["radius"])
            else:
                gc = np.array([gi["center"][0], gi["center"][1]], dtype=np.float64)
                wc = (s_centers * w_area[:, None]).sum(axis=0) / (w_area.sum() + EPS)
                dist = float(np.linalg.norm(gc - wc))
                sim = sim_from_dist_power(dist, float(gi["radius"]))

            fg_infos_out.append({
                "rid": int(g_id),
                "w_fg": float(gi["w_fg"]),
                "sim": float(sim),
                "radius": float(gi["radius"]),
                "center": gi["center"],
            })
            sims.append(sim)
            radii.append(float(gi["radius"]))

        if len(fg_infos_out) == 0:
            fg_score = 0.0
            fg_mean_radius = 1.0
        else:
            fg_score = float(np.sum(np.array([x["w_fg"] for x in fg_infos_out]) * np.array([x["sim"] for x in fg_infos_out])))
            fg_mean_radius = float(np.mean(radii)) if len(radii) else 1.0

    # bg score
    radius_bg = float(np.clip(gt_cache["mean_radius"] * BG_RADIUS_K, BG_RADIUS_MIN, BG_RADIUS_MAX))
    bg = gt_cache.get("bg_mask")
    if bg is None:
        bg = (~gt_fg_mask(gt_rgb))
    if gt_cache["bg_center"] is None:
        bg_score = SIM_FLOOR
        bg_info = None
    else:
        s_ids_in = s_labels[bg > 0]
        inter = np.bincount(s_ids_in.reshape(-1), minlength=num_s).astype(np.float64)
        inter[0] = 0.0
        overlap = inter / (s_areas + EPS)
        w = overlap_soft_weight(overlap)
        w_area = w * s_areas

        if w_area.sum() < EPS:
            bg_score = SIM_FLOOR
            bg_info = {"center": gt_cache["bg_center"], "radius": radius_bg, "sim": bg_score}
        else:
            bc = np.array([gt_cache["bg_center"][0], gt_cache["bg_center"][1]], dtype=np.float64)
            wc = (s_centers * w_area[:, None]).sum(axis=0) / (w_area.sum() + EPS)
            dist = float(np.linalg.norm(bc - wc))
            bg_score = sim_from_dist_power(dist, radius_bg)
            bg_info = {"center": gt_cache["bg_center"], "radius": radius_bg, "sim": float(bg_score)}

    lam = compute_lambda_adaptive_cached(gt_rgb, gt_cache, scribble_rgb)
    total = float((1.0 - lam) * fg_score + lam * bg_score)
    return total, float(fg_score), float(bg_score), float(lam), fg_infos_out, bg_info


# ============================================================
# Lever building (uses GT cache labels/bboxes to avoid recompute)
# ============================================================
def build_levers_bgfg_cached(gt_rgb, scribble_rgb, fg_infos, bg_info, lam, target_dynamic: float, gt_cache: dict):
    comps_by_label, comps = build_components_map(scribble_rgb)
    if len(comps) == 0:
        return [], comps_by_label

    regions = []
    for gi in fg_infos:
        w_total = (1.0 - lam) * float(gi["w_fg"])
        regions.append({
            "type": "fg",
            "rid": int(gi["rid"]),
            "w_total": float(w_total),
            "sim": float(gi["sim"]),
            "radius": float(gi["radius"]),
            "center": gi["center"],
            "color_key": int(gi["color_key"]) if "color_key" in gi else None,
        })
    if bg_info is not None:
        regions.append({
            "type": "bg",
            "rid": 0,
            "w_total": float(lam),
            "sim": float(bg_info["sim"]),
            "radius": float(bg_info["radius"]),
            "center": bg_info["center"],
        })

    ranked = []
    for r in regions:
        pr = float(r["w_total"] * max(0.0, r["sim"] - target_dynamic))
        if pr > 0:
            ranked.append((pr, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[:min(TOPK_REGIONS, len(ranked))]
    ranked_regions = [x[1] for x in ranked]
    if len(ranked_regions) == 0:
        return [], comps_by_label

    gt_labels = gt_cache["gt_labels"]
    gt_bbs = gt_cache["gt_bbs"]
    g_bg = ~gt_fg_mask(gt_rgb)

    levers = []
    for r in ranked_regions:
        best_comp = None
        best_warea = 0.0

        if r["type"] == "fg":
            g_id = r["rid"]
            if g_id not in gt_bbs:
                continue
            y0, x0, y1, x1 = gt_bbs[g_id]
            reg_bbox = (gt_labels[y0:y1+1, x0:x1+1] == g_id)

            for comp in comps:
                if SCORE_MATCH_COLOR and r.get("color_key") is not None:
                    if int(comp.get("color_key", -1)) != int(r["color_key"]):
                        continue
                cy0, cx0, cy1, cx1 = comp["bbox"]
                yy0 = max(y0, cy0); xx0 = max(x0, cx0)
                yy1 = min(y1, cy1); xx1 = min(x1, cx1)
                if yy1 < yy0 or xx1 < xx0:
                    continue

                reg_sub = reg_bbox[(yy0 - y0):(yy1 - y0 + 1), (xx0 - x0):(xx1 - x0 + 1)]
                ys, xs = comp["coords"]
                in_box = (ys >= yy0) & (ys <= yy1) & (xs >= xx0) & (xs <= xx1)
                if not np.any(in_box):
                    continue
                ys2 = ys[in_box] - yy0
                xs2 = xs[in_box] - xx0
                comp_sub = np.zeros_like(reg_sub, dtype=bool)
                comp_sub[ys2, xs2] = True
                inter = np.logical_and(comp_sub, reg_sub).sum()
                ov = float(inter / (comp["area"] + EPS))
                w = float(overlap_soft_weight(np.array([ov], np.float64))[0])
                warea = w * float(comp["area"])
                if warea > best_warea:
                    best_warea = warea
                    best_comp = comp
        else:
            for comp in comps:
                y0, x0, y1, x1 = comp["bbox"]
                sub_bg = g_bg[y0:y1+1, x0:x1+1]
                ys, xs = comp["coords"]
                comp_sub = np.zeros_like(sub_bg, dtype=bool)
                comp_sub[(ys - y0), (xs - x0)] = True
                inter = np.logical_and(comp_sub, sub_bg).sum()
                ov = float(inter / (comp["area"] + EPS))
                w = float(overlap_soft_weight(np.array([ov], np.float64))[0])
                warea = w * float(comp["area"])
                if warea > best_warea:
                    best_warea = warea
                    best_comp = comp

        if best_comp is None or best_warea <= 0:
            continue

        ry, rx = r["center"]
        cy, cx = best_comp["center"]
        vec_y = cy - ry
        vec_x = cx - rx
        norm = float(np.sqrt(vec_y * vec_y + vec_x * vec_x) + EPS)
        uy, ux = float(vec_y / norm), float(vec_x / norm)

        pr = float(r["w_total"] * max(0.0, r["sim"] - target_dynamic))
        levers.append({
            "w_total": float(r["w_total"]),
            "radius": float(r["radius"]),
            "priority": pr,
            "lever_label": int(best_comp["label"]),
            "unit_dir": (uy, ux),
        })

    levers.sort(key=lambda x: x["priority"], reverse=True)
    if len(levers) > 0:
        thr = PRUNE_LEVER_RATIO * levers[0]["priority"]
        levers = [lv for lv in levers if lv["priority"] >= thr]
    return levers, comps_by_label


def dir_candidates(uy, ux, expanded=False):
    if not expanded:
        return [(uy, ux)]
    py, px = -ux, uy
    return [(uy, ux), (-uy, -ux), (py, px), (-py, -px)]


def _get_balanced_rng():
    global _BALANCED_RNG
    if _BALANCED_RNG is None:
        seed = None if int(BALANCED_RANDOM_SEED) < 0 else int(BALANCED_RANDOM_SEED)
        _BALANCED_RNG = np.random.default_rng(seed)
    return _BALANCED_RNG


def _lever_per_max_px(lv: dict) -> int:
    return int(np.clip(int(min(GLOBAL_MAX_SHIFT, max(12.0, 1.2 * float(lv["radius"])))), 12, GLOBAL_MAX_SHIFT))


def _component_shift_caps(lv: dict, comp: dict):
    per_max = _lever_per_max_px(lv)
    if not ENABLE_BALANCED_DIVERSITY:
        return per_max, per_max, int(1e9)

    comp_scale = float(np.sqrt(max(float(comp.get("area", 1.0)), 1.0)))
    step_cap = max(
        float(BAL_STEP_MIN_PX),
        float(BAL_STEP_RADIUS_MULT) * float(lv["radius"]),
        float(BAL_STEP_AREA_MULT) * comp_scale,
    )
    step_cap = int(
        np.clip(
            min(step_cap, float(BAL_STEP_MAX_PX), float(per_max)),
            float(BAL_STEP_MIN_PX),
            float(per_max),
        )
    )

    total_cap = max(
        float(step_cap),
        float(BAL_TOTAL_RADIUS_MULT) * float(lv["radius"]),
        float(BAL_TOTAL_AREA_MULT) * comp_scale,
    )
    total_cap = int(np.clip(min(total_cap, float(BAL_TOTAL_MAX_PX)), float(step_cap), float(BAL_TOTAL_MAX_PX)))
    return per_max, step_cap, total_cap


def _select_levers_balanced(
    levers: list,
    comps_by_label: dict,
    moved_labels: set,
    coverage_target: int,
    it: int,
    last_pick_iter: dict,
    shift_acc: dict,
):
    if (not ENABLE_BALANCED_DIVERSITY) or len(levers) <= 1:
        return levers

    usable = []
    weights = []
    need_cover = len(moved_labels) < int(max(0, coverage_target))

    for lv in levers:
        lb = int(lv["lever_label"])
        comp = comps_by_label.get(lb)
        if comp is None:
            continue
        _per, _step_cap, total_cap = _component_shift_caps(lv, comp)
        remain = float(total_cap) - float(shift_acc.get(lb, 0.0)) if BAL_USE_TOTAL_CAP else float("inf")
        if remain < 1.0:
            continue

        w = float(max(float(lv["priority"]), 0.0) + float(BAL_WEIGHT_EPS)) ** float(BAL_PRIORITY_TAU)
        if need_cover:
            if lb not in moved_labels:
                w *= float(BAL_COVERAGE_BOOST)
            else:
                w *= float(BAL_COVERAGE_PENALTY)
        last_it = int(last_pick_iter.get(lb, -10**9))
        if (it - last_it) <= int(BAL_COOLDOWN_ITERS):
            w *= float(BAL_COOLDOWN_FACTOR)
        usable.append(lv)
        weights.append(max(w, 0.0))

    if len(usable) <= 1:
        return usable

    max_n = int(max(1, BAL_MAX_LEVERS_PER_ITER))
    max_n = min(max_n, len(usable))
    rand_quota = int(max(0, BAL_RANDOM_QUOTA))
    top_keep_cap = max(0, max_n - min(rand_quota, max_n))
    top_keep = int(np.clip(BAL_TOPK_KEEP, 0, top_keep_cap))
    top_keep = min(top_keep, len(usable))

    selected = usable[:top_keep]
    rest = usable[top_keep:]
    rest_weights = weights[top_keep:]
    need_n = max_n - len(selected)

    if need_n > 0 and len(rest) > 0:
        w_arr = np.asarray(rest_weights, dtype=np.float64)
        if not np.isfinite(w_arr).all() or float(w_arr.sum()) <= 0.0:
            w_arr = np.ones((len(rest),), dtype=np.float64)
        # numpy.choice is strict about the probabilities summing to exactly 1.
        w_arr = np.maximum(w_arr, 0.0)
        probs = w_arr / max(float(w_arr.sum()), EPS)
        probs = probs / max(float(probs.sum()), EPS)
        rng = _get_balanced_rng()
        idxs = rng.choice(len(rest), size=min(need_n, len(rest)), replace=False, p=probs)
        for j in np.atleast_1d(idxs).tolist():
            selected.append(rest[int(j)])

    return selected


# ============================================================
# Controller (uses cached score + cached lever builder)
# ============================================================
def offset_noise_to_target_bgfg_cached(gt_rgb, scribble_rgb, gt_cache, target_dynamic: float, tol: float):
    out = scribble_rgb.copy()
    applied = 0
    trials = 0

    total, fg, bg, lam, fg_infos, bg_info = compute_score_bgfg_cached(gt_rgb, out, gt_cache)
    if total <= target_dynamic + tol:
        return out, total, applied, trials, "skip"

    last_label = None
    last_dir = None
    last_dist_need = None
    last_per_max = None

    # Balanced mode state: encourage diverse components and cap per-component shifts.
    moved_labels = set()
    shift_acc = {}       # label -> accumulated shifted pixels (L2 step sum)
    last_pick_iter = {}  # label -> last picked iter

    def obj_fn(s):
        if s < target_dynamic - tol:
            return 10.0 + abs(s - target_dynamic)
        return abs(s - target_dynamic)

    extra_budget = MAX_ITERS_EXTRA if ENABLE_HARDCASE_EXTRA_ITER else 0
    max_total_iters = MAX_ITERS
    it = 0

    while it < max_total_iters:
        total, fg, bg, lam, fg_infos, bg_info = compute_score_bgfg_cached(gt_rgb, out, gt_cache)
        if abs(total - target_dynamic) <= tol:
            return out, total, applied, trials, "success"
        if total < target_dynamic - tol:
            return out, total, applied, trials, "overshoot"

        delta_need = total - target_dynamic
        mults = MULTS_STRONG if delta_need > STRONG_DELTA_TH else MULTS_MILD
        scale2 = SECOND_LEVER_SCALE_STRONG if delta_need > STRONG_DELTA_TH else SECOND_LEVER_SCALE_MILD

        levers, comps_by_label = build_levers_bgfg_cached(
            gt_rgb, out, fg_infos, bg_info, lam, target_dynamic, gt_cache
        )
        if len(levers) == 0:
            return out, total, applied, trials, "no_lever"

        coverage_target = 0
        selected_levers = levers
        if ENABLE_BALANCED_DIVERSITY:
            n_comps = int(len(comps_by_label))
            if n_comps > 0:
                max_cov = max(1, min(int(BAL_COVERAGE_MAX), n_comps))
                base_cov = int(np.ceil(float(BAL_COVERAGE_RATIO) * float(n_comps)))
                coverage_target = int(np.clip(base_cov, int(BAL_COVERAGE_MIN), max_cov))
            selected_levers = _select_levers_balanced(
                levers=levers,
                comps_by_label=comps_by_label,
                moved_labels=moved_labels,
                coverage_target=coverage_target,
                it=it,
                last_pick_iter=last_pick_iter,
                shift_acc=shift_acc,
            )
            if len(selected_levers) == 0:
                return out, total, applied, trials, "no_usable_lever"

        cur_obj = obj_fn(total)

        # Momentum is useful in legacy mode but conflicts with balanced diversification.
        if (
            ENABLE_MOMENTUM
            and (not ENABLE_BALANCED_DIVERSITY)
            and last_label is not None
            and last_dir is not None
            and last_dist_need is not None
        ):
            mom_mult = MOMENTUM_MULT_FAR if abs(delta_need) > MOMENTUM_NEAR_TH else MOMENTUM_MULT_NEAR
            comp = comps_by_label.get(last_label)
            if comp is not None:
                per_max = int(min(GLOBAL_MAX_SHIFT, max(12.0, last_per_max if last_per_max is not None else 48)))
                uy, ux = last_dir
                pix = int(np.clip(int(np.round(last_dist_need * mom_mult)), 1, per_max))
                dy, dx = clamp_shift(int(np.round(uy * pix)), int(np.round(ux * pix)), per_max)

                trials += 1
                cand_img, ok, ady, adx = apply_shift_to_component_no_loss(out, comp, dy, dx, out_rgb=out.copy())
                if ok:
                    cand_total = compute_score_bgfg_cached(gt_rgb, cand_img, gt_cache)[0]
                    if obj_fn(cand_total) + IMPROVE_EPS < cur_obj:
                        out = cand_img
                        total = cand_total
                        applied += 1
                        step_mag = float(np.hypot(float(ady), float(adx)))
                        shift_acc[last_label] = float(shift_acc.get(last_label, 0.0) + step_mag)
                        moved_labels.add(int(last_label))
                        last_pick_iter[int(last_label)] = int(it)
                        cur_obj = obj_fn(total)

        # recheck
        total, fg, bg, lam, fg_infos, bg_info = compute_score_bgfg_cached(gt_rgb, out, gt_cache)
        if abs(total - target_dynamic) <= tol:
            return out, total, applied, trials, "success"
        if total < target_dynamic - tol:
            return out, total, applied, trials, "overshoot"

        delta_need = total - target_dynamic
        cur_obj = obj_fn(total)
        pri_sum = float(sum(x["priority"] for x in selected_levers) + EPS)
        need_cover = bool(ENABLE_BALANCED_DIVERSITY and (len(moved_labels) < int(max(0, coverage_target))))

        def merge_stats(*items):
            out_stats = {}
            for item in items:
                for k, v in item.items():
                    out_stats[k] = out_stats.get(k, 0) + int(v)
            return out_stats

        def no_candidate_reason(stats):
            if int(stats.get("cap_exhausted", 0)) > 0 and int(stats.get("safe_shift_failed", 0)) == 0:
                return "no_candidate_cap_exhausted"
            if int(stats.get("safe_shift_failed", 0)) > 0 and int(stats.get("evaluated", 0)) == 0:
                return "no_candidate_boundary_blocked"
            if int(stats.get("missing_comp", 0)) > 0 and int(stats.get("attempts", 0)) == 0:
                return "no_candidate_missing_comp"
            if int(stats.get("zero_shift", 0)) > 0 and int(stats.get("evaluated", 0)) == 0:
                return "no_candidate_zero_shift"
            return "no_candidate_all_invalid"

        def search(expanded=False, only_top2_expand=False):
            nonlocal trials
            best = None
            second = None
            stats = {
                "levers": len(selected_levers),
                "missing_comp": 0,
                "cap_exhausted": 0,
                "attempts": 0,
                "safe_shift_failed": 0,
                "zero_shift": 0,
                "evaluated": 0,
            }
            for idx, lv in enumerate(selected_levers):
                # FAST: expanded dirs only for top-2 levers when only_top2_expand=True
                exp = expanded and ((not only_top2_expand) or (idx < 2))

                drop_alloc = float(delta_need * (lv["priority"] / pri_sum))
                drop_alloc = float(np.clip(drop_alloc, 0.02, delta_need))

                wtot = float(max(lv["w_total"], 1e-4))
                drop_sim = float(drop_alloc / wtot)
                dist_need = float(lv["radius"] * drop_sim)

                label = lv["lever_label"]
                comp = comps_by_label.get(label)
                if comp is None:
                    stats["missing_comp"] += 1
                    continue

                per_max, step_cap, total_cap = _component_shift_caps(lv, comp)
                remain = float(total_cap) - float(shift_acc.get(int(label), 0.0)) if BAL_USE_TOTAL_CAP else float("inf")
                if remain < 1.0:
                    stats["cap_exhausted"] += 1
                    continue
                pix_cap = int(
                    np.clip(
                        int(np.floor(min(float(per_max), float(step_cap), remain))),
                        1,
                        GLOBAL_MAX_SHIFT,
                    )
                )

                for m in mults:
                    pix = int(np.clip(int(np.round(dist_need * m)), 1, pix_cap))
                    for uy, ux in dir_candidates(lv["unit_dir"][0], lv["unit_dir"][1], expanded=exp):
                        dy, dx = clamp_shift(int(np.round(uy * pix)), int(np.round(ux * pix)), pix_cap)
                        trials += 1
                        stats["attempts"] += 1
                        cand_img, ok, ady, adx = apply_shift_to_component_no_loss(out, comp, dy, dx, out_rgb=out.copy())
                        if not ok:
                            stats["safe_shift_failed"] += 1
                            continue
                        step_mag = float(np.hypot(float(ady), float(adx)))
                        if step_mag < 1e-6:
                            stats["zero_shift"] += 1
                            continue
                        cand_total = compute_score_bgfg_cached(gt_rgb, cand_img, gt_cache)[0]
                        stats["evaluated"] += 1
                        o = obj_fn(cand_total)
                        if need_cover and (int(label) not in moved_labels):
                            o *= float(BAL_NEW_LABEL_OBJ_SCALE)
                        cand = (o, cand_total, cand_img, int(label), (uy, ux), dist_need, per_max, step_mag)

                        if best is None or cand[0] < best[0]:
                            if best is not None and best[3] != cand[3]:
                                second = best
                            best = cand
                        else:
                            if best is not None and cand[3] != best[3]:
                                if second is None or cand[0] < second[0]:
                                    second = cand
            return best, second, stats

        best, second, stats1 = search(expanded=False)
        if best is None:
            # FAST: expanded dirs only for top-2 levers
            best, second, stats2 = search(expanded=True, only_top2_expand=True)
        else:
            stats2 = {}

        if best is None:
            return out, total, applied, trials, no_candidate_reason(merge_stats(stats1, stats2))

        if ENABLE_NO_IMPROVE_GUARD and not (best[0] + IMPROVE_EPS < cur_obj):
            b2, s2, _stats3 = search(expanded=True, only_top2_expand=True)
            if b2 is None or not (b2[0] + IMPROVE_EPS < cur_obj):
                return out, total, applied, trials, "no_improve"
            best, second = b2, s2

        out = best[2]
        total = float(best[1])
        applied += 1

        last_label = best[3]
        last_dir = best[4]
        last_dist_need = best[5]
        last_per_max = best[6]
        shift_acc[int(last_label)] = float(shift_acc.get(int(last_label), 0.0) + float(best[7]))
        moved_labels.add(int(last_label))
        last_pick_iter[int(last_label)] = int(it)

        # dual lever
        if ENABLE_DUAL_LEVER and second is not None and second[3] != best[3]:
            comp2 = comps_by_label.get(second[3])
            if comp2 is not None:
                lv2 = None
                for _lv in selected_levers:
                    if int(_lv["lever_label"]) == int(second[3]):
                        lv2 = _lv
                        break

                uy2, ux2 = second[4]
                dist_need2 = second[5]
                if lv2 is not None:
                    per_max2, step_cap2, total_cap2 = _component_shift_caps(lv2, comp2)
                else:
                    per_max2 = int(np.clip(second[6], 12, GLOBAL_MAX_SHIFT))
                    step_cap2 = per_max2
                    total_cap2 = int(1e9)

                remain2 = float(total_cap2) - float(shift_acc.get(int(second[3]), 0.0)) if BAL_USE_TOTAL_CAP else float("inf")
                if remain2 >= 1.0:
                    pix2_cap = int(
                        np.clip(
                            int(np.floor(min(float(per_max2), float(step_cap2), remain2))),
                            1,
                            GLOBAL_MAX_SHIFT,
                        )
                    )
                    pix2 = int(np.clip(int(np.round(dist_need2 * scale2)), 1, pix2_cap))
                    dy2, dx2 = clamp_shift(int(np.round(uy2 * pix2)), int(np.round(ux2 * pix2)), pix2_cap)

                    trials += 1
                    cand2, ok2, ady2, adx2 = apply_shift_to_component_no_loss(out, comp2, dy2, dx2, out_rgb=out.copy())
                    if ok2:
                        cand2_total = compute_score_bgfg_cached(gt_rgb, cand2, gt_cache)[0]
                        if obj_fn(cand2_total) + IMPROVE_EPS < obj_fn(total):
                            out = cand2
                            total = float(cand2_total)
                            applied += 1
                            step_mag2 = float(np.hypot(float(ady2), float(adx2)))
                            shift_acc[int(second[3])] = float(shift_acc.get(int(second[3]), 0.0) + step_mag2)
                            moved_labels.add(int(second[3]))
                            last_pick_iter[int(second[3])] = int(it)

        it += 1
        if it >= max_total_iters and ENABLE_HARDCASE_EXTRA_ITER and extra_budget > 0:
            total_now = compute_score_bgfg_cached(gt_rgb, out, gt_cache)[0]
            if total_now > target_dynamic + tol and (total_now - target_dynamic) > HARDCASE_EXTRA_DELTA_TH:
                max_total_iters += 1
                extra_budget -= 1

    return out, compute_score_bgfg_cached(gt_rgb, out, gt_cache)[0], applied, trials, "max_iters"


# ============================================================
# Plotting
# ============================================================
def plot_score_distribution(before, target, after, save_path, title_suffix=""):
    before = np.asarray(before, np.float64); before = before[np.isfinite(before)]
    target = np.asarray(target, np.float64); target = target[np.isfinite(target)]
    after  = np.asarray(after,  np.float64); after  = after[np.isfinite(after)]
    if before.size < 2 or after.size < 2 or target.size < 2:
        return

    xs = np.linspace(0.0, 1.0, 512)
    plt.figure(figsize=(10, 6))

    if _HAS_SCIPY and before.size >= 5 and after.size >= 5 and target.size >= 5:
        plt.plot(xs, gaussian_kde(before)(xs), label="Before")
        plt.plot(xs, gaussian_kde(target)(xs), label="Target")
        plt.plot(xs, gaussian_kde(after)(xs),  label="After")
    else:
        hb, _ = np.histogram(before, bins=60, range=(0, 1), density=True)
        ht, _ = np.histogram(target, bins=60, range=(0, 1), density=True)
        ha, _ = np.histogram(after,  bins=60, range=(0, 1), density=True)
        xh = np.linspace(0, 1, 60)
        plt.plot(xh, hb, label="Before")
        plt.plot(xh, ht, label="Target")
        plt.plot(xh, ha, label="After")

    plt.title(f"Score Distribution (KDE){title_suffix}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_ratio_summary(df_summary: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))
    x = df_summary["ratio"].to_numpy(dtype=float)

    plt.plot(x, df_summary["mean_before"].to_numpy(dtype=float), marker="o", label="Mean Before")
    plt.plot(x, df_summary["mean_target"].to_numpy(dtype=float), marker="o", label="Mean Target")
    plt.plot(x, df_summary["mean_after"].to_numpy(dtype=float), marker="o", label="Mean After")

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, df_summary["success_rate"].to_numpy(dtype=float), marker="s", linestyle="--", label="Success Rate")
    ax2.set_ylabel("Success Rate")

    ax.set_xlabel("Ratio")
    ax.set_ylabel("Score (mean)")
    ax.set_title("Score vs Ratio (mean) + Success Rate")
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# IO
# ============================================================
def load_pair(fname: str):
    gt_path = os.path.join(GT_DIR, fname)
    sc_path = os.path.join(SCRIBBLE_DIR, fname)
    if (not os.path.exists(gt_path)) or (not os.path.exists(sc_path)):
        return None, None
    gt = np.array(Image.open(gt_path).convert("RGB"))
    sc = np.array(Image.open(sc_path).convert("RGB"))
    return gt, sc


def map_ratio_to_effective_ratio(ratio: float) -> float:
    r = float(np.clip(ratio, 0.0, 1.0))
    mode = str(TARGET_MAPPING_MODE).strip().lower()
    if mode == "legacy":
        return r

    # Reachable linear mapping:
    # requested ratio 0.0 -> effective TARGET_EFF_RATIO_MIN
    # requested ratio 1.0 -> effective 1.0
    r_eff_min = float(np.clip(TARGET_EFF_RATIO_MIN, 0.0, 1.0))
    gamma = float(max(TARGET_EFF_RATIO_GAMMA, 1e-6))
    r_adj = float(np.clip(r ** gamma, 0.0, 1.0))
    r_eff = r_eff_min + (1.0 - r_eff_min) * r_adj
    return float(np.clip(r_eff, 0.0, 1.0))


# ============================================================
# Run per ratio
# ============================================================
def run_for_ratio(files, ratio: float):
    ratio_tag = f"ratio_{ratio:.2f}"
    out_dir = os.path.join(SAVE_ROOT, ratio_tag)
    img_dir = os.path.join(out_dir, "images")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "noise_offset_simplified.csv")
    dist_png = os.path.join(csv_dir, "score_dist_kde.png")

    rows = []
    before_scores, target_scores, after_scores = [], [], []
    effective_ratios = []
    n_success = 0

    for f in tqdm(files, total=len(files), desc=f"OffsetNoise ratio={ratio:.2f}"):
        gt, sc = load_pair(f)
        if gt is None:
            continue

        gt_cache = precompute_gt(gt)  # ✅ GT cache once per image

        score_before = compute_score_bgfg_cached(gt, sc, gt_cache)[0]
        eff_ratio = map_ratio_to_effective_ratio(float(ratio))
        target_dynamic = float(np.clip(score_before * eff_ratio, 0.0, 1.0))

        out_img, score_after, applied, trials, reason = offset_noise_to_target_bgfg_cached(
            gt, sc, gt_cache, target_dynamic, TOL
        )

        # save image
        cv2.imwrite(os.path.join(img_dir, f), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

        before_minus_target = float(score_before - target_dynamic)
        after_minus_target  = float(score_after  - target_dynamic)

        denom = target_dynamic if target_dynamic > 1e-6 else 1e-6
        before_over_target = float(score_before / denom)
        after_over_target  = float(score_after  / denom)

        ok = 1 if abs(after_minus_target) <= TOL else 0
        n_success += ok

        # 简化表格：目标+增噪前后评分 + 对比目标
        rows.append({
            "file": f,
            "ratio": round(float(ratio), 4),
            "effective_ratio": round(float(eff_ratio), 4),
            "score_before": round(float(score_before), 4),
            "target": round(float(target_dynamic), 4),
            "score_after": round(float(score_after), 4),
            "before_minus_target": round(before_minus_target, 4),
            "after_minus_target": round(after_minus_target, 4),
            "before_over_target": round(before_over_target, 4),
            "after_over_target": round(after_over_target, 4),
            "ok": int(ok),
        })

        before_scores.append(float(score_before))
        target_scores.append(float(target_dynamic))
        after_scores.append(float(score_after))
        effective_ratios.append(float(eff_ratio))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_score_distribution(before_scores, target_scores, after_scores, dist_png, title_suffix=f"  ({ratio_tag})")

    summary = {
        "ratio": float(ratio),
        "effective_ratio_mean": float(np.mean(effective_ratios)) if len(effective_ratios) else np.nan,
        "mean_before": float(np.mean(before_scores)) if len(before_scores) else np.nan,
        "mean_target": float(np.mean(target_scores)) if len(target_scores) else np.nan,
        "mean_after": float(np.mean(after_scores)) if len(after_scores) else np.nan,
        "success_rate": float(n_success / max(len(before_scores), 1)),
    }
    return summary


def main():
    print("GT_DIR =", GT_DIR)
    print("SCRIBBLE_DIR =", SCRIBBLE_DIR)
    print("SAVE_ROOT =", SAVE_ROOT)
    print("RATIO_LIST =", RATIO_LIST)
    print("FAST_MODE =", FAST_MODE)
    print("MAX_ITERS =", MAX_ITERS)
    print("TOPK_REGIONS =", TOPK_REGIONS)
    print("MULTS_MILD =", MULTS_MILD)
    print("MULTS_STRONG =", MULTS_STRONG)
    print("TARGET_MAPPING_MODE =", TARGET_MAPPING_MODE)
    print("TARGET_EFF_RATIO_MIN =", TARGET_EFF_RATIO_MIN)
    print("TARGET_EFF_RATIO_GAMMA =", TARGET_EFF_RATIO_GAMMA)

    files = sorted([f for f in os.listdir(GT_DIR) if f.lower().endswith(".png")])
    if MAX_IMAGES and MAX_IMAGES > 0:
        files = files[:MAX_IMAGES]

    summaries = []
    for r in RATIO_LIST:
        summaries.append(run_for_ratio(files, float(r)))

    df_sum = pd.DataFrame(summaries)
    sum_csv = os.path.join(SAVE_ROOT, "summary_by_ratio.csv")
    df_sum.to_csv(sum_csv, index=False, encoding="utf-8-sig")

    ratio_png = os.path.join(SAVE_ROOT, "score_vs_ratio_summary.png")
    plot_ratio_summary(df_sum, ratio_png)

    print("[DONE] summary csv:", sum_csv)
    print("[DONE] ratio plot:", ratio_png)
    print("Done.")

if __name__ == "__main__":
    main()
