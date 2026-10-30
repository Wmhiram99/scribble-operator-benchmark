import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm

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
BASE_DIR = os.environ.get("VOC_DATA_ROOT", "datasets/voc2012")
GT_DIR = os.path.join(BASE_DIR, "GtMask")
SCRIBBLE_DIR = os.path.join(BASE_DIR, "auto_scribbles_idx")

OUT_ROOT = os.environ.get("VOC_OFFSET_OUT_ROOT", "outputs/voc2012/offset")
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

# ============================================================
# FAST mode knobs (speed up, keep quality)
# ============================================================
FAST_MODE = True

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


# ============================================================
# Masks / helpers
# ============================================================
def gt_fg_mask(gt_rgb: np.ndarray) -> np.ndarray:
    return np.any(gt_rgb != BLACK, axis=-1)

def scribble_fg_mask(scribble_rgb: np.ndarray) -> np.ndarray:
    return np.any(scribble_rgb != WHITE, axis=-1)

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
    ys = np.repeat(np.arange(h), w)
    xs = np.tile(np.arange(w), h)

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
def component_stats(labels: np.ndarray, num_labels: int):
    comps = []
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
        comps.append({
            "label": int(lb),
            "area": int(area),
            "center": (float(cy), float(cx)),
            "bbox": (int(y0), int(x0), int(y1), int(x1)),
            "coords": (ys, xs),
        })
    return comps

def build_components_map(scribble_rgb):
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
        return img_rgb if out_rgb is None else out_rgb, False

    if out_rgb is None:
        out_rgb = img_rgb.copy()

    ys, xs = comp["coords"]
    color = img_rgb[ys[0], xs[0]].copy()

    out_rgb[ys, xs] = WHITE
    out_rgb[ys + dy, xs + dx] = color
    return out_rgb, True


# ============================================================
# GT Precompute Cache (BIG SPEEDUP)
# ============================================================
def precompute_gt(gt_rgb: np.ndarray):
    gmask = gt_fg_mask(gt_rgb).astype(np.uint8)
    num_gt, gt_labels = connected_components(gmask)

    infos = []
    areas = []
    radii = []
    bbs = {}

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
        infos.append({
            "rid": int(g_id),
            "area": float(g_area),
            "center": (float(cy), float(cx)),
            "radius": float(rad),
        })
        areas.append(float(g_area))
        radii.append(float(rad))

    if len(infos) == 0:
        # no fg
        bg_center = None
        if gt_rgb.size > 0:
            bg = (~gt_fg_mask(gt_rgb))
            ys, xs = np.where(bg)
            if xs.size > 0:
                bg_center = (float(ys.mean()), float(xs.mean()))
        return {
            "num_gt": num_gt,
            "gt_labels": gt_labels,
            "gt_bbs": bbs,
            "fg_infos": [],
            "w_fg": [],
            "mean_radius": 1.0,
            "bg_center": bg_center
        }

    areas = np.asarray(areas, np.float64)
    w_fg = areas / (areas.sum() + EPS)
    for i in range(len(infos)):
        infos[i]["w_fg"] = float(w_fg[i])
    mean_radius = float(np.mean(radii)) if len(radii) else 1.0

    bg = (~gt_fg_mask(gt_rgb))
    ys, xs = np.where(bg)
    bg_center = (float(ys.mean()), float(xs.mean())) if xs.size > 0 else None

    return {
        "num_gt": num_gt,
        "gt_labels": gt_labels,
        "gt_bbs": bbs,
        "fg_infos": infos,
        "w_fg": w_fg,
        "mean_radius": mean_radius,
        "bg_center": bg_center
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

def compute_score_bgfg_cached(gt_rgb: np.ndarray, scribble_rgb: np.ndarray, gt_cache: dict):
    smask = scribble_fg_mask(scribble_rgb).astype(np.uint8)
    num_s, s_labels = connected_components(smask)
    if num_s <= 1:
        # no scribble fg lines at all
        fg_score = SIM_FLOOR
        bg_score = SIM_FLOOR
        lam = compute_lambda_adaptive(gt_rgb, scribble_rgb)
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

    lam = compute_lambda_adaptive(gt_rgb, scribble_rgb)
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

        cur_obj = obj_fn(total)

        # momentum
        if ENABLE_MOMENTUM and last_label is not None and last_dir is not None and last_dist_need is not None:
            mom_mult = MOMENTUM_MULT_FAR if abs(delta_need) > MOMENTUM_NEAR_TH else MOMENTUM_MULT_NEAR
            comp = comps_by_label.get(last_label)
            if comp is not None:
                per_max = int(min(GLOBAL_MAX_SHIFT, max(12.0, last_per_max if last_per_max is not None else 48)))
                uy, ux = last_dir
                pix = int(np.clip(int(np.round(last_dist_need * mom_mult)), 1, per_max))
                dy, dx = clamp_shift(int(np.round(uy * pix)), int(np.round(ux * pix)), per_max)

                trials += 1
                cand_img, ok = apply_shift_to_component_no_loss(out, comp, dy, dx, out_rgb=out.copy())
                if ok:
                    cand_total = compute_score_bgfg_cached(gt_rgb, cand_img, gt_cache)[0]
                    if obj_fn(cand_total) + IMPROVE_EPS < cur_obj:
                        out = cand_img
                        total = cand_total
                        applied += 1
                        cur_obj = obj_fn(total)

        # recheck
        total, fg, bg, lam, fg_infos, bg_info = compute_score_bgfg_cached(gt_rgb, out, gt_cache)
        if abs(total - target_dynamic) <= tol:
            return out, total, applied, trials, "success"
        if total < target_dynamic - tol:
            return out, total, applied, trials, "overshoot"

        delta_need = total - target_dynamic
        cur_obj = obj_fn(total)
        pri_sum = float(sum(x["priority"] for x in levers) + EPS)

        def search(expanded=False, only_top2_expand=False):
            nonlocal trials
            best = None
            second = None
            for idx, lv in enumerate(levers):
                # FAST: expanded dirs only for top-2 levers when only_top2_expand=True
                exp = expanded and ((not only_top2_expand) or (idx < 2))

                drop_alloc = float(delta_need * (lv["priority"] / pri_sum))
                drop_alloc = float(np.clip(drop_alloc, 0.02, delta_need))

                wtot = float(max(lv["w_total"], 1e-4))
                drop_sim = float(drop_alloc / wtot)
                dist_need = float(lv["radius"] * drop_sim)

                per_max = int(np.clip(int(min(GLOBAL_MAX_SHIFT, max(12.0, 1.2 * lv["radius"]))), 12, GLOBAL_MAX_SHIFT))
                label = lv["lever_label"]
                comp = comps_by_label.get(label)
                if comp is None:
                    continue

                for m in mults:
                    pix = int(np.clip(int(np.round(dist_need * m)), 1, per_max))
                    for uy, ux in dir_candidates(lv["unit_dir"][0], lv["unit_dir"][1], expanded=exp):
                        dy, dx = clamp_shift(int(np.round(uy * pix)), int(np.round(ux * pix)), per_max)
                        trials += 1
                        cand_img, ok = apply_shift_to_component_no_loss(out, comp, dy, dx, out_rgb=out.copy())
                        if not ok:
                            continue
                        cand_total = compute_score_bgfg_cached(gt_rgb, cand_img, gt_cache)[0]
                        o = obj_fn(cand_total)
                        cand = (o, cand_total, cand_img, label, (uy, ux), dist_need, per_max)

                        if best is None or cand[0] < best[0]:
                            if best is not None and best[3] != cand[3]:
                                second = best
                            best = cand
                        else:
                            if best is not None and cand[3] != best[3]:
                                if second is None or cand[0] < second[0]:
                                    second = cand
            return best, second

        best, second = search(expanded=False)
        if best is None:
            # FAST: expanded dirs only for top-2 levers
            best, second = search(expanded=True, only_top2_expand=True)

        if best is None:
            return out, total, applied, trials, "no_candidate"

        if ENABLE_NO_IMPROVE_GUARD and not (best[0] + IMPROVE_EPS < cur_obj):
            b2, s2 = search(expanded=True, only_top2_expand=True)
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

        # dual lever
        if ENABLE_DUAL_LEVER and second is not None and second[3] != best[3]:
            comp2 = comps_by_label.get(second[3])
            if comp2 is not None:
                uy2, ux2 = second[4]
                dist_need2 = second[5]
                per_max2 = int(np.clip(second[6], 12, GLOBAL_MAX_SHIFT))
                pix2 = int(np.clip(int(np.round(dist_need2 * scale2)), 1, per_max2))
                dy2, dx2 = clamp_shift(int(np.round(uy2 * pix2)), int(np.round(ux2 * pix2)), per_max2)

                trials += 1
                cand2, ok2 = apply_shift_to_component_no_loss(out, comp2, dy2, dx2, out_rgb=out.copy())
                if ok2:
                    cand2_total = compute_score_bgfg_cached(gt_rgb, cand2, gt_cache)[0]
                    if obj_fn(cand2_total) + IMPROVE_EPS < obj_fn(total):
                        out = cand2
                        total = float(cand2_total)
                        applied += 1

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
    n_success = 0

    for f in tqdm(files, total=len(files), desc=f"OffsetNoise ratio={ratio:.2f}"):
        gt, sc = load_pair(f)
        if gt is None:
            continue

        gt_cache = precompute_gt(gt)  # ✅ GT cache once per image

        score_before = compute_score_bgfg_cached(gt, sc, gt_cache)[0]
        target_dynamic = float(np.clip(score_before * ratio, 0.0, 1.0))

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

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_score_distribution(before_scores, target_scores, after_scores, dist_png, title_suffix=f"  ({ratio_tag})")

    summary = {
        "ratio": float(ratio),
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

    print("✅ summary csv:", sum_csv)
    print("📈 ratio plot:", ratio_png)
    print("Done.")

if __name__ == "__main__":
    main()
