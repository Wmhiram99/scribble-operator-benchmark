#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fragmentation / Connectivity scoring (GT auto RGB/IDX) + plots

Changes vs v3:
- FG: k=0 regions do NOT participate in FG fragmentation score
- FG exponential lambda = 0.4
- BG: penalize background scribbles only (exp(-lambda_bg*k), k=0 => 1)
- After scoring, generate 3 plots:
    1) score_fg_hist.png
    2) fg_covered_ratio_hist.png
    3) score_fg_vs_covered_ratio_scatter.png
"""

import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image
import cv2

# matplotlib only (no seaborn)
import matplotlib
matplotlib.use("Agg")  # 非交互式后端：只保存图片，不需要 GUI
import matplotlib.pyplot as plt


# ============================================================
# 1) 参数（写死在这里）
# ============================================================
GT_DIR = os.environ.get("VOC_FRAGMENT_GT_DIR", "datasets/voc2012/GtMask")
SCRIB_DIR = os.environ.get("VOC_FRAGMENT_SCRIBBLE_DIR", "datasets/voc2012/auto_scribbles_idx")
OUT_DIR = os.environ.get("VOC_FRAGMENT_SCORE_OUT_DIR", "outputs/voc2012/fragmentation_score")

LIMIT_N: Optional[int] = 200  # None -> full

SCRIB_BG_VALUE = 0
SCRIB_IGNORE_VALUE = 255

GT_IDX_BG_VALUE = 0
GT_IDX_IGNORE_VALUE = 255
GT_BG_COLOR_RGB = (0, 0, 0)
GT_IGNORE_COLOR_RGB = (255, 255, 255)

CONNECTIVITY = 8
MIN_SCRIBBLE_AREA = 1
TIE_EPS = 0.0

LAMBDA_FG = 0.4
LAMBDA_BG = 0.8
GAMMA_BG = 0.3

PER_IMAGE_CSV = os.path.join(OUT_DIR, "fragmentation_per_image.csv")
PER_REGION_CSV = os.path.join(OUT_DIR, "fragmentation_per_region.csv")

SAVE_DEBUG = True
DEBUG_DIR = os.path.join(OUT_DIR, "debug_vis")
DEBUG_SAVE_OVERLAY = True
DEBUG_MAX_SAVE: Optional[int] = 200

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
MAKE_PLOTS = True


# ============================================================
# 2) 工具函数
# ============================================================
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def list_common_png_files(gt_dir: str, scrib_dir: str) -> List[str]:
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(".png")}
    sc_files = {f for f in os.listdir(scrib_dir) if f.lower().endswith(".png")}
    return sorted(list(gt_files.intersection(sc_files)))


def write_csv(path: str, header: List[str], rows: List[List]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def connected_components(binary01: np.ndarray, connectivity: int = 8):
    if binary01.dtype != np.uint8:
        binary01 = binary01.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary01, connectivity=connectivity)
    return n, labels.astype(np.int32), stats


# ============================================================
# 3) GT 读取：自动兼容 RGB / Palette(P) / IDX
# ============================================================
@dataclass
class GTAuto:
    kind: str
    data: np.ndarray
    rgb_vis: np.ndarray
    pil_mode: str


def read_gt_auto(path: str) -> GTAuto:
    img = Image.open(path)
    mode = img.mode
    arr = np.array(img)
    rgb_vis = np.array(img.convert("RGB"))

    if arr.ndim == 2:
        return GTAuto(kind="IDX", data=arr.astype(np.uint8), rgb_vis=rgb_vis, pil_mode=mode)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return GTAuto(kind="RGB", data=arr.astype(np.uint8), rgb_vis=rgb_vis, pil_mode=mode)
    raise ValueError(f"[ERROR] Unsupported GT format: mode={mode}, shape={arr.shape}, path={path}")


def read_scribble_idx(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"[ERROR] Scribble must be single-channel idx PNG. got shape={arr.shape} path={path}")
    return arr.astype(np.uint8)


# ============================================================
# 4) GT 区域构建：IDX / RGB
# ============================================================
def build_gt_regions_from_idx(gt_idx: np.ndarray, bg_value: int, ignore_value: int):
    valid = (gt_idx != ignore_value)
    fg_bin = ((gt_idx != bg_value) & valid).astype(np.uint8)
    bg_bin = ((gt_idx == bg_value) & valid).astype(np.uint8)

    n_fg, lab_fg, stats_fg = connected_components(fg_bin, CONNECTIVITY)
    n_bg, lab_bg, stats_bg = connected_components(bg_bin, CONNECTIVITY)

    fg_regions = {rid: (lab_fg == rid) for rid in range(1, n_fg) if int(stats_fg[rid, cv2.CC_STAT_AREA]) > 0}
    bg_regions = {rid: (lab_bg == rid) for rid in range(1, n_bg) if int(stats_bg[rid, cv2.CC_STAT_AREA]) > 0}
    return fg_regions, bg_regions


def build_gt_regions_from_rgb(gt_rgb: np.ndarray, bg_color_rgb, ignore_color_rgb):
    bg = np.array(bg_color_rgb, dtype=np.uint8).reshape(1, 1, 3)
    ig = np.array(ignore_color_rgb, dtype=np.uint8).reshape(1, 1, 3)

    is_bg = np.all(gt_rgb == bg, axis=2)
    is_ig = np.all(gt_rgb == ig, axis=2)
    valid = ~is_ig

    fg_bin = (valid & ~is_bg).astype(np.uint8)
    bg_bin = (valid & is_bg).astype(np.uint8)

    n_fg, lab_fg, stats_fg = connected_components(fg_bin, CONNECTIVITY)
    n_bg, lab_bg, stats_bg = connected_components(bg_bin, CONNECTIVITY)

    fg_regions = {rid: (lab_fg == rid) for rid in range(1, n_fg) if int(stats_fg[rid, cv2.CC_STAT_AREA]) > 0}
    bg_regions = {rid: (lab_bg == rid) for rid in range(1, n_bg) if int(stats_bg[rid, cv2.CC_STAT_AREA]) > 0}
    return fg_regions, bg_regions


# ============================================================
# 5) Scribble 连通域
# ============================================================
@dataclass
class ScribbleCC:
    labels: np.ndarray
    stats: np.ndarray
    ids: List[int]
    scribble_pixels: int


def build_scribble_cc_idx(scrib_idx: np.ndarray) -> ScribbleCC:
    fg = ((scrib_idx != SCRIB_BG_VALUE) & (scrib_idx != SCRIB_IGNORE_VALUE)).astype(np.uint8)
    n, labels, stats = connected_components(fg, CONNECTIVITY)

    ids, total = [], 0
    for sid in range(1, n):
        area = int(stats[sid, cv2.CC_STAT_AREA])
        if area >= MIN_SCRIBBLE_AREA:
            ids.append(sid)
            total += area
    return ScribbleCC(labels=labels, stats=stats, ids=ids, scribble_pixels=total)


# ============================================================
# 6) 分配：overlap = |S ∩ G| / |S|
# ============================================================
@dataclass
class AssignResult:
    k_map: Dict[Tuple[str, int], float]
    cc_assign_types: Dict[int, Set[str]]


def assign_scribbles(scrib_cc: ScribbleCC, fg_regions, bg_regions) -> AssignResult:
    region_keys, region_masks = [], []
    for rid, m in fg_regions.items():
        region_keys.append(("FG", rid))
        region_masks.append(m)
    for rid, m in bg_regions.items():
        region_keys.append(("BG", rid))
        region_masks.append(m)

    k_map = {k: 0.0 for k in region_keys}
    cc_assign_types: Dict[int, Set[str]] = {}
    if not region_keys:
        return AssignResult(k_map=k_map, cc_assign_types=cc_assign_types)

    labels = scrib_cc.labels
    for sid in scrib_cc.ids:
        S = (labels == sid)
        s_area = int(S.sum())
        if s_area <= 0:
            continue

        best_r = -1.0
        best_idxs: List[int] = []
        for idx, G in enumerate(region_masks):
            inter = int(np.logical_and(S, G).sum())
            r = inter / float(s_area)
            if r > best_r + TIE_EPS:
                best_r = r
                best_idxs = [idx]
            elif abs(r - best_r) <= TIE_EPS:
                best_idxs.append(idx)

        types: Set[str] = set()
        for idx in best_idxs:
            k_map[region_keys[idx]] += 1.0
            types.add(region_keys[idx][0])
        cc_assign_types[sid] = types

    return AssignResult(k_map=k_map, cc_assign_types=cc_assign_types)


# ============================================================
# 7) 评分：FG(k=0不参与) + BG(只惩罚乱画)
# ============================================================
@dataclass
class RegionRow:
    name: str
    region_type: str
    region_id: int
    gt_area: int
    k_assigned: float
    c_region: float
    weight_in_type: float
    contribution_in_type: float


def region_score_fg(k: float) -> float:
    return float(math.exp(-LAMBDA_FG * (k - 1.0)))


def region_score_bg(k: float) -> float:
    return float(math.exp(-LAMBDA_BG * k))


def compute_score(name: str, fg_regions, bg_regions, k_map):
    fg_areas = {rid: int(m.sum()) for rid, m in fg_regions.items()}
    bg_areas = {rid: int(m.sum()) for rid, m in bg_regions.items()}

    fg_total_area = sum(fg_areas.values())
    bg_total_area = sum(bg_areas.values())

    # FG: only covered regions
    fg_cov_rids = [rid for rid in fg_areas if k_map.get(("FG", rid), 0.0) > 0.0]
    fg_covered_area = sum(fg_areas[rid] for rid in fg_cov_rids)
    n_fg_covered_regions = len(fg_cov_rids)

    rows: List[RegionRow] = []

    if n_fg_covered_regions == 0:
        score_fg = 1.0
        for rid, area in fg_areas.items():
            k = k_map.get(("FG", rid), 0.0)
            rows.append(RegionRow(name, "FG", rid, area, k, 0.0, 0.0, 0.0))
    else:
        denom = float(fg_covered_area)
        score_fg = 0.0
        for rid, area in fg_areas.items():
            k = k_map.get(("FG", rid), 0.0)
            if k <= 0:
                rows.append(RegionRow(name, "FG", rid, area, k, 0.0, 0.0, 0.0))
                continue
            w = area / denom
            c = region_score_fg(k)
            contrib = w * c
            score_fg += contrib
            rows.append(RegionRow(name, "FG", rid, area, k, c, w, contrib))

    # BG
    score_bg = 0.0
    k_bg_total = 0.0
    if bg_total_area > 0:
        for rid, area in bg_areas.items():
            w = area / float(bg_total_area)
            k = k_map.get(("BG", rid), 0.0)
            k_bg_total += k
            c = region_score_bg(k)
            contrib = w * c
            score_bg += contrib
            rows.append(RegionRow(name, "BG", rid, area, k, c, w, contrib))

    # mix
    if fg_total_area == 0 and bg_total_area == 0:
        score_total = 0.0
    elif fg_total_area == 0:
        score_total = score_bg
    elif bg_total_area == 0:
        score_total = score_fg
    else:
        score_total = (1.0 - GAMMA_BG) * score_fg + GAMMA_BG * score_bg

    fg_cov_ratio = (float(fg_covered_area) / float(fg_total_area)) if fg_total_area > 0 else 0.0
    return score_fg, score_bg, score_total, k_bg_total, fg_covered_area, fg_total_area, fg_cov_ratio, n_fg_covered_regions, rows


# ============================================================
# 8) Debug 可视化（略：保持你现在那套输出）
# ============================================================
def label_to_random_color_vis(label_map: np.ndarray, seed: int = 0, bg_label: int = 0) -> np.ndarray:
    out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    labels = np.unique(label_map)
    rng = np.random.default_rng(seed)
    color_dict = {int(bg_label): np.array([0, 0, 0], dtype=np.uint8)}
    for lb in labels:
        lb = int(lb)
        if lb == bg_label:
            continue
        color_dict[lb] = rng.integers(0, 256, size=(3,), dtype=np.uint8)
    for lb, col in color_dict.items():
        out[label_map == lb] = col
    return out


def build_gt_region_id_map(fg_regions, bg_regions, shape_hw):
    h, w = shape_hw
    reg = np.zeros((h, w), dtype=np.int32)
    for rid, m in fg_regions.items():
        reg[m] = int(rid)
    for rid, m in bg_regions.items():
        reg[m] = int(10000 + rid)
    return reg


def build_assigned_subset_label_map(scrib_labels, cc_assign_types, keep_type: str):
    out = np.zeros_like(scrib_labels, dtype=np.int32)
    for sid, types in cc_assign_types.items():
        if keep_type in types:
            out[scrib_labels == sid] = int(sid)
    return out


def save_debug_vis(name, gt_rgb_vis, fg_regions, bg_regions, scrib_cc, cc_assign_types):
    ensure_dir(DEBUG_DIR)
    h, w, _ = gt_rgb_vis.shape

    gt_reg_map = build_gt_region_id_map(fg_regions, bg_regions, (h, w))
    gt_reg_vis = label_to_random_color_vis(gt_reg_map, seed=123, bg_label=0)
    scrib_vis = label_to_random_color_vis(scrib_cc.labels, seed=456, bg_label=0)

    scrib_fg_map = build_assigned_subset_label_map(scrib_cc.labels, cc_assign_types, keep_type="FG")
    scrib_bg_map = build_assigned_subset_label_map(scrib_cc.labels, cc_assign_types, keep_type="BG")
    scrib_fg_vis = label_to_random_color_vis(scrib_fg_map, seed=789, bg_label=0)
    scrib_bg_vis = label_to_random_color_vis(scrib_bg_map, seed=987, bg_label=0)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__gt_regions_vis.png"), cv2.cvtColor(gt_reg_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__scrib_cc_vis.png"), cv2.cvtColor(scrib_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__scrib_cc_assigned_fg.png"), cv2.cvtColor(scrib_fg_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__scrib_cc_assigned_bg.png"), cv2.cvtColor(scrib_bg_vis, cv2.COLOR_RGB2BGR))

    if DEBUG_SAVE_OVERLAY:
        alpha = 0.55
        overlay = np.clip((1 - alpha) * gt_reg_vis.astype(np.float32) + alpha * scrib_vis.astype(np.float32), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay2 = np.clip((1 - alpha) * gt_rgb_vis.astype(np.float32) + alpha * scrib_vis.astype(np.float32), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__overlay_on_gt.png"), cv2.cvtColor(overlay2, cv2.COLOR_RGB2BGR))
        overlay_bg = np.clip((1 - alpha) * gt_reg_vis.astype(np.float32) + alpha * scrib_bg_vis.astype(np.float32), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__overlay_bg_assigned.png"), cv2.cvtColor(overlay_bg, cv2.COLOR_RGB2BGR))
        overlay_fg = np.clip((1 - alpha) * gt_reg_vis.astype(np.float32) + alpha * scrib_fg_vis.astype(np.float32), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__overlay_fg_assigned.png"), cv2.cvtColor(overlay_fg, cv2.COLOR_RGB2BGR))


# ============================================================
# 9) 三张图：hist + scatter
# ============================================================
def load_per_image_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def make_plots_from_csv(csv_path: str, out_dir: str):
    ensure_dir(out_dir)
    rows = load_per_image_csv(csv_path)

    score_fg = np.array([float(x["score_fg"]) for x in rows], dtype=np.float32)
    cov_ratio = np.array([float(x["fg_covered_ratio"]) for x in rows], dtype=np.float32)

    # 1) score_fg hist
    plt.figure()
    plt.hist(score_fg, bins=30)
    plt.xlabel("score_fg (fragmentation only)")
    plt.ylabel("count")
    plt.title("Distribution of score_fg")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_fg_hist.png"), dpi=200)
    plt.close()

    # 2) coverage ratio hist
    plt.figure()
    plt.hist(cov_ratio, bins=30)
    plt.xlabel("fg_covered_ratio")
    plt.ylabel("count")
    plt.title("Distribution of fg_covered_ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fg_covered_ratio_hist.png"), dpi=200)
    plt.close()

    # 3) scatter
    plt.figure()
    plt.scatter(cov_ratio, score_fg, s=8)
    plt.xlabel("fg_covered_ratio")
    plt.ylabel("score_fg")
    plt.title("score_fg vs fg_covered_ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_fg_vs_covered_ratio_scatter.png"), dpi=200)
    plt.close()


# ============================================================
# 10) 主流程
# ============================================================
def main():
    ensure_dir(OUT_DIR)
    if SAVE_DEBUG:
        ensure_dir(DEBUG_DIR)

    files = list_common_png_files(GT_DIR, SCRIB_DIR)
    if not files:
        raise RuntimeError("[ERROR] No common .png filenames found between GT_DIR and SCRIB_DIR.")

    if LIMIT_N is not None:
        files = files[:int(LIMIT_N)]

    per_image_rows: List[List] = []
    per_region_rows: List[List] = []

    debug_saved = 0

    for fn in files:
        gp = os.path.join(GT_DIR, fn)
        sp = os.path.join(SCRIB_DIR, fn)
        name = os.path.splitext(fn)[0]

        gt = read_gt_auto(gp)
        scrib = read_scribble_idx(sp)

        if gt.kind == "IDX":
            if gt.data.shape != scrib.shape:
                raise ValueError(f"[ERROR] shape mismatch: {fn}, gt={gt.data.shape}, scrib={scrib.shape}")
        else:
            if gt.data.shape[0] != scrib.shape[0] or gt.data.shape[1] != scrib.shape[1]:
                raise ValueError(f"[ERROR] shape mismatch: {fn}, gt={gt.data.shape}, scrib={scrib.shape}")

        if gt.kind == "IDX":
            fg_regions, bg_regions = build_gt_regions_from_idx(gt.data, GT_IDX_BG_VALUE, GT_IDX_IGNORE_VALUE)
        else:
            fg_regions, bg_regions = build_gt_regions_from_rgb(gt.data, GT_BG_COLOR_RGB, GT_IGNORE_COLOR_RGB)

        scrib_cc = build_scribble_cc_idx(scrib)
        assign_res = assign_scribbles(scrib_cc, fg_regions, bg_regions)

        score_fg, score_bg, score_total, k_bg_total, fg_cov_area, fg_total_area, fg_cov_ratio, n_fg_cov, region_rows = compute_score(
            name, fg_regions, bg_regions, assign_res.k_map
        )

        per_image_rows.append([
            name, gt.kind, gt.pil_mode,
            f"{score_fg:.6f}", f"{score_bg:.6f}", f"{score_total:.6f}",
            f"{k_bg_total:.3f}",
            int(fg_cov_area), int(fg_total_area), f"{fg_cov_ratio:.6f}",
            int(n_fg_cov), int(len(fg_regions)),
            int(len(bg_regions)),
            int(len(scrib_cc.ids)),
            int(scrib_cc.scribble_pixels),
        ])

        for rr in region_rows:
            per_region_rows.append([
                rr.name, rr.region_type, rr.region_id, rr.gt_area,
                f"{rr.k_assigned:.3f}", f"{rr.c_region:.6f}",
                f"{rr.weight_in_type:.10f}", f"{rr.contribution_in_type:.10f}",
            ])

        if SAVE_DEBUG and (DEBUG_MAX_SAVE is None or debug_saved < int(DEBUG_MAX_SAVE)):
            save_debug_vis(name, gt.rgb_vis, fg_regions, bg_regions, scrib_cc, assign_res.cc_assign_types)
            debug_saved += 1

    write_csv(
        PER_IMAGE_CSV,
        ["name", "gt_kind", "gt_pil_mode",
         "score_fg", "score_bg", "score_total",
         "k_bg_total",
         "fg_covered_area", "fg_total_area", "fg_covered_ratio",
         "n_fg_covered_regions", "n_fg_regions",
         "n_bg_regions", "n_scribble_cc", "scribble_pixels"],
        per_image_rows
    )

    write_csv(
        PER_REGION_CSV,
        ["name", "region_type", "region_id", "gt_area",
         "k_assigned", "c_region", "weight_in_type", "contribution_in_type"],
        per_region_rows
    )

    print("[OK] scoring done:", len(files))
    print("csv:", PER_IMAGE_CSV)

    if MAKE_PLOTS:
        make_plots_from_csv(PER_IMAGE_CSV, PLOTS_DIR)
        print("[OK] plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()