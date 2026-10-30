# direction_noise_axis_exp_v19_2.py
# -*- coding: utf-8 -*-

import os
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import cv2
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# VOC constants
# -------------------------
VOC_BG = 0
IGNORE = 255
IGNORE_BRIDGE_RADIUS = 3  # bridge small no-care gaps inside one object
BACKGROUND_CLASS_ID = 0
CITYSCAPES_LABELID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}


# -------------------------
# VOC colormap (Pascal VOC)
# -------------------------
def voc_colormap(n: int = 256) -> np.ndarray:
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap

_VOC_CMAP = voc_colormap(256)
_VOC_KEYS = (( _VOC_CMAP[:,0].astype(np.uint32) << 16) |
             ( _VOC_CMAP[:,1].astype(np.uint32) << 8)  |
               _VOC_CMAP[:,2].astype(np.uint32))
_VOC_KEY2ID = { int(k): int(i) for i, k in enumerate(_VOC_KEYS) }


# -------------------------
# IO / utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def angle_wrap_pi(theta: float) -> float:
    theta = float(theta)
    while theta < 0:
        theta += math.pi
    while theta >= math.pi:
        theta -= math.pi
    return theta

def angle_diff_axial(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    d = min(d, math.pi - d)
    return float(d)  # [0, pi/2]

def safe_log1p(x: float) -> float:
    return float(math.log1p(max(0.0, float(x))))

def infer_canvas_bg_value(idx: np.ndarray) -> int:
    total = idx.size
    cnt255 = int((idx == 255).sum())
    return 255 if cnt255 / (total + 1e-6) > 0.5 else 0

def read_idx_or_rgb_as_idx(path: str, *, is_gt: bool) -> Tuple[np.ndarray, float]:
    """
    Returns: (idx_map uint8 HxW, unknown_ratio)
    - Palette 'P': idx directly, unknown_ratio=0
    - Grayscale: idx directly, unknown_ratio=0
    - RGB:
        - if is_gt: map VOC colormap -> id, unknown -> 255, unknown_ratio computed
        - else: raise
    """
    im = Image.open(path)
    if im.mode == "P":
        arr = np.array(im, dtype=np.uint8)
        return arr, 0.0

    arr = np.array(im)
    if arr.ndim == 2:
        return arr.astype(np.uint8), 0.0

    if arr.ndim == 3 and arr.shape[2] >= 3:
        if not is_gt:
            raise ValueError(f"[ERROR] {path} is RGB but expected indexed scribble (HxW).")
        rgb = arr[:, :, :3].astype(np.uint8)
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        keys = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2].astype(np.uint32)

        out = np.full((h * w,), IGNORE, dtype=np.uint8)
        uniq = np.unique(keys)
        for k in uniq:
            cid = _VOC_KEY2ID.get(int(k), None)
            if cid is None:
                continue
            out[keys == k] = np.uint8(cid)

        out = out.reshape(h, w)
        unknown_ratio = float((out == IGNORE).mean())
        return out.astype(np.uint8), unknown_ratio

    raise ValueError(f"[ERROR] {path} unsupported image mode/shape: mode={im.mode} shape={arr.shape}")


def cityscapes_labelids_to_trainids(idx: np.ndarray) -> np.ndarray:
    lut = np.full((256,), IGNORE, dtype=np.uint8)
    for k, v in CITYSCAPES_LABELID_TO_TRAINID.items():
        lut[int(k)] = np.uint8(v)
    arr = idx.astype(np.uint8, copy=False)
    return lut[arr]


def maybe_align_cityscapes_ids(gt_idx: np.ndarray, scrib_idx: np.ndarray) -> np.ndarray:
    gt_valid = gt_idx[gt_idx != IGNORE]
    scrib_valid = scrib_idx[scrib_idx != IGNORE]
    if gt_valid.size == 0 or scrib_valid.size == 0:
        return gt_idx
    gt_max = int(gt_valid.max())
    scrib_max = int(scrib_valid.max())
    if gt_max > 18 and scrib_max <= 18:
        return cityscapes_labelids_to_trainids(gt_idx)
    return gt_idx


def _canonical_cityscapes_name(name: str) -> str:
    return (
        name
        .replace("_gtFine_labelTrainIds.png", "_gtFine_labelIds.png")
        .replace("_gtFine_labelTrainIds", "_gtFine_labelIds")
    )


def collect_png_pairs(gt_dir: str, scribble_dir: str) -> List[Tuple[str, str, str, str]]:
    scrib_flat = [f for f in os.listdir(scribble_dir) if f.lower().endswith(".png")]
    if scrib_flat:
        pairs = []
        for f in sorted(scrib_flat):
            gp = os.path.join(gt_dir, _canonical_cityscapes_name(f))
            sp = os.path.join(scribble_dir, f)
            if os.path.exists(gp):
                pairs.append((f, sp, gp, f))
        return pairs

    gt_map: Dict[str, str] = {}
    for root, _, files in os.walk(gt_dir):
        for f in files:
            if f.lower().endswith(".png"):
                gt_map[f] = os.path.join(root, f)

    pairs = []
    for root, _, files in os.walk(scribble_dir):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            gname = _canonical_cityscapes_name(f)
            gp = gt_map.get(gname)
            if gp is None:
                continue
            sp = os.path.join(root, f)
            rel_out = os.path.relpath(sp, scribble_dir)
            pairs.append((f, sp, gp, rel_out))
    pairs.sort(key=lambda x: x[0])
    return pairs


# -------------------------
# Skeletonize (no skimage)
# -------------------------
def skeletonize_cv(bin_img: np.ndarray) -> np.ndarray:
    img = (bin_img > 0).astype(np.uint8) * 255
    try:
        thin = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (thin > 0).astype(np.uint8)
    except Exception:
        pass

    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cur = img.copy()
    while True:
        eroded = cv2.erode(cur, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(cur, temp)
        skel = cv2.bitwise_or(skel, temp)
        cur = eroded
        if cv2.countNonZero(cur) == 0:
            break
    return (skel > 0).astype(np.uint8)


# -------------------------
# PCA angle + rho (numpy only)
# coords in (r,c)
# -------------------------
def pca_angle_rho_from_points_rc(coords_rc: np.ndarray) -> Tuple[Optional[float], float]:
    if coords_rc is None or len(coords_rc) < 5:
        return None, 1.0
    X = coords_rc.astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(1, (len(X) - 1))
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-10)
    k = int(np.argmax(vals))
    v = vecs[:, k]  # (dr, dc)
    ang = math.atan2(float(v[0]), float(v[1]))
    ang = angle_wrap_pi(ang)

    order = np.argsort(vals)[::-1]
    l1 = float(vals[order[0]])
    l2 = float(vals[order[-1]])
    rho = float(math.sqrt(l1 / max(l2, 1e-10)))
    return float(ang), float(rho)

def pca_angle_rho_from_binary(bin_mask: np.ndarray, use_skeleton: bool = True) -> Tuple[Optional[float], float]:
    if use_skeleton:
        sk = skeletonize_cv(bin_mask)
        coords = np.column_stack(np.where(sk > 0))
        if len(coords) < 5:
            coords = np.column_stack(np.where(bin_mask > 0))
    else:
        coords = np.column_stack(np.where(bin_mask > 0))
    return pca_angle_rho_from_points_rc(coords)

def center_xy_from_bin(bin_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    coords = np.column_stack(np.where(bin_mask > 0))
    if len(coords) == 0:
        return None
    mean_rc = np.mean(coords, axis=0)
    return (float(mean_rc[1]), float(mean_rc[0]))  # (x,y)


# -------------------------
# Geometry / components
# -------------------------
def iou_bin(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    uni = float(np.logical_or(a > 0, b > 0).sum())
    return inter / (uni + 1e-6)

@dataclass
class ScribbleComp:
    class_id: int
    mask: np.ndarray
    area: int
    center_xy: Tuple[float, float]

@dataclass
class GTComp:
    class_id: int
    mask: np.ndarray
    area: int
    center_xy: Tuple[float, float]

def extract_scribble_components(
    idx_img: np.ndarray,
    canvas_bg: int,
    min_area: int,
    has_bg_class0: bool = False,
) -> Tuple[List[ScribbleComp], Optional[np.ndarray], int]:
    valid = (idx_img != canvas_bg) & (idx_img != IGNORE)
    total_valid = int(valid.sum())
    if total_valid == 0:
        return [], None, 0

    # Optional: class-0 strokes can be treated as background for VOC-like datasets.
    bg_mask = ((idx_img == BACKGROUND_CLASS_ID) & valid).astype(np.uint8) if has_bg_class0 else None

    comps: List[ScribbleComp] = []
    class_ids = np.unique(idx_img[valid])
    for cid in class_ids:
        cid = int(cid)
        if has_bg_class0 and cid == BACKGROUND_CLASS_ID:
            continue
        bin_mask = (idx_img == cid).astype(np.uint8)
        nlab, labels = cv2.connectedComponents(bin_mask)
        for i in range(1, nlab):
            m = (labels == i).astype(np.uint8)
            a = int(m.sum())
            if a < int(min_area):
                continue
            c = center_xy_from_bin(m)
            if c is None:
                continue
            comps.append(ScribbleComp(class_id=cid, mask=m, area=a, center_xy=c))
    return comps, bg_mask, total_valid


def bridge_ignore_gaps(fg: np.ndarray, ignore_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Bridge small holes/gaps *only through IGNORE pixels* so one object isn't split by no-care regions.

    Method: morphological closing on fg, then only accept newly-added pixels that fall inside ignore_mask.
    """
    if radius <= 0:
        return fg.astype(np.uint8)
    fg_u8 = (fg > 0).astype(np.uint8)
    ign = (ignore_mask > 0)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
    bridged = fg_u8 | ((closed > 0) & ign).astype(np.uint8)
    return bridged.astype(np.uint8)


def extract_gt_components(gt_idx: np.ndarray, min_area: int, has_bg_class0: bool = False) -> List[GTComp]:
    if has_bg_class0:
        valid = (gt_idx != VOC_BG) & (gt_idx != IGNORE)
    else:
        valid = (gt_idx != IGNORE)
    if int(valid.sum()) == 0:
        return []
    comps: List[GTComp] = []
    class_ids = np.unique(gt_idx[valid])
    for cid in class_ids:
        cid = int(cid)
        if cid == IGNORE or (has_bg_class0 and cid == VOC_BG):
            continue
        bin_mask = (gt_idx == cid).astype(np.uint8)
        # Bridge small gaps caused by IGNORE(255) so one object isn't split into multiple GT components.
        if int(IGNORE_BRIDGE_RADIUS) > 0:
            bin_mask = bridge_ignore_gaps(bin_mask, (gt_idx == IGNORE).astype(np.uint8), radius=int(IGNORE_BRIDGE_RADIUS))
        nlab, labels = cv2.connectedComponents(bin_mask)
        for i in range(1, nlab):
            m = (labels == i).astype(np.uint8)
            a = int(m.sum())
            if a < int(min_area):
                continue
            c = center_xy_from_bin(m)
            if c is None:
                continue
            comps.append(GTComp(class_id=cid, mask=m, area=a, center_xy=c))
    return comps


# -------------------------
# Scoring (EXP) + gate weights
# -------------------------
def gate_from_rho_sigmoid(rho: float, rho0: float, k: float) -> float:
    z = float(k) * (float(rho) - float(rho0))
    if z >= 0:
        ez = math.exp(-z)
        return float(1.0 / (1.0 + ez))
    else:
        ez = math.exp(z)
        return float(ez / (1.0 + ez))

def sdir_exp_weighted(diff_rad: float, sigma_rad: float, p: float, omega: float = 1.0) -> float:
    """EXP direction score with anisotropy weight omega in [0,1]."""
    diff = float(max(0.0, diff_rad))
    sig = float(max(1e-8, sigma_rad))
    om = float(np.clip(omega, 0.0, 1.0))
    return float(math.exp(-(om * ((diff / sig) ** float(p)))))

def sdir_exp(diff_rad: float, sigma_rad: float, p: float) -> float:
    """Backwards-compatible: omega=1.0."""
    return sdir_exp_weighted(diff_rad, sigma_rad=sigma_rad, p=p, omega=1.0)

def weight_from_union_skeleton(union: np.ndarray,
                               gate: float,
                               gate_pow: float,
                               w_min: float,
                               w_max: float) -> float:
    sk = skeletonize_cv(union)
    L = float(sk.sum())
    base = float(np.clip(safe_log1p(L) / safe_log1p(220.0), w_min, w_max))
    g = float(np.clip(gate, 0.0, 1.0)) ** float(gate_pow)
    return float(max(1e-4, base * g))
# -------------------------
# Feasible-angle geometry (angle budget from inside-ratio)
# -------------------------
def _square_crop_around_center(H: int, W: int, center_xy: Tuple[float, float],
                              radius: float, pad: int = 6) -> Tuple[int, int, int, int]:
    """Return (y0,y1,x0,x1) square crop around center with half-size ~radius+pad."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = int(math.ceil(float(radius))) + int(pad)
    x0 = int(max(0, math.floor(cx) - half))
    x1 = int(min(W, math.floor(cx) + half + 1))
    y0 = int(max(0, math.floor(cy) - half))
    y1 = int(min(H, math.floor(cy) + half + 1))
    return y0, y1, x0, x1

def _max_radius_from_center(bin_mask: np.ndarray, center_xy: Tuple[float, float]) -> float:
    ys, xs = np.where(bin_mask > 0)
    if ys.size == 0:
        return 0.0
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx = np.max(np.abs(xs.astype(np.float32) - cx))
    dy = np.max(np.abs(ys.astype(np.float32) - cy))
    return float(max(dx, dy))

def _rotate_binary_crop(bin_crop: np.ndarray,
                        angle_rad: float,
                        center_xy_in_crop: Tuple[float, float]) -> np.ndarray:
    h, w = bin_crop.shape[:2]
    M = cv2.getRotationMatrix2D(center_xy_in_crop, float(np.degrees(angle_rad)), 1.0)
    rot = cv2.warpAffine(
        bin_crop.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rot > 0).astype(np.uint8)

def compute_rin_curve_roi(union_bin: np.ndarray,
                          gt_bin: np.ndarray,
                          alphas: np.ndarray,
                          center_xy: Tuple[float, float],
                          dilate_px: int = 0) -> np.ndarray:
    """Compute r_in(alpha) over alphas using a safe square crop to reduce cost."""
    H, W = union_bin.shape[:2]
    r = _max_radius_from_center(union_bin, center_xy=center_xy)
    y0, y1, x0, x1 = _square_crop_around_center(H, W, center_xy=center_xy, radius=r, pad=8)

    u = union_bin[y0:y1, x0:x1].astype(np.uint8)
    g = gt_bin[y0:y1, x0:x1].astype(np.uint8)

    if int(dilate_px) > 0:
        k = int(2 * int(dilate_px) + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        g = cv2.dilate(g, ker, iterations=1)

    cx, cy = float(center_xy[0] - x0), float(center_xy[1] - y0)
    rin = []
    for a in alphas:
        rot_u = _rotate_binary_crop(u, float(a), (cx, cy))
        den = float(rot_u.sum())
        if den <= 1e-6:
            rin.append(0.0)
            continue
        num = float((rot_u * g).sum())
        rin.append(num / den)
    return np.asarray(rin, dtype=np.float32)

def feasible_angles_from_rin(alphas: np.ndarray,
                             rin: np.ndarray,
                             thr_strict: float = 0.95,
                             thr_soft: float = 0.60) -> Tuple[float, float]:
    """Return (a0, a1) where:
    - a0: max |alpha| with rin>=thr_strict (near fully inside)
    - a1: max |alpha| with rin>=thr_soft  (partially inside)
    Both are in radians within [0, pi/2].
    """
    aa = np.abs(alphas.astype(np.float32))
    amax = float(np.pi / 2.0)

    def _amax_for_thr(t: float) -> float:
        ok = rin >= float(t)
        if not np.any(ok):
            return 0.0
        return float(min(amax, float(np.max(aa[ok]))))

    a0 = _amax_for_thr(float(thr_strict))
    a1 = _amax_for_thr(float(thr_soft))
    if a1 < a0:
        a1 = a0
    return float(a0), float(a1)

def g_from_feasible_angle(abs_alpha: float,
                          a0: float,
                          a1: float,
                          g0: float = 0.8,
                          g1: float = 0.6,
                          g2: float = 0.0,
                          p_out: float = 2.0) -> float:
    """Angle-based geometry score g(|alpha|) with anchors:
      |alpha|<=a0 -> g0
      |alpha|==a1 -> g1
      |alpha|==pi/2 -> g2
    """
    a = float(max(0.0, abs_alpha))
    amax = float(np.pi / 2.0)
    a0 = float(np.clip(a0, 0.0, amax))
    a1 = float(np.clip(a1, a0, amax))
    g0 = float(np.clip(g0, 0.0, 1.0))
    g1 = float(np.clip(g1, 0.0, 1.0))
    g2 = float(np.clip(g2, 0.0, 1.0))
    p_out = float(max(0.25, p_out))

    if a <= a0 + 1e-9:
        return g0
    if a0 < a <= a1 + 1e-9:
        if a1 <= a0 + 1e-9:
            return g1
        t = (a - a0) / (a1 - a0)
        return float((1.0 - t) * g0 + t * g1)

    if a >= amax - 1e-9:
        return g2

    if amax <= a1 + 1e-9:
        return g2

    # smooth drop from g1 at a1 to g2 at amax
    t = (a - a1) / (amax - a1)
    t = float(np.clip(t, 0.0, 1.0))
    drop = (1.0 - t) ** p_out
    return float(g2 + (g1 - g2) * drop)


def g_from_rin_soft(rin_val: float,
                    thr_strict: float = 0.90,
                    thr_soft: float = 0.55,
                    eps: float = 0.02,
                    q: float = 1.5,
                    hard_reach: bool = False) -> float:
    r = float(np.clip(rin_val, 0.0, 1.0))
    ts = float(np.clip(thr_strict, 0.0, 1.0))
    tf = float(np.clip(thr_soft, 0.0, 1.0))
    if tf > ts:
        tf = ts
    e = float(np.clip(eps, 0.0, 1.0))
    qq = float(max(0.1, q))

    if r >= ts:
        return 1.0
    if r < tf:
        return 0.0 if hard_reach else e
    if ts <= tf + 1e-9:
        return 1.0 if r >= ts else (0.0 if hard_reach else e)
    t = float((r - tf) / (ts - tf))
    t = float(np.clip(t, 0.0, 1.0))
    if hard_reach:
        return float(t ** qq)
    return float(e + (1.0 - e) * (t ** qq))


def cityscapes_group_name(class_id: int) -> str:
    cid = int(class_id)
    thin_human = {5, 6, 7, 11, 12, 17, 18}
    vehicle = {13, 14, 15, 16}
    if cid in thin_human:
        return "thin"
    if cid in vehicle:
        return "vehicle"
    return "stuff"


def profile_params_for_group(class_id: int,
                             sigma_stuff_deg: float,
                             p_stuff: float,
                             beta_stuff: float,
                             lambda_stuff: float,
                             sigma_vehicle_deg: float,
                             p_vehicle: float,
                             beta_vehicle: float,
                             lambda_vehicle: float,
                             sigma_thin_deg: float,
                             p_thin: float,
                             beta_thin: float,
                             lambda_thin: float) -> Tuple[float, float, float, float]:
    gname = cityscapes_group_name(class_id)
    if gname == "thin":
        return float(sigma_thin_deg), float(p_thin), float(beta_thin), float(lambda_thin)
    if gname == "vehicle":
        return float(sigma_vehicle_deg), float(p_vehicle), float(beta_vehicle), float(lambda_vehicle)
    return float(sigma_stuff_deg), float(p_stuff), float(beta_stuff), float(lambda_stuff)




# -------------------------
# Groups / table
# -------------------------
@dataclass
class AlphaTable:
    alphas: np.ndarray
    sg: np.ndarray
    rin: Optional[np.ndarray] = None

@dataclass
class GTGroup:
    gid: int
    class_id: int
    gt_mask: np.ndarray
    union_mask: np.ndarray
    scribs: List[ScribbleComp]
    theta_s: Optional[float]
    theta_g: Optional[float]
    gt_rho: float
    gate: float
    omega: float
    w: float
    a0: float
    a1: float
    sigma_deg: float
    p_exp: float
    geom_beta: float
    angle_lambda: float
    area_share: float
    max_abs_alpha: float
    min_abs_alpha: float
    table: AlphaTable
    # match iou stats (scrib->gt)
    ious: List[float]


def stable_seed_from_parts(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") & 0x7fffffff


def max_alpha_for_area_share(area_share: float,
                             small_share: float,
                             large_share: float,
                             small_deg: float,
                             large_deg: float,
                             jitter_deg: float,
                             rng: np.random.Generator) -> float:
    s = float(np.clip(area_share, 0.0, 1.0))
    lo = float(max(0.0, small_share))
    hi = float(max(lo + 1e-6, large_share))
    t = float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))
    deg = (1.0 - t) * float(small_deg) + t * float(large_deg)
    if jitter_deg > 0:
        deg += float(rng.uniform(-float(jitter_deg), float(jitter_deg)))
    deg = float(np.clip(deg, 0.0, 90.0))
    return float(np.deg2rad(deg))


def min_alpha_for_area_share(area_share: float,
                             small_share: float,
                             min_small_deg: float) -> float:
    if float(area_share) > float(max(0.0, small_share)):
        return 0.0
    return float(np.deg2rad(max(0.0, float(min_small_deg))))


def choose_groups_stratified(tmp: List[Tuple[int, dict]],
                             max_groups_keep: int,
                             rng: np.random.Generator,
                             random_group_keep_prob: float,
                             large_group_share: float,
                             large_group_keep_prob: float,
                             small_group_boost: float) -> List[Tuple[int, dict]]:
    if max_groups_keep <= 0 or len(tmp) <= max_groups_keep:
        return tmp

    keep_prob = float(np.clip(random_group_keep_prob, 0.10, 1.0))
    keep_n = int(max(1, min(max_groups_keep, round(1 + (max_groups_keep - 1) * keep_prob))))
    if keep_n < max_groups_keep and rng.random() < 0.35:
        keep_n = min(int(max_groups_keep), keep_n + 1)
    total_area = float(sum(max(0, int(a)) for a, _ in tmp)) + 1e-6
    shares = np.asarray([float(a) / total_area for a, _ in tmp], dtype=np.float64)

    # Always keep the largest object as an anchor, then sample the rest with
    # small-object boost and large-object down-weighting.
    selected = {int(np.argmax(shares))}
    remaining = [i for i in range(len(tmp)) if i not in selected]

    while len(selected) < keep_n and remaining:
        weights = []
        for i in remaining:
            sh = float(shares[i])
            w = 1.0 / math.sqrt(max(float(tmp[i][0]), 1.0))
            if sh >= float(large_group_share):
                w *= float(np.clip(large_group_keep_prob, 0.02, 1.0))
            else:
                w *= float(max(0.1, small_group_boost))
            weights.append(max(w, 1e-9))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= float(weights.sum())
        pick_pos = int(rng.choice(len(remaining), p=weights))
        selected.add(int(remaining.pop(pick_pos)))

    selected_list = [tmp[i] for i in sorted(selected)]
    selected_list.sort(key=lambda x: x[0], reverse=True)
    return selected_list

def build_groups_with_table_exp(
    scrib_idx: np.ndarray,
    gt_idx: np.ndarray,
    canvas_bg: int,
    min_scrib_comp_area: int,
    min_gt_comp_area: int,
    min_group_ratio: float,
    max_groups_keep: int,
    alphas: np.ndarray,
    rng_seed: int = 0,
    random_group_keep_prob: float = 0.85,
    large_group_share: float = 0.25,
    large_group_keep_prob: float = 0.35,
    small_group_boost: float = 2.0,
    small_area_share: float = 0.01,
    max_alpha_small_deg: float = 80.0,
    max_alpha_large_deg: float = 32.0,
    max_alpha_jitter_deg: float = 8.0,
    min_alpha_small_deg: float = 10.0,
    large_rin_thr_strict: float = 0.96,
    large_rin_thr_soft: float = 0.72,
    sigma_deg: float = 25.0,
    p: float = 2.0,
    rho0: float = 1.25,
    k_gate: float = 6.0,
    gate_pow: float = 1.0,
    w_min: float = 0.05,
    w_max: float = 1.0,

    # geometry via feasible-angle curve (computed from inside ratio)
    use_feasible_geom: bool = True,
    rin_thr_strict: float = 0.90,
    rin_thr_soft: float = 0.55,
    reach_eps: float = 0.02,
    reach_q: float = 1.5,
    hard_reach_gate: bool = False,
    geom_dilate_px: int = 0,

    # direction anisotropy weight (square GT -> omega≈0 -> direction ≈ 1)
    omega_pow: float = 1.0,

    # omega hard-off for near-isotropic GT (rho < rho_omega_off => omega=0)
    rho_omega_off: float = 1.25,
    omega_min: float = 0.2,

    # class-profile params (Cityscapes)
    sigma_stuff_deg: float = 22.0,
    p_stuff: float = 2.0,
    beta_stuff: float = 0.35,
    lambda_stuff: float = 0.03,
    sigma_vehicle_deg: float = 18.0,
    p_vehicle: float = 1.8,
    beta_vehicle: float = 0.55,
    lambda_vehicle: float = 0.06,
    sigma_thin_deg: float = 14.0,
    p_thin: float = 1.7,
    beta_thin: float = 0.85,
    lambda_thin: float = 0.10,

    has_bg_class0: bool = False,

    use_skeleton: bool = True,
) -> Tuple[List[GTGroup], dict]:
    """
    Returns: (groups, image_level_diag)
    image_level_diag includes:
      total_scrib_valid, unmatched_scrib_area, unmatched_ratio
    """
    H, W = scrib_idx.shape[:2]
    valid_scrib = (scrib_idx != canvas_bg) & (scrib_idx != IGNORE)
    scrib_comps, bg_mask, total_scrib_valid = extract_scribble_components(
        scrib_idx, canvas_bg, min_area=min_scrib_comp_area, has_bg_class0=bool(has_bg_class0)
    )
    gt_comps = extract_gt_components(gt_idx, min_area=min_gt_comp_area, has_bg_class0=bool(has_bg_class0))

    if total_scrib_valid == 0 or len(gt_comps) == 0:
        preserve_idx = np.full_like(scrib_idx, canvas_bg, dtype=np.uint8)
        preserve_idx[valid_scrib] = scrib_idx[valid_scrib]
        return [], dict(
            bg_mask=bg_mask,
            preserve_idx=preserve_idx,
            total_scrib_valid=total_scrib_valid,
            unmatched_scrib_area=0,
            unmatched_scrib_ratio=0.0,
        )

    assigns: Dict[int, List[ScribbleComp]] = {i: [] for i in range(len(gt_comps))}
    match_ious: Dict[int, List[float]] = {i: [] for i in range(len(gt_comps))}
    unmatched_area = 0

    for s in scrib_comps:
        best_j, best_i = None, -1.0
        for j, g in enumerate(gt_comps):
            if g.class_id != s.class_id:
                continue
            v = iou_bin(s.mask, g.mask)
            if v > best_i:
                best_i = v
                best_j = j
        if best_j is None:
            unmatched_area += int(s.area)
        else:
            assigns[best_j].append(s)
            match_ious[best_j].append(float(best_i))

    unmatched_ratio = float(unmatched_area) / float(total_scrib_valid + 1e-6)

    rng = np.random.default_rng(int(rng_seed))
    tmp = []
    for j, g in enumerate(gt_comps):
        scribs = assigns.get(j, [])
        if not scribs:
            continue

        union = np.zeros((H, W), dtype=np.uint8)
        for s in scribs:
            union[s.mask > 0] = 1
        area_u = int(union.sum())
        area_share = float(area_u) / float(total_scrib_valid + 1e-6)

        if area_u < float(min_group_ratio) * float(total_scrib_valid + 1e-6):
            continue

        theta_s, _ = pca_angle_rho_from_binary(union, use_skeleton=use_skeleton)
        theta_g, _rho_tmp = pca_angle_rho_from_binary(g.mask, use_skeleton=use_skeleton)
        # IMPORTANT: use area-based rho for anisotropy (avoid skeleton-induced bias on blocky GT)
        _ang_area, rho_g = pca_angle_rho_from_binary(g.mask, use_skeleton=False)

        gate = gate_from_rho_sigmoid(float(rho_g), rho0=float(rho0), k=float(k_gate))
        w = weight_from_union_skeleton(union, gate=gate, gate_pow=gate_pow, w_min=w_min, w_max=w_max)

        sigma_deg_c, p_c, beta_c, lambda_c = profile_params_for_group(
            class_id=int(g.class_id),
            sigma_stuff_deg=float(sigma_stuff_deg),
            p_stuff=float(p_stuff),
            beta_stuff=float(beta_stuff),
            lambda_stuff=float(lambda_stuff),
            sigma_vehicle_deg=float(sigma_vehicle_deg),
            p_vehicle=float(p_vehicle),
            beta_vehicle=float(beta_vehicle),
            lambda_vehicle=float(lambda_vehicle),
            sigma_thin_deg=float(sigma_thin_deg),
            p_thin=float(p_thin),
            beta_thin=float(beta_thin),
            lambda_thin=float(lambda_thin),
        )
        sigma_c = float(np.deg2rad(float(sigma_deg_c)))

        # --- feasible-angle geometry curve (optional) ---
        if bool(use_feasible_geom):
            rin = compute_rin_curve_roi(
                union_bin=union.astype(np.uint8),
                gt_bin=(g.mask > 0).astype(np.uint8),
                alphas=alphas,
                center_xy=center_xy_from_bin(union) or g.center_xy,
                dilate_px=int(geom_dilate_px),
            )
            a0, a1 = feasible_angles_from_rin(
                alphas=alphas,
                rin=rin,
                thr_strict=float(rin_thr_strict),
                thr_soft=float(rin_thr_soft),
            )
        else:
            rin = np.ones_like(alphas, dtype=np.float32)
            a0, a1 = float(np.pi / 2.0), float(np.pi / 2.0)

        # direction anisotropy: square/round GT -> lower gate, but keep a minimum for stability.
        if float(rho_g) < float(rho_omega_off):
            omega = float(omega_min)
        else:
            omega = float(np.clip(gate, 0.0, 1.0)) ** float(max(0.1, omega_pow))
            omega = float(max(float(omega_min), omega))

        max_abs_alpha = max_alpha_for_area_share(
            area_share=area_share,
            small_share=float(small_area_share),
            large_share=float(large_group_share),
            small_deg=float(max_alpha_small_deg),
            large_deg=float(max_alpha_large_deg),
            jitter_deg=float(max_alpha_jitter_deg),
            rng=rng,
        )
        min_abs_alpha = min_alpha_for_area_share(
            area_share=area_share,
            small_share=float(small_area_share),
            min_small_deg=float(min_alpha_small_deg),
        )

        sg_list = []
        for i_a, a in enumerate(alphas):
            rin_thr_strict_eff = float(rin_thr_strict)
            rin_thr_soft_eff = float(rin_thr_soft)
            if float(area_share) >= float(large_group_share):
                rin_thr_strict_eff = max(rin_thr_strict_eff, float(large_rin_thr_strict))
                rin_thr_soft_eff = max(rin_thr_soft_eff, float(large_rin_thr_soft))
            if bool(use_feasible_geom):
                gg = g_from_rin_soft(
                    rin_val=float(rin[i_a]),
                    thr_strict=float(rin_thr_strict_eff),
                    thr_soft=float(rin_thr_soft_eff),
                    eps=float(reach_eps),
                    q=float(reach_q),
                    hard_reach=bool(hard_reach_gate),
                )
            else:
                gg = 1.0
            if abs(float(a)) > float(max_abs_alpha) + 1e-9:
                sg_list.append(0.0)
                continue
            gg = float(max(0.0, gg)) ** float(max(0.05, beta_c))

            # direction score as function of final axial diff, with anisotropy weight omega
            if theta_s is None or theta_g is None:
                diff = 0.0
            else:
                sa = angle_wrap_pi(float(theta_s) + float(a))
                diff = angle_diff_axial(sa, float(theta_g))
            ss = sdir_exp_weighted(float(diff), sigma_rad=sigma_c, p=float(p_c), omega=omega)

            # small penalty on large angle to reduce unrealistic turns
            damp = float(np.exp(-float(lambda_c) * float(abs(a))))
            sg_list.append(float(gg) * float(ss) * float(damp))

        table = AlphaTable(
            alphas=alphas.astype(np.float32),
            sg=np.asarray(sg_list, dtype=np.float32),
            rin=np.asarray(rin, dtype=np.float32) if rin is not None else None,
        )

        tmp.append((area_u, dict(
            class_id=g.class_id,
            gt_mask=(g.mask > 0).astype(np.uint8),
            union_mask=union.astype(np.uint8),
            scribs=scribs,
            theta_s=theta_s,
            theta_g=theta_g,
            gt_rho=float(rho_g),
            gate=float(gate),
            omega=float(omega),
            w=float(w),
            a0=float(a0),
            a1=float(a1),
            sigma_deg=float(sigma_deg_c),
            p_exp=float(p_c),
            geom_beta=float(beta_c),
            angle_lambda=float(lambda_c),
            area_share=float(area_share),
            max_abs_alpha=float(max_abs_alpha),
            min_abs_alpha=float(min_abs_alpha),
            table=table,
            ious=match_ious.get(j, []),
        )))

    tmp.sort(key=lambda x: x[0], reverse=True)
    tmp = choose_groups_stratified(
        tmp=tmp,
        max_groups_keep=int(max_groups_keep),
        rng=rng,
        random_group_keep_prob=float(random_group_keep_prob),
        large_group_share=float(large_group_share),
        large_group_keep_prob=float(large_group_keep_prob),
        small_group_boost=float(small_group_boost),
    )

    groups: List[GTGroup] = []
    for gid, (_, d) in enumerate(tmp):
        groups.append(GTGroup(gid=gid, **d))

    # Pixels not represented by selected optimization groups are preserved as-is.
    selected = np.zeros((H, W), dtype=bool)
    for g in groups:
        for s in g.scribs:
            selected |= (s.mask > 0)
    preserve_mask = valid_scrib & (~selected)
    preserve_idx = np.full_like(scrib_idx, canvas_bg, dtype=np.uint8)
    preserve_idx[preserve_mask] = scrib_idx[preserve_mask]

    return groups, dict(
        bg_mask=bg_mask,
        preserve_idx=preserve_idx,
        total_scrib_valid=int(total_scrib_valid),
        unmatched_scrib_area=int(unmatched_area),
        unmatched_scrib_ratio=float(unmatched_ratio),
    )


# -------------------------
# Optimizer (coarse table coord descent)
# -------------------------
def overall_score_table(groups: List[GTGroup], idx_map: Dict[int, int]) -> float:
    if not groups:
        return 0.5
    den = float(sum(g.w for g in groups)) + 1e-8
    num = 0.0
    for g in groups:
        k = int(idx_map.get(g.gid, 0))
        num += float(g.w) * float(g.table.sg[k])
    return float(num / den)


def select_active_group_ids(groups: List[GTGroup],
                            ratio: float,
                            filename: str,
                            rng_seed: int,
                            active_group_prob: float,
                            min_active_groups: int,
                            large_group_share: float,
                            large_group_active_prob: float,
                            avoid_gids: Optional[Set[int]] = None,
                            max_avoid_overlap: float = 0.35) -> Set[int]:
    if not groups:
        return set()
    rng = np.random.default_rng(stable_seed_from_parts("active", rng_seed, filename, f"{float(ratio):.4f}"))
    p_base = float(np.clip(active_group_prob, 0.05, 1.0))
    p_large = float(np.clip(large_group_active_prob, 0.02, 1.0))
    avoid = set(int(x) for x in (avoid_gids or set()))
    need = int(max(1, min(min_active_groups, len(groups))))
    jitter = float(rng.uniform(0.60, 1.35))
    want = int(round(float(len(groups)) * p_base * jitter))
    want = int(np.clip(want, need, len(groups)))

    candidates = list(groups)
    weights = []
    for g in candidates:
        gid = int(g.gid)
        w = 1.0 / math.sqrt(max(float(g.union_mask.sum()), 1.0))
        if float(getattr(g, "area_share", 0.0)) >= float(large_group_share):
            w *= p_large
        if gid in avoid:
            w *= 0.08
        w *= float(rng.uniform(0.75, 1.25))
        weights.append(max(w, 1e-9))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= float(weights.sum())
    picks = rng.choice(len(candidates), size=want, replace=False, p=weights)
    active: Set[int] = set(int(candidates[int(i)].gid) for i in np.atleast_1d(picks))

    max_overlap = int(math.floor(float(max_avoid_overlap) * max(1, len(active))))
    overlap = sorted(active & avoid)
    if len(overlap) > max_overlap:
        replace_n = len(overlap) - max_overlap
        fresh = [g for g in groups if int(g.gid) not in active and int(g.gid) not in avoid]
        if fresh:
            drop = set(int(x) for x in rng.choice(overlap, size=min(replace_n, len(overlap)), replace=False))
            active -= drop
            fresh_w = np.asarray([1.0 / math.sqrt(max(float(g.union_mask.sum()), 1.0)) for g in fresh], dtype=np.float64)
            fresh_w /= float(fresh_w.sum())
            add_n = min(len(drop), len(fresh))
            add = rng.choice(len(fresh), size=add_n, replace=False, p=fresh_w)
            for i in np.atleast_1d(add):
                active.add(int(fresh[int(i)].gid))
    return active


def reachable_score_bounds_table(groups: List[GTGroup],
                                 idx_map: Dict[int, int],
                                 active_gids: Optional[Set[int]] = None) -> Tuple[float, float]:
    if not groups:
        return 0.5, 0.5
    den = float(sum(g.w for g in groups)) + 1e-8
    lo_num = 0.0
    hi_num = 0.0
    active = active_gids if active_gids is not None else {int(g.gid) for g in groups}
    for g in groups:
        if int(g.gid) in active:
            cand_idx = candidate_indices_for_group(g, enforce_min_alpha=True)
            cand = np.asarray(g.table.sg)[cand_idx]
            lo_num += float(g.w) * float(np.min(cand))
            hi_num += float(g.w) * float(np.max(cand))
        else:
            k = int(idx_map.get(g.gid, 0))
            v = float(g.table.sg[k])
            lo_num += float(g.w) * v
            hi_num += float(g.w) * v
    return float(min(lo_num, hi_num) / den), float(max(lo_num, hi_num) / den)


def expand_active_until_target_reachable(groups: List[GTGroup],
                                         idx_map: Dict[int, int],
                                         active_gids: Set[int],
                                         target: float,
                                         rng_seed: int,
                                         large_group_share: float,
                                         large_group_active_prob: float) -> Tuple[Set[int], float, float, bool]:
    active = set(int(x) for x in active_gids)
    lo, hi = reachable_score_bounds_table(groups, idx_map, active_gids=active)
    if not groups or (lo <= float(target) <= hi):
        return active, lo, hi, False

    rng = np.random.default_rng(int(rng_seed))
    inactive = [g for g in groups if int(g.gid) not in active]
    expanded = False

    while inactive and not (lo <= float(target) <= hi):
        weights = []
        for g in inactive:
            w = 1.0 / math.sqrt(max(float(g.union_mask.sum()), 1.0))
            # Keep large objects possible, but still slightly less likely than small ones.
            if float(getattr(g, "area_share", 0.0)) >= float(large_group_share):
                w *= float(np.clip(large_group_active_prob, 0.10, 1.0))
            weights.append(max(w, 1e-9))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= float(weights.sum())
        pick = int(rng.choice(len(inactive), p=weights))
        active.add(int(inactive.pop(pick).gid))
        expanded = True
        lo, hi = reachable_score_bounds_table(groups, idx_map, active_gids=active)

    return active, lo, hi, expanded


def candidate_indices_for_group(g: GTGroup, enforce_min_alpha: bool = False) -> np.ndarray:
    alphas = np.asarray(g.table.alphas, dtype=np.float32)
    valid = np.isfinite(g.table.sg) & (np.asarray(g.table.sg) > 0.0)
    valid &= np.abs(alphas) <= float(getattr(g, "max_abs_alpha", np.pi / 2.0)) + 1e-9
    if bool(enforce_min_alpha):
        min_abs = float(getattr(g, "min_abs_alpha", 0.0))
        if min_abs > 0:
            nonzero = valid & (np.abs(alphas) >= min_abs - 1e-9)
            if np.any(nonzero):
                valid = nonzero
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.arange(len(alphas), dtype=np.int64)
    return idx.astype(np.int64)


def coord_descent_fullscan_table(groups: List[GTGroup],
                                idx_map: Dict[int, int],
                                target: float,
                                passes: int = 2,
                                active_gids: Optional[Set[int]] = None,
                                rng_seed: int = 0) -> Tuple[Dict[int, int], float]:
    if not groups:
        return idx_map, abs(0.5 - target)

    active = active_gids if active_gids is not None else {int(g.gid) for g in groups}
    opt_groups = [g for g in groups if int(g.gid) in active]
    if not opt_groups:
        return idx_map, abs(overall_score_table(groups, idx_map) - target)

    den = float(sum(g.w for g in groups)) + 1e-8
    num = 0.0
    for g in groups:
        k = int(idx_map.get(g.gid, 0))
        num += float(g.w) * float(g.table.sg[k])

    best_err = abs(num / den - target)

    rng = np.random.default_rng(int(rng_seed))
    for _ in range(max(1, int(passes))):
        changed = False
        order = list(opt_groups)
        rng.shuffle(order)
        for g in order:
            k_cur = int(idx_map.get(g.gid, 0))
            sg_cur = float(g.table.sg[k_cur])
            base_num = num - float(g.w) * sg_cur

            cand_idx = candidate_indices_for_group(g, enforce_min_alpha=True)
            scores = (base_num + float(g.w) * g.table.sg[cand_idx]) / den
            k_best = int(cand_idx[int(np.argmin(np.abs(scores - target)))])
            if k_best != k_cur:
                idx_map[g.gid] = k_best
                num = base_num + float(g.w) * float(g.table.sg[k_best])
                changed = True

        err_now = abs(num / den - target)
        best_err = min(best_err, err_now)
        if not changed:
            break

    return idx_map, best_err


# -------------------------
# Render (NO CLIP)
# -------------------------
def rotate_label_region_no_clip(class_id: int,
                               region_bin: np.ndarray,
                               angle_rad: float,
                               center_xy: Tuple[float, float],
                               canvas_bg: int,
                               hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    lab = np.full((H, W), canvas_bg, dtype=np.uint8)
    lab[region_bin > 0] = np.uint8(class_id)
    M = cv2.getRotationMatrix2D(center_xy, float(np.degrees(angle_rad)), 1.0)
    rot = cv2.warpAffine(
        lab, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(canvas_bg),
    )
    return rot

def render_scribble_no_clip(groups: List[GTGroup],
                            idx_map: Dict[int, int],
                            canvas_bg: int,
                            hw: Tuple[int, int],
                            keep_bg_mask: Optional[np.ndarray],
                            preserve_idx: Optional[np.ndarray],
                            keep_bg_as_class0: bool = False) -> np.ndarray:
    H, W = hw
    out = np.full((H, W), canvas_bg, dtype=np.uint8)

    for g in groups:
        k = int(idx_map.get(g.gid, 0))
        alpha = float(g.table.alphas[k])
        for s in g.scribs:
            rot = rotate_label_region_no_clip(s.class_id, s.mask, alpha, s.center_xy, canvas_bg, (H, W))
            m = (rot != canvas_bg)
            out[m] = rot[m]

    if keep_bg_as_class0 and keep_bg_mask is not None:
        bgm = (keep_bg_mask > 0)
        pos = bgm & (out == canvas_bg)
        out[pos] = np.uint8(BACKGROUND_CLASS_ID)

    # Preserve scribble components that were filtered out from optimization
    # (e.g., tiny or unmatched components), so they are not silently dropped.
    if preserve_idx is not None:
        pm = (preserve_idx != canvas_bg) & (preserve_idx != IGNORE)
        pos = pm & (out == canvas_bg)
        out[pos] = preserve_idx[pos]

    return out


# -------------------------
# Diagnostics panels
# -------------------------
def draw_axis(img_bgr: np.ndarray, center_xy: Tuple[float,float], theta: float,
              length: float, color: Tuple[int,int,int], thickness: int):
    if theta is None:
        return
    cx, cy = center_xy
    dx = math.cos(theta)
    dy = math.sin(theta)
    x1 = int(round(cx - dx * length))
    y1 = int(round(cy - dy * length))
    x2 = int(round(cx + dx * length))
    y2 = int(round(cy + dy * length))
    cv2.line(img_bgr, (x1,y1), (x2,y2), color, thickness, lineType=cv2.LINE_AA)

def make_panel_for_image(
    fname: str,
    gt_idx: np.ndarray,
    scrib_idx: np.ndarray,
    canvas_bg: int,
    gt_unknown_ratio: float,
    groups: List[GTGroup],
    img_diag: dict,
    group_rows: List[dict],
    idx_map: Optional[Dict[int, int]],
    out_path: str,
):
    H, W = gt_idx.shape[:2]
    base = np.full((H, W, 3), 245, dtype=np.uint8)

    before_all = ((scrib_idx != canvas_bg) & (scrib_idx != IGNORE)).astype(np.uint8)
    preserve_idx = img_diag.get("preserve_idx", None) if img_diag is not None else None
    final_idx = render_scribble_no_clip(
        groups=groups,
        idx_map=idx_map,
        canvas_bg=canvas_bg,
        hw=(H, W),
        keep_bg_mask=None,
        preserve_idx=preserve_idx,
        keep_bg_as_class0=False,
    )
    after_all = ((final_idx != canvas_bg) & (final_idx != IGNORE)).astype(np.uint8)
    per_group_after: Dict[int, np.ndarray] = {}

    if idx_map is None:
        idx_map = {g.gid: int(np.argmin(np.abs(g.table.alphas - 0.0))) for g in groups}

    for g in groups:
        k = int(idx_map.get(g.gid, 0))
        alpha = float(g.table.alphas[k])
        gm = np.zeros((H, W), dtype=np.uint8)
        for s in g.scribs:
            rot = rotate_label_region_no_clip(s.class_id, s.mask, alpha, s.center_xy, canvas_bg, (H, W))
            gm[rot != canvas_bg] = 1
        per_group_after[int(g.gid)] = gm

    before_only = (before_all > 0) & (after_all == 0)
    after_only = (after_all > 0) & (before_all == 0)
    overlap = (before_all > 0) & (after_all > 0)
    base[overlap] = (115, 115, 115)
    base[before_only] = (0, 185, 0)   # old position: green
    base[after_only] = (0, 0, 230)    # new position: red

    def put_angle(text: str, xy: Tuple[float, float]):
        x = int(np.clip(round(xy[0]), 4, W - 120))
        y = int(np.clip(round(xy[1]), 16, H - 6))
        cv2.putText(base, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 210), 1, cv2.LINE_AA)

    for g in groups:
        k = int(idx_map.get(g.gid, 0))
        alpha_deg = float(np.degrees(float(g.table.alphas[k])))
        if abs(alpha_deg) < 0.5:
            continue
        before_c = center_xy_from_bin(g.union_mask)
        after_c = center_xy_from_bin(per_group_after.get(int(g.gid), np.zeros((H, W), dtype=np.uint8)))
        if before_c is None and after_c is None:
            continue
        if before_c is not None and after_c is not None:
            cv2.line(base, (int(round(before_c[0])), int(round(before_c[1]))),
                     (int(round(after_c[0])), int(round(after_c[1]))),
                     (0, 0, 160), 1, lineType=cv2.LINE_AA)
            label_xy = ((before_c[0] + after_c[0]) * 0.5 + 4.0, (before_c[1] + after_c[1]) * 0.5 - 4.0)
        else:
            label_xy = after_c or before_c
        put_angle(f"{alpha_deg:+.0f}deg", label_xy)

    # put text panel at right (extend canvas)
    pad = 420
    canvas = np.full((H, W + pad, 3), 245, dtype=np.uint8)
    canvas[:, :W] = base

    y = 25
    def put(line, dy=20):
        nonlocal y
        cv2.putText(canvas, line, (W + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1, cv2.LINE_AA)
        y += dy

    put(f"{fname}")
    put(f"canvas_bg={canvas_bg}  gt_unknown={gt_unknown_ratio:.4f}")
    put(f"scrib_valid={img_diag.get('total_scrib_valid',0)}  unmatched_ratio={img_diag.get('unmatched_scrib_ratio',0.0):.3f}")
    put(f"groups={len(groups)}")
    put("old=green  new=red  overlap=gray")

    put("-"*30, dy=18)
    # show top 6 lowest-sdir groups for quick inspection
    rows_sorted = sorted(group_rows, key=lambda r: float(r.get("sdir0", 1.0)))
    for r in rows_sorted[:min(6, len(rows_sorted))]:
        put(f"gid{r['gid']} cls{r['class_id']} sdir0={r['sdir0']:.3f} diff0={r['diff0_deg']:.1f}")
        put(f"  rho={r['gt_rho']:.2f} gate={r['gate']:.2f} omega={r.get('omega',1.0):.2f} w={r['w']:.3f} iou(min/mean)={r['iou_min']:.3f}/{r['iou_mean']:.3f}", dy=18)

    cv2.imwrite(out_path, canvas)


# -------------------------
# Main runner (ratio passes + diagnostics)
# -------------------------
def run_direction_noise_axis_exp_v19_2(
    GT_DIR: str,
    SCRIBBLE_DIR: str,
    OUT_ROOT: str,
    EXP_PREFIX: str,
    target_ratios: List[float],
    max_images: int = 200,

    # comps
    min_scrib_comp_area: int = 1,
    min_gt_comp_area: int = 30,
    min_group_ratio: float = 0.001,
    max_groups_keep: int = 12,
    rng_seed: int = 20260424,
    random_group_keep_prob: float = 0.85,
    active_group_prob: float = 0.35,
    min_active_groups: int = 2,
    large_group_share: float = 0.25,
    large_group_keep_prob: float = 0.35,
    large_group_active_prob: float = 0.45,
    active_avoid_recent: int = 1,
    max_active_overlap_recent: float = 0.20,
    small_group_boost: float = 2.5,
    small_area_share: float = 0.01,
    max_alpha_small_deg: float = 80.0,
    max_alpha_large_deg: float = 32.0,
    max_alpha_jitter_deg: float = 8.0,
    min_alpha_small_deg: float = 10.0,
    large_rin_thr_strict: float = 0.96,
    large_rin_thr_soft: float = 0.72,

    # exp score
    sigma_deg: float = 25.0,
    p_exp: float = 2.0,
    use_class_profiles: bool = True,
    sigma_stuff_deg: float = 22.0,
    p_stuff: float = 2.0,
    beta_stuff: float = 0.35,
    lambda_stuff: float = 0.03,
    sigma_vehicle_deg: float = 18.0,
    p_vehicle: float = 1.8,
    beta_vehicle: float = 0.55,
    lambda_vehicle: float = 0.06,
    sigma_thin_deg: float = 14.0,
    p_thin: float = 1.7,
    beta_thin: float = 0.85,
    lambda_thin: float = 0.10,

    # gate
    rho0: float = 1.25,
    k_gate: float = 6.0,
    gate_pow: float = 1.0,
    rho_omega_off: float = 1.25,
    omega_min: float = 0.2,

    # weight clamp
    w_min: float = 0.05,
    w_max: float = 1.00,

    # alpha table
    alpha_max_deg: float = 90.0,
    coarse_step_deg: float = 6.0,
    coarse_passes: int = 2,

    # tol
    use_adaptive_tol: bool = True,
    tol_fixed: float = 0.05,
    tol_min: float = 0.02,
    tol_rel: float = 0.10,
    target_clip_margin: float = 0.005,

    # background strokes
    keep_bg_strokes: bool = False,
    has_bg_class0: bool = False,

    # reach gate
    use_feasible_geom: bool = True,
    rin_thr_strict: float = 0.90,
    rin_thr_soft: float = 0.55,
    reach_eps: float = 0.02,
    reach_q: float = 1.5,
    hard_reach_gate: bool = False,
    geom_dilate_px: int = 1,

    # diagnostics thresholds
    near0_thr: float = 0.05,
    low_thr: float = 0.5,

    # panels
    make_panels: bool = True,
    max_panels_per_ratio: int = 80,
    debug_first_n: int = 2,
):
    exp_root = os.path.join(OUT_ROOT, EXP_PREFIX)
    ensure_dir(exp_root)
    ensure_dir(os.path.join(exp_root, "plots"))

    pairs = collect_png_pairs(GT_DIR, SCRIBBLE_DIR)
    if max_images and max_images > 0:
        pairs = pairs[:max_images]

    alpha_max = float(np.deg2rad(alpha_max_deg))
    coarse_step = float(np.deg2rad(coarse_step_deg))
    alphas = np.arange(-alpha_max, alpha_max + 1e-9, coarse_step, dtype=np.float32)
    alphas = np.unique(np.concatenate([alphas, np.asarray([0.0], dtype=np.float32)])).astype(np.float32)

    # cache per image (to avoid re-extract per ratio)
    cache: Dict[str, dict] = {}

    for i, (fname, sp, gp, rel_out) in enumerate(pairs):
        scrib, _ = read_idx_or_rgb_as_idx(sp, is_gt=False)
        gt, gt_unknown_ratio = read_idx_or_rgb_as_idx(gp, is_gt=True)
        gt = maybe_align_cityscapes_ids(gt, scrib)

        canvas_bg = infer_canvas_bg_value(scrib)

        if i < debug_first_n:
            print(f"[DEBUG] {fname} scrib unique(head): {np.unique(scrib)[:20]} gt unique(head): {np.unique(gt)[:20]}")
            print(f"[DEBUG] {fname} canvas_bg={canvas_bg} gt_unknown_ratio={gt_unknown_ratio:.6f}")

        groups, img_diag = build_groups_with_table_exp(
            scrib_idx=scrib,
            gt_idx=gt,
            canvas_bg=canvas_bg,
            min_scrib_comp_area=min_scrib_comp_area,
            min_gt_comp_area=min_gt_comp_area,
            min_group_ratio=min_group_ratio,
            max_groups_keep=max_groups_keep,
            alphas=alphas,
            rng_seed=stable_seed_from_parts("groups", rng_seed, fname),
            random_group_keep_prob=float(random_group_keep_prob),
            large_group_share=float(large_group_share),
            large_group_keep_prob=float(large_group_keep_prob),
            small_group_boost=float(small_group_boost),
            small_area_share=float(small_area_share),
            max_alpha_small_deg=float(max_alpha_small_deg),
            max_alpha_large_deg=float(max_alpha_large_deg),
            max_alpha_jitter_deg=float(max_alpha_jitter_deg),
            min_alpha_small_deg=float(min_alpha_small_deg),
            large_rin_thr_strict=float(large_rin_thr_strict),
            large_rin_thr_soft=float(large_rin_thr_soft),
            sigma_deg=sigma_deg,
            p=p_exp,
            rho0=rho0,
            k_gate=k_gate,
            gate_pow=gate_pow,
            w_min=w_min,
            w_max=w_max,
            use_feasible_geom=bool(use_feasible_geom),
            rin_thr_strict=float(rin_thr_strict),
            rin_thr_soft=float(rin_thr_soft),
            reach_eps=float(reach_eps),
            reach_q=float(reach_q),
            hard_reach_gate=bool(hard_reach_gate),
            geom_dilate_px=int(geom_dilate_px),
            rho_omega_off=float(rho_omega_off),
            omega_min=float(omega_min),
            sigma_stuff_deg=float(sigma_stuff_deg if use_class_profiles else sigma_deg),
            p_stuff=float(p_stuff if use_class_profiles else p_exp),
            beta_stuff=float(beta_stuff if use_class_profiles else 1.0),
            lambda_stuff=float(lambda_stuff if use_class_profiles else 0.0),
            sigma_vehicle_deg=float(sigma_vehicle_deg if use_class_profiles else sigma_deg),
            p_vehicle=float(p_vehicle if use_class_profiles else p_exp),
            beta_vehicle=float(beta_vehicle if use_class_profiles else 1.0),
            lambda_vehicle=float(lambda_vehicle if use_class_profiles else 0.0),
            sigma_thin_deg=float(sigma_thin_deg if use_class_profiles else sigma_deg),
            p_thin=float(p_thin if use_class_profiles else p_exp),
            beta_thin=float(beta_thin if use_class_profiles else 1.0),
            lambda_thin=float(lambda_thin if use_class_profiles else 0.0),
            has_bg_class0=bool(has_bg_class0),
            use_skeleton=True,
        )

        # alpha=0 init
        idx0 = {g.gid: int(np.argmin(np.abs(g.table.alphas - 0.0))) for g in groups}
        original = overall_score_table(groups, idx0)

        cache[fname] = dict(
            scrib=scrib,
            gt=gt,
            canvas_bg=int(canvas_bg),
            gt_unknown_ratio=float(gt_unknown_ratio),
            groups=groups,
            img_diag=img_diag,
            original=float(original),
            hw=scrib.shape[:2],
            out_rel=str(rel_out),
        )

    if not cache:
        print("[DONE] no valid images")
        return

    all_records: List[dict] = []

    # prepare group/image debug across ALL ratios (but group sdir0 is independent of ratio)
    # We'll dump group_debug & image_debug once under exp_root.
    group_debug_rows: List[dict] = []
    image_debug_rows: List[dict] = []

    for fname, d in cache.items():
        groups: List[GTGroup] = d["groups"]
        gt_unknown_ratio = float(d["gt_unknown_ratio"])
        img_diag = d["img_diag"]
        original = float(d["original"])

        # group rows at alpha=0 (key for your analysis)
        rows_this_img = []
        for g in groups:
            k0 = int(np.argmin(np.abs(g.table.alphas - 0.0)))
            sdir0 = float(g.table.sg[k0])
            if g.theta_s is None or g.theta_g is None:
                diff0 = 0.0
            else:
                diff0 = float(angle_diff_axial(float(g.theta_s), float(g.theta_g)))
            diff0_deg = float(np.degrees(diff0))

            ious = g.ious if g.ious else []
            iou_min = float(np.min(ious)) if ious else 0.0
            iou_mean = float(np.mean(ious)) if ious else 0.0
            iou_max = float(np.max(ious)) if ious else 0.0

            row = dict(
                filename=fname,
                gid=int(g.gid),
                class_id=int(g.class_id),
                union_area=int(g.union_mask.sum()),
                gt_area=int(g.gt_mask.sum()),
                sdir0=round(sdir0, 6),
                diff0_deg=round(diff0_deg, 3),
                theta_s_deg=(round(float(np.degrees(g.theta_s)), 3) if g.theta_s is not None else np.nan),
                theta_g_deg=(round(float(np.degrees(g.theta_g)), 3) if g.theta_g is not None else np.nan),
                gt_rho=round(float(g.gt_rho), 6),
                gate=round(float(g.gate), 6),
                omega=round(float(getattr(g, 'omega', 1.0)), 6),
                w=round(float(g.w), 6),
                area_share=round(float(getattr(g, "area_share", 0.0)), 6),
                max_abs_alpha_deg=round(float(np.degrees(getattr(g, "max_abs_alpha", np.pi / 2.0))), 3),
                min_abs_alpha_deg=round(float(np.degrees(getattr(g, "min_abs_alpha", 0.0))), 3),
                iou_min=round(iou_min, 6),
                iou_mean=round(iou_mean, 6),
                iou_max=round(iou_max, 6),
            )
            rows_this_img.append(row)
            group_debug_rows.append(row)

        if rows_this_img:
            min_sdir0 = float(min(r["sdir0"] for r in rows_this_img))
            mean_sdir0 = float(np.mean([r["sdir0"] for r in rows_this_img]))
        else:
            min_sdir0 = 1.0
            mean_sdir0 = 1.0

        image_debug_rows.append(dict(
            filename=fname,
            original_score=round(original, 6),
            group_count=int(len(groups)),
            min_group_sdir0=round(min_sdir0, 6),
            mean_group_sdir0=round(mean_sdir0, 6),
            gt_unknown_ratio=round(gt_unknown_ratio, 6),
            total_scrib_valid=int(img_diag.get("total_scrib_valid", 0)),
            unmatched_scrib_area=int(img_diag.get("unmatched_scrib_area", 0)),
            unmatched_scribble_area_ratio=round(float(img_diag.get("unmatched_scrib_ratio", 0.0)), 6),
        ))

    group_debug_df = pd.DataFrame(group_debug_rows).sort_values(["sdir0", "diff0_deg", "filename"]).reset_index(drop=True)
    image_debug_df = pd.DataFrame(image_debug_rows).sort_values(["original_score", "min_group_sdir0", "filename"]).reset_index(drop=True)

    group_debug_df.to_csv(os.path.join(exp_root, "group_debug.csv"), index=False)
    image_debug_df.to_csv(os.path.join(exp_root, "image_debug.csv"), index=False)

    # export near0 + low lists for your analysis
    near0_df = group_debug_df[group_debug_df["sdir0"] <= near0_thr].copy().reset_index(drop=True)
    low_df = group_debug_df[group_debug_df["sdir0"] < low_thr].copy().reset_index(drop=True)
    near0_df.to_csv(os.path.join(exp_root, "near0_groups.csv"), index=False)
    low_df.to_csv(os.path.join(exp_root, "low_groups.csv"), index=False)

    # ratio passes
    active_history: Dict[str, List[Set[int]]] = {fname: [] for fname in cache.keys()}
    for ratio in target_ratios:
        ratio_dir = os.path.join(exp_root, f"ratio_{ratio:.2f}")
        ensure_dir(ratio_dir)
        scrib_out_dir = os.path.join(ratio_dir, "direction_noise")
        ensure_dir(scrib_out_dir)
        panel_dir = os.path.join(ratio_dir, "panels")
        ensure_dir(panel_dir)

        records: List[dict] = []
        idx_maps_for_panel: Dict[str, Dict[int, int]] = {}

        for fname in sorted(cache.keys()):
            d = cache[fname]
            canvas_bg = int(d["canvas_bg"])
            groups: List[GTGroup] = d["groups"]
            img_diag = d["img_diag"]
            bg_mask = img_diag.get("bg_mask", None)
            preserve_idx = img_diag.get("preserve_idx", None)
            original = float(d["original"])
            hw = d["hw"]
            out_rel = str(d.get("out_rel", fname))

            target_raw = float(np.clip(original * float(ratio), 0.0, 1.0))
            target = float(target_raw)
            idx_map = {g.gid: int(np.argmin(np.abs(g.table.alphas - 0.0))) for g in groups}
            recent = active_history.get(fname, [])
            avoid_gids: Set[int] = set()
            for old in recent[-int(max(0, active_avoid_recent)):]:
                avoid_gids |= set(int(x) for x in old)
            active_gids = select_active_group_ids(
                groups=groups,
                ratio=float(ratio),
                filename=fname,
                rng_seed=int(rng_seed),
                active_group_prob=float(active_group_prob),
                min_active_groups=int(min_active_groups),
                large_group_share=float(large_group_share),
                large_group_active_prob=float(large_group_active_prob),
                avoid_gids=avoid_gids,
                max_avoid_overlap=float(max_active_overlap_recent),
            )
            active_expanded = False
            if groups:
                active_gids, s_min, s_max, active_expanded = expand_active_until_target_reachable(
                    groups=groups,
                    idx_map=idx_map,
                    active_gids=active_gids,
                    target=target_raw,
                    rng_seed=stable_seed_from_parts("expand", rng_seed, fname, f"{float(ratio):.4f}"),
                    large_group_share=float(large_group_share),
                    large_group_active_prob=float(large_group_active_prob),
                )
                lo = float(min(s_min, s_max)) + float(target_clip_margin)
                hi = float(max(s_min, s_max)) - float(target_clip_margin)
                if lo <= hi:
                    target = float(np.clip(target_raw, lo, hi))
            active_history.setdefault(fname, []).append(set(active_gids))
            target_was_clipped = bool(abs(float(target) - float(target_raw)) > 1e-9)
            tol = float(max(tol_min, tol_rel * target)) if use_adaptive_tol else float(tol_fixed)

            idx_map, best_err = coord_descent_fullscan_table(
                groups,
                idx_map,
                target,
                passes=coarse_passes,
                active_gids=active_gids,
                rng_seed=stable_seed_from_parts("coord", rng_seed, fname, f"{float(ratio):.4f}"),
            )
            idx_maps_for_panel[fname] = dict(idx_map)

            final_score = overall_score_table(groups, idx_map)
            abs_error = abs(final_score - target)
            success = bool(abs_error <= tol)

            reach_violation = np.nan
            large_angle_rate = np.nan
            if groups:
                rv = []
                la = []
                for g in groups:
                    k_sel = int(idx_map.get(g.gid, 0))
                    alpha_sel = float(abs(g.table.alphas[k_sel]))
                    if g.table.rin is not None and len(g.table.rin) > k_sel:
                        rv.append(1.0 if float(g.table.rin[k_sel]) < float(rin_thr_soft) else 0.0)
                    la.append(1.0 if alpha_sel > float(g.a1) + 1e-9 else 0.0)
                if rv:
                    reach_violation = float(np.mean(rv))
                if la:
                    large_angle_rate = float(np.mean(la))

            out = render_scribble_no_clip(
                groups=groups,
                idx_map=idx_map,
                canvas_bg=canvas_bg,
                hw=hw,
                keep_bg_mask=bg_mask if keep_bg_strokes else None,
                preserve_idx=preserve_idx,
                keep_bg_as_class0=bool(has_bg_class0)
            )
            out_path = os.path.join(scrib_out_dir, out_rel)
            ensure_dir(os.path.dirname(out_path))
            cv2.imwrite(out_path, out)

            rec = dict(
                filename=fname,
                target_ratio=float(ratio),
                original_score=round(original, 6),
                target_score_raw=round(target_raw, 6),
                target_score=round(target, 6),
                final_score=round(float(final_score), 6),
                abs_error=round(float(abs_error), 6),
                tol_used=round(float(tol), 6),
                success=bool(success),
                group_count=int(len(groups)),
                active_group_count=int(len(active_gids)),
                active_group_ids=";".join(str(int(x)) for x in sorted(active_gids)),
                active_avoid_overlap_count=int(len(active_gids & avoid_gids)),
                active_expanded_for_reach=bool(active_expanded),
                target_clipped=bool(target_was_clipped),
                best_possible_err=round(float(best_err), 6),
                reach_violation_rate=round(float(reach_violation), 6) if np.isfinite(reach_violation) else np.nan,
                large_angle_rate=round(float(large_angle_rate), 6) if np.isfinite(large_angle_rate) else np.nan,
            )
            records.append(rec)
            all_records.append(rec)

        df = pd.DataFrame(records).sort_values(["filename"]).reset_index(drop=True)
        df.to_csv(os.path.join(ratio_dir, f"direction_noise_ratio_{ratio:.2f}_long.csv"), index=False)

        summary_row = {
            "target_ratio": float(ratio),
            "n": int(len(df)),
            "success_rate": float(df["success"].mean()) if len(df) else 0.0,
            "abs_error_mean": float(df["abs_error"].mean()) if len(df) else 0.0,
            "abs_error_median": float(df["abs_error"].median()) if len(df) else 0.0,
            "reach_violation_rate": float(pd.to_numeric(df["reach_violation_rate"], errors="coerce").mean()) if len(df) else np.nan,
            "large_angle_rate": float(pd.to_numeric(df["large_angle_rate"], errors="coerce").mean()) if len(df) else np.nan,
        }
        pd.DataFrame([summary_row]).to_csv(os.path.join(ratio_dir, f"direction_noise_ratio_{ratio:.2f}_summary.csv"), index=False)

        # panels: prioritize near0 and low (ratio-independent, based on sdir0)
        if make_panels:
            # candidates from near0 first, then low
            cand_files = []
            if len(near0_df) > 0:
                cand_files += near0_df["filename"].unique().tolist()
            if len(low_df) > 0:
                for f in low_df["filename"].unique().tolist():
                    if f not in cand_files:
                        cand_files.append(f)

            cand_files = cand_files[:max_panels_per_ratio]
            for f in cand_files:
                if f not in cache:
                    continue
                dd = cache[f]
                gt = dd["gt"]
                scrib = dd["scrib"]
                canvas_bg = int(dd["canvas_bg"])
                gt_unk = float(dd["gt_unknown_ratio"])
                groups = dd["groups"]
                img_diag = dd["img_diag"]
                # group rows for this img
                gr = group_debug_df[group_debug_df["filename"] == f].to_dict("records")
                outp = os.path.join(panel_dir, f.replace(".png", "_panel.png"))
                make_panel_for_image(
                    fname=f,
                    gt_idx=gt,
                    scrib_idx=scrib,
                    canvas_bg=canvas_bg,
                    gt_unknown_ratio=gt_unk,
                    groups=groups,
                    img_diag=img_diag,
                    group_rows=gr,
                    idx_map=idx_maps_for_panel.get(f),
                    out_path=outp,
                )

        print(f"[DONE] ratio={ratio:.2f} -> {ratio_dir}")

    all_df = pd.DataFrame(all_records).sort_values(["filename", "target_ratio"]).reset_index(drop=True)
    all_df.to_csv(os.path.join(exp_root, "direction_noise_all_ratios_long.csv"), index=False)

    global_summary = (
        all_df.groupby("target_ratio", as_index=False)
              .agg(
                  n=("filename", "count"),
                  success_rate=("success", "mean"),
                  abs_error_mean=("abs_error", "mean"),
                  abs_error_median=("abs_error", "median"),
                  reach_violation_rate=("reach_violation_rate", "mean"),
                  large_angle_rate=("large_angle_rate", "mean"),
              )
    )
    global_summary.to_csv(os.path.join(exp_root, "direction_noise_all_ratios_summary.csv"), index=False)

    # -----------------------------
    # Quality plots (auto)
    # -----------------------------
    try:
        plots_dir = os.path.join(exp_root, "plots")
        ensure_dir(plots_dir)

        curve = (
            all_df.groupby("target_ratio", as_index=False)
                 .agg(
                     mean_original=("original_score", "mean"),
                     mean_target=("target_score", "mean"),
                     mean_final=("final_score", "mean"),
                     success_rate=("success", "mean"),
                     mean_abs_err=("abs_error", "mean"),
                     median_abs_err=("abs_error", "median"),
                     reach_violation_rate=("reach_violation_rate", "mean"),
                     large_angle_rate=("large_angle_rate", "mean"),
                 )
                 .sort_values("target_ratio")
        )

        # 1) mean target vs mean final
        plt.figure()
        plt.plot(curve["target_ratio"], curve["mean_target"], marker="o")
        plt.plot(curve["target_ratio"], curve["mean_final"], marker="o")
        plt.xlabel("target_ratio")
        plt.ylabel("score")
        plt.legend(["mean_target", "mean_final"])
        plt.title("Mean target vs mean final")
        plt.savefig(os.path.join(plots_dir, "plot_mean_target_final.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 2) success rate
        plt.figure()
        plt.plot(curve["target_ratio"], curve["success_rate"], marker="o")
        plt.xlabel("target_ratio")
        plt.ylabel("success_rate")
        plt.title("Success rate by target_ratio")
        plt.savefig(os.path.join(plots_dir, "plot_success_rate.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 3) abs error mean/median
        plt.figure()
        plt.plot(curve["target_ratio"], curve["mean_abs_err"], marker="o")
        plt.plot(curve["target_ratio"], curve["median_abs_err"], marker="o")
        plt.xlabel("target_ratio")
        plt.ylabel("abs_error")
        plt.legend(["mean_abs_error", "median_abs_error"])
        plt.title("Absolute error by target_ratio")
        plt.savefig(os.path.join(plots_dir, "plot_abs_error.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 4) abs error distribution boxplot (hide fliers)
        ratios_sorted = sorted(all_df["target_ratio"].unique())
        data = [all_df.loc[all_df["target_ratio"] == r, "abs_error"].values for r in ratios_sorted]
        plt.figure(figsize=(10, 4))
        plt.boxplot(data, labels=[f"{r:.1f}" for r in ratios_sorted], showfliers=False)
        plt.xlabel("target_ratio")
        plt.ylabel("abs_error")
        plt.title("Abs error distribution (boxplot, fliers hidden)")
        plt.savefig(os.path.join(plots_dir, "plot_abs_error_box.png"), dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[PLOTS] Saved to {plots_dir}")
    except Exception as e:
        print(f"[WARN] Failed to create plots: {e}")

    print("[DONE] Exp finished.")
    print(f"  exp_root={exp_root}")
    print("  diagnostics:")
    print("   - group_debug.csv / image_debug.csv")
    print("   - near0_groups.csv / low_groups.csv")
    print("   - panels under each ratio: ratio_xx/panels/")


if __name__ == "__main__":
    # === your NEW paths ===
    GT_DIR = os.environ.get("CITYSCAPES_GT_DIR", "datasets/cityscapes/GtMask")
    SCRIBBLE_DIR = os.environ.get("CITYSCAPES_SCRIBBLE_DIR", "datasets/cityscapes/auto_scribbles_idx")
    OUT_ROOT = os.environ.get("CITYSCAPES_DIRECTION_OUT_ROOT", "outputs/cityscapes/direction")
    EXP_PREFIX = "direction_noise_auto"

    target_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    run_direction_noise_axis_exp_v19_2(
        GT_DIR=GT_DIR,
        SCRIBBLE_DIR=SCRIBBLE_DIR,
        OUT_ROOT=OUT_ROOT,
        EXP_PREFIX=EXP_PREFIX,
        target_ratios=target_ratios,
        max_images=13000,

        # speed
        coarse_step_deg=6.0,
        coarse_passes=2,

        # scoring
        sigma_deg=25.0,
        p_exp=2.0,

        # gate
        rho0=1.25,
        k_gate=6.0,
        gate_pow=1.0,

        keep_bg_strokes=True,

        # diagnostics thresholds
        near0_thr=0.05,
        low_thr=0.5,

        make_panels=True,
        max_panels_per_ratio=80,
    )
