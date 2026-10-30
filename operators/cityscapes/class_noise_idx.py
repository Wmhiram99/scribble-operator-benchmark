# class_noise_idx_batch_0.1_to_0.9.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from itertools import combinations

# =========================
# Config (edit these)
# =========================
SCRIBBLE_DIR = os.environ.get("CITYSCAPES_SCRIBBLE_DIR", "datasets/cityscapes/auto_scribbles_idx")
GT_DIR       = os.environ.get("CITYSCAPES_GT_DIR", "datasets/cityscapes/GtMask")

OUT_ROOT     = os.environ.get("CITYSCAPES_CLASS_OUT_ROOT", "outputs/cityscapes/class_noise")

# target scores to generate
TARGET_SCORES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# VOC
N_CLASSES = 21
IGNORE = 255


# =========================
# IO (idx-safe)
# =========================
def read_as_idx(path: str) -> np.ndarray:
    """
    Read label as idx mask.
    - If 2D: return directly as uint8.
    - If RGB: try exact VOC palette mapping; unknown -> IGNORE.
      (for safety, but your pipeline is idx already)
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        return arr.astype(np.uint8)

    # RGB fallback: exact color match with VOC bit-mix palette
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.uint8)
        key = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)

        # build VOC colormap mapping
        pal = voc_palette_256()
        pal_key = (pal[:, 0].astype(np.uint32) << 16) | (pal[:, 1].astype(np.uint32) << 8) | pal[:, 2].astype(np.uint32)
        key2id = {int(k): int(i) for i, k in enumerate(pal_key)}

        flat = key.reshape(-1)
        out = np.full_like(flat, IGNORE, dtype=np.uint8)
        for k, v in key2id.items():
            out[flat == k] = v
        return out.reshape(rgb.shape[0], rgb.shape[1])

    raise ValueError(f"[ERROR] Unsupported image shape {arr.shape}: {path}")


def save_idx_png_whitebg(path: str, idx: np.ndarray):
    """
    Save as single-channel indexed mask in L mode:
    - 255 shows as white background
    - 0..20 show as dark lines
    """
    idx = idx.astype(np.uint8)

    # Keep original label space as-is.
    # Some datasets use sparse / non-contiguous ids (or class ids larger than N_CLASSES-1),
    # and force-clamping to IGNORE would visually "erase" valid scribbles.

    Image.fromarray(idx, mode="L").save(path)


def voc_palette_256():
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette[i] = [r, g, b]
    return palette


def infer_canvas_bg(idx: np.ndarray) -> int:
    """
    Scribble canvas background is usually 255 (white), sometimes 0.
    We infer by majority.
    """
    total = idx.size
    cnt255 = int((idx == 255).sum())
    return 255 if (cnt255 / (total + 1e-12)) > 0.5 else 0


# =========================
# Region extraction (on idx)
# =========================
def extract_regions_idx(scrib_idx: np.ndarray, gt_idx: np.ndarray, canvas_bg: int):
    """
    Extract connected regions from scribble foreground:
      fg = scrib != canvas_bg and scrib != IGNORE
    For each region:
      - area: #valid pixels (gt != IGNORE)
      - correct: #pixels where scrib == gt
      - acc: correct/area
      - class_id: dominant scrib class inside region
    """
    fg = (scrib_idx != canvas_bg) & (scrib_idx != IGNORE)
    mask = fg.astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    regions = []
    for rid in range(1, num_labels):
        region_mask = (labels == rid)
        coords = np.argwhere(region_mask)
        if coords.size == 0:
            continue

        ys = coords[:, 0]
        xs = coords[:, 1]

        scr_vals = scrib_idx[ys, xs]
        gt_vals  = gt_idx[ys, xs]

        valid = (scr_vals != canvas_bg) & (scr_vals != IGNORE) & (gt_vals != IGNORE)
        if not np.any(valid):
            continue

        scr_v = scr_vals[valid].astype(np.int32)
        gt_v  = gt_vals[valid].astype(np.int32)

        area = int(scr_v.size)
        correct = int((scr_v == gt_v).sum())

        # mode of scrib values in region
        binc = np.bincount(scr_v.clip(0, 255), minlength=256)
        class_id = int(np.argmax(binc))

        regions.append({
            "rid": rid,
            "mask": region_mask,
            "area": area,
            "correct": correct,
            "acc": correct / (area + 1e-12),
            "class_id": class_id
        })

    return regions


def build_safe_class_remap(region_info, canvas_bg: int):
    """
    Build a deterministic class remap table that never maps to canvas background
    or IGNORE. This guarantees class-noise only changes class ids, not foreground
    support.
    """
    classes = sorted({
        int(r["class_id"])
        for r in region_info
        if int(r["class_id"]) not in (int(canvas_bg), int(IGNORE))
    })
    if len(classes) <= 1:
        return {}

    # Use relative shift in the observed foreground-class list.
    shift = 3 % len(classes)
    if shift == 0:
        shift = 1

    remap = {}
    for i, cid in enumerate(classes):
        remap[cid] = int(classes[(i + shift) % len(classes)])
    return remap


# =========================
# Noise application (keep your original idea)
# =========================
def apply_noise_brute_idx(scrib_idx: np.ndarray, region_info, target_score: float, class_remap=None):
    total_pix = sum(r["area"] for r in region_info)
    if total_pix == 0:
        return scrib_idx.copy(), 0.0

    n = len(region_info)
    best_combo = None
    best_diff = 1.0

    for k in range(n + 1):
        for combo in combinations(range(n), k):
            kept_area = sum(region_info[i]["area"] for i in combo)
            score = kept_area / total_pix
            diff = abs(score - target_score)
            if diff < best_diff:
                best_combo = combo
                best_diff = diff
                if diff < 0.01:
                    break
        if best_diff < 0.01:
            break

    keep = set(best_combo) if best_combo is not None else set()
    noisy = scrib_idx.copy()

    for i, r in enumerate(region_info):
        if i not in keep:
            cid = int(r["class_id"])
            if class_remap is not None and cid in class_remap:
                new_class = int(class_remap[cid])
            elif 0 <= cid < N_CLASSES:
                new_class = int((cid + 3) % N_CLASSES)
            else:
                continue
            noisy[r["mask"]] = np.uint8(new_class)

    final_score = sum(region_info[i]["area"] for i in keep) / total_pix
    return noisy, float(final_score)


def _partial_replace_region_pixels(noisy_idx: np.ndarray, region_mask: np.ndarray, new_class: int, n_replace: int):
    """
    Replace exactly n_replace pixels inside one region.
    We use evenly spaced flat indices for deterministic and stable behavior.
    """
    if n_replace <= 0:
        return 0

    flat = np.flatnonzero(region_mask.reshape(-1))
    if flat.size == 0:
        return 0

    n = int(min(max(0, n_replace), flat.size))
    if n <= 0:
        return 0

    if n == flat.size:
        chosen = flat
    else:
        sel = np.linspace(0, flat.size - 1, num=n, dtype=np.int64)
        chosen = flat[sel]

    noisy_flat = noisy_idx.reshape(-1)
    noisy_flat[chosen] = np.uint8(new_class)
    return int(n)


def apply_noise_hybrid_idx(scrib_idx: np.ndarray, region_info, target_score: float, class_remap=None):
    """
    Region-first replacement with pixel-level fine adjustment.
    This removes quantization errors from pure region-level selection and
    substantially improves hit-rate at high target ratios (0.7~0.9).
    """
    total_pix = int(sum(r["area"] for r in region_info))
    if total_pix == 0:
        return scrib_idx.copy(), 0.0

    target_score = float(np.clip(target_score, 0.0, 1.0))
    target_keep = int(round(target_score * total_pix))
    target_keep = int(np.clip(target_keep, 0, total_pix))
    target_remove = int(total_pix - target_keep)

    noisy = scrib_idx.copy()
    if target_remove <= 0:
        return noisy, 1.0

    # Prefer replacing lower-accuracy regions first.
    regions = sorted(region_info, key=lambda r: (r["acc"], r["area"]))

    removed = 0
    for r in regions:
        if removed >= target_remove:
            break

        cid = int(r["class_id"])
        if class_remap is not None and cid in class_remap:
            new_class = int(class_remap[cid])
        elif 0 <= cid < N_CLASSES:
            new_class = int((cid + 3) % N_CLASSES)
        else:
            continue

        area = int(r["area"])
        remain = int(target_remove - removed)

        if area <= remain:
            noisy[r["mask"]] = np.uint8(new_class)
            removed += area
        else:
            # Fine-grained adjustment inside one region to hit target exactly.
            k = _partial_replace_region_pixels(noisy, r["mask"], new_class, remain)
            removed += int(k)
            break

    final_keep = int(total_pix - removed)
    final_score = float(final_keep / (total_pix + 1e-12))
    return noisy, final_score


# =========================
# Main loop: generate 0.1..0.9
# =========================
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    fnames = [f for f in os.listdir(SCRIBBLE_DIR) if f.lower().endswith(".png")]
    fnames.sort()

    for target_score in TARGET_SCORES:
        out_dir = os.path.join(OUT_ROOT, f"class_noise_{target_score:.1f}")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "score_comparison.csv")
        records = []

        for fname in tqdm(fnames, desc=f"class_noise target={target_score:.1f}"):
            sp = os.path.join(SCRIBBLE_DIR, fname)
            gp = os.path.join(GT_DIR, fname)
            if not os.path.isfile(gp):
                continue

            scrib_idx = read_as_idx(sp)
            gt_idx = read_as_idx(gp)

            canvas_bg = infer_canvas_bg(scrib_idx)

            region_info = extract_regions_idx(scrib_idx, gt_idx, canvas_bg=canvas_bg)
            class_remap = build_safe_class_remap(region_info, canvas_bg=canvas_bg)

            # original score definition: sum(correct pixels of "good regions") / sum(area)
            orig_area = sum(r["correct"] for r in region_info if r["acc"] >= 0.8)
            orig_total = sum(r["area"] for r in region_info)
            orig_score = (orig_area / orig_total) if orig_total else 0.0

            noisy_idx, final_score = apply_noise_hybrid_idx(
                scrib_idx, region_info, target_score, class_remap=class_remap
            )
            method = "hybrid_partial"

            # IMPORTANT: keep canvas background unchanged
            # Our region masks exclude background/ignore already, but be safe:
            noisy_idx[scrib_idx == canvas_bg] = canvas_bg
            noisy_idx[scrib_idx == IGNORE] = IGNORE

            save_idx_png_whitebg(os.path.join(out_dir, fname), noisy_idx)

            records.append({
                "filename": fname,
                "method": method,
                "canvas_bg": int(canvas_bg),
                "original_score": round(float(orig_score), 6),
                "target_score": float(target_score),
                "actual_score": round(float(final_score), 6),
                "num_regions": int(len(region_info)),
            })

        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"[DONE] target={target_score:.1f} -> {out_dir}")
        print(f"[DONE] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
