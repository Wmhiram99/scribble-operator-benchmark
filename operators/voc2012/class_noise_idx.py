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
SCRIBBLE_DIR = os.environ.get("VOC_SCRIBBLE_DIR", "datasets/voc2012/auto_scribbles_idx")
GT_DIR       = os.environ.get("VOC_GT_DIR", "datasets/voc2012/GtMask")

OUT_ROOT     = os.environ.get("VOC_CLASS_OUT_ROOT", "outputs/voc2012/class_noise")

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

    # safety clamp: anything not in [0..20] and not IGNORE -> IGNORE
    bad = (idx != IGNORE) & (idx >= N_CLASSES)
    if np.any(bad):
        idx[bad] = IGNORE

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


# =========================
# Noise application (keep your original idea)
# =========================
def class_replacement(cid: int) -> int:
    return (int(cid) + 3) % N_CLASSES


def corrupted_diag_contribution(gt_idx: np.ndarray, region, new_class: int) -> int:
    region_mask = region["mask"]
    gt_vals = gt_idx[region_mask]
    valid = gt_vals != IGNORE
    if not np.any(valid):
        return 0
    gt_v = gt_vals[valid].astype(np.int32)
    correct = int((gt_v == int(new_class)).sum())
    acc = correct / (int(valid.sum()) + 1e-12)
    return correct if acc >= 0.8 else 0


def diag_contributions_idx(gt_idx: np.ndarray, region_info):
    retained = []
    corrupted = []
    for r in region_info:
        retained.append(int(r["correct"]) if float(r["acc"]) >= 0.8 else 0)
        cid = int(r["class_id"])
        if cid < 0 or cid >= N_CLASSES:
            corrupted.append(0)
        else:
            corrupted.append(corrupted_diag_contribution(gt_idx, r, class_replacement(cid)))
    return retained, corrupted


def apply_noise_brute_idx(scrib_idx: np.ndarray, gt_idx: np.ndarray, region_info, target_score: float):
    total_pix = sum(r["area"] for r in region_info)
    if total_pix == 0:
        return scrib_idx.copy(), 0.0

    n = len(region_info)
    best_combo = None
    best_diff = 1.0
    retained_contrib, corrupted_contrib = diag_contributions_idx(gt_idx, region_info)

    for k in range(n + 1):
        for combo in combinations(range(n), k):
            keep = set(combo)
            score = (
                sum(retained_contrib[i] if i in keep else corrupted_contrib[i] for i in range(n))
                / total_pix
            )
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
            if cid < 0 or cid >= N_CLASSES:
                continue
            new_class = class_replacement(cid)
            noisy[r["mask"]] = np.uint8(new_class)

    final_score = (
        sum(retained_contrib[i] if i in keep else corrupted_contrib[i] for i in range(n))
        / total_pix
    )
    return noisy, float(final_score)


def apply_noise_greedy_idx(scrib_idx: np.ndarray, gt_idx: np.ndarray, region_info, target_score: float):
    total_pix = sum(r["area"] for r in region_info)
    if total_pix == 0:
        return scrib_idx.copy(), 0.0

    n = len(region_info)
    retained_contrib, corrupted_contrib = diag_contributions_idx(gt_idx, region_info)
    keep = set()
    current = sum(corrupted_contrib) / total_pix
    current_diff = abs(current - target_score)

    # Deterministic approximation for large component sets: repeatedly add the
    # region that most improves the diagnostic target error.
    improved = True
    while improved:
        improved = False
        best_i = None
        best_score = current
        best_diff = current_diff
        for i in range(n):
            if i in keep:
                continue
            candidate_score = (
                current
                + (retained_contrib[i] - corrupted_contrib[i]) / total_pix
            )
            candidate_diff = abs(candidate_score - target_score)
            if candidate_diff + 1e-12 < best_diff:
                best_i = i
                best_score = candidate_score
                best_diff = candidate_diff
        if best_i is not None:
            keep.add(best_i)
            current = best_score
            current_diff = best_diff
            improved = True

    noisy = scrib_idx.copy()
    for i, r in enumerate(region_info):
        if i not in keep:
            cid = int(r["class_id"])
            if cid < 0 or cid >= N_CLASSES:
                continue
            new_class = class_replacement(cid)
            noisy[r["mask"]] = np.uint8(new_class)

    return noisy, float(current)


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

            # diagnostic score definition: sum(correct pixels of "good regions") / sum(area)
            orig_area = sum(r["correct"] for r in region_info if r["acc"] >= 0.8)
            orig_total = sum(r["area"] for r in region_info)
            orig_score = (orig_area / orig_total) if orig_total else 0.0
            target_diag_score = float(target_score) * float(orig_score)

            # choose method
            if len(region_info) > 18:
                noisy_idx, final_score = apply_noise_greedy_idx(scrib_idx, gt_idx, region_info, target_diag_score)
                method = "greedy"
            else:
                noisy_idx, final_score = apply_noise_brute_idx(scrib_idx, gt_idx, region_info, target_diag_score)
                method = "brute"

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
                "target_ratio": float(target_score),
                "target_score": round(float(target_diag_score), 6),
                "actual_score": round(float(final_score), 6),
                "abs_error": round(float(abs(final_score - target_diag_score)), 6),
                "num_regions": int(len(region_info)),
            })

        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"[DONE] target={target_score:.1f} -> {out_dir}")
        print(f"[DONE] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
