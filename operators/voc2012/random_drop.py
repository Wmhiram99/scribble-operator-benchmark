import os
import glob
import argparse
import numpy as np
from PIL import Image
import cv2

from skimage.morphology import skeletonize, disk
from skimage.morphology import dilation as sk_dilation


def is_color_image(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] in (3, 4)


def load_mask(path: str):
    img = Image.open(path)
    arr = np.array(img)
    if is_color_image(arr):
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return "color", arr.astype(np.uint8)
    else:
        return "single", arr


def save_mask(path: str, mode: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == "color":
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        Image.fromarray(arr).save(path)


def detect_bg_val_single(label_map: np.ndarray, ignore_val: int):
    """
    Heuristic: background is the most frequent value excluding ignore.
    Works for (bg=255, fg=0) or (bg=0, fg>0) etc.
    """
    flat = label_map.reshape(-1)
    if ignore_val is not None:
        flat = flat[flat != ignore_val]
    if flat.size == 0:
        return 0
    vals, cnts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(cnts)])


def estimate_radius_from_original(original_bin: np.ndarray, skel: np.ndarray) -> int:
    dt = cv2.distanceTransform(original_bin.astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt[skel.astype(bool)]
    if vals.size == 0:
        return 1
    r = int(np.clip(np.median(vals), 1, 10))
    return r


def make_random_order_indices(skel_mask: np.ndarray, seed: int):
    ys, xs = np.where(skel_mask)
    n = len(ys)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return ys, xs, perm


def synth_from_k_points(original_bin: np.ndarray,
                        ys: np.ndarray, xs: np.ndarray, perm: np.ndarray,
                        k: int,
                        keep_width: bool,
                        fixed_radius: int):
    """
    Take first k points in perm order, optionally dilate and intersect with original.
    Returns binary uint8 mask.
    """
    k = int(max(1, min(k, len(perm))))
    kept = np.zeros_like(original_bin, dtype=bool)
    pick = perm[:k]
    kept[ys[pick], xs[pick]] = True

    if not keep_width:
        return kept.astype(np.uint8)

    if fixed_radius > 0:
        r = fixed_radius
    else:
        r = estimate_radius_from_original(original_bin, kept.astype(np.uint8))
    thick = sk_dilation(kept, disk(r)).astype(np.uint8)
    thick = thick & original_bin.astype(np.uint8)  # never grow outside original stroke
    return thick


def binary_search_k_for_target_pixels(original_bin: np.ndarray,
                                      target_pixels: int,
                                      seed: int,
                                      keep_width: bool,
                                      fixed_radius: int,
                                      tol_pixels: int):
    """
    Deterministic monotonic search using a fixed random order (perm).
    Assumes output pixels are (mostly) non-decreasing with k.
    """
    if target_pixels <= 0:
        # keep at least 1 pixel if original exists
        target_pixels = 1

    skel = skeletonize(original_bin.astype(bool))
    if skel.sum() == 0:
        # fallback: treat original as skeleton (rare)
        skel = original_bin.astype(bool)

    order = make_random_order_indices(skel, seed)
    if order is None:
        return original_bin
    ys, xs, perm = order
    n = len(perm)

    # boundaries
    lo, hi = 1, n
    best_mask = None
    best_err = 10**18

    # quick checks
    m1 = synth_from_k_points(original_bin, ys, xs, perm, 1, keep_width, fixed_radius)
    c1 = int(m1.sum())
    if abs(c1 - target_pixels) <= tol_pixels:
        return m1

    mn = synth_from_k_points(original_bin, ys, xs, perm, n, keep_width, fixed_radius)
    cn = int(mn.sum())
    if cn <= target_pixels:
        # even keeping all skel points can't reach target (shouldn't happen with intersect),
        # return max possible
        return mn

    # binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        mm = synth_from_k_points(original_bin, ys, xs, perm, mid, keep_width, fixed_radius)
        cm = int(mm.sum())
        err = abs(cm - target_pixels)

        if err < best_err:
            best_err = err
            best_mask = mm

        if err <= tol_pixels:
            return mm

        if cm < target_pixels:
            lo = mid + 1
        else:
            hi = mid - 1

    return best_mask if best_mask is not None else original_bin


def shrink_single_channel_precise(label_map: np.ndarray,
                                  target_keep: float = None,
                                  target_coverage: float = None,
                                  tol: float = 0.002,
                                  seed: int = 0,
                                  keep_width: bool = True,
                                  fixed_radius: int = 0,
                                  bg_val: int = None,
                                  ignore_val: int = 255):
    """
    Returns a new label_map with shrunk scribbles, preserving class ids.
    Coverage is computed as (foreground pixels excluding bg and ignore) / total pixels.
    """
    H, W = label_map.shape[:2]
    total_pixels = H * W

    if bg_val is None:
        bg_val = detect_bg_val_single(label_map, ignore_val)

    # compute per-class pixel counts (foreground only)
    mask_valid = (label_map != bg_val)
    if ignore_val is not None:
        mask_valid = mask_valid & (label_map != ignore_val)

    orig_fg = int(mask_valid.sum())
    if orig_fg == 0:
        return label_map.copy()

    if target_coverage is not None:
        target_fg = int(round(float(target_coverage) * total_pixels))
    else:
        # default via target_keep; if not set, infer from "shrink_ratio-like" behavior
        if target_keep is None:
            raise ValueError("Need --target_keep or --target_coverage (or --shrink_ratio).")
        target_fg = int(round(orig_fg * float(target_keep)))

    target_fg = max(1, min(target_fg, orig_fg))
    tol_pixels = max(1, int(round(tol * total_pixels)))

    out = np.full_like(label_map, bg_val)

    # keep ignore as is
    if ignore_val is not None:
        out[label_map == ignore_val] = ignore_val

    # per-class allocation proportional to original pixels
    classes = np.unique(label_map[mask_valid])
    orig_counts = {}
    for c in classes:
        orig_counts[int(c)] = int((label_map == c).sum())

    # distribute target pixels
    # initial proportional allocation
    alloc = {}
    for c, cnt in orig_counts.items():
        alloc[c] = max(1, int(round(target_fg * (cnt / orig_fg))))

    # fix rounding mismatch
    cur_sum = sum(alloc.values())
    diff = target_fg - cur_sum
    # adjust by adding/subtracting 1 to largest classes
    sorted_cls = sorted(orig_counts.keys(), key=lambda k: orig_counts[k], reverse=True)
    idx = 0
    while diff != 0 and idx < len(sorted_cls) * 5:
        c = sorted_cls[idx % len(sorted_cls)]
        if diff > 0:
            alloc[c] += 1
            diff -= 1
        else:
            if alloc[c] > 1:
                alloc[c] -= 1
                diff += 1
        idx += 1

    # shrink each class with deterministic different seed offset
    for j, c in enumerate(sorted_cls):
        original_bin = (label_map == c).astype(np.uint8)
        if original_bin.sum() == 0:
            continue
        cls_target = min(alloc[c], int(original_bin.sum()))
        shr = binary_search_k_for_target_pixels(
            original_bin=original_bin,
            target_pixels=cls_target,
            seed=seed + 10007 * j,
            keep_width=keep_width,
            fixed_radius=fixed_radius,
            tol_pixels=max(1, int(round(tol_pixels * (orig_counts[c] / orig_fg))))
        )
        out[shr.astype(bool)] = c

    return out


def shrink_color_precise(rgb: np.ndarray,
                         target_keep: float = None,
                         target_coverage: float = None,
                         tol: float = 0.002,
                         seed: int = 0,
                         keep_width: bool = True,
                         fixed_radius: int = 0,
                         background_rgb=(255, 255, 255)):
    H, W, _ = rgb.shape
    total_pixels = H * W
    bg = np.array(background_rgb, dtype=np.uint8)

    fg_mask = np.any(rgb != bg, axis=2)
    orig_fg = int(fg_mask.sum())
    if orig_fg == 0:
        return rgb.copy()

    if target_coverage is not None:
        target_fg = int(round(float(target_coverage) * total_pixels))
    else:
        if target_keep is None:
            raise ValueError("Need --target_keep or --target_coverage (or --shrink_ratio).")
        target_fg = int(round(orig_fg * float(target_keep)))

    target_fg = max(1, min(target_fg, orig_fg))
    tol_pixels = max(1, int(round(tol * total_pixels)))

    out = np.zeros_like(rgb, dtype=np.uint8)
    out[:] = bg

    # unique colors excluding bg
    flat = rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    colors = [tuple(c.tolist()) for c in uniq if not np.all(c == bg)]
    if not colors:
        return out

    # per-color pixel counts
    orig_counts = {}
    for c in colors:
        m = np.all(rgb == np.array(c, dtype=np.uint8), axis=2)
        orig_counts[c] = int(m.sum())

    # proportional allocation
    alloc = {}
    for c, cnt in orig_counts.items():
        alloc[c] = max(1, int(round(target_fg * (cnt / orig_fg))))

    cur_sum = sum(alloc.values())
    diff = target_fg - cur_sum
    sorted_colors = sorted(colors, key=lambda k: orig_counts[k], reverse=True)

    idx = 0
    while diff != 0 and idx < len(sorted_colors) * 5:
        c = sorted_colors[idx % len(sorted_colors)]
        if diff > 0:
            alloc[c] += 1
            diff -= 1
        else:
            if alloc[c] > 1:
                alloc[c] -= 1
                diff += 1
        idx += 1

    for j, c in enumerate(sorted_colors):
        col = np.array(c, dtype=np.uint8)
        original_bin = np.all(rgb == col, axis=2).astype(np.uint8)
        if original_bin.sum() == 0:
            continue
        cls_target = min(alloc[c], int(original_bin.sum()))
        shr = binary_search_k_for_target_pixels(
            original_bin=original_bin,
            target_pixels=cls_target,
            seed=seed + 10007 * j,
            keep_width=keep_width,
            fixed_radius=fixed_radius,
            tol_pixels=max(1, int(round(tol_pixels * (orig_counts[c] / orig_fg))))
        )
        out[shr.astype(bool)] = col

    return out


def process_dir(in_dir: str,
                out_dir: str,
                target_keep: float = None,
                target_coverage: float = None,
                shrink_ratio: float = None,
                tol: float = 0.002,
                seed: int = 0,
                keep_width: bool = True,
                fixed_radius: int = 0,
                bg_val: int = None,
                ignore_val: int = 255,
                exts=("png",)):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, f"*.{e}"))
    paths.sort()
    print(f"Found {len(paths)} images")

    # backward compatible: shrink_ratio -> target_keep
    if shrink_ratio is not None:
        target_keep = 1.0 - float(shrink_ratio)

    if target_keep is None and target_coverage is None:
        raise ValueError("You must provide --target_keep or --target_coverage (or --shrink_ratio).")

    for p in paths:
        mode, data = load_mask(p)

        if mode == "single":
            out = shrink_single_channel_precise(
                data,
                target_keep=target_keep,
                target_coverage=target_coverage,
                tol=tol,
                seed=seed,
                keep_width=keep_width,
                fixed_radius=fixed_radius,
                bg_val=bg_val,
                ignore_val=ignore_val
            )
        else:
            out = shrink_color_precise(
                data,
                target_keep=target_keep,
                target_coverage=target_coverage,
                tol=tol,
                seed=seed,
                keep_width=keep_width,
                fixed_radius=fixed_radius,
                background_rgb=(255, 255, 255)
            )

        save_mask(os.path.join(out_dir, os.path.basename(p)), mode, out)

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    # Choose ONE of these (or use shrink_ratio as legacy)
    ap.add_argument("--target_keep", type=float, default=None,
                    help="Keep ratio relative to original coverage, e.g. 0.3 means keep 30% foreground pixels.")
    ap.add_argument("--target_coverage", type=float, default=None,
                    help="Absolute coverage (fg/total), e.g. 0.01.")
    ap.add_argument("--shrink_ratio", type=float, default=None,
                    help="Legacy: 0~1, shrink_ratio=0.7 means keep 0.3 (same as target_keep=0.3).")

    ap.add_argument("--tol", type=float, default=0.002,
                    help="Tolerance in absolute coverage units, default 0.002 (0.2% of pixels).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_width", action="store_true")
    ap.add_argument("--fixed_radius", type=int, default=0)

    # For single-channel
    ap.add_argument("--bg_val", type=int, default=None,
                    help="Background value for single-channel. If omitted, auto-detect.")
    ap.add_argument("--ignore_val", type=int, default=255,
                    help="Ignore value for single-channel (default 255). Set to -1 if none.")
    args = ap.parse_args()

    ignore_val = args.ignore_val
    if ignore_val is not None and int(ignore_val) < 0:
        ignore_val = None

    process_dir(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        target_keep=args.target_keep,
        target_coverage=args.target_coverage,
        shrink_ratio=args.shrink_ratio,
        tol=args.tol,
        seed=args.seed,
        keep_width=args.keep_width,
        fixed_radius=args.fixed_radius,
        bg_val=args.bg_val,
        ignore_val=ignore_val
    )
