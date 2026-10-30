import os
import glob
import argparse
import numpy as np
from PIL import Image
import cv2
from collections import deque

from skimage.morphology import skeletonize, disk
from skimage.morphology import dilation as sk_dilation


def load_single_channel(path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError("Expected single-channel image, got shape: {}".format(arr.shape))
    return arr


def save_single_channel(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


def detect_bg_val(arr, ignore_val=None):
    flat = arr.reshape(-1)
    if ignore_val is not None:
        flat = flat[flat != ignore_val]
    if flat.size == 0:
        return 255
    vals, cnts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(cnts)])


def make_binary_fg(arr, bg_val, ignore_val=None):
    fg = (arr != bg_val)
    if ignore_val is not None:
        fg = fg & (arr != ignore_val)
    return fg.astype(np.uint8)


def neighbors8(y, x, H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def neighbor_count8(bin_u8):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(bin_u8.astype(np.uint8), -1, kernel)


def find_endpoints(skel_u8):
    nb = neighbor_count8(skel_u8)
    pts = np.argwhere((skel_u8 == 1) & (nb == 1))
    return pts  # (K,2)


def bfs_dist(skel_u8, start_yx):
    H, W = skel_u8.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    sy, sx = int(start_yx[0]), int(start_yx[1])
    dist[sy, sx] = 0
    q.append((sy, sx))

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for ny, nx in neighbors8(y, x, H, W):
            if skel_u8[ny, nx] == 0:
                continue
            if dist[ny, nx] != -1:
                continue
            dist[ny, nx] = d + 1
            q.append((ny, nx))
    return dist


def pick_diameter_endpoints(skel_u8, endpoints):
    """
    Two-pass BFS to approximate skeleton diameter endpoints.
    """
    e0 = endpoints[0]
    dist0 = bfs_dist(skel_u8, e0)

    e1, best_d = e0, -1
    for e in endpoints:
        d = dist0[int(e[0]), int(e[1])]
        if d > best_d:
            best_d = d
            e1 = e

    dist1 = bfs_dist(skel_u8, e1)

    e2, best_d = e1, -1
    for e in endpoints:
        d = dist1[int(e[0]), int(e[1])]
        if d > best_d:
            best_d = d
            e2 = e

    return e1, e2, dist1


def estimate_radius(original_bin_u8, skel_bool):
    dt = cv2.distanceTransform(original_bin_u8.astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt[skel_bool]
    if vals.size == 0:
        return 1
    r = int(np.clip(np.median(vals), 1, 10))
    return r


def reconstruct_width_by_nearest_skeleton(original_bin_u8, full_skel_bool, keep_skel_bool):
    """
    Width-preserving reconstruction:
    assign each original foreground pixel to its nearest source-skeleton pixel,
    then keep pixels whose nearest skeleton point is in keep_skel_bool.
    """
    if int(original_bin_u8.sum()) == 0:
        return original_bin_u8.astype(np.uint8)

    full_skel = (full_skel_bool > 0)
    keep_skel = (keep_skel_bool > 0)
    if not np.any(full_skel) or not np.any(keep_skel):
        return original_bin_u8.astype(np.uint8)

    # distanceTransformWithLabels uses zero-valued pixels as sources.
    seed = np.ones_like(original_bin_u8, dtype=np.uint8)
    seed[full_skel] = 0
    _, labels = cv2.distanceTransformWithLabels(
        seed, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )

    keep_ids = np.unique(labels[keep_skel])
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return original_bin_u8.astype(np.uint8)

    keep_mask = np.isin(labels, keep_ids)
    out = ((original_bin_u8 > 0) & keep_mask).astype(np.uint8)
    if int(out.sum()) == 0:
        return original_bin_u8.astype(np.uint8)
    return out


def linear_trim_precise(original_bin_u8, target_pixels, tol_pixels, keep_width=True, fixed_radius=0):
    """
    Shrink by trimming skeleton equally from both ends, and binary-search trim to match target fg pixels.

    - If skeleton has <2 endpoints (closed loop / blob), returns original unchanged.
    - Guarantees at least 1 pixel if original has pixels.
    """
    if int(original_bin_u8.sum()) == 0:
        return original_bin_u8

    skel = skeletonize(original_bin_u8.astype(bool))
    if int(skel.sum()) == 0:
        return original_bin_u8

    skel_u8 = skel.astype(np.uint8)
    endpoints = find_endpoints(skel_u8)
    if endpoints.shape[0] < 2:
        # do not process closed loops / blobs
        return original_bin_u8

    head, tail, distH = pick_diameter_endpoints(skel_u8, endpoints)
    distT = bfs_dist(skel_u8, tail)

    ht_dist = int(distH[int(tail[0]), int(tail[1])])
    if ht_dist <= 2:
        return original_bin_u8

    max_t = ht_dist // 2

    if keep_width:
        r = fixed_radius if fixed_radius > 0 else estimate_radius(original_bin_u8, skel)
        se = disk(r)

    def synth(t):
        keep_skel = (skel_u8 == 1) & (distH >= t) & (distT >= t)
        if not keep_skel.any():
            # keep a midpoint-like pixel
            m = np.minimum(distH, distT).astype(np.int32)
            m[skel_u8 == 0] = -1
            yx = np.unravel_index(int(np.argmax(m)), m.shape)
            keep_skel = np.zeros_like(keep_skel, dtype=bool)
            keep_skel[yx] = True

        if not keep_width:
            return keep_skel.astype(np.uint8)

        # Primary path: preserve original stroke width by nearest-skeleton assignment.
        thick = reconstruct_width_by_nearest_skeleton(
            original_bin_u8=original_bin_u8,
            full_skel_bool=skel,
            keep_skel_bool=keep_skel
        ).astype(np.uint8)

        # Fallback for edge cases.
        if int(thick.sum()) == 0:
            thick = sk_dilation(keep_skel, se).astype(np.uint8)
            thick = thick & original_bin_u8
        return thick

    # clamp target
    orig = int(original_bin_u8.sum())
    target_pixels = int(max(1, min(int(target_pixels), orig)))

    lo, hi = 0, int(max_t)
    best = None
    best_err = 10**18

    while lo <= hi:
        mid = (lo + hi) // 2
        out = synth(mid)
        c = int(out.sum())
        err = abs(c - target_pixels)

        if err < best_err:
            best_err = err
            best = out

        if err <= tol_pixels:
            return out

        if c > target_pixels:
            lo = mid + 1  # trim more
        else:
            hi = mid - 1  # trim less

    return best if best is not None else original_bin_u8


def mode_value(vals: np.ndarray, fallback: int = 0) -> int:
    if vals.size == 0:
        return int(fallback)
    u, c = np.unique(vals, return_counts=True)
    return int(u[np.argmax(c)])


def allocate_targets_per_cc(sizes, total_target, ensure_min1=True):
    """
    Distribute total_target pixels across components proportional to their original size.
    Ensure each component keeps at least 1 pixel if ensure_min1.
    """
    sizes = np.array(sizes, dtype=np.int64)
    orig_total = int(sizes.sum())
    n = int(sizes.size)
    if orig_total <= 0 or n == 0:
        return []

    raw = sizes.astype(np.float64) * (float(total_target) / float(orig_total))
    targets = np.rint(raw).astype(np.int64)

    if ensure_min1:
        targets = np.maximum(targets, 1)

    targets = np.minimum(targets, sizes)

    cur = int(targets.sum())
    diff = int(total_target - cur)

    if diff == 0:
        return targets.tolist()

    if diff > 0:
        room = sizes - targets
        order = np.argsort(-room)
        for idx in order:
            if diff <= 0:
                break
            if room[idx] <= 0:
                continue
            add = min(int(room[idx]), diff)
            targets[idx] += add
            diff -= add
    else:
        need = -diff
        room = targets - (1 if ensure_min1 else 0)
        order = np.argsort(-room)
        for idx in order:
            if need <= 0:
                break
            if room[idx] <= 0:
                continue
            rm = min(int(room[idx]), need)
            targets[idx] -= rm
            need -= rm

    return targets.tolist()


def make_disk_offsets(r: int):
    r = int(max(1, r))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    m = (yy*yy + xx*xx) <= (r*r)
    ys, xs = np.where(m)
    ys = ys - r
    xs = xs - r
    return list(zip(ys.tolist(), xs.tolist()))


def dot_compensate_on_skeleton(original_cc_u8, kept_cc_u8, target_pixels, dot_radius, rng):
    """
    Add small disks centered on skeleton pixels to increase kept pixels,
    but strictly within original_cc_u8 (no new growth).
    """
    orig = int(original_cc_u8.sum())
    cur = int(kept_cc_u8.sum())
    if orig == 0 or cur >= target_pixels:
        return kept_cc_u8

    # skeleton candidates from ORIGINAL cc (not the kept one)
    skel = skeletonize(original_cc_u8.astype(bool))
    skel_pts = np.argwhere(skel)
    if skel_pts.size == 0:
        return kept_cc_u8

    # shuffle skeleton points
    idxs = np.arange(len(skel_pts))
    rng.shuffle(idxs)

    offsets = make_disk_offsets(dot_radius)

    out = kept_cc_u8.copy()
    H, W = out.shape
    need = int(target_pixels - cur)
    if need <= 0:
        return out

    # add disks at skeleton points, count newly added pixels
    for k in idxs:
        cy, cx = map(int, skel_pts[k])
        newly = 0
        for dy, dx in offsets:
            y = cy + dy
            x = cx + dx
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            if original_cc_u8[y, x] == 0:
                continue
            if out[y, x] == 1:
                continue
            out[y, x] = 1
            newly += 1
            need -= 1
            if need <= 0:
                return out

    return out


def process_one_image_cc(arr_u8, target_keep=None, target_coverage=None, tol=0.002,
                         bg_val=None, ignore_val=None, keep_width=True, fixed_radius=0,
                         enable_dot_comp=False, dot_radius=1, dot_only_for_high_keep=0.8,
                         dot_trigger_gap_pixels=1, seed=0):
    H, W = arr_u8.shape
    total = H * W

    if bg_val is None:
        bg_val = detect_bg_val(arr_u8, ignore_val)

    fg_bin = make_binary_fg(arr_u8, bg_val, ignore_val)
    orig_fg = int(fg_bin.sum())
    if orig_fg == 0:
        return arr_u8.copy(), 0.0, 0.0

    num, labels = cv2.connectedComponents(fg_bin.astype(np.uint8), connectivity=8)
    if num <= 1:
        return arr_u8.copy(), orig_fg / float(total), orig_fg / float(total)

    comps = []
    sizes = []
    for cc in range(1, num):
        mask = (labels == cc)
        s = int(mask.sum())
        if s <= 0:
            continue
        comps.append(mask)
        sizes.append(s)

    if not comps:
        return arr_u8.copy(), orig_fg / float(total), orig_fg / float(total)

    # decide per-component targets
    if target_coverage is not None:
        total_target = int(round(float(target_coverage) * total))
        total_target = int(max(1, min(total_target, orig_fg)))
        cc_targets = allocate_targets_per_cc(sizes, total_target, ensure_min1=True)
        global_keep_ratio = None
    else:
        if target_keep is None:
            raise ValueError("Need target_keep or target_coverage.")
        cc_targets = []
        for s in sizes:
            t = int(round(float(target_keep) * float(s)))
            t = max(1, min(t, s))
            cc_targets.append(t)
        global_keep_ratio = float(target_keep)

    tol_total_pixels = int(max(1, round(float(tol) * total)))

    rng = np.random.default_rng(int(seed))

    # store per-cc kept masks so we can dot-compensate cc-by-cc safely
    kept_cc_list = []
    for mask, s, t in zip(comps, sizes, cc_targets):
        tol_cc = int(max(1, round(tol_total_pixels * (float(s) / float(orig_fg)))))
        cc_bin = mask.astype(np.uint8)

        shr_cc = linear_trim_precise(
            cc_bin,
            target_pixels=t,
            tol_pixels=tol_cc,
            keep_width=keep_width,
            fixed_radius=fixed_radius
        ).astype(np.uint8)

        # ---- Scheme B: dot compensation only if needed ----
        if enable_dot_comp:
            # Only trigger for high keep ratio (e.g., 0.8/0.9) to avoid touching low ratio cases
            high_ok = (global_keep_ratio is None) or (global_keep_ratio >= float(dot_only_for_high_keep))
            if high_ok:
                cur = int(shr_cc.sum())
                gap = int(t - cur)
                # trigger only if we are meaningfully below target (avoid tiny jitter)
                if gap >= int(dot_trigger_gap_pixels):
                    shr_cc = dot_compensate_on_skeleton(
                        original_cc_u8=cc_bin,
                        kept_cc_u8=shr_cc,
                        target_pixels=t,
                        dot_radius=int(dot_radius),
                        rng=rng
                    ).astype(np.uint8)

        kept_cc_list.append(shr_cc)

    # union
    shr_union = np.zeros_like(fg_bin, dtype=np.uint8)
    for mask, shr_cc in zip(comps, kept_cc_list):
        # ensure within component
        shr_union |= (shr_cc & mask.astype(np.uint8))

    # reconstruct output: preserve per-cc label value
    out = np.full_like(arr_u8, bg_val, dtype=np.uint8)

    order = np.argsort(-np.array(sizes, dtype=np.int64))
    for idx in order:
        mask = comps[idx]
        vals = arr_u8[mask]
        if ignore_val is not None:
            vals = vals[vals != ignore_val]
        vals = vals[vals != bg_val]
        fg_val = mode_value(vals, fallback=0)

        kept = (kept_cc_list[idx].astype(bool) & mask)
        out[kept] = fg_val

    cov_before = orig_fg / float(total)
    cov_after = int(shr_union.sum()) / float(total)
    return out, cov_before, cov_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--target_keep", type=float, default=None,
                    help="Keep ratio per connected component, e.g. 0.9 keeps ~90% pixels of each CC.")
    ap.add_argument("--target_coverage", type=float, default=None,
                    help="Absolute coverage (fg/total) for the whole image; distributed to CCs by size.")

    ap.add_argument("--tol", type=float, default=0.002, help="Absolute coverage tolerance, default 0.002.")
    ap.add_argument("--bg_val", type=int, default=None, help="Background value (default auto-detect).")
    ap.add_argument("--ignore_val", type=int, default=-1, help="Ignore value, -1 means none.")
    ap.add_argument("--keep_width", action="store_true")
    ap.add_argument("--fixed_radius", type=int, default=0)
    ap.add_argument("--ext", default="png")

    # ---- Scheme B params ----
    ap.add_argument("--enable_dot_comp", action="store_true",
                    help="Enable Scheme-B compensation: add small dots on skeleton to reach target pixels.")
    ap.add_argument("--dot_radius", type=int, default=1,
                    help="Radius of compensation dots (in pixels). Recommended 1 (max 2).")
    ap.add_argument("--dot_only_for_high_keep", type=float, default=0.8,
                    help="Only apply dot compensation if target_keep >= this value (default 0.8).")
    ap.add_argument("--dot_trigger_gap_pixels", type=int, default=2,
                    help="Only compensate if missing pixels >= this threshold (default 2).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for dot placement (only affects compensation).")

    args = ap.parse_args()

    ignore_val = None if int(args.ignore_val) < 0 else int(args.ignore_val)

    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.{}".format(args.ext))))
    if not paths:
        raise RuntimeError("No input images found in {}".format(args.in_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    cov_b_list, cov_a_list = [], []
    for p in paths:
        arr = load_single_channel(p)
        out, cb, ca = process_one_image_cc(
            arr,
            target_keep=args.target_keep,
            target_coverage=args.target_coverage,
            tol=args.tol,
            bg_val=args.bg_val,
            ignore_val=ignore_val,
            keep_width=args.keep_width,
            fixed_radius=args.fixed_radius,
            enable_dot_comp=args.enable_dot_comp,
            dot_radius=args.dot_radius,
            dot_only_for_high_keep=args.dot_only_for_high_keep,
            dot_trigger_gap_pixels=args.dot_trigger_gap_pixels,
            seed=args.seed
        )
        save_single_channel(os.path.join(args.out_dir, os.path.basename(p)), out)
        cov_b_list.append(cb)
        cov_a_list.append(ca)

    cov_b = np.array(cov_b_list, dtype=np.float64)
    cov_a = np.array(cov_a_list, dtype=np.float64)
    ratio = cov_a / np.maximum(cov_b, 1e-12)

    def summarize(name, x):
        print("\n=== {} ===".format(name))
        print("count =", x.size)
        print("mean = {:.6f}, std = {:.6f}, min = {:.6f}, max = {:.6f}".format(
            float(x.mean()), float(x.std()), float(x.min()), float(x.max())))
        print("percentile [10,25,50,75,90] =", np.percentile(x, [10, 25, 50, 75, 90]).round(6))

    summarize("Coverage BEFORE", cov_b)
    summarize("Coverage AFTER", cov_a)
    summarize("AFTER/BEFORE ratio", ratio)

    print("\nDone. Output ->", args.out_dir)


if __name__ == "__main__":
    main()
