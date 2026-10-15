"""Shared helpers: image IO, resizing, connected-component cleanup, crops."""
from __future__ import annotations
import os
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
LINK_PX = 60  # beetle parts within this distance of the main blob are kept (a leg hidden behind the ~40 px rod)


def load_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def save_mask(mask: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def load_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def upscale_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    return m > 0


def upscale_prob(prob: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- colour maps
def foreground_score(rgb: np.ndarray) -> np.ndarray:
    """0..1 score of how 'not white background' a pixel is.

    Combines darkness (1 - L) and chroma in Lab. Works for a dark metallic beetle
    on a bright, low-chroma background. The white rod scores ~0.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 128.0
    b = (lab[..., 2] - 128.0) / 128.0
    chroma = np.sqrt(a * a + b * b)
    dark = 1.0 - L
    return np.clip(np.maximum(dark, chroma * 2.0), 0, 1)


# --------------------------------------------------------------------------- components
def _label(mask: np.ndarray):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    return lab, n


def remove_border_components(mask: np.ndarray, border: int = 2) -> np.ndarray:
    lab, n = _label(mask)
    if n == 0:
        return mask
    touch = set(np.unique(lab[:border, :])) | set(np.unique(lab[-border:, :])) \
        | set(np.unique(lab[:, :border])) | set(np.unique(lab[:, -border:]))
    touch.discard(0)
    out = mask.copy()
    for t in touch:
        out[lab == t] = False
    return out


def keep_main_component(mask: np.ndarray, link_radius: int = 0, min_frac: float = 0.0) -> np.ndarray:
    """Keep the largest component plus any components within link_radius px of it.

    Components smaller than min_frac * largest are dropped even if close (noise). The distance
    test uses a distance transform inside the padded bbox, so a large link radius (a leg cut
    off from the body by the mounting rod, ~60 px) costs nothing.
    """
    lab, n = _label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    out = lab == main
    if link_radius > 0 and n > 1:
        b = bbox(mask)
        h, w = mask.shape
        x0, y0 = max(0, b[0] - link_radius), max(0, b[1] - link_radius)
        x1, y1 = min(w, b[2] + link_radius), min(h, b[3] + link_radius)
        dist = ndi.distance_transform_edt(~out[y0:y1, x0:x1])
        near = np.zeros_like(out); near[y0:y1, x0:x1] = dist <= link_radius
        for i in range(1, n + 1):
            if i == main or sizes[i - 1] < min_frac * sizes[main - 1]:
                continue
            comp = lab == i
            if np.any(comp & near):
                out |= comp
    return out


def fill_holes(mask: np.ndarray, max_hole_frac: float = 1.0) -> np.ndarray:
    if max_hole_frac >= 1.0:
        return ndi.binary_fill_holes(mask)
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    lab, n = _label(holes)
    if n == 0:
        return mask
    sizes = ndi.sum(holes, lab, index=np.arange(1, n + 1))
    area = mask.sum()
    out = mask.copy()
    for i in range(1, n + 1):
        if sizes[i - 1] <= max_hole_frac * area:
            out[lab == i] = True
    return out


def clean(mask: np.ndarray, link_radius: int = 15, open_px: int = 0) -> np.ndarray:
    """Standard post-processing: drop border blobs, keep main blob (+neighbours), fill holes."""
    m = mask.astype(bool)
    if open_px > 0:
        m = ndi.binary_opening(m, iterations=open_px)
    m = remove_border_components(m)
    m = keep_main_component(m, link_radius=link_radius, min_frac=0.002)
    m = fill_holes(m, max_hole_frac=0.3)
    return m


def bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_box(box, shape_hw, pad_frac=0.5, min_pad=64):
    x0, y0, x1, y1 = box
    h, w = shape_hw
    bw, bh = x1 - x0, y1 - y0
    px = max(int(bw * pad_frac), min_pad)
    py = max(int(bh * pad_frac), min_pad)
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def square_box(box, shape_hw):
    """Expand box to a square (helps models that resize to square inputs)."""
    x0, y0, x1, y1 = box
    h, w = shape_hw
    s = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
    x1, y1 = min(w, x0 + s), min(h, y0 + s)
    x0, y0 = max(0, x1 - s), max(0, y1 - s)
    return x0, y0, x1, y1


def paste(full_shape_hw, box, crop_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(full_shape_hw, dtype=bool)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = crop_mask
    return out


def scale_box(box, s: float):
    return tuple(int(round(v * s)) for v in box)


# --------------------------------------------------------------------------- rod removal
def _rim_components(mask, bright, min_frac):
    """Bright components inside mask that touch the outside of the mask."""
    cand = bright & mask
    lab_c, n = _label(cand)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndi.sum(cand, lab_c, index=np.arange(1, n + 1))
    dil = ndi.binary_dilation(cand, iterations=1) & ~mask
    touch_ids = np.unique(lab_c[ndi.binary_dilation(dil, iterations=1) & cand])
    out = np.zeros_like(mask)
    area = mask.sum()
    for i in touch_ids:
        if i == 0 or sizes[i - 1] < min_frac * area:
            continue
        out |= lab_c == i
    return out


def _fit_line(pts_yx):
    """PCA line fit. Returns (centre, direction, length, halfwidth, elongation)."""
    p = pts_yx.astype(np.float32)
    c = p.mean(0)
    q = p - c
    cov = q.T @ q / len(q)
    w, v = np.linalg.eigh(cov)
    d = v[:, 1]  # major axis
    along = q @ d
    perp = q @ v[:, 0]
    length = np.percentile(along, 99) - np.percentile(along, 1)
    halfw = np.percentile(np.abs(perp), 95)
    elong = np.sqrt(w[1] / max(w[0], 1e-6))
    return c, d, length, halfw, elong


def detect_rod(rgb: np.ndarray, mask: np.ndarray, down: int = 2):
    """Find the mounting rod as a pair of long parallel Canny/Hough edges near the mask.

    Returns (centre_yx, dir_yx, halfwidth) in full-res coords, or None.
    """
    b = bbox(mask)
    if b is None:
        return None
    diam = max(b[2] - b[0], b[3] - b[1])
    pb = pad_box(b, mask.shape, pad_frac=1.2, min_pad=200)
    x0, y0, x1, y1 = pb
    crop = rgb[y0:y1, x0:x1]
    small = cv2.resize(crop, ((x1 - x0) // down, (y1 - y0) // down), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    edges = cv2.Canny(gray, 40, 120)
    # collect segments at two scales (whole-rod masks have a huge bbox, beetle-only masks a small one)
    segs = []
    mc = np.array([((b[0] + b[2]) / 2 - x0) / down, ((b[1] + b[3]) / 2 - y0) / down])
    for frac in (0.5, 0.25):
        min_len = max(30, int(frac * diam / down))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(30, min_len // 2),
                                minLineLength=min_len, maxLineGap=min_len // 4)
        if lines is None:
            continue
        for xa, ya, xb, yb in lines.reshape(-1, 4).astype(np.float32):
            d = np.array([xb - xa, yb - ya]); n = np.linalg.norm(d)
            if n < 1:
                continue
            d /= n
            nrm = np.array([-d[1], d[0]])
            off = float(np.dot(np.array([xa, ya]) - mc, nrm))
            ang = np.arctan2(d[1], d[0]) % np.pi
            # merge with an existing collinear segment (same angle, same offset): rod edges are long
            merged = False
            for s in segs:
                da = abs(s[0] - ang); da = min(da, np.pi - da)
                if da < np.deg2rad(1.5) and abs(abs(s[1]) - abs(off)) < 3 and np.sign(s[1]) == np.sign(off):
                    s[2] += n; merged = True; break
            if not merged:
                segs.append([ang, off, n, xa, ya, xb, yb, d, nrm])
    if not segs:
        return None
    # rod edges must lie outside the thick body core (the raw blob may already contain the rod)
    core = ndi.binary_opening(mask, structure=np.ones((61, 61), bool))
    blob_dil = ndi.binary_dilation(core if core.any() else mask, iterations=8)
    best = None
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a1, o1, n1 = segs[i][:3]; a2, o2, n2 = segs[j][:3]
            da = abs(a1 - a2); da = min(da, np.pi - da)
            if da > np.deg2rad(3.5):
                continue
            sep = abs(o1 - o2)
            if not (3 <= sep <= 90):          # rod width 6..180 px at full res (thin metal rod too)
                continue
            if max(abs(o1), abs(o2)) > 0.6 * diam / down:   # centreline must pass through the beetle
                continue
            # the rod's edges lie OUTSIDE the beetle blob: segment midpoints must not be inside it
            inside = 0
            for sg in (segs[i], segs[j]):
                mx, my = (sg[3] + sg[5]) / 2, (sg[4] + sg[6]) / 2
                fx, fy = int(mx * down + x0), int(my * down + y0)
                if 0 <= fy < mask.shape[0] and 0 <= fx < mask.shape[1] and blob_dil[fy, fx]:
                    inside += 1
            if inside:
                continue
            score = n1 + n2
            if best is None or score > best[0]:
                best = (score, i, j, sep)
    if best is None:
        return None
    _, i, j, sep = best
    s1, s2 = segs[i], segs[j]
    d = s1[7]
    nrm = s1[8]
    centre_off = (s1[1] + s2[1]) / 2
    c_small = mc + nrm * centre_off
    c_full = np.array([c_small[1] * down + y0, c_small[0] * down + x0])  # y, x
    d_full = np.array([d[1], d[0]])  # y, x
    return c_full, d_full, sep * down / 2


def _rod_removal(m, rod, L, chroma, L_band, chroma_thresh, step=6, stop_after=5):
    """Walk along the rod line from both ends of the mask and clear rod-coloured (non-dark,
    low-chroma) mask pixels in a band around the line.

    The band half-width is estimated from the mask itself at the walk's starting end (the
    Hough half-width is often the inner shading edges of a thick white rod). Walking stops
    after `stop_after` consecutive windows in which dark beetle pixels dominate the band,
    so legs crossing the rod (a few windows) do not stop it but the body does.
    """
    c, d, halfw = rod
    Wb = 3 * halfw + 40
    b = bbox(m)
    yy, xx = np.mgrid[b[1]:b[3], b[0]:b[2]]
    qy, qx = (yy - c[0]).astype(np.float32), (xx - c[1]).astype(np.float32)
    perp = np.abs(qy * d[1] - qx * d[0])
    along = qy * d[0] + qx * d[1]
    sub = m[b[1]:b[3], b[0]:b[2]]
    dark = ~((L[b[1]:b[3], b[0]:b[2]] > L_band) & (chroma[b[1]:b[3], b[0]:b[2]] < chroma_thresh + 5))
    in_wide = sub & (perp <= Wb)
    if not in_wide.any():
        return np.zeros_like(m)
    t = along[in_wide]
    t0, t1 = float(t.min()), float(t.max())
    removal = np.zeros_like(sub)
    for sign in (+1, -1):
        pos = t0 if sign > 0 else t1
        # 1. estimate the rod half-width from rod-coloured pixels in the first windows
        widths = []
        for k in range(6):
            lo, hi = (pos + k * step, pos + (k + 1) * step) if sign > 0 else (pos - (k + 1) * step, pos - k * step)
            win = in_wide & (along >= lo) & (along < hi) & ~dark
            if win.sum() >= step:
                widths.append(np.percentile(perp[win], 95))
        if not widths:
            continue  # no rod on this end
        hw = float(np.clip(np.median(widths), halfw, Wb)) + 6
        band = sub & (perp <= hw)
        band_area = 2.0 * hw * step
        # 2. walk
        dark_run = 0
        while (pos - t0) >= -1e-3 and (t1 - pos) >= -1e-3:
            lo, hi = (pos, pos + step) if sign > 0 else (pos - step, pos)
            win = (along >= lo) & (along < hi)
            bw = band & win
            n_dark = (dark & bw).sum()
            if n_dark > 0.5 * band_area:
                dark_run += 1
                if dark_run >= stop_after:
                    break
            else:
                dark_run = 0
            removal |= bw & ~dark
            pos += sign * step
    out = np.zeros_like(m)
    out[b[1]:b[3], b[0]:b[2]] = removal
    return out


def fill_holes_except(mask: np.ndarray, exclude: np.ndarray, max_hole_frac: float = 0.3) -> np.ndarray:
    """Fill enclosed holes, except holes that mostly overlap `exclude` (e.g. removed rod pixels)."""
    m = mask.astype(bool)
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    lab, n = _label(holes)
    if n == 0:
        return m
    area = max(1, m.sum())
    out = m.copy()
    for i in range(1, n + 1):
        h = lab == i
        hs = h.sum()
        if hs > max_hole_frac * area:
            continue
        if exclude is not None and (h & exclude).sum() > 0.3 * hs:
            continue
        out |= h
    return out


def remove_wall_holes(mask: np.ndarray, L: np.ndarray, chroma: np.ndarray, min_frac: float = 0.003) -> np.ndarray:
    """Background seen through the legs is often enclosed by the mask and gets 'filled'.
    Bright, low-chroma, texture-less regions inside the mask whose lightness matches the
    surrounding wall are background -> removed. Specular highlights on the beetle are textured
    (and usually darker than the wall), so they stay.
    """
    m = mask.astype(bool)
    if not m.any():
        return m
    bright = (L > 150) & (chroma < 15) & m
    lab, n = _label(bright)
    if n == 0:
        return m
    ring = ndi.binary_dilation(m, iterations=60) & ~m
    wallL = float(np.median(L[ring])) if ring.any() else 240.0
    sizes = ndi.sum(bright, lab, index=np.arange(1, n + 1))
    area = m.sum()
    lap = np.abs(cv2.Laplacian(L.astype(np.float32), cv2.CV_32F))
    out = m.copy()
    for i in range(1, n + 1):
        if sizes[i - 1] < min_frac * area:
            continue
        comp = lab == i
        Lc = L[comp]
        if Lc.mean() > wallL - 30 and np.percentile(Lc, 10) > 160 and lap[comp].mean() < 10:
            out[comp] = False
    return out


def strip_bright_rim(mask: np.ndarray, rgb: np.ndarray, L_thresh: int = 165, chroma_thresh: int = 15,
                     min_frac: float = 0.05, L_band: int = 110, thin_r: int = 5, zone_r: int = 12) -> np.ndarray:
    """Shared post-process: remove the mounting rod, background seen through the legs, and thin
    residue; keep the main blob; fill genuine holes.

    1. Detect the rod as two long parallel edges (Hough) near the mask -> walk along it and clear
       rod-coloured (non-dark, low-chroma) mask pixels. Legs wrapped around the rod are dark and
       stay. Fallback without a rod line: remove large bright low-chroma components that touch
       the outside of the mask.
    2. Thin residue next to the removed region (shadowed rod edge) is dropped.
    3. Wall regions enclosed by legs are removed (remove_wall_holes); holes are filled except
       where the rod was removed.
    """
    m = mask.astype(bool).copy()
    if not m.any():
        return m
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    rod = detect_rod(rgb, m)
    if rod is not None:
        removal = _rod_removal(m, rod, L, chroma, L_band, chroma_thresh)
    else:
        bright = (L > L_thresh) & (chroma < chroma_thresh)
        removal = _rim_components(m, bright, min_frac)
    m &= ~removal
    m = keep_main_component(m, link_radius=LINK_PX, min_frac=0.002)
    removed = mask.astype(bool) & ~m
    if removed.any():
        zone = ndi.binary_dilation(removed, iterations=zone_r)
        se = np.ones((2 * thin_r + 1, 2 * thin_r + 1), bool)
        thin = m & ~ndi.binary_opening(m, structure=se)
        m = m & ~(thin & zone & (L > 100))   # only grey residue; dark tarsi/claws next to the rod stay
        m = keep_main_component(m, link_radius=LINK_PX, min_frac=0.002)
    m = fill_holes_except(m, removal, max_hole_frac=0.3)
    m = remove_wall_holes(m, L, chroma)
    m = keep_main_component(m, link_radius=LINK_PX, min_frac=0.002)
    return m


def clean_crop(mask: np.ndarray, coarse: np.ndarray, link_radius: int = 15) -> np.ndarray:
    """Cleanup for masks computed inside a crop, guided by the coarse beetle blob.

    Keeps the components that overlap the (dilated) coarse blob; if the result is much larger
    than the coarse blob (leak into wall / rod that is connected to the beetle), it is clipped
    to a generous dilation of the coarse blob. Then holes are filled.
    """
    m = mask.astype(bool)
    if not m.any():
        return m
    if not coarse.any():
        return clean(m, link_radius=link_radius)
    cb = bbox(coarse)
    diam = max(cb[2] - cb[0], cb[3] - cb[1])
    near = ndi.binary_dilation(coarse, iterations=max(5, int(0.1 * diam)))
    lab, n = _label(m)
    ids = np.unique(lab[near]); ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros_like(m)
    out = np.isin(lab, ids)
    if out.sum() > 3.0 * coarse.sum():
        out &= ndi.binary_dilation(coarse, iterations=max(10, int(0.25 * diam)))
    out = keep_main_component(out, link_radius=max(link_radius, LINK_PX), min_frac=0.002)
    return fill_holes(out, max_hole_frac=0.002)  # tiny holes only; real hole logic is in strip_bright_rim


def strip_large_white(mask: np.ndarray, rgb: np.ndarray, L_thresh: int = 165, chroma_thresh: int = 15,
                      min_frac: float = 0.05) -> np.ndarray:
    """Safeguard for SAM masks: remove any large (>= min_frac of the mask) near-white, low-chroma
    component that touches the outside of the mask, i.e. a chunk of the mounting rod that SAM
    segmented together with the beetle. Specular highlights are far smaller than 5 %, so they
    never trigger it. Then keep the main blob (+ fragments within one rod width)."""
    m = mask.astype(bool)
    if not m.any():
        return m
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    bright = (L > L_thresh) & (chroma < chroma_thresh)
    rim = _rim_components(m, bright, min_frac)
    if not rim.any():
        return m
    m = m & ~rim
    m = keep_main_component(m, link_radius=LINK_PX, min_frac=0.002)
    # thin grey residue along the removed rod edge (shadowed rim): drop pixels that vanish under a
    # small opening, only within a zone around the removed chunk and only if not dark (legs stay)
    zone = ndi.binary_dilation(rim, iterations=20)
    thin = m & ~ndi.binary_opening(m, structure=np.ones((25, 25), bool))
    m = m & ~(thin & zone & (L > 100))
    return keep_main_component(m, link_radius=LINK_PX, min_frac=0.002)
