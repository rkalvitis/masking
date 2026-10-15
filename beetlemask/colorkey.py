"""Pure colour keying against the white background (no neural network).

1. Local white estimate: per-channel grey closing + blur at 1/8 res (the beetle is filled in by the
   surrounding wall). Dividing by it flattens vignetting and wall shading: background -> ~1.0.
2. Key score per pixel at FULL resolution: how far from white, as max(darkness, chroma) of the
   normalised colour. White wall and white rod -> ~0; beetle body (dark or metallic green) -> high.
3. Hysteresis threshold: strong pixels (score >= high) seed the mask, connected weaker pixels
   (>= low) join it: hairs, leg edges, pale underside.
4. Blob choice as in the locator: border-touching cores are clutter, the core must contain
   dark pixels, fragments within one rod width (hidden leg) are kept.
5. Rod handling: the rod is white so it is mostly not keyed at all; its shadowed edge is a weak
   (low-band) strip. Low-band pixels are kept only within `reach` px of strong pixels, which keeps
   pale beetle parts (next to dark body) and cuts the rod strip to a short stub.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from . import common as C
from .enhance import white_map


def key_score(rgb: np.ndarray) -> np.ndarray:
    wm = white_map(rgb)                                   # local white per channel, full res
    n = np.clip(rgb.astype(np.float32) / wm, 0, 1.2)
    dark = 1.0 - n.min(-1)                                # darkest channel vs local white
    chroma = n.max(-1) - n.min(-1)                        # colourfulness (green sheen)
    return np.maximum(dark, 1.5 * chroma)


def mask_by_color(rgb: np.ndarray, low: float = 0.18, high: float = 0.45, reach: int = 40,
                  core_open: int = 61):
    """Returns (mask, info). mask is full-res bool."""
    s = key_score(rgb)
    s = cv2.GaussianBlur(s, (0, 0), 1.0)
    strong = s >= high
    m = apply_hysteresis_threshold(s, low, high)
    H, W = m.shape
    # --- choose the beetle blob (same logic as the locator, at 1/4 res for speed)
    m4 = cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0
    s4 = cv2.resize(s, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    core = ndi.binary_opening(m4, structure=np.ones((core_open // 4 | 1, core_open // 4 | 1), bool))
    core = C.remove_border_components(core)
    lab, n = ndi.label(core, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(m), {"note": "no core"}
    sizes = ndi.sum(core, lab, index=np.arange(1, n + 1))
    maxs = ndi.maximum(s4, lab, index=np.arange(1, n + 1))
    ranked = np.where(maxs >= high, sizes, 0)
    if ranked.max() == 0:
        ranked = sizes
    main_core4 = lab == (int(np.argmax(ranked)) + 1)
    cb = C.bbox(main_core4); diam4 = max(cb[2] - cb[0], cb[3] - cb[1])
    reach4 = ndi.binary_dilation(main_core4, iterations=int(1.5 * diam4))
    # full-res component containing the core, clipped to reach
    main_core = C.upscale_mask(main_core4, (H, W))
    reach_full = C.upscale_mask(reach4, (H, W))
    labm, nm = ndi.label(m, structure=np.ones((3, 3)))
    ids = np.unique(labm[main_core]); ids = ids[ids > 0]
    blob = np.isin(labm, ids) & reach_full
    # fragments (hidden leg): smaller, non-border components within LINK_PX of the blob
    border_ids = set(np.unique(np.concatenate([labm[0], labm[-1], labm[:, 0], labm[:, -1]])))
    bb = C.bbox(blob); L = C.LINK_PX
    x0, y0 = max(0, bb[0] - L), max(0, bb[1] - L); x1, y1 = min(W, bb[2] + L), min(H, bb[3] + L)
    near = np.zeros_like(blob); near[y0:y1, x0:x1] = ndi.distance_transform_edt(~blob[y0:y1, x0:x1]) <= L
    for cid in set(np.unique(labm[near & m & ~blob])) - border_ids - set(ids) - {0}:
        comp = labm == cid
        if comp.sum() < blob.sum():
            blob |= comp & reach_full
    # --- rod strip: weak pixels only within `reach` of strong pixels
    bb = C.bbox(blob)
    x0, y0 = max(0, bb[0] - reach), max(0, bb[1] - reach); x1, y1 = min(W, bb[2] + reach), min(H, bb[3] + reach)
    dist = ndi.distance_transform_edt(~(strong & blob)[y0:y1, x0:x1])
    keep = np.zeros_like(blob); keep[y0:y1, x0:x1] = dist <= reach
    blob &= keep
    blob = C.keep_main_component(blob, link_radius=C.LINK_PX, min_frac=0.002)
    blob = C.fill_holes(blob, max_hole_frac=0.002)
    return blob, {"score": s}


def mask_by_color_boxed(rgb: np.ndarray, low: float = 0.12, high: float = 0.45, band: int = 80,
                        pad: float = 0.5):
    """Box first, then key against white inside the crop.

    1. Locator (beetlemask.locate) -> box + coarse blob (rod-stripped, hidden-leg fragments linked).
    2. Square crop around the box; local white estimated INSIDE the crop (the crop background is
       white even when the frame is not); key score at full res; hysteresis threshold.
    3. Keep keyed pixels within `band` px of the coarse blob: the boundary is decided by colour
       at full resolution, but the rod edge, rod holder and clutter outside the band cannot leak in.
    """
    from .locate import locate_classical
    box, coarse = locate_classical(rgb)
    cbox = C.square_box(C.pad_box(box, rgb.shape[:2], pad_frac=pad, min_pad=96), rgb.shape[:2])
    x0, y0, x1, y1 = cbox
    crop = rgb[y0:y1, x0:x1]; cc = coarse[y0:y1, x0:x1]
    wm = background_white(crop, cc)
    n = np.clip(crop.astype(np.float32) / wm, 0, 1.2)
    s = np.maximum(1.0 - n.min(-1), 1.5 * (n.max(-1) - n.min(-1)))
    s = cv2.GaussianBlur(s, (0, 0), 0.6)   # light blur only: hairs and claw tips are 1-3 px wide
    m = apply_hysteresis_threshold(s, low, high)
    allowed = ndi.binary_dilation(cc, iterations=band) if cc.any() else np.ones_like(m)
    # the shaded rod edge is grey, not white: inside the rod band (line known from the locator's
    # Hough detection) the mask may not grow beyond the rod-stripped coarse blob
    rod = C.detect_rod(crop, cc) if cc.any() else None
    if rod is not None:
        c, d, halfw = rod
        hw = max(halfw, 15.0) + 10.0
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        perp = np.abs((yy - c[0]) * d[1] - (xx - c[1]) * d[0])
        band = perp <= hw
        # next to the thick body core the band may hold real beetle (elytra side where the rod
        # enters): no colour rule there. Further out along the rod, grey is rod.
        core = ndi.binary_opening(cc, structure=np.ones((61, 61), bool))
        near_core = ndi.binary_dilation(core if core.any() else cc, iterations=40)
        allowed &= ~(band & ~cc & ~near_core)
        # rod pieces are bright (>= 0.62 of white) and colourless; the elytra side next to the rod is
        # dark (~0.5) so it survives. Applied over the whole band, including next to the body.
        grey = (n.min(-1) > 0.62) & ((n.max(-1) - n.min(-1)) < 0.15)
        allowed &= ~(band & grey)
    else:
        band = None; near_core = None
    m &= allowed
    if band is not None:
        # long thin structures inside the rod band are rod edges whatever their colour
        thin_b = m & band & ~ndi.binary_opening(m, structure=np.ones((9, 9), bool))
        labb, nb = ndi.label(thin_b, structure=np.ones((3, 3)))
        if nb:
            objs = ndi.find_objects(labb)
            long_ids = [i + 1 for i, sl in enumerate(objs) if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) > 120]
            m &= ~np.isin(labb, long_ids)
    # thin grey structures = rod (thin metal rod or shadowed rod edge): low chroma, not dark, and
    # gone under a 9x9 opening. Legs are dark, the pale underside is wide -> both untouched.
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    Lc = lab[..., 0]; chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    thin = m & ~ndi.binary_opening(m, structure=np.ones((9, 9), bool))
    tg = thin & (Lc > 110) & (chroma < 12)
    labt, nt = ndi.label(tg, structure=np.ones((3, 3)))
    if nt:
        # only LONG thin grey structures are rod; short ones are hairs / leg tips -> keep
        objs = ndi.find_objects(labt)
        long_ids = [i + 1 for i, sl in enumerate(objs) if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) > 60]
        m &= ~np.isin(labt, long_ids)
    m = C.clean_crop(m, cc, link_radius=C.LINK_PX)
    if band is not None:
        # detached fragments lying inside the rod band are rod remnants, not beetle parts
        labf, nf = ndi.label(m, structure=np.ones((3, 3)))
        if nf > 1:
            sizes = ndi.sum(m, labf, index=np.arange(1, nf + 1)); main = int(np.argmax(sizes)) + 1
            inb = ndi.sum(band & m, labf, index=np.arange(1, nf + 1))
            drop = [i + 1 for i in range(nf) if i + 1 != main and inb[i] >= 0.8 * sizes[i]]
            m &= ~np.isin(labf, drop)
    m = C.strip_large_white(m, crop)          # thick white rod chunk touching the mask edge
    forbid = (band & ~near_core) if band is not None else None
    m = fill_body_holes(m, brightness=n.min(-1), forbid=forbid)   # grey sheen enclosed by body yes; wall between legs / rod in the band no
    return C.paste(rgb.shape[:2], cbox, m), {"box": box}


def fill_body_holes(m: np.ndarray, brightness: np.ndarray = None, max_hole_frac: float = 0.3, thick_open: int = 25,
                    max_thin_frac: float = 0.3, max_white: float = 0.9, forbid: np.ndarray = None):
    """Fill enclosed holes whose border is mostly thick body (specular highlights on the elytra).
    A hole bordered largely by thin structures (legs) is background seen between the legs."""
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    lab, n = ndi.label(holes)
    if n == 0:
        return m
    thick = ndi.binary_opening(m, structure=np.ones((thick_open, thick_open), bool))
    thin = m & ~thick
    area = max(1, m.sum())
    out = m.copy()
    for i in range(1, n + 1):
        h = lab == i
        if h.sum() > max_hole_frac * area:
            continue
        if brightness is not None and np.median(brightness[h]) >= max_white:
            continue  # pure white inside the beetle outline = wall seen between legs, not a highlight
        if forbid is not None and (h & forbid).sum() > 0.5 * h.sum():
            continue  # hole lies on the rod, away from the body: it is rod, not a highlight
        ring = ndi.binary_dilation(h, iterations=3) & m
        if ring.sum() and (ring & thin).sum() / ring.sum() <= max_thin_frac:
            out |= h
    return out


def background_white(crop: np.ndarray, coarse: np.ndarray, down: int = 4, sigma: int = 40, margin: int = 30) -> np.ndarray:
    """Per-channel white reference estimated from BACKGROUND pixels only: the coarse beetle blob
    (dilated by `margin`) is excluded and the reference is interpolated across it by normalised
    convolution at 1/`down` res. Unlike a morphological closing, this cannot mistake a large
    beetle body for background, so the elytra centre is never normalised against itself."""
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (w // down, h // down), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = ~ndi.binary_dilation(coarse, iterations=margin) if coarse.any() else np.ones(crop.shape[:2], bool)
    bg = cv2.resize(bg.astype(np.uint8), (w // down, h // down), interpolation=cv2.INTER_NEAREST) > 0
    # only bright background samples (drop shadows / dark clutter that survived)
    bright = small.min(-1) > np.percentile(small.min(-1)[bg], 30) if bg.any() else np.ones_like(bg)
    valid = (bg & bright).astype(np.float32)
    s = sigma / down
    num = cv2.GaussianBlur(small * valid[..., None], (0, 0), s)
    den = cv2.GaussianBlur(valid, (0, 0), s)[..., None]
    wm = num / np.maximum(den, 1e-3)
    # where no valid sample is nearby (deep inside the beetle) fall back to the crop's global white
    glob = np.percentile(small.reshape(-1, 3)[valid.ravel() > 0], 50, axis=0) if valid.any() else np.array([255, 255, 255.0])
    wm = np.where(den > 0.05, wm, glob)
    wm = cv2.resize(wm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(wm, 40.0)


def refine_with_color(rgb: np.ndarray, sam_mask: np.ndarray, band: int = 14, low: float = 0.30, high: float = 0.45):
    """Hybrid: SAM decides WHAT is beetle (topology, shadows, highlights); colour decides the exact
    edge within +-`band` px of SAM's outline (claws, hairs, leg tips at native resolution).

    final = (SAM eroded by band)  U  (colour-keyed pixels within band of SAM's outline that connect
            to the eroded SAM core), then holes that SAM had filled stay filled.
    """
    from .locate import locate_classical
    box, coarse = locate_classical(rgb)
    cbox = C.square_box(C.pad_box(C.bbox(sam_mask) or box, rgb.shape[:2], pad_frac=0.3, min_pad=96), rgb.shape[:2])
    x0, y0, x1, y1 = cbox
    crop = rgb[y0:y1, x0:x1]; sm = sam_mask[y0:y1, x0:x1]; cc = coarse[y0:y1, x0:x1]
    wm = background_white(crop, sm | cc)
    n = np.clip(crop.astype(np.float32) / wm, 0, 1.2)
    s = np.maximum(1.0 - n.min(-1), 1.5 * (n.max(-1) - n.min(-1)))
    s = cv2.GaussianBlur(s, (0, 0), 0.6)
    keyed = s >= low   # strict in the edge band: clearly non-white only (no halo, no contact shadow)
    core = ndi.binary_erosion(sm, iterations=band)
    ring = ndi.binary_dilation(sm, iterations=band) & ~core
    # rod band (line from the Hough detector): nothing is ever ADDED inside it, and grey pixels
    # SAM had inside it are removed (shaded rod edge along SAM's outline)
    rod = C.detect_rod(crop, sm | cc)
    rodband = None
    if rod is not None:
        rc, rd, rhw = rod
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        perp = np.abs((yy - rc[0]) * rd[1] - (xx - rc[1]) * rd[0])
        rodband = perp <= max(rhw, 15.0) + 10.0
        keyed &= ~rodband
    cand = core | (keyed & ring)
    # beetle fragments SAM never had (a leg behind the rod): strong keyed components within one
    # rod width of the mask, smaller than it, not touching the crop border, outside the rod band
    strong = s >= high
    labs, ns = ndi.label(strong & ~ndi.binary_dilation(sm, iterations=band), structure=np.ones((3, 3)))
    if ns:
        near = ndi.binary_dilation(sm, iterations=C.LINK_PX)
        border = set(np.unique(np.concatenate([labs[0], labs[-1], labs[:, 0], labs[:, -1]]))) - {0}
        sizes = ndi.sum(strong, labs, index=np.arange(1, ns + 1))
        inband = ndi.sum(rodband, labs, index=np.arange(1, ns + 1)) if rodband is not None else np.zeros(ns)
        for i in range(1, ns + 1):
            if i in border or sizes[i - 1] < 150 or sizes[i - 1] > 0.5 * sm.sum() or inband[i - 1] > 0.5 * sizes[i - 1]:
                continue
            comp = labs == i
            if (comp & near).any():
                cand |= comp
    # keep what connects to SAM's core, plus the accepted fragments (within LINK_PX of the mask)
    lab, nlab = ndi.label(cand, structure=np.ones((3, 3)))
    ids = np.unique(lab[core]); ids = ids[ids > 0]
    out = np.isin(lab, ids) | (cand & ~ndi.binary_dilation(sm, iterations=band))
    # SAM's interior decisions are kept: anything SAM marked that is not in the outer ring stays,
    # except grey rod pixels inside the rod band
    keep_int = sm & ~ring
    if rodband is not None:
        grey = (n.min(-1) > 0.62) & ((n.max(-1) - n.min(-1)) < 0.15)
        keep_int &= ~(rodband & grey)
    out |= keep_int
    # pure white, colourless patches within `gap_reach` px of SAM's outline are wall seen through
    # a gap (inside a curled claw, between leg segments): colour overrules SAM there. Deeper inside
    # the body nothing is removed, so specular highlights stay.
    white = (n.min(-1) > 0.9) & ((n.max(-1) - n.min(-1)) < 0.06)
    white = ndi.binary_opening(white, structure=np.ones((3, 3), bool))
    near_edge = ndi.binary_dilation(~sm, iterations=40) & sm
    labw, nw = ndi.label(white & sm, structure=np.ones((3, 3)))
    if nw:
        touch = np.unique(labw[near_edge & white & sm]); touch = touch[touch > 0]
        out &= ~np.isin(labw, touch)
    out = C.keep_main_component(out, link_radius=C.LINK_PX, min_frac=0.001)
    return C.paste(rgb.shape[:2], cbox, out)
