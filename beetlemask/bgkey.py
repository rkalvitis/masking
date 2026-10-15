"""Background-colour keying methods: the background is known to be (near) white, so model
the background explicitly and key everything that deviates from it.

fn(rgb_fullres, ctx) -> bool mask at full res. All work on a crop around the coarse box.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_sauvola, apply_hysteresis_threshold
from sklearn.mixture import GaussianMixture
from . import common as C

CROP_PAD = 0.8


def _crop(rgb, ctx, pad=CROP_PAD):
    box = C.pad_box(ctx["box"], rgb.shape[:2], pad_frac=pad, min_pad=128)
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1], box, ctx["coarse_mask"][y0:y1, x0:x1]


def _finish(m, coarse, box, shape_hw, link=40):
    m = C.clean_crop(m, coarse, link_radius=link)
    return C.paste(shape_hw, box, m)


# ---------------------------------------------------------------- 1. flat-field + Lab distance key
def flatfield_key(rgb, ctx, deg=2, low=0.12, high=0.30):
    """Fit a smooth 2-D polynomial illumination surface to background pixels (everything outside
    the dilated coarse blob) for each RGB channel, divide the image by it (flat-field), then key
    on distance from pure white in the normalised image with hysteresis."""
    crop, box, coarse = _crop(rgb, ctx)
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    cs = cv2.resize(coarse.astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_NEAREST) > 0
    bg = ~ndi.binary_dilation(cs, iterations=12)
    ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    yn, xn = ys / small.shape[0], xs / small.shape[1]
    terms = [xn ** i * yn ** j for i in range(deg + 1) for j in range(deg + 1 - i)]
    A = np.stack([t.ravel() for t in terms], 1)
    flat = np.empty_like(small)
    for c in range(3):
        v = small[..., c].ravel()
        sel = bg.ravel() & (v >= np.percentile(v[bg.ravel()], 20))  # bright background samples only (>=: enhanced bg is all 255)
        coef, *_ = np.linalg.lstsq(A[sel], v[sel], rcond=None)
        surf = (A @ coef).reshape(small.shape[:2])
        flat[..., c] = surf
    flat_full = cv2.resize(flat, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(crop.astype(np.float32) / np.maximum(flat_full, 1.0), 0, 1.5)
    # distance from white in normalised space: 1 - min channel, plus chroma
    dist = np.maximum(1.0 - norm.min(-1), (norm.max(-1) - norm.min(-1)))
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
    m = apply_hysteresis_threshold(dist, low, high)
    return _finish(m, coarse, box, rgb.shape[:2])


# ---------------------------------------------------------------- 2. morphological background estimation
def morph_bg_key(rgb, ctx, low=18, high=45):
    """Estimate the background by a grey-scale closing with a structuring element larger than
    the beetle (fills the dark beetle with surrounding wall), subtract, hysteresis-threshold."""
    crop, box, coarse = _crop(rgb, ctx)
    cb = C.bbox(coarse)
    diam = max(cb[2] - cb[0], cb[3] - cb[1]) if cb else 300
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    # work at 1/4 res for the big closing
    h, w = L.shape
    Ls = cv2.resize(L, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = int(diam / 4 * 0.6) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(Ls, cv2.MORPH_CLOSE, se)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg_full = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff = bg_full - L.astype(np.float32)          # darker than local background
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    score = np.maximum(diff, chroma * 2.0)
    score = cv2.GaussianBlur(score, (0, 0), 1.0)
    m = apply_hysteresis_threshold(score, low, high)
    return _finish(m, coarse, box, rgb.shape[:2])


# ---------------------------------------------------------------- 3. Gaussian mixture in Lab
def gmm_lab_key(rgb, ctx, k_bg=2, k_fg=3):
    """Two Gaussian mixtures in Lab: background GMM fit on pixels outside the dilated coarse blob,
    foreground GMM on pixels inside the eroded blob. Per-pixel log-likelihood ratio + hysteresis."""
    crop, box, coarse = _crop(rgb, ctx)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    X = lab.reshape(-1, 3)
    bg_sel = ~ndi.binary_dilation(coarse, iterations=40)
    fg_sel = ndi.binary_erosion(coarse, iterations=6)
    if fg_sel.sum() < 200:
        fg_sel = coarse
    rng = np.random.default_rng(0)
    bgX = X[bg_sel.ravel()]; fgX = X[fg_sel.ravel()]
    bgX = bgX[rng.choice(len(bgX), min(20000, len(bgX)), replace=False)]
    fgX = fgX[rng.choice(len(fgX), min(20000, len(fgX)), replace=False)]
    g_bg = GaussianMixture(k_bg, covariance_type="full", random_state=0, reg_covar=1.0).fit(bgX)
    g_fg = GaussianMixture(k_fg, covariance_type="full", random_state=0, reg_covar=1.0).fit(fgX)
    llr = (g_fg.score_samples(X) - g_bg.score_samples(X)).reshape(h, w)
    llr = cv2.GaussianBlur(llr.astype(np.float32), (0, 0), 1.0)
    m = apply_hysteresis_threshold(llr, -1.0, 3.0)
    return _finish(m, coarse, box, rgb.shape[:2])


# ---------------------------------------------------------------- 4. Sauvola local threshold
def sauvola_key(rgb, ctx, window=151, k=0.15):
    """Sauvola adaptive threshold (document-binarisation classic) on the min-RGB channel,
    then hysteresis so faint shading is not foreground."""
    crop, box, coarse = _crop(rgb, ctx)
    g = crop.min(-1)   # darkest channel: dark beetle vs white wall
    t = threshold_sauvola(g, window_size=window, k=k)
    dark = g < t
    # require a real darkness margin too (Sauvola alone fires on paper texture)
    s = C.foreground_score(crop)
    m = dark & (s > 0.25)
    m = ndi.binary_opening(m, iterations=1)
    return _finish(m, coarse, box, rgb.shape[:2])


# ---------------------------------------------------------------- 5. HSV key
def hsv_key(rgb, ctx, v_max=0.55, s_min=0.28):
    """Classic chroma-key style rule in HSV: foreground if value is low (dark) or saturation is
    high (coloured) relative to the white background; hysteresis on a combined score."""
    crop, box, coarse = _crop(rgb, ctx)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0; V = hsv[..., 2] / 255.0
    score = np.maximum((v_max - V) / v_max, (S - s_min) / (1 - s_min))
    score = cv2.GaussianBlur(score, (0, 0), 1.0)
    m = apply_hysteresis_threshold(score, 0.0, 0.35)
    return _finish(m, coarse, box, rgb.shape[:2])


# ---------------------------------------------------------------- 6. Difference-from-background-colour (global white model)
def white_distance_key(rgb, ctx, low=28, high=70):
    """Simplest keying: estimate the background colour as the median of the crop's outer ring,
    Euclidean RGB distance from it, hysteresis threshold."""
    crop, box, coarse = _crop(rgb, ctx)
    h, w = crop.shape[:2]
    r = max(10, min(h, w) // 10)
    ring = np.zeros((h, w), bool); ring[:r] = ring[-r:] = ring[:, :r] = ring[:, -r:] = True
    bgc = np.median(crop[ring].reshape(-1, 3), 0)
    dist = np.linalg.norm(crop.astype(np.float32) - bgc, axis=-1)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
    m = apply_hysteresis_threshold(dist, low, high)
    return _finish(m, coarse, box, rgb.shape[:2])


METHODS = {
    "bg_flatfield": flatfield_key,
    "bg_morph": morph_bg_key,
    "bg_gmm_lab": gmm_lab_key,
    "bg_sauvola": sauvola_key,
    "bg_hsv": hsv_key,
    "bg_whitedist": white_distance_key,
}
