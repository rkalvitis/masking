"""Mathematical / classical segmentation methods.

Every method: fn(rgb_fullres, ctx) -> bool mask at full resolution.
ctx carries the cached coarse localisation (see run_methods.py).
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import slic, random_walker, morphological_chan_vese, watershed, felzenszwalb
from skimage.morphology import disk
from sklearn.cluster import KMeans
from . import common as C

CROP_PAD = 0.6  # relative padding around the coarse box for crop-based methods


def _crop(rgb, ctx, pad=CROP_PAD):
    box = C.pad_box(ctx["box"], rgb.shape[:2], pad_frac=pad, min_pad=96)
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1], box


def _coarse_in_crop(ctx, box):
    x0, y0, x1, y1 = box
    return ctx["coarse_mask"][y0:y1, x0:x1]


# ---------------------------------------------------------------- 1. global Otsu on colour score
def otsu_score(rgb, ctx):
    """Otsu threshold of a darkness/chroma score at half res, border-blob removal, main blob."""
    half = C.downscale(rgb, 2)
    s = C.foreground_score(half)
    s = cv2.GaussianBlur(s, (0, 0), 0.8)
    t = max(threshold_otsu(s), 0.2)
    m = C.clean(s > t, link_radius=20, open_px=1)
    return C.upscale_mask(m, rgb.shape[:2])


# ---------------------------------------------------------------- 2. local Otsu inside crop (full res)
def otsu_crop(rgb, ctx):
    """Same score, but thresholded only inside the crop at full res so the threshold adapts."""
    crop, box = _crop(rgb, ctx)
    s = C.foreground_score(crop)
    s = cv2.GaussianBlur(s, (0, 0), 1.2)
    t = max(threshold_otsu(s), 0.2)
    m = C.clean_crop(ndi.binary_opening(s > t, iterations=2), _coarse_in_crop(ctx, box), link_radius=40)
    return C.paste(rgb.shape[:2], box, m)


# ---------------------------------------------------------------- 3. K-means in Lab inside crop
def kmeans_lab(rgb, ctx, k=4):
    """K-means on Lab pixels in the crop; background = cluster(s) with highest L and low chroma."""
    crop, box = _crop(rgb, ctx)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    X = small.reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X[::3])
    cent = km.cluster_centers_
    chroma = np.hypot(cent[:, 1] - 128, cent[:, 2] - 128)
    # background clusters: bright (L within 35 of max) and low chroma
    bg = (cent[:, 0] > cent[:, 0].max() - 35) & (chroma < 18)
    labels = km.predict(lab.reshape(-1, 3)).reshape(h, w)
    fg = ~np.isin(labels, np.where(bg)[0])
    m = C.clean_crop(ndi.binary_opening(fg, iterations=2), _coarse_in_crop(ctx, box), link_radius=40)
    return C.paste(rgb.shape[:2], box, m)


# ---------------------------------------------------------------- 4. GrabCut
def grabcut(rgb, ctx, iters=6):
    """OpenCV GrabCut in the crop, initialised from the coarse mask (sure fg = eroded coarse)."""
    crop, box = _crop(rgb, ctx)
    coarse = _coarse_in_crop(ctx, box)
    h, w = coarse.shape
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    dil = ndi.binary_dilation(coarse, iterations=40)
    ero = ndi.binary_erosion(coarse, iterations=8)
    gc[dil] = cv2.GC_PR_FGD
    gc[ero] = cv2.GC_FGD
    gc[~ndi.binary_dilation(coarse, iterations=120)] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    # run at half res for speed, then refine 1 iteration at full res
    cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    m = C.clean_crop(m, coarse, link_radius=40)
    return C.paste(rgb.shape[:2], box, m)


# ---------------------------------------------------------------- 5. SLIC superpixels + background colour model
def slic_bgmodel(rgb, ctx, n_segments=1500):
    """SLIC superpixels in crop; superpixels are background if close (Mahalanobis) to the
    colour model of the crop's outer ring, and not dark. Foreground = rest, cleaned."""
    crop, box = _crop(rgb, ctx)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    seg = slic(crop, n_segments=n_segments, compactness=12, sigma=1, start_label=0, channel_axis=-1)
    n = seg.max() + 1
    feats = np.stack([ndi.mean(lab[..., c], seg, index=np.arange(n)) for c in range(3)], 1)
    ring = np.zeros((h, w), bool)
    r = max(8, min(h, w) // 12)
    ring[:r, :] = ring[-r:, :] = ring[:, :r] = ring[:, -r:] = True
    ring_ids = np.unique(seg[ring])
    bgX = feats[ring_ids]
    mu = bgX.mean(0)
    cov = np.cov(bgX.T) + np.eye(3) * 100.0   # floor of 10 Lab units std: wall shading is not foreground
    icov = np.linalg.inv(cov)
    d = feats - mu
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))
    fg_sp = maha > 3.0
    fg = fg_sp[seg]
    m = C.clean_crop(fg, _coarse_in_crop(ctx, box), link_radius=40)
    return C.paste(rgb.shape[:2], box, m)


# ---------------------------------------------------------------- 6. Random walker
def random_walk(rgb, ctx):
    """Random walker with fg seeds = eroded coarse mask, bg seeds = far ring; on Lab at half res."""
    crop, box = _crop(rgb, ctx)
    coarse = _coarse_in_crop(ctx, box)
    half = C.downscale(crop, 2)
    ch = cv2.resize(coarse.astype(np.uint8), (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    lab = cv2.cvtColor(half, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    labels = np.zeros(ch.shape, np.int32)
    labels[ndi.binary_erosion(ch, iterations=5)] = 1
    labels[~ndi.binary_dilation(ch, iterations=45)] = 2
    if labels.max() < 2 or not (labels == 1).any():
        return C.upscale_mask(ch, rgb.shape[:2])
    out = random_walker(lab, labels, beta=250, mode="cg_j", channel_axis=-1)
    m = out == 1
    m = C.clean_crop(m, ch, link_radius=20)
    mfull = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    return C.paste(rgb.shape[:2], box, mfull)


# ---------------------------------------------------------------- 7. Morphological Chan-Vese (active contour)
def chan_vese(rgb, ctx, iters=60):
    """Morphological ACWE on the foreground score inside the crop, initialised by the coarse mask."""
    crop, box = _crop(rgb, ctx)
    coarse = _coarse_in_crop(ctx, box)
    half = C.downscale(crop, 2)
    ch = cv2.resize(coarse.astype(np.uint8), (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    s = C.foreground_score(half)
    s = cv2.GaussianBlur(s, (0, 0), 1.0)
    init = ndi.binary_dilation(ch, iterations=6)
    m = morphological_chan_vese(s, num_iter=iters, init_level_set=init, smoothing=1, lambda1=1, lambda2=1)
    m = C.clean_crop(m.astype(bool), ch, link_radius=20)
    mfull = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    return C.paste(rgb.shape[:2], box, mfull)


# ---------------------------------------------------------------- 8. Marker watershed on gradient
def watershed_markers(rgb, ctx):
    """Marker-based watershed on a Sobel gradient of the score map (crop, full res)."""
    crop, box = _crop(rgb, ctx)
    coarse = _coarse_in_crop(ctx, box)
    s = C.foreground_score(crop)
    s = cv2.GaussianBlur(s, (0, 0), 1.5)
    grad = sobel(s)
    markers = np.zeros(s.shape, np.int32)
    markers[ndi.binary_erosion(coarse, iterations=10)] = 1
    markers[~ndi.binary_dilation(coarse, iterations=90)] = 2
    if not (markers == 1).any():
        return C.paste(rgb.shape[:2], box, coarse)
    ws = watershed(grad, markers)
    m = C.clean_crop(ws == 1, coarse, link_radius=20)
    return C.paste(rgb.shape[:2], box, m)


# ---------------------------------------------------------------- 9. Felzenszwalb graph segmentation
def felzenszwalb_graph(rgb, ctx):
    """Felzenszwalb graph-based segmentation on the crop (half res); regions whose mean score is
    high and overlap the coarse mask are foreground."""
    crop, box = _crop(rgb, ctx)
    coarse = _coarse_in_crop(ctx, box)
    half = C.downscale(crop, 2)
    ch = cv2.resize(coarse.astype(np.uint8), (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    seg = felzenszwalb(half, scale=200, sigma=0.8, min_size=60)
    n = seg.max() + 1
    s = C.foreground_score(half)
    mean_s = ndi.mean(s, seg, index=np.arange(n))
    ov = ndi.mean(ndi.binary_dilation(ch, iterations=4).astype(float), seg, index=np.arange(n))
    fg_ids = (mean_s > 0.3) & (ov > 0.3)
    m = fg_ids[seg]
    m = C.clean_crop(m, ch, link_radius=20)
    mfull = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    return C.paste(rgb.shape[:2], box, mfull)


def hysteresis_global(rgb, ctx):
    """The coarse locator itself: hysteresis threshold of the score map at 1/4 res, upsampled."""
    return ctx["coarse_mask"].copy()


METHODS = {
    "hysteresis_global": hysteresis_global,
    "otsu_score": otsu_score,
    "otsu_crop": otsu_crop,
    "kmeans_lab": kmeans_lab,
    "grabcut": grabcut,
    "slic_bgmodel": slic_bgmodel,
    "random_walker": random_walk,
    "chan_vese": chan_vese,
    "watershed": watershed_markers,
    "felzenszwalb": felzenszwalb_graph,
}
