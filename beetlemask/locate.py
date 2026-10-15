"""Coarse beetle localisation: returns a full-res bbox around the beetle."""
from __future__ import annotations
import numpy as np
import cv2
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from scipy import ndimage as ndi
from . import common as C

DOWN = 4  # locate on 1008x756


def locate_classical(rgb: np.ndarray, low=0.35, high=0.6):
    """Hysteresis threshold of the 'not-white' score at low res: seeds must be really dark
    (high), grown to a looser level (low). Border-touching blobs are dropped (floor, cardboard).
    The main blob is the one with the largest *core* (opened), so a beetle beats shaded wall.
    Then the mounting rod is stripped at full res."""
    small = C.downscale(rgb, DOWN)
    s = C.foreground_score(small)
    s = cv2.GaussianBlur(s, (0, 0), 1.0)
    m = apply_hysteresis_threshold(s, low, high)
    # Body cores: a large opening (15 px at 1/4 res = 60 px) breaks the connection between the
    # beetle body and the rod / legs. Cores touching the frame border are clutter (floor slats,
    # cardboard, the rod running out of the frame, shaded wall in a corner) and are dropped, but
    # the beetle core survives even when the rod connects it to the border.
    core = ndi.binary_opening(m, structure=np.ones((15, 15), bool))
    core = C.remove_border_components(core)
    lab, n = ndi.label(core, structure=np.ones((3, 3)))
    if n == 0:  # fall back to the old behaviour (small opening after border removal)
        m2 = C.remove_border_components(m)
        core = ndi.binary_opening(m2, structure=np.ones((7, 7), bool))
        lab, n = ndi.label(core, structure=np.ones((3, 3)))
    if n == 0:
        h, w = small.shape[:2]
        b = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
        return C.scale_box(b, DOWN), C.upscale_mask(m, rgb.shape[:2])
    sizes = ndi.sum(core, lab, index=np.arange(1, n + 1))
    # a real beetle core contains genuinely dark pixels (score >= high); a dim wall wedge does not,
    # even if it is larger. Prefer the largest core with a true seed.
    maxs = ndi.maximum(s, lab, index=np.arange(1, n + 1))
    ranked = np.where(maxs >= high, sizes, 0)
    if ranked.max() == 0:
        ranked = sizes
    main_core = lab == (int(np.argmax(ranked)) + 1)
    cb = C.bbox(main_core)
    diam = max(cb[2] - cb[0], cb[3] - cb[1])
    # blob = hysteresis component(s) touching the main core, limited to 1.5 diam around it
    reach = ndi.binary_dilation(main_core, iterations=int(1.5 * diam))
    labm, nm = ndi.label(m, structure=np.ones((3, 3)))
    ids = np.unique(labm[main_core]); ids = ids[ids > 0]
    blob = np.isin(labm, ids) & reach
    # beetle fragments cut off by the rod (a leg behind the needle): other components of m within
    # ~one rod width of the blob, smaller than the blob and NOT touching the frame border
    # (cardboard, floor and cables always do). The blob itself stays anchored to the dark core.
    H, W = m.shape
    border_ids = set(np.unique(np.concatenate([labm[0], labm[-1], labm[:, 0], labm[:, -1]])))
    link = C.LINK_PX // DOWN
    bb = C.bbox(blob)
    x0, y0 = max(0, bb[0] - link), max(0, bb[1] - link); x1, y1 = min(W, bb[2] + link), min(H, bb[3] + link)
    near = np.zeros_like(blob); near[y0:y1, x0:x1] = ndi.distance_transform_edt(~blob[y0:y1, x0:x1]) <= link
    cand = set(np.unique(labm[near & m & ~blob])) - border_ids - set(ids) - {0}
    blob_size = blob.sum()
    for cid in cand:
        comp = labm == cid
        if comp.sum() < blob_size:
            blob |= comp & reach
    full = C.upscale_mask(blob, rgb.shape[:2])
    full = C.strip_bright_rim(full, rgb)
    full = C.fill_holes(full, 0.3)
    b = C.bbox(full)
    return b, full


_owl = None


def locate_owlv2(rgb: np.ndarray, device="cpu", queries=("a beetle", "an insect", "a bug")):
    """OWLv2 open-vocabulary detector. Returns full-res box of best detection and its score."""
    global _owl
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    if _owl is None:
        proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
        _owl = (proc, model)
    proc, model = _owl
    small = C.downscale(rgb, 2)
    inputs = proc(text=[list(queries)], images=small, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    h, w = small.shape[:2]
    # OWLv2 pads to square; target size must be the padded square size
    side = max(h, w)
    res = proc.image_processor.post_process_object_detection(out, threshold=0.02, target_sizes=torch.tensor([[side, side]]))[0]
    if len(res["scores"]) == 0:
        return None, 0.0, None
    i = int(torch.argmax(res["scores"]))
    box = res["boxes"][i].tolist()
    box = [max(0, min(box[0], w)), max(0, min(box[1], h)), max(0, min(box[2], w)), max(0, min(box[3], h))]
    return C.scale_box(box, 2), float(res["scores"][i]), queries[int(res["labels"][i])]
