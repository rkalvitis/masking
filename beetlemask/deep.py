"""Deep-learning segmentation methods (SAM, BiRefNet, rembg models, OWLv2 detection).

All heavy models run on a crop around the coarse box so the ~1-2 %-of-frame beetle gets
full input resolution; results are pasted back into a full-res mask.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import torch
from scipy import ndimage as ndi
from . import common as C
from .locate import locate_owlv2

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = os.environ.get("BEETLE_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
CROP_PAD = 0.6
_cache: dict = {}


def _crop(rgb, ctx, pad=CROP_PAD, square=True):
    box = C.pad_box(ctx["box"], rgb.shape[:2], pad_frac=pad, min_pad=96)
    if square:
        box = C.square_box(box, rgb.shape[:2])
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1], box


def _box_in_crop(box_full, crop_box):
    x0, y0, _, _ = crop_box
    bx0, by0, bx1, by1 = box_full
    return [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]


# ---------------------------------------------------------------- SAM
def _sam(name="facebook/sam-vit-large"):
    key = "sam:" + name
    if key not in _cache:
        from transformers import SamModel, SamProcessor
        proc = SamProcessor.from_pretrained(name)
        model = SamModel.from_pretrained(name).to(DEVICE).eval()
        _cache[key] = (proc, model)
    return _cache[key]


def _sam_predict(crop_rgb, boxes=None, points=None, labels=None, multimask=True, name="facebook/sam-vit-large"):
    proc, model = _sam(name)
    kw = {}
    if boxes is not None:
        kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
    if points is not None:
        kw["input_points"] = [[[list(map(float, p)) for p in points]]]
        kw["input_labels"] = [[list(map(int, labels))]]
    inputs = proc(crop_rgb, return_tensors="pt", **kw)
    inputs = {k: (v.to(torch.float32) if getattr(v, "dtype", None) == torch.float64 else v) for k, v in inputs.items()}
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, multimask_output=multimask)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
    scores = out.iou_scores.cpu()[0]
    # masks: (n_prompts, n_masks, H, W)
    return masks.numpy(), scores.numpy()


def _sam_pick(masks, scores, ref=None):
    """Pick mask: highest predicted IoU, but if a reference blob is given prefer the candidate
    with highest IoU to it (SAM's 3 outputs are part/whole/sub-part; we want the whole beetle)."""
    cands = masks[0]
    sc = scores[0]
    if ref is not None:
        ious = [(c & ref).sum() / max(1, (c | ref).sum()) for c in cands]
        j = int(np.argmax(np.array(ious) + 0.2 * sc))
    else:
        j = int(np.argmax(sc))
    return cands[j].astype(bool)


def sam_box(rgb, ctx):
    """SAM (ViT-L) with the coarse classical box as prompt, on the crop."""
    crop, cbox = _crop(rgb, ctx)
    b = _box_in_crop(ctx["box"], cbox)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    masks, scores = _sam_predict(crop, boxes=[b])
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def sam_points(rgb, ctx):
    """SAM with positive points inside the coarse blob (distance-transform peaks) and negative
    points on bright pixels near the blob (rod / wall), plus the box."""
    crop, cbox = _crop(rgb, ctx)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    b = _box_in_crop(ctx["box"], cbox)
    dt = ndi.distance_transform_edt(coarse)
    pts, labs = [], []
    # up to 3 positive points: global max of DT, then maxima after suppressing a neighbourhood
    d = dt.copy()
    for _ in range(3):
        if d.max() <= 0:
            break
        y, x = np.unravel_index(np.argmax(d), d.shape)
        pts.append((x, y)); labs.append(1)
        r = max(20, int(dt.max() * 2))
        d[max(0, y - r):y + r, max(0, x - r):x + r] = 0
    # negative points: bright pixels in a ring 15-60 px outside the blob
    ring = ndi.binary_dilation(coarse, iterations=60) & ~ndi.binary_dilation(coarse, iterations=15)
    bright = (crop.min(-1) > 200) & ring
    ys, xs = np.where(bright)
    if len(ys) > 0:
        rng = np.random.default_rng(0)
        for i in rng.choice(len(ys), size=min(4, len(ys)), replace=False):
            pts.append((xs[i], ys[i])); labs.append(0)
    if not pts:
        return sam_box(rgb, ctx)
    masks, scores = _sam_predict(crop, boxes=[b], points=pts, labels=labs)
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def insectsam_box(rgb, ctx):
    """InsectSAM (martintomov/InsectSAM: SAM ViT-B fine-tuned on insects) with the coarse box, on the crop."""
    crop, cbox = _crop(rgb, ctx)
    b = _box_in_crop(ctx["box"], cbox)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    masks, scores = _sam_predict(crop, boxes=[b], name="martintomov/InsectSAM")
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def owlv2_sam(rgb, ctx):
    """OWLv2 ('a beetle'/'an insect') box -> SAM on crop. Mirrors the original pipeline's
    detector; falls back to the classical box if OWLv2 finds nothing."""
    if "owl_box" not in ctx:
        ctx["owl_box"], ctx["owl_score"], ctx["owl_label"] = locate_owlv2(rgb, device=DEVICE)
    box = ctx["owl_box"] if ctx["owl_box"] is not None else ctx["box"]
    sub = dict(ctx); sub["box"] = box
    crop, cbox = _crop(rgb, sub)
    b = _box_in_crop(box, cbox)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    masks, scores = _sam_predict(crop, boxes=[b])
    m = _sam_pick(masks, scores, ref=coarse if coarse.any() else None)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def owlv2_sam_full(rgb, ctx):
    """Previous pipeline, as a single candidate: OWLv2 box -> SAM ViT-L on the FULL frame
    (SAM downsamples the 4032 px frame to 1024 px, so the beetle is ~100 px wide)."""
    if "owl_box" not in ctx:
        ctx["owl_box"], ctx["owl_score"], ctx["owl_label"] = locate_owlv2(rgb, device=DEVICE)
    box = ctx["owl_box"] if ctx["owl_box"] is not None else ctx["box"]
    half = C.downscale(rgb, 2)   # 2016 px is still > SAM's 1024 input, identical result, less RAM
    b = [v / 2 for v in box]
    masks, scores = _sam_predict(half, boxes=[b])
    coarse_half = C.downscale(ctx["coarse_mask"].astype(np.uint8) * 255, 2) > 127
    m = _sam_pick(masks, scores, ref=coarse_half if coarse_half.any() else None)
    m = C.clean_crop(m, coarse_half, link_radius=5)
    return C.upscale_mask(m, rgb.shape[:2])


# ---------------------------------------------------------------- SAM 2.1 / HQ-SAM / SAM 3
def _to_dev(inputs):
    inputs = {k: (v.to(torch.float32) if getattr(v, "dtype", None) == torch.float64 else v) for k, v in inputs.items()}
    return {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _generic_box_predict(model_name, cls_model, cls_proc, crop_rgb, box, **fwd):
    key = "gen:" + model_name
    if key not in _cache:
        proc = cls_proc.from_pretrained(model_name)
        model = cls_model.from_pretrained(model_name).to(DEVICE, dtype=torch.float32).eval()
        _cache[key] = (proc, model)
    proc, model = _cache[key]
    inputs = _to_dev(proc(images=crop_rgb, input_boxes=[[list(map(float, box))]], return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs, multimask_output=True, **fwd)
    if "reshaped_input_sizes" in inputs:   # SAM / HQ-SAM processors
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
    else:                                   # SAM 2 processor
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    masks = masks.numpy()
    if masks.dtype != bool:
        masks = masks > 0
    return masks, out.iou_scores.cpu()[0].numpy()


def sam2_box(rgb, ctx):
    """SAM 2.1 (hiera-large) with the coarse box prompt, on the crop."""
    from transformers import Sam2Model, Sam2Processor
    crop, cbox = _crop(rgb, ctx)
    b = _box_in_crop(ctx["box"], cbox)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    masks, scores = _generic_box_predict("facebook/sam2.1-hiera-large", Sam2Model, Sam2Processor, crop, b)
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def grey_enhance(crop, coarse, lo=0.15, hi=0.85, gamma=1.3):
    """Greyscale + contrast stretch: luminance divided by the background white reference
    (estimated from background pixels only), then levels: <= lo -> black, >= hi -> white, gamma
    darkens the mid-tones. Returned as a 3-channel uint8 image for the model."""
    from .colorkey import background_white
    wm = background_white(crop, coarse)
    lum = (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]).astype(np.float32)
    wl = 0.299 * wm[..., 0] + 0.587 * wm[..., 1] + 0.114 * wm[..., 2]
    g = np.clip(lum / np.maximum(wl, 1.0), 0, 1.2)
    g = np.clip((g - lo) / (hi - lo), 0, 1) ** gamma
    g8 = (g * 255 + 0.5).astype(np.uint8)
    return np.stack([g8] * 3, -1)


def sam2_box_grey(rgb, ctx):
    """SAM 2.1 box prompt on a grey, contrast-stretched crop (user idea: whiter whites, darker darks)."""
    from transformers import Sam2Model, Sam2Processor
    crop, cbox = _crop(rgb, ctx)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    b = _box_in_crop(ctx["box"], cbox)
    ge = grey_enhance(crop, coarse)
    ctx["_grey_crop"] = (ge, cbox)
    masks, scores = _generic_box_predict("facebook/sam2.1-hiera-large", Sam2Model, Sam2Processor, ge, b)
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def linear_enhance(img, alpha, beta, grey=False):
    """cv2.convertScaleAbs: out = clip(alpha*x + beta, 0, 255) per channel; optionally on greyscale."""
    if grey:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.stack([cv2.convertScaleAbs(g, alpha=alpha, beta=beta)] * 3, -1)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _sam2_box_linear(rgb, ctx, alpha, beta, grey=False):
    from transformers import Sam2Model, Sam2Processor
    crop, cbox = _crop(rgb, ctx)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    b = _box_in_crop(ctx["box"], cbox)
    masks, scores = _generic_box_predict("facebook/sam2.1-hiera-large", Sam2Model, Sam2Processor, linear_enhance(crop, alpha, beta, grey), b)
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def sam2_box_lin15(rgb, ctx):
    """SAM 2.1 on crop after convertScaleAbs(alpha=1.5, beta=-20)."""
    return _sam2_box_linear(rgb, ctx, 1.5, -20)


def sam2_box_lin05(rgb, ctx):
    """SAM 2.1 on crop after convertScaleAbs(alpha=0.5, beta=-47) (slider setting)."""
    return _sam2_box_linear(rgb, ctx, 0.5, -47)


def sam2_box_g07p100(rgb, ctx):
    """SAM 2.1 on greyscale crop after convertScaleAbs(alpha=0.7, beta=+100)."""
    return _sam2_box_linear(rgb, ctx, 0.7, 100, grey=True)


def sam2_box_g145m100(rgb, ctx):
    """SAM 2.1 on greyscale crop after convertScaleAbs(alpha=1.45, beta=-100)."""
    return _sam2_box_linear(rgb, ctx, 1.45, -100, grey=True)


def sam2_box_g125m100(rgb, ctx):
    """SAM 2.1 on greyscale crop after convertScaleAbs(alpha=1.25, beta=-100)."""
    return _sam2_box_linear(rgb, ctx, 1.25, -100, grey=True)


def sam2_box_lin125m100(rgb, ctx):
    """SAM 2.1 on RGB crop after convertScaleAbs(alpha=1.25, beta=-100)."""
    return _sam2_box_linear(rgb, ctx, 1.25, -100, grey=False)


def _sam3_box_predict(crop, box, coarse):
    from transformers import Sam3Model, Sam3Processor
    key = "sam3"
    if key not in _cache:
        _cache[key] = (Sam3Processor.from_pretrained("facebook/sam3"), Sam3Model.from_pretrained("facebook/sam3").to(DEVICE, dtype=torch.float32).eval())
    proc, model = _cache[key]
    inputs = _to_dev(proc(images=crop, input_boxes=[[list(map(float, box))]], input_boxes_labels=[[1]], return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=inputs["original_sizes"].cpu().tolist())[0]
    if len(res["masks"]) == 0:
        return np.zeros(crop.shape[:2], bool)
    ms = res["masks"].cpu().numpy().astype(bool)
    ious = [(mm & coarse).sum() / max(1, (mm | coarse).sum()) for mm in ms]
    return ms[int(np.argmax(ious))]


def _sam3_box_generic(rgb, ctx, alpha=None, beta=None, grey=False):
    crop, cbox = _crop(rgb, ctx)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    b = _box_in_crop(ctx["box"], cbox)
    inp = crop if alpha is None else linear_enhance(crop, alpha, beta, grey)
    m = _sam3_box_predict(inp, b, coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def sam3_box(rgb, ctx):
    """SAM 3 with the coarse box as prompt (no text), on the crop."""
    return _sam3_box_generic(rgb, ctx)


def sam3_box_g125m100(rgb, ctx): return _sam3_box_generic(rgb, ctx, 1.25, -100, True)
def sam3_box_lin125m100(rgb, ctx): return _sam3_box_generic(rgb, ctx, 1.25, -100, False)
def sam3_box_g18m62(rgb, ctx): return _sam3_box_generic(rgb, ctx, 1.8, -62, True)
def sam3_box_lin18m62(rgb, ctx): return _sam3_box_generic(rgb, ctx, 1.8, -62, False)
def sam2_box_g18m62(rgb, ctx): return _sam2_box_linear(rgb, ctx, 1.8, -62, grey=True)
def sam2_box_lin18m62(rgb, ctx): return _sam2_box_linear(rgb, ctx, 1.8, -62, grey=False)


def sam3_text_insect(rgb, ctx):
    """SAM 3 with the text prompt 'insect' (scores more consistently than 'beetle' on this data)."""
    return sam3_text(rgb, ctx, prompt="insect")


def _sam3_text_enh(rgb, ctx, alpha, beta, grey, prompt="insect"):
    """SAM 3 text prompt on a linearly enhanced (optionally greyscale) crop."""
    from transformers import Sam3Model, Sam3Processor
    key = "sam3"
    if key not in _cache:
        _cache[key] = (Sam3Processor.from_pretrained("facebook/sam3"), Sam3Model.from_pretrained("facebook/sam3").to(DEVICE, dtype=torch.float32).eval())
    proc, model = _cache[key]
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    inp = linear_enhance(crop, alpha, beta, grey)
    inputs = _to_dev(proc(images=inp, text=prompt, return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=inputs["original_sizes"].cpu().tolist())[0]
    if len(res["masks"]) == 0:
        return np.zeros(rgb.shape[:2], bool)
    ms = res["masks"].cpu().numpy().astype(bool)
    ious = [(mm & coarse).sum() / max(1, (mm | coarse).sum()) for mm in ms]
    m = ms[int(np.argmax(ious))]
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def sam3_insect_g125m100(rgb, ctx): return _sam3_text_enh(rgb, ctx, 1.25, -100, True)
def sam3_insect_lin125m100(rgb, ctx): return _sam3_text_enh(rgb, ctx, 1.25, -100, False)
def sam3_insect_g18m62(rgb, ctx): return _sam3_text_enh(rgb, ctx, 1.8, -62, True)
def sam3_insect_lin18m62(rgb, ctx): return _sam3_text_enh(rgb, ctx, 1.8, -62, False)
def sam3_insect_g07p100(rgb, ctx): return _sam3_text_enh(rgb, ctx, 0.7, 100, True)


def sam2_points(rgb, ctx):
    """SAM 2.1 with box + positive points inside the beetle + negative points on bright pixels
    near it (rod / wall), on the crop."""
    from transformers import Sam2Model, Sam2Processor
    key = "gen:facebook/sam2.1-hiera-large"
    if key not in _cache:
        _cache[key] = (Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large"),
                       Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(DEVICE, dtype=torch.float32).eval())
    proc, model = _cache[key]
    crop, cbox = _crop(rgb, ctx)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    b = _box_in_crop(ctx["box"], cbox)
    dt = ndi.distance_transform_edt(coarse)
    pts, labs = [], []
    d = dt.copy()
    for _ in range(3):
        if d.max() <= 0:
            break
        y, x = np.unravel_index(np.argmax(d), d.shape)
        pts.append([float(x), float(y)]); labs.append(1)
        r = max(20, int(dt.max() * 2))
        d[max(0, y - r):y + r, max(0, x - r):x + r] = 0
    ring = ndi.binary_dilation(coarse, iterations=60) & ~ndi.binary_dilation(coarse, iterations=15)
    ys, xs = np.where((crop.min(-1) > 200) & ring)
    if len(ys):
        rng = np.random.default_rng(0)
        for i in rng.choice(len(ys), size=min(4, len(ys)), replace=False):
            pts.append([float(xs[i]), float(ys[i])]); labs.append(0)
    if not pts:
        return sam2_box(rgb, ctx)
    inputs = _to_dev(proc(images=crop, input_boxes=[[list(map(float, b))]], input_points=[[pts]], input_labels=[[labs]], return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0].numpy()
    masks = masks > 0 if masks.dtype != bool else masks
    m = _sam_pick(masks, out.iou_scores.cpu()[0].numpy(), ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def samhq_box(rgb, ctx):
    """HQ-SAM (ViT-L, trained for finer boundaries) with the coarse box prompt, on the crop."""
    from transformers import SamHQModel, SamHQProcessor
    crop, cbox = _crop(rgb, ctx)
    b = _box_in_crop(ctx["box"], cbox)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    masks, scores = _generic_box_predict("syscv-community/sam-hq-vit-large", SamHQModel, SamHQProcessor, crop, b)
    m = _sam_pick(masks, scores, ref=coarse)
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


def sam3_text(rgb, ctx, prompt="beetle"):
    """SAM 3 promptable concept segmentation: text prompt 'beetle' on the crop, no box.
    Needs access to the gated facebook/sam3 weights (accept licence + huggingface-cli login)."""
    from transformers import Sam3Model, Sam3Processor
    key = "sam3"
    if key not in _cache:
        proc = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE, dtype=torch.float32).eval()
        _cache[key] = (proc, model)
    proc, model = _cache[key]
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    coarse = ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]]
    inputs = _to_dev(proc(images=crop, text=prompt, return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                  target_sizes=inputs["original_sizes"].cpu().tolist())[0]
    if len(res["masks"]) == 0:
        return np.zeros(rgb.shape[:2], bool)
    # pick the instance overlapping the coarse blob most (SAM 3 may also return the rod or nothing)
    ms = res["masks"].cpu().numpy().astype(bool)
    ious = [(mm & coarse).sum() / max(1, (mm | coarse).sum()) for mm in ms]
    m = ms[int(np.argmax(ious))]
    return C.paste(rgb.shape[:2], cbox, C.clean_crop(m, coarse, link_radius=10))


# ---------------------------------------------------------------- BiRefNet (HF, transformers)
def _birefnet(name="ZhengPeng7/BiRefNet"):
    key = "biref:" + name
    if key not in _cache:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True)
        model = model.to(DEVICE, dtype=torch.float32).eval()
        _cache[key] = model
    return _cache[key]


def _birefnet_prob(rgb_uint8, size=1024):
    model = _birefnet()
    h, w = rgb_uint8.shape[:2]
    x = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(DEVICE)
    with torch.no_grad():
        pred = model(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
    return C.upscale_prob(pred, (h, w))


def birefnet_full(rgb, ctx):
    """BiRefNet on the whole frame (1024x1024 input). Beetle is tiny at that scale."""
    p = _birefnet_prob(C.downscale(rgb, 2))
    coarse_half = C.downscale(ctx["coarse_mask"].astype(np.uint8) * 255, 2) > 127
    m = C.clean_crop(p > 0.5, coarse_half, link_radius=5)   # NOT clean(): the rod touches the frame border
    return C.upscale_mask(m, rgb.shape[:2])


def birefnet_crop(rgb, ctx):
    """BiRefNet on the square crop around the coarse box (crop resized to 1024)."""
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    p = _birefnet_prob(crop)
    m = C.clean_crop(p > 0.5, ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]], link_radius=10)
    return C.paste(rgb.shape[:2], cbox, m)


# ---------------------------------------------------------------- rembg (ONNX) models
def _rembg_session(model):
    key = "rembg:" + model
    if key not in _cache:
        from rembg import new_session
        _cache[key] = new_session(model)
    return _cache[key]


def _rembg_mask(crop_rgb, model):
    from rembg import remove
    from PIL import Image
    sess = _rembg_session(model)
    out = remove(Image.fromarray(crop_rgb), session=sess, only_mask=True, post_process_mask=False)
    return np.asarray(out) > 127


def rembg_isnet(rgb, ctx):
    """IS-Net (DIS) general-use via rembg, on the crop."""
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    m = C.clean_crop(_rembg_mask(crop, "isnet-general-use"), ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]], link_radius=10)
    return C.paste(rgb.shape[:2], cbox, m)


def rembg_u2net(rgb, ctx):
    """U2-Net via rembg, on the crop."""
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    m = C.clean_crop(_rembg_mask(crop, "u2net"), ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]], link_radius=10)
    return C.paste(rgb.shape[:2], cbox, m)


def rembg_birefnet(rgb, ctx):
    """BiRefNet-general ONNX via rembg, on the crop (independent implementation of BiRefNet)."""
    crop, cbox = _crop(rgb, ctx, pad=0.8)
    m = C.clean_crop(_rembg_mask(crop, "birefnet-general"), ctx["coarse_mask"][cbox[1]:cbox[3], cbox[0]:cbox[2]], link_radius=10)
    return C.paste(rgb.shape[:2], cbox, m)


METHODS = {
    "sam_box": sam_box,
    "sam_points": sam_points,
    "insectsam_box": insectsam_box,
    "owlv2_sam": owlv2_sam,
    "owlv2_sam_full": owlv2_sam_full,
    "sam2_box": sam2_box,
    "sam2_points": sam2_points,
    "sam2_box_grey": sam2_box_grey,
    "sam2_box_lin15": sam2_box_lin15,
    "sam2_box_lin05": sam2_box_lin05,
    "sam2_box_g07p100": sam2_box_g07p100,
    "sam2_box_g145m100": sam2_box_g145m100,
    "sam2_box_g125m100": sam2_box_g125m100,
    "sam2_box_lin125m100": sam2_box_lin125m100,
    "sam2_box_g18m62": sam2_box_g18m62,
    "sam2_box_lin18m62": sam2_box_lin18m62,
    "sam3_box": sam3_box,
    "sam3_text_insect": sam3_text_insect,
    "sam3_insect_g125m100": sam3_insect_g125m100,
    "sam3_insect_lin125m100": sam3_insect_lin125m100,
    "sam3_insect_g18m62": sam3_insect_g18m62,
    "sam3_insect_lin18m62": sam3_insect_lin18m62,
    "sam3_insect_g07p100": sam3_insect_g07p100,
    "sam3_box_g125m100": sam3_box_g125m100,
    "sam3_box_lin125m100": sam3_box_lin125m100,
    "sam3_box_g18m62": sam3_box_g18m62,
    "sam3_box_lin18m62": sam3_box_lin18m62,
    "samhq_box": samhq_box,
    "sam3_text": sam3_text,
    "birefnet_full": birefnet_full,
    "birefnet_crop": birefnet_crop,
    "rembg_isnet": rembg_isnet,
    "rembg_u2net": rembg_u2net,
    "rembg_birefnet": rembg_birefnet,
}
