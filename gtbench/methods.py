"""Masking methods for the ground-truth benchmark (DIS5K Insect group).

Every method receives the SAME input: a square crop around the object (padded ground-truth box,
same crop rule as the Popillia pipeline) and the object box in crop coordinates. It returns a
boolean mask of the crop. No white-background assumptions, no rod logic, no post-processing:
the raw model output is what gets scored. Models are the largest public variants.

Method                 model / weights                                   prompt
sam_box                facebook/sam-vit-huge                             box
sam_points             facebook/sam-vit-huge                             box + positive point at box centre
sam2_box               facebook/sam2.1-hiera-large                       box
sam2_points            facebook/sam2.1-hiera-large                       box + positive point at box centre
samhq_box              HQ-SAM ViT-H (lkeab/hq-sam sam_hq_vit_h.pth, segment-anything-hq)      box
samhq2_box             HQ-SAM 2 hiera-large (lkeab/hq-sam sam2.1_hq_hiera_large.pt, SysCV sam-hq2 package, 2024-11 beta)  box
sam3_box               facebook/sam3 (gated)                             box
sam3_text              facebook/sam3 (gated)                             text "insect"
birefnet_crop          ZhengPeng7/BiRefNet (1024 input)                  none
birefnet_hr            ZhengPeng7/BiRefNet_HR (2048 input)               none
rembg_birefnet         rembg birefnet-general (ONNX)                     none
rembg_isnet            rembg isnet-general-use (ONNX; trained ON DIS5K)  none
rembg_u2net            rembg u2net (ONNX)                                none
grabcut                OpenCV GrabCut, 8 iterations                      box
"""
from __future__ import annotations
import os
import numpy as np
import cv2

DEVICE = os.environ.get("GTBENCH_DEVICE") or None
_cache: dict = {}


def device():
    global DEVICE
    if DEVICE is None:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return DEVICE


def _to_dev(inputs):
    import torch
    inputs = {k: (v.to(torch.float32) if getattr(v, "dtype", None) == torch.float64 else v) for k, v in inputs.items()}
    return {k: (v.to(device()) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _pick(masks, scores):
    """SAM-family multimask output -> the candidate with the highest predicted IoU."""
    return masks[0][int(np.argmax(scores[0]))].astype(bool)


def _centre(box):
    return [[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]]


# ---------------------------------------------------------------- SAM 1 (ViT-H)
def _sam1():
    if "sam1" not in _cache:
        import torch
        from transformers import SamModel, SamProcessor
        name = "facebook/sam-vit-huge"
        _cache["sam1"] = (SamProcessor.from_pretrained(name), SamModel.from_pretrained(name).to(device(), dtype=torch.float32).eval())
    return _cache["sam1"]


def _sam1_predict(crop, box, points=None):
    import torch
    proc, model = _sam1()
    kw = dict(input_boxes=[[list(map(float, box))]])
    if points:
        kw["input_points"] = [[points]]; kw["input_labels"] = [[[1] * len(points)]]
    inputs = _to_dev(proc(images=crop, return_tensors="pt", **kw))
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0].numpy()
    return _pick(masks, out.iou_scores.cpu()[0].numpy())


def sam_box(crop, box):
    return _sam1_predict(crop, box)


def sam_points(crop, box):
    return _sam1_predict(crop, box, points=_centre(box))


# ---------------------------------------------------------------- SAM 2.1 (hiera-large)
def _sam2():
    if "sam2" not in _cache:
        import torch
        from transformers import Sam2Model, Sam2Processor
        name = "facebook/sam2.1-hiera-large"
        _cache["sam2"] = (Sam2Processor.from_pretrained(name), Sam2Model.from_pretrained(name).to(device(), dtype=torch.float32).eval())
    return _cache["sam2"]


def _sam2_predict(crop, box, points=None):
    import torch
    proc, model = _sam2()
    kw = dict(input_boxes=[[list(map(float, box))]])
    if points:
        kw["input_points"] = [[points]]; kw["input_labels"] = [[[1] * len(points)]]
    inputs = _to_dev(proc(images=crop, return_tensors="pt", **kw))
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0].numpy()
    masks = masks > 0 if masks.dtype != bool else masks
    return _pick(masks, out.iou_scores.cpu()[0].numpy())


def sam2_box(crop, box):
    return _sam2_predict(crop, box)


def sam2_points(crop, box):
    return _sam2_predict(crop, box, points=_centre(box))


# ---------------------------------------------------------------- HQ-SAM (ViT-H, official package)
def _samhq(model_type="vit_h"):
    key = "samhq:" + model_type
    if key not in _cache:
        import torch
        from huggingface_hub import hf_hub_download
        from segment_anything_hq import sam_model_registry, SamPredictor
        ckpt = hf_hub_download("lkeab/hq-sam", f"sam_hq_{model_type}.pth")
        sam = sam_model_registry[model_type](checkpoint=ckpt).to(device()).eval()
        _cache[key] = SamPredictor(sam)
    return _cache[key]


def samhq_box(crop, box):
    """HQ-SAM ViT-H. Authors' recommendation for a single object: multimask_output=False, hq_token_only=True
    (HQ output alone; False would only correct the SAM output)."""
    import torch
    pred = _samhq(os.environ.get("GTBENCH_HQSAM", "vit_h"))
    pred.set_image(crop)
    with torch.no_grad():
        masks, scores, _ = pred.predict(box=np.array(box, dtype=np.float32), multimask_output=False, hq_token_only=True)
    return masks[0].astype(bool)


# ---------------------------------------------------------------- HQ-SAM 2 (SysCV sam-hq2 fork of SAM 2.1, hiera-large)
def _samhq2():
    if "samhq2" not in _cache:
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2                      # SysCV/sam-hq/sam-hq2 package (installs as `sam2`)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = hf_hub_download("lkeab/hq-sam", "sam2.1_hq_hiera_large.pt")
        _cache["samhq2"] = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hq_hiera_l.yaml", ckpt, device=device()))
    return _cache["samhq2"]


def samhq2_box(crop, box):
    """HQ-SAM 2 (hiera-large) with the box prompt; same single-object settings as HQ-SAM (hq_token_only=True)."""
    import torch
    pred = _samhq2()
    with torch.inference_mode():
        pred.set_image(crop)
        masks, scores, _ = pred.predict(box=np.array(box, dtype=np.float32), multimask_output=False, hq_token_only=True)
    return masks[0].astype(bool)


# ---------------------------------------------------------------- SAM 3 (gated weights)
def _sam3():
    if "sam3" not in _cache:
        import torch
        from transformers import Sam3Model, Sam3Processor
        _cache["sam3"] = (Sam3Processor.from_pretrained("facebook/sam3"), Sam3Model.from_pretrained("facebook/sam3").to(device(), dtype=torch.float32).eval())
    return _cache["sam3"]


def _sam3_pick(res, box, shape):
    """Among SAM 3 instances take the one that best covers the prompt box (highest IoU with the box region)."""
    if len(res["masks"]) == 0:
        return np.zeros(shape, bool)
    ms = res["masks"].cpu().numpy().astype(bool)
    b = np.zeros(shape, bool); x0, y0, x1, y1 = [int(round(v)) for v in box]; b[max(0, y0):y1, max(0, x0):x1] = True
    ious = [(m & b).sum() / max(1, (m | b).sum()) for m in ms]
    return ms[int(np.argmax(ious))]


def sam3_box(crop, box):
    import torch
    proc, model = _sam3()
    inputs = _to_dev(proc(images=crop, input_boxes=[[list(map(float, box))]], input_boxes_labels=[[1]], return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=inputs["original_sizes"].cpu().tolist())[0]
    return _sam3_pick(res, box, crop.shape[:2])


def sam3_text(crop, box, prompt=None):
    import torch
    proc, model = _sam3()
    prompt = prompt or os.environ.get("GTBENCH_SAM3_PROMPT", "insect")
    inputs = _to_dev(proc(images=crop, text=prompt, return_tensors="pt"))
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=inputs["original_sizes"].cpu().tolist())[0]
    return _sam3_pick(res, box, crop.shape[:2])


# ---------------------------------------------------------------- BiRefNet (transformers remote code)
def _birefnet(name):
    key = "biref:" + name
    if key not in _cache:
        import torch
        from transformers import AutoModelForImageSegmentation
        _cache[key] = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True).to(device(), dtype=torch.float32).eval()
    return _cache[key]


def _birefnet_mask(crop, name, size):
    import torch
    model = _birefnet(name)
    h, w = crop.shape[:2]
    x = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(device())
    with torch.no_grad():
        p = model(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def birefnet_crop(crop, box):
    return _birefnet_mask(crop, "ZhengPeng7/BiRefNet", 1024)


def birefnet_hr(crop, box):
    return _birefnet_mask(crop, "ZhengPeng7/BiRefNet_HR", 2048)


# ---------------------------------------------------------------- rembg (ONNX)
def _rembg(crop, model):
    from rembg import remove, new_session
    from PIL import Image
    key = "rembg:" + model
    if key not in _cache:
        _cache[key] = new_session(model)
    out = remove(Image.fromarray(crop), session=_cache[key], only_mask=True, post_process_mask=False)
    return np.asarray(out) > 127


def rembg_birefnet(crop, box): return _rembg(crop, "birefnet-general")
def rembg_isnet(crop, box): return _rembg(crop, "isnet-general-use")
def rembg_u2net(crop, box): return _rembg(crop, "u2net")


# ---------------------------------------------------------------- GrabCut (classical floor)
def grabcut(crop, box, iters=8, max_side=1024):
    h, w = crop.shape[:2]; s = min(1.0, max_side / max(h, w))
    img = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else crop
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    x0, y0, x1, y1 = [int(round(v * s)) for v in box]
    rect = (max(0, x0), max(0, y0), max(1, min(bgr.shape[1], x1) - max(0, x0)), max(1, min(bgr.shape[0], y1) - max(0, y0)))
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if s < 1 else m


METHODS = {
    "sam_box": sam_box, "sam_points": sam_points,
    "sam2_box": sam2_box, "sam2_points": sam2_points,
    "samhq_box": samhq_box, "samhq2_box": samhq2_box,
    "sam3_box": sam3_box, "sam3_text": sam3_text,
    "birefnet_crop": birefnet_crop, "birefnet_hr": birefnet_hr,
    "rembg_birefnet": rembg_birefnet, "rembg_isnet": rembg_isnet, "rembg_u2net": rembg_u2net,
    "grabcut": grabcut,
}
GROUPS = {
    "sam": ["sam_box", "sam_points", "sam2_box", "sam2_points", "samhq_box", "samhq2_box"],
    "sam3": ["sam3_box", "sam3_text"],
    "salient": ["birefnet_crop", "birefnet_hr", "rembg_birefnet", "rembg_isnet", "rembg_u2net"],
    "classical": ["grabcut"],
}
