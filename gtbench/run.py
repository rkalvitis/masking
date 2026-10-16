"""Run every benchmark method on the DIS5K Insect set (or any images+masks folder).

    python -m gtbench.run --data DIS5K_insect --out out/gtbench --weights /path/for/weights \
        [--methods short|all|sam,sam3,salient,classical|name,name] [--device cuda] [--limit N] [--only stem,stem] [--overwrite]

Per image: square crop around the ground-truth box (pad 60 % of the box, min 96 px, same rule as
the Popillia pipeline: the object fills the model input), every method on that crop with the box
in crop coordinates, the crop mask pasted back into a full-size mask. Layout (one subfolder per method):
    <out>/<method>/masks/<stem>.png   <out>/<method>/timing.json   <out>/<method>/run.log
    <out>/_inputs/crops/<stem>.jpg + <out>/_inputs/crops.json      (the exact crops/boxes every method saw)
Ground truth is used ONLY for the box; nothing else from it reaches the methods.
"""
import argparse, csv, glob, json, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="DIS5K_insect", help="folder with images/, masks/ (same stems) and optional index.csv; on the server: <mask_bench>/gt")
ap.add_argument("--out", default="out/gtbench", help="results root, one subfolder per method; on the server: <mask_bench>/results")
ap.add_argument("--weights", default=None, help="where model weights/caches are stored (HF_HOME, U2NET_HOME, TORCH_HOME set from this)")
ap.add_argument("--methods", default="short", help="short (default: sam2_box, sam3_box, samhq_box, samhq2_box, birefnet_crop, rembg_isnet), all, a group, or names")
ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only", default=None, help="comma list of stems")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--pad", type=float, default=0.6, help="crop padding as a fraction of the box size")
ap.add_argument("--min-pad", type=int, default=96)
ap.add_argument("--sam3-prompt", default="insect")
ap.add_argument("--hqsam", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
args = ap.parse_args()

if args.weights:
    w = os.path.abspath(args.weights); os.makedirs(w, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(w, "hf"))
    os.environ.setdefault("U2NET_HOME", os.path.join(w, "rembg"))
    os.environ.setdefault("TORCH_HOME", os.path.join(w, "torch"))
if args.device:
    os.environ["GTBENCH_DEVICE"] = args.device
os.environ["GTBENCH_SAM3_PROMPT"] = args.sam3_prompt
os.environ["GTBENCH_HQSAM"] = args.hqsam

import numpy as np, cv2
from gtbench import methods as M

names = []
for k in args.methods.split(","):
    if k == "all": names += list(M.METHODS)
    elif k in M.GROUPS: names += M.GROUPS[k]
    elif k in M.METHODS: names.append(k)
    else: sys.exit(f"unknown method/group: {k}")
names = list(dict.fromkeys(names))

imgs = sorted(glob.glob(os.path.join(args.data, "images", "*")))
if args.only:
    keep = set(args.only.split(",")); imgs = [p for p in imgs if os.path.splitext(os.path.basename(p))[0] in keep]
if args.limit:
    imgs = imgs[:args.limit]
INP = os.path.join(args.out, "_inputs"); os.makedirs(os.path.join(INP, "crops"), exist_ok=True)
logs = {}
def mdir(m): return os.path.join(args.out, m)
def say(s, m=None):
    print(s, flush=True)
    for mm in ([m] if m else names):
        if mm not in logs:
            os.makedirs(mdir(mm), exist_ok=True); logs[mm] = open(os.path.join(mdir(mm), "run.log"), "a")
        logs[mm].write(s + "\n"); logs[mm].flush()

def tpath(m): return os.path.join(mdir(m), "timing.json")
timing = {m: (json.load(open(tpath(m))) if os.path.exists(tpath(m)) else {}) for m in names}
cpath = os.path.join(INP, "crops.json"); crops = json.load(open(cpath)) if os.path.exists(cpath) else {}
say(f"== gtbench run {time.strftime('%F %T')}: {len(imgs)} images, methods {names}, device {M.device()}")


def crop_box(gt, shape, pad_frac, min_pad):
    ys, xs = np.where(gt)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    px, py = max(min_pad, int(bw * pad_frac)), max(min_pad, int(bh * pad_frac))
    cx0, cy0, cx1, cy1 = x0 - px, y0 - py, x1 + px, y1 + py
    side = max(cx1 - cx0, cy1 - cy0)                       # square
    cxm, cym = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    cx0, cy0 = int(round(cxm - side / 2)), int(round(cym - side / 2))
    H, W = shape
    cx0 = min(max(0, cx0), max(0, W - side)); cy0 = min(max(0, cy0), max(0, H - side))
    cx1, cy1 = min(W, cx0 + side), min(H, cy0 + side)
    return (int(cx0), int(cy0), int(cx1), int(cy1)), (int(x0), int(y0), int(x1), int(y1))


for i, p in enumerate(imgs, 1):
    stem = os.path.splitext(os.path.basename(p))[0]
    todo = [m for m in names if args.overwrite or not os.path.exists(os.path.join(mdir(m), "masks", stem + ".png"))]
    if not todo:
        continue
    rgb = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(args.data, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE) > 127
    cbox, gbox = crop_box(gt, gt.shape, args.pad, args.min_pad)
    crop = np.ascontiguousarray(rgb[cbox[1]:cbox[3], cbox[0]:cbox[2]])
    box = [gbox[0] - cbox[0], gbox[1] - cbox[1], gbox[2] - cbox[0], gbox[3] - cbox[1]]
    cp = os.path.join(INP, "crops", stem + ".jpg")
    if not os.path.exists(cp):
        cv2.imwrite(cp, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    crops[stem] = dict(image=os.path.relpath(p, args.data), full_size=[int(rgb.shape[1]), int(rgb.shape[0])], crop=list(cbox), gt_box=list(gbox), box_in_crop=box)
    say(f"[{i}/{len(imgs)}] {stem}: full {rgb.shape[1]}x{rgb.shape[0]}, crop {crop.shape[1]}x{crop.shape[0]}, box {box}")
    for m in todo:
        t0 = time.time()
        try:
            cm = M.METHODS[m](crop, box)
            dt = time.time() - t0
            full = np.zeros(gt.shape, bool); full[cbox[1]:cbox[3], cbox[0]:cbox[2]] = cm
            od = os.path.join(mdir(m), "masks"); os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, stem + ".png"), full.astype(np.uint8) * 255)
            timing[m][stem] = round(dt, 3)
            say(f"    {m:15s} area {cm.mean()*100:5.1f}% of crop  {dt:6.1f}s", m)
            json.dump(timing[m], open(tpath(m), "w"), indent=1)
        except Exception as e:
            say(f"    {m:15s} FAILED: {type(e).__name__}: {e}", m)
            logs[m].write(traceback.format_exc())
    json.dump(crops, open(cpath, "w"), indent=1)
say("== done")
