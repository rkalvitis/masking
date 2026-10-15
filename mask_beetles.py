"""Production masking: beetle on white background, mounting rod excluded.

Pipeline per image:  classical locator (hysteresis on a darkness/chroma score, 1/4 res)
                     -> square crop around the beetle
                     -> SAM 2.1 (hiera-large) with the box prompt (or box + points)
                     -> light cleanup only (main blob + fragments within one rod width, tiny holes),
                        plus a safeguard that drops a large near-white rod chunk touching the mask edge.
                        SAM 2's box never contains the rod, so the raw mask already excludes it;
                        the heavy rod-strip post-process (--postprocess full) damages these masks.
Output: <out>/<stem>.png, 8-bit L, 255 = beetle, same size as the input, same stem as the image.

Usage:
  python mask_beetles.py --images <dir with .tif/.jpg/.png> --out <mask dir> [--prompt box|points]
                         [--preview <dir>] [--limit N] [--overwrite] [--device mps|cuda|cpu]
"""
import argparse, glob, os, sys, time, traceback
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="sam2", choices=["sam2", "sam3"], help="sam2: SAM 2.1 with the locator box; sam3: SAM 3 with the text prompt 'beetle' (gated weights)")
ap.add_argument("--prompt", default="box", choices=["box", "points"])
ap.add_argument("--preview", default=None, help="also write cut-out previews (beetle kept, rest white) here")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
ap.add_argument("--postprocess", default="none", choices=["none", "full"], help="none (default): raw SAM 2 mask with light cleanup only; full: rod-strip/hole post-process (hurts SAM 2 masks, kept for reference)")
args = ap.parse_args()
if args.device:
    os.environ["BEETLE_DEVICE"] = args.device

from beetlemask import common as C
from beetlemask.locate import locate_classical
from beetlemask import deep

exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.JPG")
paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.images, e)))
if args.limit:
    paths = paths[:args.limit]
os.makedirs(args.out, exist_ok=True)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
segment = deep.sam3_text if args.model == "sam3" else (deep.sam2_box if args.prompt == "box" else deep.sam2_points)
print(f"{len(paths)} images -> {args.out}  ({'SAM 3 text prompt' if args.model == 'sam3' else 'SAM 2.1 ' + args.prompt + ' prompt'}, device {deep.DEVICE})", flush=True)

ok = fail = skipped = 0
t_all = time.time()
for i, p in enumerate(paths, 1):
    stem = os.path.splitext(os.path.basename(p))[0]
    outp = os.path.join(args.out, stem + ".png")
    if os.path.exists(outp) and not args.overwrite:
        skipped += 1
        continue
    t0 = time.time()
    try:
        rgb = C.load_rgb(p)
        box, coarse = locate_classical(rgb)
        ctx = {"box": box, "coarse_mask": coarse, "name": stem}
        raw = segment(rgb, ctx)              # already: main blob + fragments within one rod width, tiny holes filled
        m = C.strip_bright_rim(raw, rgb) if args.postprocess == "full" else C.strip_large_white(raw, rgb)
        if m.sum() < 0.05 * max(1, coarse.sum()):      # SAM returned (almost) nothing: fall back to the coarse blob
            m = coarse
            note = " [fallback: coarse blob]"
        else:
            note = ""
        C.save_mask(m, outp)
        if args.preview:
            bb = C.bbox(m) or box
            pb = C.pad_box(bb, m.shape, pad_frac=0.25, min_pad=50)
            cut = rgb[pb[1]:pb[3], pb[0]:pb[2]].copy(); cut[~m[pb[1]:pb[3], pb[0]:pb[2]]] = 255
            Image.fromarray(cut).save(os.path.join(args.preview, stem + ".jpg"), quality=90)
        ok += 1
        area = m.mean() * 100
        if area < 0.3 or area > 2.0:
            note += f" [WARNING: unusual mask area, check preview]"
        print(f"[{i}/{len(paths)}] {stem}: mask {area:.2f}% of frame, {time.time()-t0:.1f}s{note}", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(paths)}] {stem}: FAILED {e}", flush=True)
        traceback.print_exc()
print(f"done: {ok} masks, {fail} failed, {skipped} skipped, {(time.time()-t_all)/60:.1f} min", flush=True)
sys.exit(1 if fail else 0)
