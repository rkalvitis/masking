"""Colour-only beetle masking (white background keying). Same output format as mask_beetles.py."""
import argparse, glob, os, time
from multiprocessing import Pool
import numpy as np
from PIL import Image
from beetlemask import common as C
from beetlemask.colorkey import mask_by_color, mask_by_color_boxed

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--preview", default=None); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.images, e)))
os.makedirs(args.out, exist_ok=True)
if args.preview: os.makedirs(args.preview, exist_ok=True)

def work(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    outp = os.path.join(args.out, stem + ".png")
    if os.path.exists(outp) and not args.overwrite:
        return f"{stem}: skipped"
    t0 = time.time()
    try:
        rgb = C.load_rgb(p)
        m, info = mask_by_color_boxed(rgb)
        C.save_mask(m, outp)
        if args.preview and m.any():
            pb = C.pad_box(C.bbox(m), m.shape, pad_frac=0.25, min_pad=50)
            cut = rgb[pb[1]:pb[3], pb[0]:pb[2]].copy(); cut[~m[pb[1]:pb[3], pb[0]:pb[2]]] = 255
            Image.fromarray(cut).save(os.path.join(args.preview, stem + ".jpg"), quality=90)
        area = m.mean() * 100
        warn = " [WARNING: unusual mask area]" if (area < 0.3 or area > 2.0) else ""
        return f"{stem}: mask {area:.2f}% of frame, {time.time()-t0:.1f}s{warn}"
    except Exception as e:
        return f"{stem}: FAILED {e}"

if __name__ == "__main__":
    print(f"{len(paths)} images -> {args.out} (colour keying, {args.workers} workers)", flush=True)
    with Pool(args.workers) as pool:
        for i, line in enumerate(pool.imap(work, paths), 1):
            print(f"[{i}/{len(paths)}] {line}", flush=True)
    print("done", flush=True)
