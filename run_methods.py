"""Run every masking method on data/images and save masks to out/masks/<method>/<name>.png."""
import argparse, glob, json, os, time, traceback
import numpy as np
from PIL import Image
from beetlemask import common as C
from beetlemask.locate import locate_classical
from beetlemask import classical

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images")
ap.add_argument("--out", default="out/masks_raw")
ap.add_argument("--methods", default="all", help="comma list or 'all' / 'classical' / 'deep'")
ap.add_argument("--only", default=None, help="comma list of image stems")
ap.add_argument("--enhance", action="store_true", help="pre-enhance (white background -> pure white, contrast stretch); methods saved with _enh suffix")
args = ap.parse_args()

from beetlemask import bgkey
methods = {}
if args.methods in ("all", "classical"):
    methods.update(classical.METHODS)
if args.methods in ("all", "bgkey"):
    methods.update(bgkey.METHODS)
if args.methods in ("all", "deep"):
    from beetlemask import deep
    methods.update(deep.METHODS)
if args.methods not in ("all", "classical", "deep", "bgkey"):
    from beetlemask import deep
    groups = {"classical": classical.METHODS, "bgkey": bgkey.METHODS, "deep": deep.METHODS}
    allm = {**classical.METHODS, **bgkey.METHODS, **deep.METHODS}
    for k in args.methods.split(","):
        methods.update(groups[k] if k in groups else {k: allm[k]})

paths = sorted(glob.glob(os.path.join(args.images, "*.tif")))
if args.only:
    keep = set(args.only.split(","))
    paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in keep]

timing = {}
tpath = os.path.join(args.out, "timing.json")
if os.path.exists(tpath):
    timing = json.load(open(tpath))

for p in paths:
    name = os.path.splitext(os.path.basename(p))[0]
    rgb = C.load_rgb(p)
    if args.enhance:
        from beetlemask.enhance import enhance
        rgb = enhance(rgb)
    t0 = time.time()
    box, coarse = locate_classical(rgb)
    ctx = {"box": box, "coarse_mask": coarse, "name": name}
    C.save_mask(coarse, os.path.join(args.out, "_coarse" + ("_enh" if args.enhance else ""), name + ".png"))
    print(f"{name}: coarse box {box} ({time.time()-t0:.1f}s)", flush=True)
    for mname0, fn in methods.items():
        mname = mname0 + ("_enh" if args.enhance else "")
        t0 = time.time()
        try:
            m_raw = fn(rgb, ctx)
            dt = time.time() - t0
            C.save_mask(m_raw, os.path.join(args.out, mname, name + ".png"))
            timing.setdefault(mname, {})[name] = dt
            print(f"  {mname:16s} raw area {m_raw.mean()*100:5.2f}%  {dt:5.1f}s", flush=True)
        except Exception as e:
            print(f"  {mname:16s} FAILED: {e}", flush=True)
            traceback.print_exc()
    json.dump(timing, open(tpath, "w"), indent=1)
