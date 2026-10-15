"""Apply the shared post-process (rod strip, wall holes, cleanup) to raw masks and write
out/masks/<method>/<name>.png plus cut-out previews in out/cutouts/<method>/<name>.jpg.
Runs one worker per image (each worker loads its image once and processes every method)."""
import argparse, glob, os, json
from multiprocessing import Pool
import numpy as np
from PIL import Image
from beetlemask import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images")
ap.add_argument("--raw", default="out/masks_raw")
ap.add_argument("--out", default="out/masks")
ap.add_argument("--methods", default=None, help="comma list; default all dirs in --raw")
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

methods = args.methods.split(",") if args.methods else sorted(d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)))
methods = [m for m in methods if not m.startswith("_")]
names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.images, "*.tif")))


def work(name):
    rgb = C.load_rgb(os.path.join(args.images, name + ".tif"))
    rgb_enh = None
    stats, lines = {}, []
    for mname in methods:
        p = os.path.join(args.raw, mname, name + ".png")
        if not os.path.exists(p):
            continue
        if mname.endswith("_enh") and rgb_enh is None:
            from beetlemask.enhance import enhance
            rgb_enh = enhance(rgb)
        proc_rgb = rgb_enh if mname.endswith("_enh") else rgb
        raw = C.load_mask(p)
        m = C.strip_bright_rim(raw, proc_rgb)
        C.save_mask(m, os.path.join(args.out, mname, name + ".png"))
        bb = C.bbox(m) or C.bbox(raw) or (0, 0, 100, 100)
        pb = C.pad_box(bb, m.shape, pad_frac=0.25, min_pad=50)
        cut = rgb[pb[1]:pb[3], pb[0]:pb[2]].copy(); cut[~m[pb[1]:pb[3], pb[0]:pb[2]]] = 255
        os.makedirs(os.path.join("out/cutouts", mname), exist_ok=True)
        Image.fromarray(cut).save(os.path.join("out/cutouts", mname, name + ".jpg"), quality=90)
        stats[mname] = dict(raw=int(raw.sum()), final=int(m.sum()))
        lines.append(f"{name} {mname:20s} raw {raw.mean()*100:5.2f}% -> {m.mean()*100:5.2f}%  (removed {(raw.sum()-m.sum())/max(1,raw.sum())*100:5.1f}%)")
    return name, stats, lines


if __name__ == "__main__":
    for m in methods:
        os.makedirs(os.path.join(args.out, m), exist_ok=True)
    all_stats = {}
    with Pool(args.workers) as pool:
        for name, stats, lines in pool.imap_unordered(work, names):
            print("\n".join(lines), flush=True)
            for m, s in stats.items():
                all_stats.setdefault(m, {})[name] = s
    json.dump(all_stats, open(os.path.join(args.out, "postprocess_stats.json"), "w"), indent=1)
    if os.path.exists(os.path.join(args.raw, "timing.json")):
        import shutil; shutil.copy(os.path.join(args.raw, "timing.json"), os.path.join(args.out, "timing.json"))
