"""Agreement benchmark for beetle masks.

No ground truth: every method is scored against the leave-one-out majority consensus of the
other methods, plus pairwise IoU, boundary F-score, and sanity metrics. Also writes contact
sheets (per image: every method's mask on a crop) for visual review.
"""
import argparse, glob, json, os
import numpy as np, cv2, pandas as pd
from scipy import ndimage as ndi
from beetlemask import common as C, viz

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images")
ap.add_argument("--masks", default="out/masks")
ap.add_argument("--baseline", default="data/masks_baseline")
ap.add_argument("--out", default="out/bench")
ap.add_argument("--exclude", default="_coarse", help="comma list of method dirs to skip")
ap.add_argument("--no-consensus", default="baseline,_coarse,ensemble", help="methods not used to build the consensus")
ap.add_argument("--down", type=int, default=2, help="metric resolution divisor")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)

names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.images, "*.tif")))
excl = set(args.exclude.split(","))
methods = sorted(d for d in os.listdir(args.masks) if os.path.isdir(os.path.join(args.masks, d)) and d not in excl)
if args.baseline and os.path.isdir(args.baseline):
    methods = ["baseline"] + methods
no_cons = set(args.no_consensus.split(","))
cons_methods = [m for m in methods if m not in no_cons]
print("methods:", methods)
print("consensus from:", cons_methods)


def load(method, name):
    p = os.path.join(args.baseline if method == "baseline" else os.path.join(args.masks, method), name + ".png")
    if not os.path.exists(p):
        return None
    m = C.load_mask(p)
    if args.down > 1:
        m = cv2.resize(m.astype(np.uint8), (m.shape[1] // args.down, m.shape[0] // args.down), interpolation=cv2.INTER_NEAREST) > 0
    return m


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def boundary_f(a, b, tol=3):
    """Boundary F-score: precision/recall of contour pixels within tol px."""
    def contour(m):
        return m & ~ndi.binary_erosion(m)
    ca, cb = contour(a), contour(b)
    if ca.sum() == 0 or cb.sum() == 0:
        return 0.0
    da = ndi.binary_dilation(ca, iterations=tol)
    db = ndi.binary_dilation(cb, iterations=tol)
    p = (ca & db).sum() / ca.sum()
    r = (cb & da).sum() / cb.sum()
    return 2 * p * r / (p + r) if p + r else 0.0


rows = []
pair = {m: {n: [] for n in methods} for m in methods}
for name in names:
    rgb = C.load_rgb(os.path.join(args.images, name + ".tif"))
    small = cv2.resize(rgb, (rgb.shape[1] // args.down, rgb.shape[0] // args.down), interpolation=cv2.INTER_AREA)
    masks = {m: load(m, name) for m in methods}
    masks = {m: v for m, v in masks.items() if v is not None}
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]; chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    white = (L > 200) & (chroma < 12)
    stack = {m: masks[m] for m in cons_methods if m in masks}
    votes = np.sum([v for v in stack.values()], axis=0)
    for m, mk in masks.items():
        # leave-one-out consensus
        if m in stack:
            v = votes - mk.astype(int); k = len(stack) - 1
        else:
            v = votes; k = len(stack)
        cons = v > k / 2
        soft = v / max(1, k)
        area = mk.sum()
        lab_c, ncomp = ndi.label(mk)
        rows.append(dict(
            image=name, method=m,
            iou_consensus=iou(mk, cons),
            bf_consensus=boundary_f(mk, cons),
            mean_vote_inside=float(soft[mk].mean()) if area else 0.0,       # how many others agree with my fg pixels
            recall_of_consensus=float((mk & cons).sum() / max(1, cons.sum())),
            area_pct=100 * area / mk.size,
            n_components=int(ncomp),
            white_frac=float(white[mk].mean()) if area else 0.0,           # rod / wall leakage indicator
            empty=int(area < 50),
        ))
        for n, mn in masks.items():
            pair[m][n].append(iou(mk, mn))
    # contact sheet: crop around consensus + all masks
    cons_all = votes > len(stack) / 2
    b, crop, cms = viz.crop_to(small, [cons_all if cons_all.any() else masks[m] for m in masks], pad=0.3)
    cms = [masks[m][b[1]:b[3], b[0]:b[2]] for m in masks]
    tiles = [viz.label(crop, name)]
    tiles.append(viz.label(viz.overlay(crop, cons_all[b[1]:b[3], b[0]:b[2]], (255, 200, 0)), "consensus"))
    for m, cm in zip(masks, cms):
        r = [x for x in rows if x["image"] == name and x["method"] == m][0]
        tiles.append(viz.label(viz.overlay(crop, cm, (255, 0, 0)), f"{m} {r['iou_consensus']:.2f}"))
    from PIL import Image
    Image.fromarray(viz.grid(tiles, 6, 420)).save(os.path.join(args.out, "sheets", name + ".jpg"), quality=85)
    print(name, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out, "per_image.csv"), index=False)
timing = json.load(open(os.path.join(args.masks, "timing.json"))) if os.path.exists(os.path.join(args.masks, "timing.json")) else {}
summ = df.groupby("method").agg(
    iou_consensus_mean=("iou_consensus", "mean"), iou_consensus_min=("iou_consensus", "min"),
    bf_consensus_mean=("bf_consensus", "mean"), mean_vote_inside=("mean_vote_inside", "mean"),
    recall_of_consensus=("recall_of_consensus", "mean"), white_frac=("white_frac", "mean"),
    n_components=("n_components", "mean"), empty=("empty", "sum"), area_pct=("area_pct", "mean"))
summ["mean_pair_iou"] = [np.mean([np.mean(pair[m][n]) for n in methods if n != m and pair[m][n]]) for m in summ.index]
summ["sec_per_img"] = [np.mean(list(timing[m].values())) if m in timing else np.nan for m in summ.index]
summ = summ.sort_values("iou_consensus_mean", ascending=False)
summ.to_csv(os.path.join(args.out, "summary.csv"))
pd.DataFrame({m: {n: np.mean(pair[m][n]) if pair[m][n] else np.nan for n in methods} for m in methods}).to_csv(os.path.join(args.out, "pairwise_iou.csv"))
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(summ.round(3).to_string())
