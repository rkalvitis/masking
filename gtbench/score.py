"""Score benchmark masks against ground truth and write the report.

    python -m gtbench.score --data DIS5K_insect --out out/gtbench [--methods ...] [--sheets N]

Metrics per image and method (full resolution):
  iou        intersection over union with the GT mask
  bf3        boundary F-score, contour pixels matched within 3 px (how well edges/legs are traced)
  prec/rec   pixel precision / recall (rec < 1 = parts of the insect missed, prec < 1 = background included)
  thin_rec   recall restricted to THIN GT structures (GT pixels less than 6 px from the GT boundary on both
             sides = legs, antennae, wing edges): the metric that separates fine-structure quality
  mae        mean absolute error (fraction of crop pixels wrong), computed inside the crop
Every mask is scored twice: RAW (exactly what the model returned) and LIGHT (the Popillia production
cleanup, no colour logic: keep the largest component plus components within a link radius of 2 % of the
crop side, fill holes smaller than 0.2 % of the mask). Method names get the suffix `+light`.
Reads <out>/<method>/masks/ and <out>/_inputs/crops.json (layout of gtbench.run).
Outputs: <out>/results.json (the ranking: per method mean IoU and the other metrics, raw and +light, plus
per-category means and run metadata; this is the file to read), <out>/_report/{REPORT.md, summary.csv,
per_image.csv, per_category.csv, sheets/<stem>.jpg} and <out>/<method>/score.csv (that method's per-image metrics).
"""
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np, cv2
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="DIS5K_insect"); ap.add_argument("--out", default="out/gtbench")
ap.add_argument("--methods", default=None, help="comma list (default: every folder in <out>/masks)")
ap.add_argument("--sheets", type=int, default=1000, help="how many contact sheets to write (0 = none)")
ap.add_argument("--tile", type=int, default=360)
args = ap.parse_args()

methods = args.methods.split(",") if args.methods else sorted(d for d in os.listdir(args.out) if not d.startswith("_") and os.path.isdir(os.path.join(args.out, d, "masks")))
INP = os.path.join(args.out, "_inputs"); REP = os.path.join(args.out, "_report")
crops = json.load(open(os.path.join(INP, "crops.json")))
timing = {}
for m in methods:
    tp = os.path.join(args.out, m, "timing.json")
    if os.path.exists(tp): timing[m] = json.load(open(tp))
index = {}
if os.path.exists(os.path.join(args.data, "index.csv")):
    for r in csv.DictReader(open(os.path.join(args.data, "index.csv"))):
        index[r["stem"]] = r


def contour(m):
    return m & ~ndi.binary_erosion(m, border_value=0)


def boundary_f(a, b, tol=3):
    ca, cb = contour(a), contour(b)
    if ca.sum() == 0 or cb.sum() == 0:
        return 0.0
    da = ndi.binary_dilation(ca, iterations=tol); db = ndi.binary_dilation(cb, iterations=tol)
    p = (ca & db).sum() / ca.sum(); r = (cb & da).sum() / cb.sum()
    return 2 * p * r / (p + r) if p + r else 0.0


def thin_region(gt, width=6):
    """GT pixels that belong to structures thinner than ~2*width px (distance to background < width)."""
    dt = ndi.distance_transform_edt(gt)
    return gt & (dt < width)


def light_clean(m, link_radius, min_frac=0.002, hole_frac=0.002):
    """Largest component + components within link_radius px of it (not smaller than min_frac of it), tiny holes filled."""
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1)); main = int(np.argmax(sizes)) + 1
    out = lab == main
    if n > 1:
        near = ndi.distance_transform_edt(~out) <= link_radius
        for i in range(1, n + 1):
            if i != main and sizes[i - 1] >= min_frac * sizes[main - 1] and np.any((lab == i) & near):
                out |= lab == i
    filled = ndi.binary_fill_holes(out); holes = filled & ~out
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, index=np.arange(1, hn + 1)); area = out.sum()
        for i in range(1, hn + 1):
            if hs[i - 1] <= hole_frac * area:
                out[hl == i] = True
    return out


def overlay(rgb, m, color, alpha=0.35, thick=2):
    out = rgb.copy(); mb = m.astype(bool)
    out[mb] = (out[mb] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thick); return out


def label(img, text):
    img = img.copy(); s = max(0.45, img.shape[0] / 800)
    cv2.rectangle(img, (0, 0), (int(len(text) * 15 * s) + 8, int(28 * s)), (0, 0, 0), -1)
    cv2.putText(img, text, (4, int(21 * s)), cv2.FONT_HERSHEY_SIMPLEX, s * 0.8, (255, 255, 255), max(1, int(2 * s))); return img


rows = []
stems = sorted(crops)
os.makedirs(os.path.join(REP, "sheets"), exist_ok=True)
for k, stem in enumerate(stems):
    c = crops[stem]; x0, y0, x1, y1 = c["crop"]
    gt_full = cv2.imread(os.path.join(args.data, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE) > 127
    gt = gt_full[y0:y1, x0:x1]
    thin = thin_region(gt)
    rgb = None
    tiles = []
    for m in methods:
        mp = os.path.join(args.out, m, "masks", stem + ".png")
        if not os.path.exists(mp):
            continue
        pm = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127)[y0:y1, x0:x1]
        link = max(10, int(0.02 * max(gt.shape)))
        for suffix, mm in (("", pm), ("+light", light_clean(pm, link))):
            inter = (mm & gt).sum(); union = (mm | gt).sum()
            rows.append(dict(stem=stem, category=index.get(stem, {}).get("category", stem.split("_")[0]), method=m + suffix,
                             iou=inter / max(1, union), bf3=boundary_f(mm, gt),
                             prec=inter / max(1, mm.sum()), rec=inter / max(1, gt.sum()),
                             thin_rec=(mm & thin).sum() / max(1, thin.sum()), mae=(mm != gt).mean(),
                             area_ratio=mm.sum() / max(1, gt.sum()), seconds=timing.get(m, {}).get(stem, "")))
        if k < args.sheets:
            if rgb is None:
                rgb = cv2.cvtColor(cv2.imread(os.path.join(INP, "crops", stem + ".jpg")), cv2.COLOR_BGR2RGB)
                tiles.append(label(cv2.resize(overlay(rgb, gt, (0, 200, 0)), (args.tile, args.tile), interpolation=cv2.INTER_AREA), f"{stem[:28]} GT"))
            r = rows[-2]
            tiles.append(label(cv2.resize(overlay(rgb, pm, (255, 0, 0)), (args.tile, args.tile), interpolation=cv2.INTER_AREA), f"{m} iou {r['iou']:.3f} bf {r['bf3']:.2f}"))
    if tiles:
        ncol = 5
        while len(tiles) % ncol: tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)], 0)
        cv2.imwrite(os.path.join(REP, "sheets", stem + ".jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if k % 20 == 0:
        print(f"scored {k+1}/{len(stems)}", flush=True)

keys = ["iou", "bf3", "prec", "rec", "thin_rec", "mae"]
with open(os.path.join(REP, "per_image.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for m in methods:
    mine = [r for r in rows if r["method"] in (m, m + "+light")]
    if mine:
        with open(os.path.join(args.out, m, "score.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mine[0])); w.writeheader(); w.writerows(mine)

by_m = defaultdict(list)
for r in rows: by_m[r["method"]].append(r)
summary = []
for m, rs in by_m.items():
    s = dict(method=m, n=len(rs))
    for kk in keys: s[kk] = float(np.mean([r[kk] for r in rs]))
    s["iou_min"] = float(min(r["iou"] for r in rs)); s["iou_p10"] = float(np.percentile([r["iou"] for r in rs], 10))
    s["fail_iou<0.5"] = int(sum(r["iou"] < 0.5 for r in rs))
    secs = [r["seconds"] for r in rs if r["seconds"] != ""]; s["s_per_img"] = float(np.median(secs)) if secs else ""
    summary.append(s)
summary.sort(key=lambda s: -s["iou"])
with open(os.path.join(REP, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

by_cm = defaultdict(list)
for r in rows: by_cm[(r["category"], r["method"])].append(r["iou"])
cats = sorted({r["category"] for r in rows})
with open(os.path.join(REP, "per_category.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["category", "n"] + [s["method"] for s in summary])
    for c in cats:
        n = len(by_cm[(c, summary[0]["method"])]) if (c, summary[0]["method"]) in by_cm else 0
        w.writerow([c, n] + [f"{np.mean(by_cm[(c, s['method'])]):.3f}" if (c, s["method"]) in by_cm else "" for s in summary])

with open(os.path.join(REP, "REPORT.md"), "w") as f:
    f.write(f"# Ground-truth benchmark: {args.data} ({len(stems)} images)\n\n")
    f.write("Every method got the same square crop around the padded GT box (the Popillia crop rule) and, where it takes a prompt, the GT box in crop coordinates. Raw model output, no post-processing. **Ranking = mean IoU over all images** (GT vs method mask, full resolution inside the crop); the other columns are means too.\n\n")
    f.write("`iou` mean IoU, `bf3` boundary F-score at 3 px, `prec`/`rec` pixel precision/recall, `thin_rec` recall on thin GT structures (<6 px from the GT boundary: legs, antennae), `mae` fraction of crop pixels wrong, `iou_min`/`iou_p10` worst and 10th-percentile image, `fail` images with IoU < 0.5, `s/img` median seconds.\n\n")
    for title, sel in (("Raw model output", [s for s in summary if not s["method"].endswith("+light")]),
                       ("With the light production cleanup (largest blob + linked fragments, tiny holes filled)", [s for s in summary if s["method"].endswith("+light")])):
        f.write(f"## {title}\n\n| rank | method | n | iou | bf3 | prec | rec | thin_rec | mae | iou_min | iou_p10 | fail | s/img |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, s in enumerate(sel, 1):
            sec = f"{s['s_per_img']:.1f}" if s["s_per_img"] != "" else ""
            f.write(f"| {i} | `{s['method']}` | {s['n']} | {s['iou']:.3f} | {s['bf3']:.3f} | {s['prec']:.3f} | {s['rec']:.3f} | {s['thin_rec']:.3f} | {s['mae']:.4f} | {s['iou_min']:.3f} | {s['iou_p10']:.3f} | {s['fail_iou<0.5']} | {sec} |\n")
        f.write("\n")
    summary_all = summary
    summary = [s for s in summary if not s["method"].endswith("+light")]   # category table: raw only
    f.write("\n## Mean IoU per category (raw)\n\n| category | n | " + " | ".join(f"`{s['method']}`" for s in summary) + " |\n|---|---|" + "---|" * len(summary) + "\n")
    for c in cats:
        n = len(by_cm[(c, summary[0]["method"])]) if (c, summary[0]["method"]) in by_cm else 0
        f.write(f"| {c} | {n} | " + " | ".join(f"{np.mean(by_cm[(c, s['method'])]):.3f}" if (c, s["method"]) in by_cm else "" for s in summary) + " |\n")
    f.write("\n## Hardest images (lowest mean IoU over methods)\n\n")
    by_s = defaultdict(list)
    for r in rows:
        if not r["method"].endswith("+light"): by_s[r["stem"]].append(r["iou"])
    for stem, v in sorted(by_s.items(), key=lambda kv: np.mean(kv[1]))[:15]:
        f.write(f"- `{stem}`: mean IoU {np.mean(v):.3f}, best {max(v):.3f} (`sheets/{stem}.jpg`)\n")
    f.write("\nContact sheets: `sheets/<stem>.jpg` (green = GT, red = method). Per-image numbers: `per_image.csv`.\n")
import datetime, platform
res = dict(
    benchmark="gtbench", data=os.path.abspath(args.data), results_dir=os.path.abspath(args.out),
    scored_at=datetime.datetime.now().isoformat(timespec="seconds"), host=platform.node(),
    n_images=len(stems), ranking_metric="mean IoU over all images (raw model output)",
    ranking=[s["method"] for s in summary],
    methods={s["method"]: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items() if k != "method"} for s in summary},
    methods_light={s["method"]: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items() if k != "method"} for s in summary_all if s["method"].endswith("+light")},
    per_category_iou={c: {s["method"]: round(float(np.mean(by_cm[(c, s["method"])])), 5) for s in summary if (c, s["method"]) in by_cm} for c in cats},
    metrics=dict(iou="mean IoU vs GT", bf3="boundary F-score, 3 px", prec="pixel precision", rec="pixel recall",
                 thin_rec="recall on GT pixels < 6 px from the GT boundary (legs, antennae)", mae="fraction of crop pixels wrong",
                 iou_min="worst image", iou_p10="10th percentile image", s_per_img="median seconds per image"),
)
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
print("wrote", os.path.join(args.out, "results.json"))
print(open(os.path.join(REP, "REPORT.md")).read()[:3000])
