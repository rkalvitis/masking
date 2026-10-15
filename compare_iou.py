"""Per-frame IoU agreement between mask sets (baseline vs SAM 2, baseline vs SAM 3, SAM 2 vs SAM 3).
Writes out/bench/iou_pairs.csv and out/bench/iou_pairs.png."""
import glob, os, sys
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

D = "/Users/robertskalvitis/Library/CloudStorage/OneDrive-USI/SUPSI/dataset_popillia_15_09_2026_FS"
SETS = {"baseline": f"{D}/masks", "SAM 2": f"{D}/masks_sam2", "SAM 3": f"{D}/masks_sam3", "colour": "out/color_masks_full"}
PAIRS = [("baseline", "SAM 2"), ("baseline", "SAM 3"), ("SAM 2", "SAM 3"), ("colour", "SAM 2"), ("colour", "SAM 3"), ("colour", "baseline")]
CHARTS = {"iou_pairs.png": ["baseline vs SAM 2", "baseline vs SAM 3", "SAM 2 vs SAM 3"],
          "iou_pairs_colour.png": ["colour vs SAM 2", "colour vs SAM 3", "colour vs baseline"]}
COLORS = {"baseline vs SAM 2": "#2a78d6", "baseline vs SAM 3": "#eb6834", "SAM 2 vs SAM 3": "#1baf7a",
          "colour vs SAM 2": "#2a78d6", "colour vs SAM 3": "#eb6834", "colour vs baseline": "#1baf7a"}  # validated palette slots 1-3, one chart per trio

def load(path):
    m = np.asarray(Image.open(path).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_NEAREST) > 0

names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{SETS['SAM 2']}/*.png"))
rows = []
for n in names:
    ms = {k: load(f"{v}/{n}.png") for k, v in SETS.items() if os.path.exists(f"{v}/{n}.png")}
    r = {"image": n, "set": n.split("_")[0], "view": n.split("_")[1]}
    for a, b in PAIRS:
        if a in ms and b in ms:
            r[f"{a} vs {b}"] = (ms[a] & ms[b]).sum() / max(1, (ms[a] | ms[b]).sum())
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("out/bench/iou_pairs.csv", index=False)
cols = [f"{a} vs {b}" for a, b in PAIRS if f"{a} vs {b}" in df.columns]
summ = df[cols].describe(percentiles=[.05, .25, .5, .75]).T[["mean", "5%", "25%", "50%", "75%", "min"]]
summ["frames < 0.90"] = [(df[c] < 0.90).sum() for c in cols]
print(summ.round(3).to_string())
print("\nper view (median):"); print(df.groupby("view")[cols].median().round(3).to_string())

for fname, ccols in CHARTS.items():
    ccols = [c for c in ccols if c in df.columns]
    if not ccols:
        continue
    # ---- chart: per-frame IoU (dots, ordered by frame) + distribution (boxes)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
    fig = plt.figure(figsize=(14, 7.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0]); ax.set_facecolor("#fcfcfb")
    x = np.arange(len(df))
    for c in ccols:
        ax.plot(x, df[c], ".", ms=4, color=COLORS[c], alpha=0.85, label=c)
    ax.set_ylim(0.5, 1.005); ax.set_xlim(-2, len(df) + 1)
    ax.set_ylabel("IoU between the two mask sets"); ax.set_xlabel("frame (ordered by set s00…s35, view p0…p8)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # set boundaries every 9 frames as faint ticks
    ax.set_xticks(x[::9]); ax.set_xticklabels([df.image[i].split("_")[0] for i in x[::9]], rotation=90, fontsize=8)
    ax.legend(loc="lower left", frameon=False, ncol=3, markerscale=2.5)
    ax.set_title("How much the mask sets agree, per frame (324 frames)", loc="left", fontsize=12, color="#0b0b0b")
    ax2 = fig.add_subplot(gs[1], sharey=ax); ax2.set_facecolor("#fcfcfb")
    bp = ax2.boxplot([df[c].dropna() for c in ccols], widths=0.55, patch_artist=True, showfliers=True,
                     flierprops=dict(marker=".", ms=3, markeredgecolor="#9a9891"), medianprops=dict(color="#0b0b0b", lw=1.5))
    for patch, c in zip(bp["boxes"], ccols):
        patch.set_facecolor(COLORS[c]); patch.set_alpha(0.55); patch.set_edgecolor(COLORS[c])
    ax2.set_xticks([1, 2, 3]); ax2.set_xticklabels([c.replace(" vs ", "\nvs ") + f"\nmedian {df[c].median():.3f}" for c in ccols], fontsize=8)
    ax2.tick_params(labelleft=False); ax2.grid(axis="y", color="#e6e5e0", lw=0.8); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    ax2.set_title("distribution", loc="left", fontsize=10, color="#52514e")
    fig.savefig(f"out/bench/{fname}", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"chart: out/bench/{fname}")
