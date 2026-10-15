"""Turn out/bench/summary.csv + per_image.csv into out/bench/REPORT.md."""
import pandas as pd, numpy as np, os, json
S = pd.read_csv("out/bench/summary.csv", index_col=0)
P = pd.read_csv("out/bench/per_image.csv")
pair = pd.read_csv("out/bench/pairwise_iou.csv", index_col=0)

GROUP = {}
for m in S.index:
    base = m[:-4] if m.endswith("_enh") else m
    if base in ("baseline",): g = "previous pipeline (combined)"
    elif base == "samhq_box": g = "deep learning (HF port unreliable: masks span the whole crop)"
    elif base in ("owlv2_sam_full", "owlv2_sam", "birefnet_full", "birefnet_crop"): g = "previous pipeline components"
    elif base.startswith("sam") or base.startswith("rembg"): g = "deep learning"
    elif base.startswith("bg_"): g = "background-colour keying"
    else: g = "classical / mathematical"
    GROUP[m] = g

lines = ["# Beetle masking benchmark (10 hardest images)", "",
         "No ground truth exists, so every method is scored against the **leave-one-out majority consensus** of all other methods "
         "(the method itself is excluded from the vote). Metrics are computed at half resolution; the rod is stripped from every "
         "method's output by the same shared post-process, so the comparison is about the segmentation itself.", "",
         "Columns: `iou_cons` = mean IoU vs consensus, `iou_min` = worst image, `bf` = boundary F-score (3 px tolerance), "
         "`vote_in` = mean fraction of other methods that also mark this method's foreground pixels, `recall` = fraction of consensus "
         "foreground covered, `white` = fraction of mask pixels that are near-white (rod / wall leakage), `comps` = connected components, "
         "`area%` = mask area, `pair_iou` = mean pairwise IoU with every other method, `s/img` = seconds per image on the M3 Pro (method only, "
         "excluding the shared locator and post-process). `_enh` = run on the white-enhanced image.", "",
         "## Ranking", "",
         "| rank | method | group | iou_cons | iou_min | bf | vote_in | recall | white | comps | area% | pair_iou | s/img |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, (m, r) in enumerate(S.iterrows(), 1):
    lines.append(f"| {i} | `{m}` | {GROUP[m]} | {r.iou_consensus_mean:.3f} | {r.iou_consensus_min:.3f} | {r.bf_consensus_mean:.3f} | "
                 f"{r.mean_vote_inside:.3f} | {r.recall_of_consensus:.3f} | {r.white_frac:.3f} | {r.n_components:.1f} | {r.area_pct:.2f} | "
                 f"{r.mean_pair_iou:.3f} | {'' if np.isnan(r.sec_per_img) else f'{r.sec_per_img:.1f}'} |")

# enhancement effect
lines += ["", "## Effect of white enhancement (same method, plain vs `_enh`)", "", "| method | plain iou_cons | enh iou_cons | delta | plain bf | enh bf |", "|---|---|---|---|---|---|"]
for m in S.index:
    if m.endswith("_enh") or m + "_enh" not in S.index: continue
    a, b = S.loc[m], S.loc[m + "_enh"]
    lines.append(f"| `{m}` | {a.iou_consensus_mean:.3f} | {b.iou_consensus_mean:.3f} | {b.iou_consensus_mean - a.iou_consensus_mean:+.3f} | {a.bf_consensus_mean:.3f} | {b.bf_consensus_mean:.3f} |")

# per-image table for top methods
top = list(S.index[:12])
lines += ["", "## Per-image IoU vs consensus (top 12 methods)", "", "| image | " + " | ".join(f"`{m}`" for m in top) + " |", "|---|" + "---|" * len(top)]
for img, g in P.groupby("image"):
    g = g.set_index("method")
    lines.append(f"| {img} | " + " | ".join(f"{g.loc[m, 'iou_consensus']:.3f}" if m in g.index else "" for m in top) + " |")

lines += ["", "## Visual review", "", "Contact sheets per image (every method's mask drawn on the crop) are in `out/bench/sheets/<image>.jpg`; "
          "per-method cut-outs (beetle kept, rest white) are in `out/cutouts/<method>/<image>.jpg`.", ""]
open("out/bench/REPORT.md", "w").write("\n".join(lines))
print("\n".join(lines[:40]))
