# Beetle masking (Popillia japonica on a white box, mounting rod excluded)

Produces binary masks (8-bit PNG, 255 = beetle, same size and stem as the image) for Masked-3DGS.

## Chosen pipeline: `mask_beetles.py`

1. **Locate** (`beetlemask/locate.py`): hysteresis threshold of a darkness/chroma score at 1/4 res.
   A 15x15 opening separates the beetle body core from the rod and legs; cores touching the frame
   border are clutter (floor, cardboard, rod running out of frame, shaded wall) and are dropped; among
   the rest the largest core that contains genuinely dark pixels wins (a dim wall wedge cannot).
   The blob is the hysteresis component of that core, plus nearby non-border fragments smaller than
   it (a leg hidden behind the rod). The rod is stripped from the blob -> coarse blob + box.
2. **Segment** (`beetlemask/deep.py: sam2_box`): SAM 2.1 hiera-large with the box prompt on a
   square crop around the beetle (so the beetle fills the model's 1024 px input instead of being
   ~100 px wide in the full frame). The SAM candidate closest to the coarse blob is taken.
3. **Light cleanup only** (inside `sam2_box`): main blob plus fragments within one rod width
   (a leg hidden behind the rod), tiny holes filled. SAM 2's box prompt never contains the rod, so
   the raw mask already excludes it. The heavy rod-strip/hole post-process in
   `beetlemask/common.py: strip_bright_rim` (used for the benchmark of the other methods) damages
   SAM 2 masks (it misreads the shiny grey elytra side as rod), so it is off by default
   (`--postprocess full` re-enables it).

```bash
conda activate masking     # torch + transformers + opencv + scikit-image (see below)
python mask_beetles.py --images <dir> --out <mask dir> [--preview <dir>] [--prompt box|points] [--device cuda]
```
About 4 s/image on an M3 Pro (MPS); the first image also downloads/loads the model. Re-running skips
existing masks unless `--overwrite`. If SAM returns (almost) nothing, the script falls back to the
coarse blob and says so in the log.

Environment: `conda create -n masking python=3.11 && pip install torch torchvision transformers timm kornia
einops opencv-python-headless scikit-image scikit-learn scipy tifffile huggingface_hub rembg onnxruntime
accelerate matplotlib pandas`.

## Benchmark that led here

`run_methods.py` (all candidate methods, raw masks to `out/masks_raw`), `postprocess.py` (shared
post-process to `out/masks` + cut-outs), `benchmark.py` (agreement metrics, no ground truth: each
method vs the leave-one-out majority consensus of the others; contact sheets in `out/bench/sheets`),
`write_report.py` (`out/bench/REPORT.md`). Candidates: classical (`beetlemask/classical.py`),
background-colour keying (`beetlemask/bgkey.py`), deep models incl. the previous OWLv2+SAM+BiRefNet
pipeline's parts as separate entries (`beetlemask/deep.py`), each optionally on a white-enhanced
image (`beetlemask/enhance.py`, `--enhance`).

Findings on the 10 hardest images: SAM 2.1 (box, crop) and IS-Net were the most consistent
(IoU vs consensus ~0.98); crop-based colour keying (Otsu on a darkness/chroma score, Chan-Vese)
was within 0.01 at 1/50 of the cost; full-frame SAM/BiRefNet and the previous combined pipeline were
the weakest (rod included, tiny beetle at 1024 px); HQ-SAM through transformers is unusable; SAM 3
is gated (accept licence at huggingface.co/facebook/sam3 + `hf auth login`, then `sam3_text` works).
White enhancement did not help SAM 2 or the keying methods.

## Result on the full dataset (2026-09-23)

`masks_sam2/` next to `stacked/`: 324 masks, no failures, mask area 0.54-1.99 % of frame. The
script prints a WARNING for masks outside 0.3-2 % so locator failures are visible in the log.
Validation sheets: `out/bench/compare_baseline_sam2_sam3_*.jpg`, `out/bench/final_random36.jpg`.

## SAM 3 masks and agreement report (2026-09-24)

`masks_sam3/` (SAM 3, text prompt "beetle", `mask_beetles.py --model sam3`) next to `masks_sam2/`.
`compare_iou.py` -> `out/bench/iou_pairs.csv` / `.png` (per-frame IoU: baseline vs SAM 2, baseline vs
SAM 3, SAM 2 vs SAM 3); `make_iou_report.py` -> `out/bench/iou_pairs.xlsx` (red < 0.90, orange < 0.95,
thresholds editable) and `out/bench/compare_big_disagreement_overlay.jpg` (frames with IoU < 0.90).
Production cleanup also drops a large near-white rod chunk touching the mask edge plus its grey rim
(`strip_large_white`); it changed exactly one SAM 2 frame (s03_p8) and no SAM 3 frame.

## Ground-truth benchmark on DIS5K insects (`gtbench/`, 2026-10-15)

The agreement benchmark above has no ground truth. `DIS5K_insect/` holds the 196 images of the
DIS5K "Insect" group (ant, beetle, bee, butterfly, caterpillar, centipede, dragonfly, locust,
mantis, mosquito, spider; one snake) with pixel-accurate hand masks that follow legs and antennae
(`images/`, `masks/`, `index.csv`; full dataset in `DIS5K_full/`, zip `DIS5K.zip`). These are
natural-background photos, so the white-box locator, the colour keying methods and the rod
post-process are NOT part of this benchmark.

- `gtbench/methods.py`: the candidates, largest public variants, raw output, no cleanup:
  SAM ViT-H (box, box+point), SAM 2.1 hiera-large (box, box+point), HQ-SAM ViT-H (`segment-anything-hq`,
  `lkeab/hq-sam`), HQ-SAM 2 hiera-large (SysCV `sam-hq2` package, `sam2.1_hq_hiera_large.pt`, 2024-11 beta),
  SAM 3 (box; text "insect"; gated weights -> `hf auth login`), BiRefNet (1024) and
  BiRefNet_HR (2048), rembg BiRefNet / IS-Net / U2-Net, GrabCut as the classical floor.
  Note IS-Net was trained on DIS5K, so its score is optimistic.
- `gtbench/run.py`: crops a square around the padded GT box (the Popillia crop rule, 60 % padding,
  min 96 px) and runs every method on that crop with the GT box as the prompt; GT is used for nothing
  else. Writes `out/gtbench/masks/<method>/<stem>.png`, `crops/`, `crops.json`, `timing.json`, `run.log`.
  `--weights DIR` stores all model caches under DIR (HF_HOME, U2NET_HOME, TORCH_HOME). Output layout:
  one folder per method (`<out>/<method>/{masks,run.log,timing.json,score.csv}`), the shared inputs in
  `<out>/_inputs/`, the report in `<out>/_report/`.
- `gtbench/score.py`: **methods are ranked by mean IoU over all images** (the GT mask vs the method
  mask, averaged over the 196 images); boundary F-score, precision/recall, thin-structure recall, MAE,
  per-category means, hardest images and contact sheets are reported alongside -> `out/gtbench/REPORT.md`.
  Each mask is scored raw and with the light production cleanup (`+light`).
- Server (rhea) layout: code `/home/robertsk/masking` (git pull or `sync_to_rhea.sh code`), container
  `/home/robertsk/containers/gtbench.sif`, bench dir `/media/white/nanodrones/roberts.kalvitis/mask_bench`
  with `gt/{images,masks,index.csv}` (= `DIS5K_insect/`, `sync_to_rhea.sh data`), `results/<method>/`
  and `weights/`. `gtbench/gtbench.def` is the Singularity container (CUDA 12.4, python 3.11, torch 2.6,
  transformers 5, rembg GPU, segment-anything-hq; the lab's 3dgs containers are too old);
  `python -m gtbench.selfcheck` (run once with `singularity exec --nv`) verifies CUDA and the model imports,
  which `singularity build` cannot test because it has no GPU.
  `bash gtbench/sync_to_rhea.sh cmd` prints the build and run commands, `... results` pulls
  `results/` back to `out/gtbench_rhea/`. `gtbench/server_setup.sh` is the conda alternative.

```bash
# on the server, inside the container (/workspace = code, /bench = mask_bench):
python -m gtbench.run --data /bench/gt --out /bench/results --weights /bench/weights --device cuda
python -m gtbench.score --data /bench/gt --out /bench/results      # -> /bench/results/_report/REPORT.md
```
