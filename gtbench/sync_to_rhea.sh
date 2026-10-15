#!/bin/bash
# Mac side helpers for the benchmark on rhea.
#   bash gtbench/sync_to_rhea.sh data      # DIS5K_insect/{images,masks,index.csv} -> $BENCH/gt/
#   bash gtbench/sync_to_rhea.sh code      # gtbench/ + README -> $SRV/   (or just git push + git pull)
#   bash gtbench/sync_to_rhea.sh results   # $BENCH/results/ -> out/gtbench_rhea/
#   bash gtbench/sync_to_rhea.sh cmd       # print the build + run commands for the server
# Server layout (2026-10-15):
#   $BENCH = /media/white/nanodrones/roberts.kalvitis/mask_bench
#       gt/images, gt/masks   the DIS5K insect set      results/<method>/   one folder per method
#       weights/              model caches              container: $CONT, code: $SRV
set -eu
HOST=${HOST:-rhea}
SRV=${SRV:-/home/robertsk/masking}
BENCH=${BENCH:-/media/white/nanodrones/roberts.kalvitis/mask_bench}
CONT=${CONT:-/home/robertsk/containers/gtbench.sif}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN="singularity exec --nv --cleanenv --contain --bind $SRV:/workspace --bind $BENCH:/bench $CONT bash -c 'cd /workspace && python -m gtbench.run --data /bench/gt --out /bench/results --weights /bench/weights --device cuda && python -m gtbench.score --data /bench/gt --out /bench/results'"
case "${1:-}" in
  data)
    rsync -av --progress "$HERE/DIS5K_insect/" "$HOST:$BENCH/gt/"
    echo "== dataset in $HOST:$BENCH/gt/ (images/, masks/, index.csv)";;
  code)
    rsync -av --exclude __pycache__ "$HERE/gtbench" "$HERE/README.md" "$HOST:$SRV/"
    echo "== code in $HOST:$SRV/gtbench";;
  results)
    mkdir -p "$HERE/out/gtbench_rhea"
    rsync -av "$HOST:$BENCH/results/" "$HERE/out/gtbench_rhea/"
    echo "== results in out/gtbench_rhea/ (_report/REPORT.md, <method>/score.csv)";;
  cmd)
    echo "# build once:"; echo "singularity build --fakeroot $CONT $SRV/gtbench/gtbench.def"
    echo "# SAM 3 (gated): accept the licence on huggingface.co/facebook/sam3, then once:"
    echo "singularity exec --bind $BENCH:/bench $CONT bash -c 'HF_HOME=/bench/weights/hf hf auth login'"
    echo "# run everything (inside screen):"; echo "export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0"
    echo "$RUN 2>&1 | tee $BENCH/results/run_\$(date +%m%d_%H%M).log";;
  *) echo "usage: $0 data|code|results|cmd"; exit 1;;
esac
