#!/bin/bash
# Mac side: push the benchmark code + the DIS5K insect set to the server, pull results back.
#   bash gtbench/sync_to_rhea.sh code      # gtbench/ + DIS5K_insect/ -> $HOST:$SRV
#   bash gtbench/sync_to_rhea.sh results   # $HOST:$SRV/out/gtbench -> out/gtbench_rhea
# Set SRV (server folder for this benchmark) and WEIGHTS (server folder for model weights) to the paths you choose.
set -eu
HOST=${HOST:-rhea}
SRV=${SRV:?set SRV=<server folder for the benchmark, e.g. /home/robertsk/gtbench>}
HERE=$(cd "$(dirname "$0")/.." && pwd)
case "${1:-}" in
  code)
    ssh "$HOST" "mkdir -p $SRV"
    rsync -av --exclude __pycache__ "$HERE/gtbench" "$HOST:$SRV/"
    rsync -av "$HERE/DIS5K_insect" "$HOST:$SRV/"
    echo "== synced. On $HOST (container route, see gtbench/gtbench.def):"
    echo "   singularity build --fakeroot ~/containers/gtbench.sif $SRV/gtbench/gtbench.def      # once"
    echo "   screen -S gtbench"
    echo "   export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0"
    echo "   singularity exec --nv --cleanenv --contain --bind $SRV:/workspace --bind \${WEIGHTS:-<weights dir>}:/weights ~/containers/gtbench.sif \\"
    echo "     bash -c 'cd /workspace && python -m gtbench.run --data DIS5K_insect --out out/gtbench --weights /weights --device cuda && python -m gtbench.score --data DIS5K_insect --out out/gtbench' 2>&1 | tee $SRV/run_\$(date +%m%d_%H%M).log";;
  results)
    mkdir -p "$HERE/out/gtbench_rhea"
    rsync -av "$HOST:$SRV/out/gtbench/" "$HERE/out/gtbench_rhea/"
    echo "== results in out/gtbench_rhea (REPORT.md, summary.csv, sheets/)";;
  *) echo "usage: $0 code|results"; exit 1;;
esac
