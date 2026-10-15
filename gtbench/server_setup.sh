#!/bin/bash
# One-time environment for the GT benchmark on a CUDA server (tested target: rhea, RTX 4080).
# The 3dgs/masked-3dgs containers (python 3.8, torch 2.0) are too old for SAM 2 / SAM 3 / transformers 5,
# so the benchmark gets its own conda env.
#
#   bash gtbench/server_setup.sh [env name, default gtbench]
#
# Afterwards (SAM 3 weights are gated: accept the licence at https://huggingface.co/facebook/sam3 once, then):
#   conda activate gtbench && hf auth login
set -eu
ENV=${1:-gtbench}
conda create -y -n "$ENV" python=3.11
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=5.0" "huggingface_hub>=1.0" timm kornia einops accelerate \
            opencv-python-headless scikit-image scipy pandas pillow \
            "rembg[gpu]" onnxruntime-gpu segment-anything-hq
python - <<'PY'
import torch, transformers, rembg, segment_anything_hq
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
PY
echo "== env $ENV ready. Next: conda activate $ENV && hf auth login   (for facebook/sam3)"
