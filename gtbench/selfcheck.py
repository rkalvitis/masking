"""Runtime self-check for the gtbench container: run once with the GPU visible.

    singularity exec --nv ... gtbench.sif bash -c "cd /workspace && python -m gtbench.selfcheck"

Checks what `singularity build` cannot (there is no GPU during the build): CUDA visibility,
transformers model-class imports (they touch torch.accelerator at import time), HQ-SAM 2 import,
onnxruntime CUDA provider. Does NOT download any weights.
"""
import sys


def main() -> int:
    ok = True
    import torch
    print(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  cuda OK: {p.name}, {p.total_memory / 2**30:.1f} GB, sm_{p.major}{p.minor}")
    else:
        print("  cuda NOT available (did you pass --nv ?)")
        ok = False

    try:
        import transformers
        from transformers import Sam2Model, Sam3Model, SamModel  # noqa: F401
        print(f"transformers {transformers.__version__}: SamModel, Sam2Model, Sam3Model import OK")
    except Exception as e:  # noqa: BLE001
        print(f"transformers model import FAILED: {type(e).__name__}: {e}")
        ok = False

    try:
        from sam2.build_sam import build_sam2  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        print("HQ-SAM 2 (sam2 package from SysCV/sam-hq) import OK")
    except Exception as e:  # noqa: BLE001
        print(f"HQ-SAM 2 import FAILED: {type(e).__name__}: {e}")
        ok = False

    try:
        import segment_anything_hq  # noqa: F401
        print("HQ-SAM v1 (segment_anything_hq) import OK")
    except Exception as e:  # noqa: BLE001
        print(f"segment_anything_hq import FAILED: {type(e).__name__}: {e}")
        ok = False

    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        print(f"onnxruntime {ort.__version__} providers: {prov}")
        if "CUDAExecutionProvider" not in prov:
            print("  WARNING: no CUDAExecutionProvider, rembg will run on CPU")
    except Exception as e:  # noqa: BLE001
        print(f"onnxruntime import FAILED: {type(e).__name__}: {e}")
        ok = False

    try:
        import rembg  # noqa: F401
        import gtbench.methods as m
        print(f"rembg OK; gtbench.methods lists {len(m.METHODS)} methods: {', '.join(m.METHODS)}")
    except Exception as e:  # noqa: BLE001
        print(f"gtbench.methods import FAILED: {type(e).__name__}: {e}")
        ok = False

    print("SELFCHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
