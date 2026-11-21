"""Download model files from Hugging Face into MODELS_DIR (default /tmp/models).

Usage: python scripts/download_models.py taardis lora [ref_q6] [ref_q4]
"""
import os
import sys

from huggingface_hub import hf_hub_download

MODELS = {
    # Model under test (as released: V2 weights + Doctors-V3 correction adapter)
    "taardis": ("CodeMasterCody3D/taardis-27b-full-ternary", "TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf"),
    "lora": ("CodeMasterCody3D/taardis-27b-full-ternary", "doctors/TAARDIS-27B-Doctors-V3.lora.gguf"),
    # Reference for "original model" (Q8 does not fit on free Kaggle 2xT4; Q6_K is near-lossless)
    "ref_q6": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q6_K.gguf"),
    # Standard 4-bit baseline, for context on how TAARDIS compares to a common quant
    "ref_q4": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
}


def main(names):
    out = os.environ.get("MODELS_DIR", "/tmp/models")
    os.makedirs(out, exist_ok=True)
    for name in names:
        repo, fname = MODELS[name]
        path = hf_hub_download(repo, fname, local_dir=out)
        print(f"{name}: {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["taardis", "lora"])
