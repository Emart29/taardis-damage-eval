# Kaggle kernel: smoke test for TAARDIS-27B (GPU T4 x2, internet on).
# All logic lives in the GitHub repo so the run is reproducible outside Kaggle too.
import subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("nvidia-smi")
sh("git clone --depth 1 https://github.com/Emart29/taardis-damage-eval /tmp/repo")
sh("bash /tmp/repo/scripts/setup_llama.sh 75")
sh("pip install -q -U huggingface_hub")
sh("python /tmp/repo/scripts/download_models.py taardis lora")
sh("python /tmp/repo/scripts/smoke_test.py")
