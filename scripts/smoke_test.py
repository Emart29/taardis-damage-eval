"""Smoke test: does the TAARDIS build produce readable text and a sane perplexity?

Runs a handful of prompts through llama-server (with and without the Doctors-V3 LoRA),
then wikitext-2 perplexity on a small slice. Results go to $OUT_DIR/smoke_test.json.
Nothing here is a damage measurement yet; it only checks the setup before spending GPU hours.
"""
import json
import os
import re
import subprocess
import time
import urllib.request

LLAMA = os.environ.get("LLAMA_DIR", "/tmp/llama.cpp") + "/build/bin"
MODELS = os.environ.get("MODELS_DIR", "/tmp/models")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
MODEL = f"{MODELS}/TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf"
LORA = f"{MODELS}/doctors/TAARDIS-27B-Doctors-V3.lora.gguf"
PORT = 8080

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "A tank holds 120 litres and drains at 8 litres per minute. How many minutes until it is empty? Show the calculation.",
    "Explain in two sentences what a refrigerant does in an air conditioner.",
]


def readable_fraction(text):
    """Crude gibberish check: share of whitespace-separated tokens that look like words/numbers/code."""
    toks = text.split()
    if not toks:
        return 0.0
    ok = sum(bool(re.fullmatch(r"[\w.,;:!?()'\"`\-+=*/<>\[\]{}#%$&|^~\\]+", t)) for t in toks)
    return ok / len(toks)


def start_server(lora):
    cmd = [f"{LLAMA}/llama-server", "-m", MODEL, "-ngl", "99", "-c", "4096", "--port", str(PORT), "--jinja"]
    if lora:
        cmd += ["--lora", LORA]
    log = open(f"{OUT}/server_{'lora' if lora else 'nolora'}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited; see server log")
        time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def chat(prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def wikitext_file():
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    path = "/tmp/wikitext2_test.txt"
    with open(path, "w") as f:
        f.write("\n".join(ds["text"]))
    return path


def perplexity(textfile, lora, chunks=40):
    cmd = [f"{LLAMA}/llama-perplexity", "-m", MODEL, "-f", textfile, "-ngl", "99", "-c", "512", "--chunks", str(chunks)]
    if lora:
        cmd += ["--lora", LORA]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    return {"ppl": float(m.group(1)), "stderr": float(m.group(2))} if m else {"error": out[-2000:]}


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"llama_commit": open("/tmp/llama_commit.txt").read().strip(), "generation": {}, "perplexity": {}}
    for lora in (True, False):
        key = "with_lora" if lora else "no_lora"
        proc = start_server(lora)
        try:
            gens = []
            for p in PROMPTS:
                t = time.time()
                text = chat(p)
                gens.append({"prompt": p, "output": text, "seconds": round(time.time() - t, 1),
                             "readable_fraction": round(readable_fraction(text), 3)})
                print(f"\n=== [{key}] {p}\n{text}", flush=True)
            results["generation"][key] = gens
        finally:
            proc.terminate()
            proc.wait()
    wt = wikitext_file()
    for lora in (True, False):
        key = "with_lora" if lora else "no_lora"
        results["perplexity"][key] = perplexity(wt, lora)
        print(f"\nwikitext-2 PPL [{key}]: {results['perplexity'][key]}", flush=True)
    with open(f"{OUT}/smoke_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved", f"{OUT}/smoke_test.json")


if __name__ == "__main__":
    main()
