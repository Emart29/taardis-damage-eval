"""Build the per-domain text sets used for the KL-divergence damage map.

Each domain gets the same number of tokens (Qwen3.8 tokenizer) so domains are compared on equal footing.
Text is fetched from pinned, public sources and written to data/domains/<domain>.txt, with
data/domains/manifest.json recording sources, licences and token counts. The text files are committed,
so every run (and anyone re-running the eval) scores exactly the same text.

Usage: python scripts/build_corpora.py [--tokens 9000]
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

from tokenizers import Tokenizer

UA = "taardis-damage-eval/0.1 (research eval corpus builder)"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "domains")

# English Wikipedia articles, grouped by subject. Several articles per domain so no single article dominates.
WIKI_EN = {
    "physics": ["Quantum mechanics", "General relativity", "Thermodynamics", "Electromagnetism", "Special relativity", "Nuclear fission"],
    "chemistry": ["Chemical bond", "Organic chemistry", "Acid–base reaction", "Periodic table", "Redox", "Chemical equilibrium"],
    "biology": ["DNA", "Cell (biology)", "Evolution", "Photosynthesis", "Protein", "Mitochondrion"],
    "medicine": ["Hypertension", "Diabetes", "Antibiotic", "Myocardial infarction", "Vaccine", "Asthma"],
    "law": ["Contract", "Tort", "Due process", "Habeas corpus", "Common law", "Criminal law"],
    "history": ["French Revolution", "Roman Empire", "World War I", "Industrial Revolution", "Cold War", "Ming dynasty"],
    "economics": ["Inflation", "Supply and demand", "Gross domestic product", "Monetary policy", "Comparative advantage", "Game theory"],
    "philosophy": ["Epistemology", "Utilitarianism", "Existentialism", "Immanuel Kant", "Ethics", "Free will"],
    "computer_science": ["Algorithm", "Hash table", "Operating system", "Compiler", "TCP/IP", "Big O notation"],
    "hvac": ["Heating, ventilation, and air conditioning", "Vapor-compression refrigeration", "Heat pump", "Refrigerant",
             "Psychrometrics", "Chiller", "Coefficient of performance", "Air handler"],
    "ml_quantization": ["Quantization (signal processing)", "Large language model", "Transformer (deep learning architecture)",
                        "Neural network (machine learning)", "Backpropagation", "Floating-point arithmetic"],
}

# Non-English Wikipedia articles (same topics across languages where possible).
WIKI_INTL = {
    "lang_french": ("fr", ["Révolution française", "Photosynthèse", "Mécanique quantique", "Empire romain", "Inflation"]),
    "lang_german": ("de", ["Französische Revolution", "Photosynthese", "Quantenmechanik", "Römisches Reich", "Inflation"]),
    "lang_spanish": ("es", ["Revolución francesa", "Fotosíntesis", "Mecánica cuántica", "Imperio romano", "Inflación"]),
    "lang_chinese": ("zh", ["法国大革命", "光合作用", "量子力学", "罗马帝国", "通货膨胀"]),
    "lang_japanese": ("ja", ["フランス革命", "光合成", "量子力学", "ローマ帝国", "インフレーション"]),
}

# Source files from popular repositories, pinned to a release tag.
GH = "https://raw.githubusercontent.com/"
CODE = {
    "code_python": [GH + "python/cpython/v3.13.0/Lib/" + f for f in
                    ["textwrap.py", "heapq.py", "fractions.py", "statistics.py", "bisect.py", "shlex.py"]],
    "code_javascript": [GH + "expressjs/express/4.21.1/lib/" + f for f in
                        ["router/index.js", "application.js", "response.js", "request.js", "router/route.js"]],
    "code_cpp": [GH + "fmtlib/fmt/11.0.2/" + f for f in ["src/format.cc", "include/fmt/ranges.h", "include/fmt/chrono.h"]],
    "code_rust": [GH + "BurntSushi/ripgrep/14.1.1/crates/" + f for f in
                  ["core/main.rs", "searcher/src/searcher/mod.rs", "globset/src/lib.rs", "ignore/src/walk.rs"]],
    "code_go": [GH + "golang/go/go1.23.0/src/" + f for f in
                ["strings/strings.go", "sort/sort.go", "bufio/bufio.go", "bytes/buffer.go"]],
    "code_shell": [GH + "nvm-sh/nvm/v0.40.1/" + f for f in ["install.sh", "nvm.sh"]],
}

LICENSES = {
    "wiki": "CC BY-SA 4.0 (Wikipedia)",
    "code_python": "PSF License", "code_javascript": "MIT", "code_cpp": "MIT", "code_rust": "MIT / Unlicense",
    "code_go": "BSD-3-Clause", "code_shell": "MIT",
    "math_gsm8k": "MIT", "math_competition": "MIT", "tool_calls": "Apache-2.0", "general_wikitext": "CC BY-SA 3.0",
}


CACHE = os.environ.get("CORPUS_CACHE", "/tmp/corpus_cache")


def get(url):
    """Fetch a URL with a polite delay, backoff on rate limits, and an on-disk cache for re-runs."""
    import hashlib

    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest())
    if os.path.exists(cached):
        return open(cached, encoding="utf-8").read()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(7):
        try:
            time.sleep(1.5)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8")
            with open(cached, "w", encoding="utf-8") as f:
                f.write(body)
            return body
        except Exception:
            if attempt == 6:
                raise
            time.sleep(5 * 2 ** attempt)


def wiki_text(lang, title):
    q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "extracts|revisions", "rvprop": "ids",
                                "explaintext": 1, "redirects": 1, "titles": title})
    page = next(iter(json.loads(get(f"https://{lang}.wikipedia.org/w/api.php?{q}"))["query"]["pages"].values()))
    if "extract" not in page:
        raise RuntimeError(f"missing article {lang}:{title}")
    # Drop reference-style tail sections, which are mostly lists of citations.
    text = page["extract"]
    for tail in ("\n== See also", "\n== References", "\n== Notes", "\n== Voir aussi", "\n== Notes et références",
                 "\n== Siehe auch", "\n== Literatur", "\n== Einzelnachweise", "\n== Véase también", "\n== Referencias",
                 "\n== 参见", "\n== 参考文献", "\n== 脚注", "\n== 関連項目"):
        text = text.split(tail)[0]
    return text.strip(), page["revisions"][0]["revid"]


def interleave(docs, tok, n_tokens):
    """Take text from every doc in turn (~1/len(docs) of the budget each) so no single source dominates."""
    per_doc = n_tokens // len(docs) + 64
    parts = []
    for d in docs:
        ids = tok.encode(d).ids[:per_doc]
        parts.append(tok.decode(ids))
    text = "\n\n".join(parts)
    ids = tok.encode(text).ids
    if len(ids) < n_tokens:
        raise RuntimeError(f"only {len(ids)} tokens, need {n_tokens}")
    return tok.decode(ids[:n_tokens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=9000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer.from_str(get("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json"))
    manifest = {"tokenizer": "Qwen/Qwen3.8-27B", "tokens_per_domain": args.tokens, "domains": {}}

    def save(name, docs, sources, license_):
        text = interleave(docs, tok, args.tokens)
        with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
            f.write(text)
        manifest["domains"][name] = {"sources": sources, "license": license_, "chars": len(text),
                                     "tokens": len(tok.encode(text).ids)}
        print(f"{name:20s} {len(text):7d} chars", flush=True)

    for name, titles in WIKI_EN.items():
        docs, srcs = [], []
        for t in titles:
            text, rev = wiki_text("en", t)
            docs.append(text)
            srcs.append(f"https://en.wikipedia.org/w/index.php?oldid={rev} ({t})")
        save(name, docs, srcs, LICENSES["wiki"])

    for name, (lang, titles) in WIKI_INTL.items():
        docs, srcs = [], []
        for t in titles:
            text, rev = wiki_text(lang, t)
            docs.append(text)
            srcs.append(f"https://{lang}.wikipedia.org/w/index.php?oldid={rev} ({t})")
        save(name, docs, srcs, LICENSES["wiki"])

    for name, urls in CODE.items():
        save(name, [get(u) for u in urls], urls, LICENSES[name])

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    save("math_gsm8k", ["\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in gsm.select(range(200)))],
         ["openai/gsm8k main/test, first 200 items"], LICENSES["math_gsm8k"])

    subjects = ["algebra", "geometry", "number_theory", "counting_and_probability", "precalculus", "intermediate_algebra"]
    math_docs = []
    for s in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", s, split="test")
        math_docs.append("\n\n".join(f"Problem: {r['problem']}\nSolution: {r['solution']}" for r in ds.select(range(60))))
    save("math_competition", math_docs, [f"EleutherAI/hendrycks_math {s}/test, first 60 items" for s in subjects],
         LICENSES["math_competition"])

    glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    calls = [f"{r['system']}\n{r['chat']}" for r in glaive.select(range(400)) if "<functioncall>" in r["chat"]][:80]
    save("tool_calls", ["\n\n".join(calls)], ["glaiveai/glaive-function-calling-v2 train, first 400 rows with a function call"],
         LICENSES["tool_calls"])

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    save("general_wikitext", ["\n".join(wt["text"])], ["Salesforce/wikitext wikitext-2-raw-v1 test"],
         LICENSES["general_wikitext"])

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
