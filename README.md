# TAARDIS damage eval

An independent evaluation of **where [TAARDIS-27B](https://huggingface.co/CodeMasterCody3D/taardis-27b-full-ternary)
lost knowledge** when Qwen3.8-27B was compressed to full ternary weights (−1, 0, +1 × scale).

The question is not "how good is TAARDIS?" but "which subjects and skills took the most damage
compared with the original model?", so that further training can be aimed where it is needed.

## Approach

**Damage = reference score − TAARDIS score**, on the same inputs, prompts and decoding settings.
A low TAARDIS score is only damage if the original model did better on the same items.

| Model | Role |
|---|---|
| TAARDIS-27B V2 1.75-bit + Doctors-V3 LoRA | Model under test, as released |
| TAARDIS-27B V2 1.75-bit, no LoRA | Shows what the correction adapter fixes |
| Qwen3.8-27B Q6_K (Unsloth GGUF) | Reference for the original model |
| Qwen3.8-27B Q4_K_M (Unsloth GGUF) | Common 4-bit quant, for context |

The reference is Q6_K rather than Q8_0 or BF16 because the eval runs on free Kaggle GPUs (2× T4, 32 GB),
where Q8_0 (29 GB) does not fit with room for context. Q6_K is close to lossless, but it is not the
original weights; see *Limitations*.

### Measurements (planned)

1. **Per-domain KL divergence**: `llama-perplexity --kl-divergence` against the reference's saved
   output distributions, over one text set per domain (sciences, humanities, law, medicine, code by
   language, math, multilingual). Measures how far TAARDIS's next-token predictions drift from the
   original, with no answer keys needed.
2. **Per-subject multiple choice**: MMLU-style questions scored from answer-letter logprobs, reported
   as accuracy drop per subject, also normalised against chance.
3. **Skill checks**: small generation subsets for math (GSM8K), code (HumanEval) and tool calling.

All comparisons are paired (same items for every model) and reported with bootstrap confidence intervals.
Damage is only called where the interval excludes zero.

## Status

- [x] Repo, build script, model downloader
- [ ] Smoke test on Kaggle: build works, output is readable, perplexity is sane
- [ ] Reference distributions (Q6_K) per domain
- [ ] KL divergence runs
- [ ] Per-subject multiple choice
- [ ] Skill checks
- [ ] Analysis and write-up

## Running it

Everything runs on a Kaggle notebook with **GPU T4 ×2** and internet enabled. Each `kaggle/NN_*/run.py`
is a thin kernel that clones this repo and calls the scripts in `scripts/`, so the same steps also run
on any Linux machine with a CUDA GPU:

```bash
bash scripts/setup_llama.sh 75        # builds Cody's llama.cpp fork (75 = T4; 80 A100, 86 RTX 30xx, 89 RTX 40xx)
python scripts/download_models.py taardis lora
OUT_DIR=results python scripts/smoke_test.py
```

## Limitations

Filled in as results come in. Known so far: the reference is Q6_K, not the original BF16 weights.
