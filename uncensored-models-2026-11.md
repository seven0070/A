# Top uncensored models that still hold up on every capability — November 2026

*Snapshot: 27 Nov 2026. "Uncensored" here means **weight-level refusal removal** (abliteration / directional ablation / Heretic / "CRACK" builds) — not prompt jailbreaks, not system-prompt tricks. Only models with a published refusal measurement and a published capability-retention measurement are ranked.*

---

## Read this first: uncensoring does not add capability

Abliteration is a rank-1..N projection edit: `W' = W − s·QQᵀW` applied to residual-stream writes. It removes one (or a few) refusal directions in activation space. It **cannot** add reasoning, coding, or knowledge ability. So the only sane way to rank "uncensored and more powerful in all capabilities" is:

1. **Start from a strong base** (its benchmark floor is your ceiling), then
2. **Pick the edit that loses the least capability** (measured as KL divergence on harmless prompts + benchmark delta vs. base), while
3. **Actually killing refusals** (HarmBench / AdvBench / XSTest numbers, and over-refusal on benign prompts).

Most "best uncensored model" listicles fail step 2 and 3 — they quote download counts. Below, the numbers come from the publishers' cards.

**Rule of thumb:** a 4-bit quant hurts quality more than a good abliteration does. If a model matters, run it at FP8/NVFP4/Q8, not Q4.

---

## The list — ranked by "frontier capability × actually uncensored × usable"

| # | Model | Base / arch | Refusals after edit | Capability retention | Where it runs | License |
|---|---|---|---|---|---|---|
| 1 | **GLM 5.3 Flash Uncensored** (`dealignai/GLM-5.3-Flash-UNCENSORED-FP8`) | GLM-5.3-Flash, 320B total / 18B active, hybrid MoE, 1M ctx, vision, MTP | **0/320 HarmBench** (100%), incl. copyright set; stable at temp 1.0 | **MMLU 86.74% → 87.33%** (above base), no degeneration | 4×H200 / 2×B200 FP8, or API $0.13/$0.53 per 1M | check GLM licence |
| 2 | **DeepSeek V4.1 Flash 1017 Uncensored** (`mrfakename/...-abliterated`, `dealignai/...-UNCENSORED-FP8`) | DeepSeek-V4.1-Flash-1017, 763B MoE, FP8, 1M ctx, vision | **97→4 /100** (chat), 98→9/100 (think) on the gentle build; **0/320** on dealignai's | **KL 0.067** on harmless first token (gentle build); dealignai trades ~3.8 MMLU pts for zero refusals | 8×H200 FP8; or 2.9 bpw EXL3 ≈ 4×80GB | check DeepSeek licence |
| 3 | **Qwen3.5-397B-A17B Heretic** (`RadicalNotionAI/...-heretic`, `trohrbaugh/...`) | Qwen3.5 397B total / 17B active, Gated DeltaNet + MoE, 262K→1M, vision | **99→2 /100** | KL 0.38 (the honest caveat) — base is very strong: MMLU-Pro 87.8, GPQA 88.4, SWE-bench Verified 76.4, 201 languages | 8×H100 FP8 (~400 GB) or Q4 GGUF ~200 GB | Apache 2.0 |
| 4 | **GLM 5.3 (753B) Abliterated** | GLM 5.3, 753B MoE, text-only, 1M ctx | **12–13% still refuse** (soft refusals) — GLM resists abliteration | MMLU 85.6% → 87.4%; still the best open agentic coder (TB 4.0 41.9%) | FP8 ~750 GB, NVFP4 ~391 GB | check GLM licence |
| 5 | **Qwen3.8-Flash-Next Uncensored** (`orcarouter/...`, `SC117/...` quants) | Qwen4-arch preview, 512 experts / 10+1 active, Gated DeltaNet + QSA, native vision, 1M ctx, tool calling | rank-1 edit transplanted; refusals removed (SC117 A/B: 5/5 refuse → 5/5 comply) | 144 of 1224 tensors edited; rest byte-identical to base | Q4_K_M ≈ 110 GB, Q8 ≈ 188 GB (llama.cpp `qwen4exp` after 2026-08-27) | Apache 2.0 |
| 6 | **Gemma 4 31B it Heretic** (`coder3101/gemma-4-31B-it-heretic`) | Gemma 4 31B dense, 256K, vision, Apache 2.0 | **99→15 /100**; UGU **W/10 = 10.0**, UGI **65.69** (#6 of 1,329) | **KL 0.0434** — the cleanest low-damage edit in the whole list | bf16 ~62 GB; Q4 ~20 GB → one 24 GB card | Apache 2.0 |
| 7 | **Inkling-2-Mini abliterated** (`orcarouter/Inkling-2-Mini-abliterated`) | Thinking Machines 26.4B / 3.4B active, looped MoE, 1M ctx, vision | 82.5% → **11.9%** avg (HarmBench-standard 5.3%, XSTest over-refusal 7.2→3.2%) | MMLU 86.3→86.0, MMLU-Pro 80.0→79.6, GSM8K 88.7→90.7, CMMLU −2.2 | bf16 **53 GB** → one 80 GB GPU | Apache 2.0 |
| 8 | **Huihui Qwen3.8-27B abliterated** (`huihui-ai/...`) | Qwen 3.8 27B, 256K, vision | crude PoC abliteration; layers 18–51 only, first 15 layers untouched, MTP+vision unmodified | good general/vision; **weak agentic coding** (TB 4.0 5.6%) | Q4 ~17 GB → 24 GB card / Ollama | Apache 2.0 |
| 9 | **Qwen3.6-35B-A3B abliterated (MTP)** (`huihui-ai/...`) | 35B MoE, ~3B active | as above | fast, strong long-document throughput; 4.25M downloads/mo | Q4 ~20 GB → 24 GB card | Apache 2.0 |
| 10 | **gpt-oss-120b-heretic** (`kldzj/...-heretic-v2`) | 117B / 5.1B active, 131K, Apache 2.0 | **97→19/100** (v1), 22/100 (v2) — poor uncensoring | KL 0.92 (v1) / 0.53 (v2) — highest damage here | MXFP4 ~63 GB → one H100 80 GB | Apache 2.0 |

**Also worth knowing:** `ArliAI/GLM-4.6-Derestricted-v3` tops the OpenUGI leaderboard for open weights (UGI **69.82** with think-prefill, **W/10 8.8**; 65.91 / 9.8 without) — older 355B base, but still the highest measured "willingness × breadth" score you can download.

---

## Best in each capability (uncensored builds only)

| Capability | Pick | Why |
|---|---|---|
| Raw intelligence / general reasoning | **DeepSeek V4.1 Flash 1017 Uncensored** | Highest-scoring open base (AA Intelligence Index 32.6; FrontierCode 2.0 28.5%; HLE-Diamond 25.3%), and the rank-2 abliteration keeps KL at 0.067 |
| Agentic coding / SWE / terminal | **GLM 5.3 Flash Uncensored** (or GLM 5.3 if you accept 12% refusals) | TB 4.0 32.8% / 41.9%, SciCode 51.6% / 59.0% — the strongest open coder family of 2026 |
| Math & STEM | **Qwen3.5-397B-A17B Heretic** | GPQA 88.4, AIME26 91.3, HMMT-25 94.8 on base; only 2/100 refusals after edit |
| Vision / documents / UI | **Gemma 4 31B it Heretic** | MMMU-Pro 76.9%, OmniDocBench 0.131, 256K ctx — best vision per GPU-dollar you can fully self-host |
| 1M-token long context | **GLM 5.3 Flash Uncensored**, then **Qwen3.8-Flash-Next** / **Inkling-2-Mini** | all three ship 1M context; GLM also keeps vision + MTP |
| Multilingual breadth | **Qwen3.5-397B-A17B Heretic** | 201 languages/dialects; MMMLU 88.5, MMLU-ProX 84.7 |
| Speed (single consumer GPU) | **Gemma 4 31B it Heretic** (24 GB), **Inkling-2-Mini** (3.4B active, 1M ctx) | dense 31B at Q4 ≈ 20 GB; Inkling activates 3.4B/token |
| Creative writing / roleplay | **Anubis-70B-v1.1** (UGI 50.79, W 8.5), **Cydonia-24B-v4.3** (44.52, W 7.0), **Behemoth-X-123B-v2** (53.6, W 8.2) | TheDrummer models win EQ-Bench-style nuance, not STEM |
| Steerable / not "preachy" | **Hermes 4 405B** (UGI 54–55, W 4.8) | Most neutral, least-lecturing frontier open model; 14B/70B variants exist |
| Cheapest hosted uncensored | GLM 5.3 Flash Uncensored **$0.13 / $0.53**; DeepSeek V4.1 Flash Uncensored **$0.14 / $0.52** per 1M | both fully uncensored at default effort |

---

## If you don't need open weights (least-censored frontier APIs)

These are *not* uncensored, but they're the least-restrictive hosted models measured:

- **Grok 4.20 multi-agent** — UGI 70.0, W/10 6.5 (top of the board)
- **Grok 4 (0709)** — UGI 67.8, W/10 6.0
- **DeepSeek V4 Pro (reasoning on)** — UGI 62.3, W/10 3.2
- **Mistral Large Instruct 2411** — highest reported willingness (W/10 7.5) among frontier models

---

## Fit by hardware

| Your rig | Best pick | Notes |
|---|---|---|
| 8–12 GB | Gemma 4 12B heretic-abliterated, Qwen3.5-9B abliterated | expect real quality loss at these sizes |
| 16–24 GB | **Gemma 4 31B it Heretic** @ Q4 (~20 GB); Qwen3.8-27B abliterated; Qwen3.6-35B-A3B abliterated | Gemma 4 31B is the sweet spot |
| 32–48 GB | Gemma 4 31B @ Q6/Q8; Inkling-2-Mini @ NVFP4 | Inkling gives you 1M ctx + vision |
| 80 GB (1×H100/MI300X) | **Inkling-2-Mini bf16 (53 GB)**, gpt-oss-120b (MXFP4) | Inkling ≫ gpt-oss-heretic on refusals *and* KL |
| 2×80–192 GB | **Qwen3.8-Flash-Next** @ Q4/Q5; GLM 5.3 Flash @ NVFP4 | best capability per watt |
| 4–8×H100/H200 | **GLM 5.3 Flash Uncensored** (FP8, 320 GB) → **DeepSeek V4.1 Flash 1017** (763 GB FP8) → **Qwen3.5-397B her** | this is where "frontier + zero refusals" actually exists |

---

## How to vet any new "uncensored" release (30 seconds)

1. **Is the edit in the weights?** Cards that mention LoRA adapters, steering vectors, `model.py` hooks, or "system prompt" are not weight-level.
2. **What's the refusal number, and on what set?** HarmBench-320 or AdvBench-100 with n and decoding params. "No refusals" with no set = marketing.
3. **What's the cost?** KL divergence on harmless prompts (want ≲0.1 for gentle, >0.5 is a different model) **and** a benchmark delta vs. base (MMLU/MMLU-Pro). Best builds publish both — GLM 5.3 Flash CRACK went *up* 0.59 pp; DeepSeek dealignai went *down* 3.8 pp.
4. **Did they ablate reasoning and chat separately?** Chat-mode and think-mode refusals live in different directions; single-direction edits leak refusals in think mode (that's why mrfakename used rank-2).
5. **Check quantization.** An "uncensored" Q2 GGUF of a 700B model is worse than a clean Q8 of a 30B model.

---

## Caveats, honestly

- **Abliteration isn't free.** Even the best edits shift the distribution; long-chain reasoning and code correctness degrade first, and it's invisible in a 20-prompt vibe check.
- **"Uncensored" ≠ "accurate."** Refusal removal also removes some hedging, so hallucination and overconfidence go up. DeepSeek V4.1 Flash already hallucinates on 64% of the knowledge questions it misses; abliterated builds don't improve that.
- **Some bases can't be cracked cleanly.** GLM 5.3 (753B) still refuses 12–13% even in the best public build. GLM 5.3 Flash only reaches 0% at reasoning-off/max effort — low effort keeps refusals by design.
- **Licences vary and some prohibit harmful use.** Gemma 4, Qwen 3.5/3.8, Inkling and gpt-oss are Apache 2.0; GLM and DeepSeek carry their own terms. Self-hosting shifts liability to you — add your own moderation if anyone else touches the endpoint.
- **Hosted "private" routes still log.** If the point is privacy, run weights locally; the API convenience and the privacy claim are usually in tension.

---

## Sources

- [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8) — HarmBench-320 320/320, MMLU 86.74→87.33, 320B/18B, 1M ctx
- [mrfakename/DeepSeek-V4.1-Flash-1017-abliterated](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) — rank-2 subspace, 97→4/100, KL 0.067, LoRA + GGUF
- [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8) — zero-refusal variant of the same base
- [RadicalNotionAI/Qwen3.5-397B-A17B-heretic](https://huggingface.co/RadicalNotionAI/Qwen3.5-397B-A17B-heretic) — Heretic v1.2.0, 99→2/100, base benchmark table
- [coder3101/gemma-4-31B-it-heretic](https://huggingface.co/coder3101/gemma-4-31B-it-heretic) — ARA ablation, 99→15/100, KL 0.0434, Gemma 4 benchmarks
- [orcarouter/Inkling-2-Mini-abliterated](https://huggingface.co/orcarouter/Inkling-2-Mini-abliterated) — 4-D refusal subspace, full refusal + capability tables
- [orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) and [SC117/Qwen3.8-Flash-Next-abliterated-UD-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-abliterated-UD-GGUF) — tensor-level edit accounting
- [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) — layers 18–51 edit, 4.6M downloads/30d
- [kldzj/gpt-oss-120b-heretic](https://huggingface.co/kldzj/gpt-oss-120b-heretic) — 97→19/100, KL 0.92 (v2: 22/100, 0.53)
- [OpenUGI leaderboard](https://openugi.com/) — UGI / W-10 / ideology scores (Grok 4.20 70.0, ArliAI GLM-4.6-Derestricted-v3 69.82, Gemma 4 31B heretic 65.69, Hermes 4 405B 54.07, Anubis 70B 50.79, Cydonia 24B 44.52)
- [AllModels abliterated guide](https://allmodels.io/guides/best-abliterated-models) — Artificial Analysis Intelligence Index, TB 4.0, SciCode, refusal/capability trade-offs (Nov 2026)
- [Hugging Face: trending abliterated models](https://huggingface.co/models?search=abliterated&sort=trending) — download/adoption signal
