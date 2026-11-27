# GLM 5.3 Flash Uncensored — installer

Everything here installs and serves **model #1** from `../uncensored-models-2026-11.md`:

| What | Repo | Size | Runtime |
|---|---|---|---|
| FP8 (canonical, card-documented) | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` | 321 GB | vLLM, TP=4 |
| NVFP4 (Blackwell/1×B200, 2×H200) | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | 165 GB | vLLM |
| GGUF Q2–Q8 (CPU/CUDA/Metal/ROCm) | `orcarouter/GLM-5.3-Flash-Uncensored-GGUF` | 117–341 GB | llama.cpp |
| MLX (Apple Silicon) | `Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2` | ~321 B params | mlx-lm |

Base: **GLM-5.3-Flash** — 320 B total / **18 B active** MoE, 1 M context, vision tower, MTP draft head. MIT licence (via `zai-org/GLM-5.3-Flash`).

> **Not installed here.** This sandbox is 2 vCPU / 3 GB RAM / 20 GB disk, with no GPU and no route to `huggingface.co`. Even Q2 is 117 GB. The scripts below are for your own box; they were syntax-checked here and the client was tested end-to-end against a mock vLLM, but the weights themselves cannot be pulled from this environment.

---

## 1. Pick the tier

| Rig (total memory) | Tier | Download | Notes |
|---|---|---|---|
| 4–8× H100/H200, 2–4× B200 (≥380 GiB) | `fp8` | 321 GB | native FP8, MTP speculative decoding, 1 M ctx, documented path |
| 1× B200 192 GB / 2× H200 (≥175 GiB) | `nvfp4` | 165 GB | MMLU 86.16 → 85.28 (−0.88); 0 % refusals at default/max effort |
| 256 GiB RAM host (manual tier) | `gguf-q4` | 193 GB | Q4_K_M = best quality/size; KLD 0.087 — needs ~200 GB total |
| 150–200 GiB total | `gguf-q3` | 153 GB | KLD 0.176 |
| 120–150 GiB total | `gguf-q2` | 117 GB | KLD 0.486 — visibly degraded, fit-driven only |
| Mac Studio 192 GB+ | MLX repo above | — | MTP drafter included |
| under ~120 GB | hosted API | 0 | see §6 |

Auto-selection follows this ladder; `--tier` overrides it. 4×3090/4090 (96 GB) cannot hold
even Q2 — use the hosted route or the REAP50 (~90 GB at Q4, 50 % of experts pruned) variant.

Auto-detection: `./install.sh --dry-run` prints what it would do for your box.

## 2. One-command install (bare metal)

```bash
cd glm53flash
./install.sh --dry-run          # verify the plan + disk space first
./install.sh                    # detects GPU, recommends tier, downloads, writes glm.env
./serve.sh                      # vLLM (or llama.cpp) on 0.0.0.0:8000
./smoke_test.sh                 # waits for ready, then runs 5 probes
```

Requirements: CUDA 12.x driver, `cmake` + `nvcc` for the FP8 path (DeepGEMM JITs a block-FP8 kernel at startup), ~1 h on a 1 Gbit link for 321 GB. `hf transfer` is enabled automatically; downloads resume if you re-run.

## 3. Docker (least friction)

```bash
hf download dealignai/GLM-5.3-Flash-UNCENSORED-FP8 \
  --local-dir ./weights/GLM-5.3-Flash-UNCENSORED-FP8
docker compose up -d
./smoke_test.sh
```

Serve command used (from the model card, verbatim):

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-FP8 \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## 4. llama.cpp / GGUF (CPU, Mac, mixed GPU+CPU)

```bash
./install.sh --tier gguf-q4
./serve.sh                # builds llama.cpp if needed, adds --mmproj for vision
```

The whole file must fit in RAM or be mmap'd; `-ngl 999` offloads every layer. Add `EXTRA_LLAMA_ARGS="--no-kv-offload"` for pure CPU, or drop `-ngl` in `serve.sh` for CPU-only.

## 5. Five serving gotchas (these will bite you)

Straight from the model cards, all handled by `glm_client.py`:

1. **`reasoning_effort` only accepts `low` or `high`.** `medium`, `xhigh`, `off`, or omitting it → silently resolves to **max**, the deepest mode.
2. **At max effort, `max_tokens < 6000` returns an EMPTY reply** (`finish_reason=length`, all budget spent inside `<think>`). In multi-turn the empty turn poisons history and the chat never recovers. Use 6000+, or `low`/`high` (fine at 2000).
3. **Never send `enable_thinking`** — no such variable in this template. It (or a top-level `reasoning_effort: "none"`) turns the parser off while the model still thinks, dumping raw reasoning into `content`.
4. **Reasoning arrives in `message.reasoning`**, not `reasoning_content`; `clear_thinking` must be nested inside `chat_template_kwargs`.
5. **Video: `media_io_kwargs.video.fps` must be below the clip's own fps** (use `fps: 2`). Setting it at/above does not error — it triples multimodal tokens and kills the engine (`EngineDeadError`). Pixel limits go in `mm_processor_kwargs.max_image_tokens`; `max_pixels`/`detail` are accepted and ignored.

```bash
./glm_client.py -p "sketch a CUDA softmax kernel" --effort high
./glm_client.py -p "read this chart" --image chart.png   # vision
./glm_client.py -p "weather in Bengaluru?" --tools       # function calling
```

## 6. No GPU? Hosted route (works in 2 minutes)

Same weights, someone else's H200s. Any OpenAI-compatible endpoint:

```bash
export GLM_BASE=https://your-router.example/v1
export GLM_KEY=sk-...
./glm_client.py --check --base-url "$GLM_BASE" --api-key "$GLM_KEY"
./glm_client.py -p "..." --base-url "$GLM_BASE" --api-key "$GLM_KEY"
```

The model is listed on OrcaRouter and AllModels at roughly **$0.13 / $0.53 per 1 M tokens** (GLM 5.3 Flash Uncensored), which is cheaper per task than the full GLM 5.3 ($5.01) by ~8×. Verify the provider actually serves the *uncensored* build — several list base GLM 5.3 Flash under a similar name.

## 7. What you should expect

| Metric | Value | Source |
|---|---|---|
| MMLU (FP8 CRACK) | 86.74 → **87.33** (+0.59 vs base) | card, 1,026 questions, logit scoring |
| MMLU (NVFP4 CRACK) | 86.16 → **85.28** (−0.88) | card |
| HarmBench-320, reasoning-off / max | **0/320 refused** (FP8); 0 % (NVFP4 at those modes) | cards |
| HarmBench-320, high / low effort | ~4 % / ~9 % refused — *deliberate*, to protect quality | NVFP4 card |
| GGUF uncensoring | ~89 % → low teens refusal; **reduced, not removed** (different, single-direction edit) | GGUF card |
| Q6_K / Q4_K_M / Q3_K_M / Q2_K KLD | 0.026 / 0.087 / 0.176 / 0.486 | GGUF card |
| Throughput | 163 tok/s single-stream, 211 with MTP; ~19.4 k tok/s prefill (TP4, H200) | FP8 card |
| MTP acceptance | 75.9 % (FP8) / 81.7 % (NVFP4) | cards |

Note the FP8 and GGUF builds are **different abliterations by different publishers**. FP8 = dealignai "CRACK" (fully 0 % at default modes). GGUF = orcarouter (refusals reduced to low-teens). If zero refusals matter, take FP8/NVFP4.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unsupported architecture: glm5_next` | vLLM older than the GLM-5.3-Flash merge | `./install.sh --nightly`, or the docker image |
| DeepGEMM / `nvcc not found` at startup | CUDA toolkit absent | install CUDA toolkit, or use docker |
| Empty assistant replies | max effort + small `max_tokens` | ≥ 6000 tokens, or `--effort high` |
| Raw `<think>` text in `content` | parser disabled (`enable_thinking`, or `reasoning_effort: none`) | remove those keys |
| `CUDA out of memory` on load | TP too small / ctx too large | raise `--tensor-parallel-size`, lower `MAX_LEN` in `glm.env`, or drop a tier |
| Engine dies on a video request | `fps` ≥ clip fps | `media_io_kwargs.video.fps = 2` |
| Garbage/repetition on NVFP4 | old files (pre-2026-08-29) | re-download; the fixed release is `85.28 %` MMLU |
| Slow first token | 321 GB of weights + 1 M ctx KV | expected; use MTP + `--gpu-memory-utilization 0.92` |

## 9. Legal / safety

Weights are weight-modified, not retrained: the base MIT licence carries over. dealignai ships these for alignment and safety research; you own what you serve. If anything other than you can reach the endpoint, put moderation in front of it — this build will comply with requests the stock model refuses, including copyright and harmful-content categories.

---

### Files

| File | Purpose |
|---|---|
| `install.sh` | hardware detection → tier → deps → download → `glm.env` |
| `serve.sh` | vLLM or llama.cpp launch with the exact documented flags |
| `docker-compose.yml` | FP8 on 4×H200 |
| `glm_client.py` | dependency-free client + `--check` smoke suite |
| `smoke_test.sh` | waits for readiness, then runs the suite |
| `tests/mock_vllm.py` | fake endpoint so you can test the client with no GPU |
