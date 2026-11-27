---
license: mit
base_model: orcarouter/GLM-5.3-Flash-Uncensored-NVFP4
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
language: [en, ru]
tags: [moe, nvfp4, compressed-tensors, expert-pruning, reap, agentic, tool-calling, code, glm5, uncensored]
---

<div align="center">

# GLM-5.3-Flash · Uncensored · NVFP4 · Agentic Expert-Pruned

**92 of 288 routed experts per layer · ~107B total / ~18B active parameters · 62 GiB · fits one 96 GB GPU**

*Pruned for agents, tool calling and code — not for trivia.*

</div>

> **Revision 2 (2026-10-13).** This repository was rebuilt from scratch. The previous revision (80 experts, ~62 GiB)
> was selected with weight-norm heuristics on ~200 English tokens and lost multilingual and tool-calling ability;
> it is still available in the git history. The repo name keeps "62GB" for link stability — the weights are now 62 GiB.

## What is this

An expert-pruned version of [`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-NVFP4)
(abliterated GLM-5.3-Flash, 314B MoE, NVFP4 experts) that fits entirely in the memory of a single 96 GB GPU
(tested on an RTX PRO 6000 Blackwell 96 GB; the bundled engine uses 67.5 GiB after loading and ~76 GB while generating, so 80 GB cards are very tight and untested).

| | Full model | This model |
|---|---|---|
| Routed experts / MoE layer | 288 | **92** |
| Total parameters | ~314B | **~107B** |
| Active parameters / token | ~18B | ~18B (top-8 routing unchanged) |
| Weights on disk | 177.5 GiB (+13.5 GiB MTP head) | **62.0 GiB** |
| Expert / attention precision | NVFP4 / BF16 | NVFP4 / **FP8** |
| Fits a 96 GB GPU | ❌ | ✅ |

To fit more experts into ~62 GiB, the large attention projections (KDA q/k/v/o, MLA q_b/o), the shared experts and
the dense MLPs of layers 0–2 are stored as **FP8 E4M3 with per-output-channel scales** (compressed-tensors
`float-quantized`, near-lossless) instead of BF16. This frees ~6.6 GiB = 12 extra experts per layer compared with a
plain BF16 build of the same size.

Untouched: routed experts' NVFP4 weights (packed bytes copied bit-exact; router rows sliced and re-indexed), small
attention projections, norms, hyper-connections, vision tower, tokenizer and chat template. The MTP speculative head is removed.

## How the experts were chosen

1. **Exact layer-wise forward of the full 191 GiB model** on one GPU (one decoder layer resident at a time, NVFP4 experts
   dequantized on the fly) over a domain-tagged calibration set: SWE-agent trajectories, Hermes/Glaive function calling,
   **the same function-calling dialogues rendered in GLM's native tool-call format** (`tools=`, `<tool_call>…<arg_key>…`,
   tool results), Magicoder/CodeFeedback code, OpenCodeReasoning traces, RU chat (GrandMaster-PRO-MAX), EN chat
   (UltraChat) and some math — 284 calibration sequences × 2048 tokens.
2. **REAP saliency** per expert and domain: Σ over routed tokens of `gate_weight · ‖expert_output‖₂`, normalized per layer.
3. **Priority-weighted selection** per layer — native tool calling 0.20, SWE trajectories 0.20, code 0.18,
   code reasoning 0.10, RU chat 0.10, EN chat 0.10, Hermes tool text 0.06, math 0.06 — top-92 kept.
   Knowledge-heavy domains (science, encyclopedic facts, rare languages) were deliberately **not** protected.

The kept expert indices for every layer are in [`keep.json`](keep.json).

## Evaluation

### Held-out perplexity (60 sequences × 2048 tokens never used for selection)

| Domain | Full model | This model | Δ | Top-1 agreement with full model |
|---|---|---|---|---|
| Tool calling (GLM native format) | 19.13 | 19.22 | +0.5% | 62.8% |
| SWE agent trajectories | 3.41 | 3.20 | -6.1% | 78.9% |
| Code (instruct) | 5.34 | 4.51 | -15.4% | 76.4% |
| Code reasoning traces | 2.98 | 3.24 | +8.7% | 82.2% |
| Tool calling (Hermes/Glaive text) | 5.96 | 9.68 | +62.4% | 65.9% |
| Chat RU | 10.85 | 11.23 | +3.5% | 55.6% |
| Chat EN | 11.76 | 14.45 | +22.9% | 57.2% |
| Math | 5.91 | 6.37 | +7.7% | 75.5% |

Lower perplexity than the full model on some domains does **not** mean the model got smarter — top-1 agreement shows it
does deviate from the full model. Treat these as "nothing broke" signals, not as benchmark scores.

### Functional agent tests (real tools, objective success checks, temperature 0)

Tools: `list_dir`, `read_file`, `write_file`, `run_python`, `run_tests`; the agent loop feeds real tool results back.

| Task | Result | Steps | Tool calls | Malformed calls |
|---|---|---|---|---|
| Fix 3 bugs in a Python repo until unit tests pass (task in Russian) | ✅ solved | 5 | 5 | 0 |
| Aggregate 3 CSV files into a revenue report file (task in Russian) | ✅ solved | 4 | 5 | 0 |

Small, sanity-level tests — run your own benchmarks (SWE-bench, BFCL, LiveCodeBench, …) before relying on it.

## Usage

### Tested: bundled single-GPU engine (OpenAI-compatible API)

`inference/` contains the engine used for all tests above: plain PyTorch + 🤗 Transformers modules, NVFP4 experts kept
packed on the GPU and dequantized per token. Requires `transformers>=5.16`, `torch>=2.11`, `safetensors`;
`flash-linear-attention` strongly recommended.

```bash
huggingface-cli download Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB --local-dir glm-agentic
mkdir -p /tmp/work && cp glm-agentic/inference/*.py /tmp/work/
python /tmp/work/serve.py glm-agentic     # loads in ~1-2 min, serves http://127.0.0.1:8000/v1/chat/completions
```

```python
import json, urllib.request
req = {"messages": [{"role": "user", "content": "Какая погода в Казани?"}],
       "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
                 "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
       "max_tokens": 300}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}))
print(json.load(r)["choices"][0]["message"])   # -> tool_calls: get_weather(city="Казань")
```

Speed with this engine: ~4 tokens/s decode on an RTX PRO 6000 (it is a reference implementation, not a fast kernel).

### vLLM (tested, ~46.7 tokens/s single stream, ~176.5 tokens/s with 8 parallel requests)

The checkpoint is stored in the layout of the bundled engine, so vLLM (0.31, which natively supports `Glm5NextForConditionalGeneration`)
needs a **one-time conversion** (~5 min, writes ~69 GiB). `inference/convert_vllm.py` does it and fixes four things that otherwise
produce a crash or NaN logits (every one of them was observed while bringing this model up):

1. FP8 attention / shared-expert / dense-MLP weights → BF16. vLLM's Glm5Next loader expects *block-wise* FP8 attention scales
   and expands per-channel scales to a 1 TiB tensor; its FP8 linear kernel also needs an NVRTC JIT build.
2. Per expert, `gate_proj` and `up_proj` have **different** NVFP4 global scales (up to 2×); vLLM fuses them into one `w13` tensor
   and needs a single scale. The converter sets both to `min(Gg, Gu)` and rescales the block scales of the other projection.
3. `input_activations` is removed from the quantization config. The checkpoint is **weight-only** (it has no input scales); with
   activation quantization declared, vLLM takes the W4A4 path, turns the missing scales into `inf` and returns NaN → garbage text.
   Without it vLLM uses the W4A16 (Marlin) path.
4. The sandbox needs the CUDA headers of the same version as `nvcc` for DeepGEMM's JIT (`nvidia-cuda-cccl==<nvcc version>`).

```bash
pip install uv && uv venv --python-preference only-managed -p 3.12 /tmp/vv
uv pip install --python /tmp/vv/bin/python vllm instanttensor
V=$(/tmp/vv/bin/python -c "import flashinfer;print(flashinfer.__version__)")
uv pip install --python /tmp/vv/bin/python "flashinfer-jit-cache==$V" --extra-index-url https://flashinfer.ai/whl/cu130

huggingface-cli download Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB --local-dir glm-src
python glm-src/inference/convert_vllm.py glm-src glm-vllm          # any python with torch + safetensors

/tmp/vv/bin/vllm serve glm-vllm --served-model-name glm --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 --load-format instanttensor --no-enable-flashinfer-autotune --trust-remote-code \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-auto-tool-choice --tool-call-parser glm47
```

Measured on one RTX PRO 6000 Blackwell (96 GB), converted checkpoint 68.5 GiB, KV cache ~6 GiB at 0.90 utilization.
Tool calls come back as standard OpenAI `tool_calls`; without a reasoning parser the reasoning is returned inline before `</think>`.

Agent tests through vLLM (same two tasks as above, temperature 0, real tools, standard OpenAI tool-call history):

| Task | Result | Steps | Tool calls | Malformed calls |
|---|---|---|---|---|
| Fix 3 bugs in a Python repo until unit tests pass (task in Russian) | ✅ solved | 5 | 5 | 0 |
| Aggregate 3 CSV files into a revenue report file (task in Russian) | ✅ solved | 4 | 5 | 0 |

Notes: the converted checkpoint is 6.5 GiB larger than the published one (BF16 instead of FP8 for the dense parts) and the block-scale
rounding in step 2 adds noise comparable to a few % on the affected scale tensors; no benchmark suite was run on the vLLM path.

### Not yet verified

- **Plain `transformers` `from_pretrained`** — not tested.

## Limitations

- The base model is **abliterated (refusal-removed)**: outputs are unfiltered. You are responsible for how you use it.
- Expert pruning trades breadth for size: expect weaker world knowledge, science and rare languages.
- Evaluation above is teacher-forced perplexity plus two functional agent tasks — not a benchmark suite.

---

### Кратко по-русски

Урезанная до 92 экспертов на слой версия GLM-5.3-Flash Uncensored NVFP4 (~107 млрд параметров, 62 ГиБ),
целиком помещается в одну видеокарту на 96 ГБ. Эксперты отбирались под агентность, вызовы инструментов в родном формате
GLM, код, русский и английский диалог; научные и энциклопедические знания сознательно не защищались. Проверено:
perplexity на отложенных данных и агентные задачи с реальными инструментами (обе решены, без ошибок в вызовах).
Запуск — движок из папки `inference/` (OpenAI-совместимый API).
