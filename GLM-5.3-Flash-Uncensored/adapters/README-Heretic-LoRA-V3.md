---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
  - uncensored
  - abliterated
  - heretic
  - lora
  - gguf
  - glm
---

# GLM-5.3-Flash Heretic — LoRA adapter V3

> **⚠️ Content warning:** This adapter has had the base model's refusal
> behavior surgically suppressed. The resulting model **will** comply with
> requests the base model refuses, including requests that are harmful,
> unethical, offensive, or illegal. It has reduced safety guardrails. See
> [Responsible use](#responsible-use--disclaimer) below — **you are solely
> responsible for what you do with it.**

This is a **rank-1 LoRA adapter** that decensors / "abliterates"
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B total / 18B active MoE, MIT license), produced with **heretic-gguf** —
a GGUF-native port of [Heretic](https://github.com/p-e-w/heretic)'s
Optuna-optimized directional ablation, which runs the whole search directly
on quantized GGUF weights via llama.cpp.

> **What's new in V3.** The adapter works with **mainline llama.cpp** —
> no fork and no patch required (the V2 release needed an
> [Unsloth-fork build](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) of
> PR #27754 plus a one-line LoRA patch). Refusals are down to
> **14/140 (10.00%)** at a much lower KL than before — **0.2380** — so this
> adapter strictly dominates the V2 operating point. Note that no
> CoT-skip (thinking-suppressed) evaluation was run this time, so the
> numbers are **not directly comparable to the V1 card**; the scores below
> come from the study's thinking-enabled evaluation.

**This repository contains only the adapter.** You need the base model
separately — any GGUF quant of GLM-5.3-Flash works, since the adapter is
applied in f32/f16 compute regardless of the base quant (it was tuned and
evaluated against
[UD-IQ4_XS](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)). The LoRA
form is the **lossless** option: the base weights are never modified or
requantized, and the download is ~86 MB instead of ~160 GB.

**heretic-gguf is available at
[github.com/MoriNoNushi/heretic-gguf](https://github.com/MoriNoNushi/heretic-gguf)** —
the full tool, so the method can be applied to other GGUF models.

## Results

Measured on the study's **thinking-enabled** evaluation: **140 harmful
prompts** (100 from `mlabonne/harmful_behaviors` test + 40 custom) and
**100 harmless prompts** (`mlabonne/harmless_alpaca` test), reasoning effort
**high**, refusal scored on the **final answer** after `</think>`:

| | Refusal rate (harmful) | KL divergence (harmless) |
|---|---|---|
| Base model (thinking) | 96.43% (135/140) | 0 (by definition) |
| **glm-5.3-heretic-lora-v3.gguf** | **10.00% (14/140)** | **0.2380** |

(For reference, the V2 adapter measured 35/140 refusals at KL 0.3030 on
this same thinking-enabled benchmark — V3 cuts refusals ~2.5× while
lowering KL.)

> **Note on comparability:** no CoT-skip (thinking-suppressed) evaluation
> was run for this release, so these numbers are not comparable to the
> "without thinking" table in the V1/V2 cards.

Refusals are counted by refusal-keyword matching (English + Chinese +
first-person-negation markers such as "I'm not going to / able to ...",
GLM-5.3's dominant refusal phrasing). KL divergence is measured on
first-token logits on harmless prompts.

> **Note on KL:** the KL divergence above (and the optimization objective
> itself) was measured against the **UD-IQ4_XS** quant. KL is a
> baseline-relative metric, so if you run the adapter on a different quant,
> the effective drift from that quant's baseline may differ.

## Usage

```bash
llama-server \
    -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
    --lora glm-5.3-heretic-lora-v3.gguf \
    --jinja
```

Add your usual offload/context flags (`-ngl 999`, `-c`, tensor splits,
etc.) — nothing model-specific is required, and no special sampling
parameters are needed. Simply omitting `--lora` restores the base model
exactly.

**llama.cpp requirements.** This release works with **mainline
llama.cpp** — GLM-5.3-Flash (`glm5next`) support and LoRA application on
all adapter tensors are included upstream. No fork, no PR build, and no
patch is needed. (The V2 adapter required a patched build; that is no
longer the case.)

## How it was made

- **Method:** directional ablation ("abliteration") — the refusal direction
  in residual space (difference of means over 480 harmful / 480 harmless
  prompts, 5% winsorized, orthogonalized against the harmless mean,
  captured at the pre-thinking position with reasoning effort high) is
  projected out of the attention output and MoE down-projection weights.
  Strengths, layer kernels, and direction selection were tuned by
  multi-objective Optuna TPE (minimize refusal rate and KL jointly) under
  a **thinking-enabled** evaluation. The study is ongoing; this adapter is
  a later trial of study `glm53think2`.
- **Configuration** (study `glm53think2`; per-layer direction
  scope; per-expert strengths scaled by measured harmful/harmless routing
  frequency; `row_normalization = "pre"`):
  - attn.o_proj: max weight 6.49 @ layer 27.4 of 45.
  - routed MLP down-proj: max weight 2.24 @ layer 34.7.
  - shared-expert down-proj: max weight 5.09 @ layer 31.5.
- **Why a LoRA:** heretic-gguf expresses ablation as a rank-1 LoRA overlay,
  the same math stock Heretic writes into PEFT adapters. Shipping the
  adapter avoids requantizing the ~160 GB base entirely — bit-identical base
  weights, instant to apply. (A merged export of the IQ4_XS quant is also
  technically impossible — imatrix quants have no requantizer — so the
  adapter is the only lossless way to ship this configuration.) The adapter
  embeds its full provenance (study, trial, parameters, scores, commit
  hashes) as `adapter.heretic.*` GGUF metadata keys; inspect with
  `strings glm-5.3-heretic-lora-v3.gguf | grep adapter.heretic`.

## Responsible use & disclaimer

- **This adapter can make the base model generate content that is
  offensive, disturbing, hateful, sexually explicit, violent, or otherwise
  objectionable, including detailed instructions for harmful or illegal
  acts.** That is the direct and intended consequence of removing refusal
  behavior.
- The ablation suppresses *refusals*, not the base model's knowledge —
  outputs on dangerous topics may be wrong, hallucinated, or incoherent.
  Nothing the model says should be treated as accurate, safe, or legal
  advice.
- **Do not deploy models using this adapter in any production system,
  public-facing service, or multi-user setting.** It is intended for
  personal research, red-teaming, and evaluation purposes.
- **You, the user, are solely responsible for any output the model produces
  and for any consequences of using this adapter.** The authors of this
  release, of heretic-gguf, of Heretic, of Unsloth, and of Z.ai accept
  no liability whatsoever. Using this adapter to produce illegal content
  or to harm others is your choice and your legal exposure — ensure your use
  complies with all applicable laws in your jurisdiction.
- By downloading or using this adapter you acknowledge the above.

## License

The base model is MIT-licensed (see the
[base repo](https://huggingface.co/zai-org/GLM-5.3-Flash));
this adapter inherits those terms. The heretic-gguf tooling used to produce
it is AGPL-3.0-or-later.
