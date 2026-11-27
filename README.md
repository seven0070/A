# A

Reference notes and tooling for uncensored open-weight LLMs (snapshot: Nov 2026).

| File | What it is |
|---|---|
| [`uncensored-models-2026-11.md`](uncensored-models-2026-11.md) | Ranked list of uncensored/abliterated models that keep their base capability, with refusal + benchmark numbers per build |
| [`glm53flash/`](glm53flash/) | Installer for #1 on that list: GLM 5.3 Flash Uncensored (320B/18B-active MoE, 1M ctx, vision) — hardware-aware tiers (FP8 / NVFP4 / GGUF / MLX), vLLM + llama.cpp serving, dependency-free client and test suite |

```bash
cd glm53flash && ./install.sh --dry-run   # prints the plan for your hardware
./tests/run_tests.sh                      # 20 checks, no GPU required
```
