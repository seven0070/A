# GLM 5.3 Flash Uncensored (Smallest Editions — Repo & Google Drive Co-Shared)

This repository and its co-shared Google Drive workspace contain the **smallest editions** of **GLM 5.3 Flash Uncensored** (based on Z.ai's `GLM-5.3-Flash`, 320B total / 18B active Mixture-of-Experts with 1M context window and hybrid linear + sparse attention).

## 🔗 Co-Shared Locations (Both Sides)

- **GitHub Repository**: [`seven0070/A`](https://github.com/seven0070/A) (`/GLM-5.3-Flash-Uncensored`)
- **Google Drive Workspace**: [`GLM-5.3-Flash-Uncensored` (`1Ki1iVbSUX2niQw_j0OkyQIHzqda6eWVO`)](https://drive.google.com/drive/folders/1Ki1iVbSUX2niQw_j0OkyQIHzqda6eWVO)
  - `adapters/` (`1-W9relsFcVMaNrKIT85V2Dqc-ptXfKQz`)
  - `inference/` (`1B24kNwkR-umwmmLklvDV_B1p0O-4nYR0`)
  - `gguf/` (`1lNywtpdgpnLat8o9YVBetV1W_iGchshp`)

---

## 📦 Included Smallest Model Variants

Because the unquantized `GLM-5.3-Flash` base model is over 320 GB, we selected and packaged the **three smallest uncensored tiers** so you can run or deploy it at any scale:

### 1. Ultra-Lightweight Uncensored LoRA Adapters (Included Directly in Repo & Drive)
Located in [`GLM-5.3-Flash-Uncensored/adapters/`](GLM-5.3-Flash-Uncensored/adapters/):
- **`abliteration-lora.gguf` (7.3 MB)** & **`abliteration-edit.npz` (35 KB)** (`mrfakename/GLM-5.3-Flash-abliterated-GGUF`): Rank-2 directional refusal ablation LoRA for `llama.cpp`.
- **`glm-5.3-heretic-lora-v3.gguf` (65 MB)** (`MorinoNushi/GLM-5.3-Flash-Heretic-Abliterated-LoRA-V3-GGUF`): Optuna-optimized rank-1 Heretic uncensored GGUF LoRA adapter compatible with mainline `llama.cpp`.

**Run with `llama.cpp`:**
```bash
llama-server \
  -m GLM-5.3-Flash-IQ1S-00001-of-00002.gguf \
  --lora GLM-5.3-Flash-Uncensored/adapters/glm-5.3-heretic-lora-v3.gguf \
  --jinja
```

### 2. Smallest Full Standalone Model — Pruned NVFP4 (62 GiB, 92/288 Experts)
Repository configs, tokenizer, shard index, and bundled inference engine from `Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB` (~107B total / ~18B active parameters, 62.0 GiB on disk — fits a single GPU).
- Configs & Tokenizer: [`config.json`](GLM-5.3-Flash-Uncensored/config.json), [`tokenizer.json`](GLM-5.3-Flash-Uncensored/tokenizer.json), [`chat_template.jinja`](GLM-5.3-Flash-Uncensored/chat_template.jinja), [`keep.json`](GLM-5.3-Flash-Uncensored/keep.json), [`model.safetensors.index.json`](GLM-5.3-Flash-Uncensored/model.safetensors.index.json)
- Bundled Engine & vLLM Converter: [`GLM-5.3-Flash-Uncensored/inference/`](GLM-5.3-Flash-Uncensored/inference/)

**Download all 62 GiB shards & serve locally:**
```bash
./GLM-5.3-Flash-Uncensored/sync_and_install.sh pruned-62gb
python3 GLM-5.3-Flash-Uncensored/inference/serve.py GLM-5.3-Flash-Uncensored
```

### 3. Smallest Full 320B GGUF Quantization — `IQ1_S` (~71.5 GiB)
Metadata and checksums in [`GLM-5.3-Flash-Uncensored/gguf/`](GLM-5.3-Flash-Uncensored/gguf/) from `BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF` (2 shards + `mmproj-GLM-5.3-Flash-F16.gguf` vision projector).

**Download & run with `llama.cpp`:**
```bash
./GLM-5.3-Flash-Uncensored/sync_and_install.sh iq1s-gguf
llama-server -m GLM-5.3-Flash-Uncensored/gguf/GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf --jinja
```

---

## 🔄 Syncing Between Repo and Google Drive

Use [`GLM-5.3-Flash-Uncensored/sync_and_install.sh`](GLM-5.3-Flash-Uncensored/sync_and_install.sh) to pull weights or push updates between the repository and Google Drive:

```bash
# Sync local files and any downloaded shards to the Google Drive folder
./GLM-5.3-Flash-Uncensored/sync_and_install.sh sync-to-drive

# Inspect files stored on the Google Drive side
./GLM-5.3-Flash-Uncensored/sync_and_install.sh pull-from-drive
```
