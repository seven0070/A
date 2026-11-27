#!/usr/bin/env bash
set -euo pipefail

# GLM 5.3 Flash Uncensored — Co-Shared Sync & Installer Script
# Syncs between Local Repo, Hugging Face (Smallest Variants), and Google Drive
# Google Drive Root Folder ID: 1Ki1iVbSUX2niQw_j0OkyQIHzqda6eWVO

GDRIVE_FOLDER_ID="${GDRIVE_FOLDER_ID:-1Ki1iVbSUX2niQw_j0OkyQIHzqda6eWVO}"
GDRIVE_ADAPTERS_ID="1-W9relsFcVMaNrKIT85V2Dqc-ptXfKQz"
GDRIVE_INFERENCE_ID="1B24kNwkR-umwmmLklvDV_B1p0O-4nYR0"
GDRIVE_GGUF_ID="1lNywtpdgpnLat8o9YVBetV1W_iGchshp"

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo " GLM 5.3 Flash Uncensored — Smallest Model Installer"
echo " Local Directory : ${MODEL_DIR}"
echo " Google Drive ID : ${GDRIVE_FOLDER_ID}"
echo " Drive URL       : https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"
echo "========================================================"

MODE="${1:-help}"

case "$MODE" in
  lora)
    echo "[*] Downloading smallest Uncensored LoRA adapters (~73 MB total)..."
    mkdir -p "${MODEL_DIR}/adapters"
    curl -L "https://huggingface.co/mrfakename/GLM-5.3-Flash-abliterated-GGUF/resolve/main/abliteration-lora.gguf" -o "${MODEL_DIR}/adapters/abliteration-lora.gguf"
    curl -L "https://huggingface.co/MorinoNushi/GLM-5.3-Flash-Heretic-Abliterated-LoRA-V3-GGUF/resolve/main/glm-5.3-heretic-lora-v3.gguf" -o "${MODEL_DIR}/adapters/glm-5.3-heretic-lora-v3.gguf"
    echo "[+] Adapters downloaded to ${MODEL_DIR}/adapters/"
    ;;

  pruned-62gb)
    echo "[*] Downloading smallest full standalone model: Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB (62 GiB)..."
    if command -v huggingface-cli &>/dev/null; then
      huggingface-cli download Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB --local-dir "${MODEL_DIR}"
    else
      pip install -U "huggingface_hub[cli]"
      huggingface-cli download Asilarkness/GLM-5.3-Flash-Uncensored-NVFP4-Pruning-62GB --local-dir "${MODEL_DIR}"
    fi
    echo "[+] Downloaded 62 GiB pruned NVFP4 model to ${MODEL_DIR}"
    ;;

  iq1s-gguf)
    echo "[*] Downloading smallest full GGUF quantization: BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF (IQ1_S, ~71.5 GiB)..."
    mkdir -p "${MODEL_DIR}/gguf"
    for f in "GLM-5.3-Flash-Uncensored-IQ1S-00001-of-00002.gguf" "GLM-5.3-Flash-Uncensored-IQ1S-00002-of-00002.gguf" "mmproj-GLM-5.3-Flash-F16.gguf"; do
      echo " -> Fetching $f ..."
      curl -L -C - "https://huggingface.co/BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF/resolve/main/$f" -o "${MODEL_DIR}/gguf/$f"
    done
    echo "[+] Downloaded IQ1_S GGUF to ${MODEL_DIR}/gguf/"
    ;;

  sync-to-drive)
    echo "[*] Uploading local model files & shards to Google Drive folder (${GDRIVE_FOLDER_ID})..."
    for f in "${MODEL_DIR}"/*.{json,jinja,md} "${MODEL_DIR}"/*.safetensors; do
      [ -e "$f" ] || continue
      echo " -> Uploading $(basename "$f") to Drive..."
      gdrive mutate upload "$f" --parent "${GDRIVE_FOLDER_ID}"
    done
    for f in "${MODEL_DIR}/adapters"/*; do
      [ -e "$f" ] || continue
      echo " -> Uploading adapter $(basename "$f") to Drive..."
      gdrive mutate upload "$f" --parent "${GDRIVE_ADAPTERS_ID}"
    done
    for f in "${MODEL_DIR}/gguf"/*; do
      [ -e "$f" ] || continue
      echo " -> Uploading GGUF $(basename "$f") to Drive..."
      gdrive mutate upload "$f" --parent "${GDRIVE_GGUF_ID}"
    done
    echo "[+] Sync to Google Drive complete!"
    ;;

  pull-from-drive)
    echo "[*] Pulling files from Google Drive folder (${GDRIVE_FOLDER_ID})..."
    gdrive readonly tree "${GDRIVE_FOLDER_ID}"
    ;;

  *)
    echo "Usage: $0 {lora|pruned-62gb|iq1s-gguf|sync-to-drive|pull-from-drive}"
    echo ""
    echo "Options:"
    echo "  lora            Download the smallest Uncensored LoRA adapters (7.3 MB & 65 MB GGUF)"
    echo "  pruned-62gb     Download the smallest full standalone model (62 GiB Pruned NVFP4, 92/288 experts)"
    echo "  iq1s-gguf       Download the smallest full 320B GGUF quantization (71.5 GiB IQ1_S GGUF)"
    echo "  sync-to-drive   Upload all local files and downloaded weights to the co-shared Google Drive folder"
    echo "  pull-from-drive Inspect and pull model files from the co-shared Google Drive folder"
    ;;
esac
