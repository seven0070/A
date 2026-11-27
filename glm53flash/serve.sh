#!/usr/bin/env bash
# serve.sh — start GLM 5.3 Flash Uncensored with the settings the model cards require.
# Reads glm.env written by install.sh. Binds 0.0.0.0 so it works behind a proxy.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVFILE="$HERE/glm.env"
[[ -f "$ENVFILE" ]] || { echo "run ./install.sh first (no $ENVFILE)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENVFILE"

: "${TIER:?}" "${RUNTIME:?}" "${WEIGHTS:?}" "${TP_SIZE:=1}" "${PORT:=8000}" "${MAX_LEN:=131072}"

export VLLM_NO_USAGE_STATS=1
# Long-context MoE on multi-GPU: keep the KV cache from fragmenting.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

case "$RUNTIME" in
  vllm)
    MODEL_DIR="$WEIGHTS/${REPO##*/}"
    if [[ ! -d "$MODEL_DIR" ]]; then echo "missing weights at $MODEL_DIR" >&2; exit 1; fi

    echo "→ vLLM · tier=$TIER · tp=$TP_SIZE · port=$PORT · max_len=$MAX_LEN"
    echo "  first start JITs a DeepGEMM block-FP8 kernel (needs nvcc) — allow a few minutes"

    exec vllm serve "$MODEL_DIR" \
      --served-model-name glm-5.3-flash-uncensored \
      --host 0.0.0.0 --port "$PORT" \
      --tensor-parallel-size "$TP_SIZE" \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
      --limit-mm-per-prompt '{"image":4,"video":1}' \
      ${EXTRA_VLLM_ARGS:-}
    ;;

  llamacpp)
    SERVER="$HERE/llama.cpp/build/bin/llama-server"
    if [[ ! -x "$SERVER" ]]; then echo "missing $SERVER — re-run install.sh" >&2; exit 1; fi

    MODEL="$(find "$WEIGHTS/glm53flash-gguf" -name "*${QUANT}*-00001-of-*.gguf" -o -name "*${QUANT}*.gguf" \
             | sort | head -1)"
    if [[ -z "$MODEL" ]]; then echo "no *${QUANT}*.gguf found under $WEIGHTS/glm53flash-gguf" >&2; exit 1; fi
    MMPROJ="$(find "$WEIGHTS/glm53flash-gguf" -name "*mmproj*.gguf" | head -1 || true)"

    echo "→ llama.cpp · quant=$QUANT · model=$(basename "$MODEL") · port=$PORT"

    # -ngl 999: offload everything; drop it for CPU-only (use -ngl 0 + --no-kv-offload).
    ARGS=(-m "$MODEL" --host 0.0.0.0 --port "$PORT" -c "$MAX_LEN" -ngl 999
          --jinja --temp 1.0 --top-p 0.95 --alias glm-5.3-flash-uncensored)
    if [[ -n "$MMPROJ" ]]; then ARGS+=(--mmproj "$MMPROJ"); fi
    exec "$SERVER" "${ARGS[@]}" ${EXTRA_LLAMA_ARGS:-}
    ;;

  *) echo "unknown RUNTIME '$RUNTIME' in $ENVFILE" >&2; exit 1 ;;
esac
