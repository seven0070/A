#!/usr/bin/env bash
# install.sh — GLM 5.3 Flash Uncensored installer with hardware-aware tier selection.
#
#   ./install.sh                 # detect hardware, recommend a tier, ask, install
#   ./install.sh --dry-run       # print the plan, touch nothing
#   ./install.sh --tier nvfp4 -y # non-interactive
#
# Tiers and their real on-disk sizes (from the published repos):
#   fp8      dealignai/GLM-5.3-Flash-UNCENSORED-FP8     321 GB  · vLLM  · 4xH200 class
#   nvfp4    dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4   165 GB  · vLLM  · 1xB200 / 2xH200
#   gguf-q6  orcarouter/GLM-5.3-Flash-Uncensored-GGUF   263 GB  · llama.cpp · near-lossless
#   gguf-q4  orcarouter/GLM-5.3-Flash-Uncensored-GGUF   193 GB  · llama.cpp · best balance
#   gguf-q3  orcarouter/GLM-5.3-Flash-Uncensored-GGUF   153 GB  · llama.cpp
#   gguf-q2  orcarouter/GLM-5.3-Flash-Uncensored-GGUF   117 GB  · llama.cpp · last resort
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
WEIGHTS="$HERE/weights"
ENVFILE="$HERE/glm.env"

TIER=""
EXPLICIT_TIER=0
DRY_RUN=0
ASSUME_YES=0
NIGHTLY=0
PORT="${PORT:-8000}"

REPO_FP8="dealignai/GLM-5.3-Flash-UNCENSORED-FP8"
REPO_NVFP4="dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4"
REPO_GGUF="orcarouter/GLM-5.3-Flash-Uncensored-GGUF"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
info() { printf '\033[36m→\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="${2:?--tier needs a value}"; EXPLICIT_TIER=1; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --nightly) NIGHTLY=1; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# ---------------------------------------------------------------- detect hardware
info "detecting hardware"
N_GPU=0; VRAM_MB=0; GPU_NAME="none"; CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  N_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
             | awk '{s+=$1} END {print s+0}')"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
fi
TOTAL_GB=$(( VRAM_MB / 1024 ))
RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
DISK_FREE_GB="$(df -BG --output=avail "$HERE" 2>/dev/null | tail -1 | tr -dc '0-9')"
[[ -n "$DISK_FREE_GB" ]] || DISK_FREE_GB=0

printf '   gpus: %s x %s (%s GB total VRAM, compute %s)\n' "$N_GPU" "$GPU_NAME" "$TOTAL_GB" "${CC:-?}"
printf '   host: %s GB RAM · %s GB free disk\n' "$RAM_GB" "$DISK_FREE_GB"

# ---------------------------------------------------------------- resolve tier
tier_spec() {
  case "$1" in
    fp8)     NEED_GB=321; REPO="$REPO_FP8";   RUNTIME="vllm";     QUANT="" ;;
    nvfp4)   NEED_GB=165; REPO="$REPO_NVFP4"; RUNTIME="vllm";     QUANT="" ;;
    gguf-q6) NEED_GB=263; REPO="$REPO_GGUF";  RUNTIME="llamacpp"; QUANT="Q6_K" ;;
    gguf-q4) NEED_GB=193; REPO="$REPO_GGUF";  RUNTIME="llamacpp"; QUANT="Q4_K_M" ;;
    gguf-q3) NEED_GB=153; REPO="$REPO_GGUF";  RUNTIME="llamacpp"; QUANT="Q3_K_M" ;;
    gguf-q2) NEED_GB=117; REPO="$REPO_GGUF";  RUNTIME="llamacpp"; QUANT="Q2_K" ;;
    *) die "invalid tier '$1' (fp8|nvfp4|gguf-q6|gguf-q4|gguf-q3|gguf-q2)" ;;
  esac
}

# What this box can actually hold: weights + KV cache + activations headroom.
# Thresholds are GiB of total VRAM (sizes above are decimal GB from the repos):
#   fp8 321 GB needs ~380 GiB with KV · nvfp4 165 GB fits one 179 GiB B200
#   q3 153 GB needs ~150 GiB · q2 117 GB needs ~120 GiB · else mmap from 256 GiB RAM
if   (( TOTAL_GB >= 380 )); then RECOMMEND="fp8"
elif (( TOTAL_GB >= 175 )); then RECOMMEND="nvfp4"
elif (( TOTAL_GB >= 150 )); then RECOMMEND="gguf-q3"
elif (( TOTAL_GB >= 120 )); then RECOMMEND="gguf-q2"
elif (( RAM_GB  >= 256 )); then RECOMMEND="gguf-q2"      # CPU/RAM-only host, mmap the file
else RECOMMEND="api"
fi

if (( EXPLICIT_TIER == 0 )); then
  if [[ "$RECOMMEND" == "api" ]]; then
    warn "this box cannot hold even the Q2 build (117 GB): ${TOTAL_GB} GB VRAM / ${RAM_GB} GB RAM"
    cat <<'EOF'

  What does work from here:
    · Hosted endpoint  — README.md § "No GPU? Hosted route" (2 minutes, no download)
    · Rented node      — 8xH100 → fp8 ; 1xB200 → nvfp4 ; 2xH200 → nvfp4
    · Smaller sibling  — patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF (~90 GB at Q4,
                         50% of experts pruned; quality is not the same model)

  Force a tier anyway with:  ./install.sh --tier gguf-q2
EOF
    exit 2
  fi
  TIER="$RECOMMEND"
  info "recommended tier for this box: $TIER"
fi

tier_spec "$TIER"

if [[ $DRY_RUN -eq 1 ]]; then
  cat <<EOF

  PLAN (dry run — nothing will be downloaded or installed)
    tier        : $TIER
    repo        : $REPO
    runtime     : $RUNTIME$( [[ "$RUNTIME" == llamacpp ]] && printf ' (quant %s)' "$QUANT" )
    download    : ~$NEED_GB GB  (free: $DISK_FREE_GB GB)
    weights     : $WEIGHTS
    venv        : $VENV
    endpoint    : http://0.0.0.0:$PORT/v1
    recommended : $RECOMMEND  (based on ${TOTAL_GB} GB VRAM / ${RAM_GB} GB RAM)
EOF
  if (( NEED_GB + 20 > DISK_FREE_GB )); then
    warn "not enough disk: need ~$((NEED_GB + 20)) GB, have $DISK_FREE_GB GB"
  fi
  if [[ "$RUNTIME" == "vllm" ]] && (( TOTAL_GB < NEED_GB + 20 )); then
    warn "VRAM tight: ~$NEED_GB GB of weights on $TOTAL_GB GB total"
  fi
  exit 0
fi

if [[ "$TIER" != "$RECOMMEND" && "$RECOMMEND" != "api" ]]; then
  warn "you chose $TIER but this box fits $RECOMMEND (${TOTAL_GB} GB VRAM / ${RAM_GB} GB RAM)"
fi

# ---------------------------------------------------------------- confirm + disk
if (( NEED_GB + 20 > DISK_FREE_GB )); then
  die "need ~$((NEED_GB + 20)) GB free on $HERE, only $DISK_FREE_GB GB available.
       Point the installer at a bigger volume:  WEIGHTS_DIR=/mnt/models $0 --tier $TIER"
fi

cat <<EOF

  About to install:
    tier     : $TIER
    repo     : $REPO
    download : ~$NEED_GB GB
    runtime  : $RUNTIME

EOF
if (( ASSUME_YES == 0 )); then
  read -r -p "  Proceed? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || die "aborted"
fi

mkdir -p "$WEIGHTS"

# ---------------------------------------------------------------- dependencies
info "installing downloader + runtime"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; install it manually: https://docs.astral.sh/uv/"

[[ -d "$VENV" ]] || uv venv "$VENV" --python 3.11
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "huggingface_hub[hf_xet]" hf_transfer

if [[ "$RUNTIME" == "vllm" ]]; then
  if (( NIGHTLY == 1 )); then
    info "installing vLLM nightly (needed if stable rejects arch glm5_next)"
    uv pip install -q -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
  else
    uv pip install -q -U vllm --torch-backend=auto
  fi
  command -v nvcc >/dev/null 2>&1 || \
    warn "nvcc not on PATH — DeepGEMM JITs a block-FP8 kernel at startup and will fail without it.
       Install CUDA toolkit, or use a container: vllm/vllm-openai:latest"
else
  info "building llama.cpp (glm5next support: mainline since 2026-09)"
  if [[ ! -d "$HERE/llama.cpp" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HERE/llama.cpp"
  fi
  cmake -S "$HERE/llama.cpp" -B "$HERE/llama.cpp/build" -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
  cmake --build "$HERE/llama.cpp/build" --config Release -j "$(nproc)" \
    --target llama-cli llama-mtmd-cli llama-server
fi

# ---------------------------------------------------------------- download weights
info "downloading $REPO (~$NEED_GB GB) — resumable, re-run to continue"
export HF_HUB_ENABLE_HF_TRANSFER=1
if [[ "$RUNTIME" == "llamacpp" ]]; then
  hf download "$REPO" --local-dir "$WEIGHTS/glm53flash-gguf" \
    --include "*${QUANT}*" "*mmproj*" "*.json"
else
  hf download "$REPO" --local-dir "$WEIGHTS/${REPO##*/}"
fi
ok "weights ready in $WEIGHTS"

# ---------------------------------------------------------------- write config
TP=1
if (( N_GPU > 1 )); then TP="$N_GPU"; fi

cat > "$ENVFILE" <<EOF
# generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
TIER=$TIER
REPO=$REPO
RUNTIME=$RUNTIME
QUANT=${QUANT:-}
WEIGHTS=$WEIGHTS
TP_SIZE=$TP
N_GPU=$N_GPU
VRAM_GB=$TOTAL_GB
PORT=$PORT
MAX_LEN=${MAX_LEN:-131072}
EOF
ok "wrote $ENVFILE"

cat <<EOF

  Installed. Next:

    ./serve.sh              # starts the endpoint on 0.0.0.0:$PORT
    ./smoke_test.sh         # after the server reports ready

  Read README.md § "Serving gotchas" before your first request — the reasoning
  template will silently return empty replies if you get max_tokens wrong.

EOF
