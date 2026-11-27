#!/usr/bin/env bash
# smoke_test.sh — wait for the endpoint, then verify the behaviours that actually break.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/glm.env" ]]; then set -a; . "$HERE/glm.env"; set +a; fi
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-8000}/v1}"
WAIT="${WAIT:-1800}"   # 30 min: first FP8 start JITs DeepGEMM, big models load slowly

echo "→ waiting for $BASE_URL/models (up to $((WAIT / 60)) min)"
deadline=$(( $(date +%s) + WAIT ))
until curl -fsS -m 5 "$BASE_URL/models" >/dev/null 2>&1; do
  if (( $(date +%s) > deadline )); then
    echo "error: endpoint never came up. Check the serve log:" >&2
    echo "  · 'unsupported architecture glm5_next' → install vLLM nightly: ./install.sh --nightly" >&2
    echo "  · DeepGEMM / nvcc error            → CUDA toolkit missing, or use the docker path" >&2
    echo "  · CUDA out of memory               → lower MAX_LEN in glm.env, or drop a tier" >&2
    exit 1
  fi
  printf '.'
  sleep 5
done
echo " up"

exec "$HERE/glm_client.py" --check --base-url "$BASE_URL" "$@"
