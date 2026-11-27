#!/usr/bin/env bash
# tests/run_tests.sh — proves the installer/client logic without a GPU or weights.
#   · install.sh tier selection + dry-run plans (fake 4xH200 and fake 2x4090)
#   · serve.sh emits the exact required vLLM flags (stubbed vllm binary)
#   · glm_client.py payload rules + a full --check against tests/mock_vllm.py
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
PORT="${TEST_PORT:-8231}"
MOCK_PID=""
pass=0; fail=0

cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

ok()   { pass=$((pass+1)); printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

# ---------------------------------------------------------------- fake hardware
mkfake() { # $1=dir $2=name $3=vram_mb $4=count $5=disk_gb
  mkdir -p "$1"
  cat > "$1/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  *compute_cap*) echo "9.0" ;;
  *memory.total*) for _ in \$(seq $4); do echo "$3"; done ;;
  *name*)        for _ in \$(seq $4); do echo "$2"; done ;;
  *) echo "$2" ;;
esac
EOF
  cat > "$1/df" <<EOF
#!/usr/bin/env bash
echo "Filesystem 1G-blocks Used Available Use% Mounted on"
echo "/dev/fake 2000 10 $5 1% /"
EOF
  cat > "$1/awk" <<'EOF'
#!/usr/bin/env bash
# pass through, except rewrite MemTotal to look like a 512 GB host
# print the bare number: arithmetic context would otherwise try to evaluate "MemTotal"
if [[ "$*" == *MemTotal* ]]; then echo 536870912; exit 0; fi
exec /usr/bin/awk "$@"
EOF
  chmod +x "$1/nvidia-smi" "$1/df" "$1/awk"
}

echo "1) tier selection and dry-run plans"
mkfake "$TMP/big" "NVIDIA H200" 143771 4 900
out="$(PATH="$TMP/big:$PATH" "$ROOT/install.sh" --dry-run 2>&1)" && rc=0 || rc=$?
check "4xH200 auto-selects fp8"            "grep -q 'tier        : fp8' <<<\"\$out\""
check "fp8 plan quotes the canonical repo" "grep -q 'dealignai/GLM-5.3-Flash-UNCENSORED-FP8' <<<\"\$out\""
check "plan reports the right size"        "grep -q '321 GB' <<<\"\$out\""

mkfake "$TMP/mid" "NVIDIA B200" 183359 1 900
out="$(PATH="$TMP/mid:$PATH" "$ROOT/install.sh" --dry-run 2>&1)" && rc=0 || rc=$?
check "1xB200 auto-selects nvfp4"          "grep -q 'tier        : nvfp4' <<<\"\$out\""
check "nvfp4 plan reports 165 GB"          "grep -q '165 GB' <<<\"\$out\""

mkfake "$TMP/small" "NVIDIA RTX 4090" 24564 2 900
out="$(PATH="$TMP/small:$PATH" "$ROOT/install.sh" --dry-run 2>&1)" && rc=0 || rc=$?
check "2x4090 (48 GB) recommends q2/exit-2 path" "grep -q 'cannot hold even the Q2' <<<\"\$out\" || grep -q 'tier        : gguf' <<<\"\$out\""

mkfake "$TMP/tiny" "none" 0 0 20
"$ROOT/install.sh" --dry-run >/dev/null 2>&1 && rc=0 || rc=$?
check "no GPU exits 2 with hosted-route advice" "test $rc -eq 2"

"$ROOT/install.sh" --tier nope --dry-run >/dev/null 2>&1 && rc=0 || rc=$?
check "invalid tier rejected (exit 1)"      "test $rc -eq 1"

# ---------------------------------------------------------------- serve.sh flags
echo "2) serve.sh flag emission (stubbed vllm)"
mkdir -p "$TMP/fakebin" "$TMP/weights/GLM-5.3-Flash-UNCENSORED-FP8"
cat > "$TMP/fakebin/vllm" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$VLLM_ARGS_OUT"
EOF
chmod +x "$TMP/fakebin/vllm"

cat > "$TMP/env" <<EOF
TIER=nvfp4
REPO=dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4
RUNTIME=vllm
QUANT=
WEIGHTS=$TMP/weights
TP_SIZE=4
N_GPU=4
VRAM_GB=575
PORT=$PORT
MAX_LEN=131072
EOF
mkdir -p "$TMP/weights/GLM-5.3-Flash-UNCENSORED-NVFP4"
cp "$ROOT/serve.sh" "$TMP/serve.sh"; cp "$TMP/env" "$TMP/glm.env"
VLLM_ARGS_OUT="$TMP/args" PATH="$TMP/fakebin:$PATH" bash "$TMP/serve.sh" >/dev/null 2>&1 || true
args="$(cat "$TMP/args" 2>/dev/null || true)"
check "passes --tool-call-parser glm47"       "grep -qx -- '--tool-call-parser' <<<\"\$args\""
check "passes --reasoning-parser glm45"       "grep -qx -- '--reasoning-parser' <<<\"\$args\""
check "passes --enable-auto-tool-choice"      "grep -qx -- '--enable-auto-tool-choice' <<<\"\$args\""
check "passes MTP speculative config"         "grep -q 'num_speculative_tokens' <<<\"\$args\""
check "passes --tensor-parallel-size"         "grep -qx -- '--tensor-parallel-size' <<<\"\$args\""
check "binds 0.0.0.0"                         "grep -qx -- '0.0.0.0' <<<\"\$args\""

# ---------------------------------------------------------------- client payloads
echo "3) client payload rules"
python3 - "$ROOT" <<'PY' && ok "payload rules" || bad "payload rules"
import importlib.util, sys
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("c", f"{root}/glm_client.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
p = c.build_payload("hi", model="m", effort=None, max_tokens=200, image=None, tools=False, system=None)
assert p["max_tokens"] == 6000, "max-effort budget guard"
assert "reasoning_effort" not in p["chat_template_kwargs"]
assert p["chat_template_kwargs"]["clear_thinking"] is True
assert "enable_thinking" not in p
p2 = c.build_payload("hi", model="m", effort="low", max_tokens=200, image=None, tools=False, system=None)
assert p2["max_tokens"] == 200 and p2["chat_template_kwargs"]["reasoning_effort"] == "low"
try:
    c.build_payload("hi", model="m", effort="medium", max_tokens=1, image=None, tools=False, system=None)
except SystemExit:
    pass
else:
    raise AssertionError("effort=medium must be rejected")
assert c.reasoning_of({"reasoning_content": "x"}) == "x"
PY

# ---------------------------------------------------------------- end-to-end check
echo "4) glm_client --check against mock vLLM"
python3 "$HERE/mock_vllm.py" --port "$PORT" >/dev/null 2>&1 &
MOCK_PID=$!
for _ in $(seq 40); do curl -fsS -m 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 0.25; done
out="$("$ROOT/glm_client.py" --check --base-url "http://127.0.0.1:$PORT/v1" 2>&1)" && rc=0 || rc=$?
check "smoke suite exits 0"        "test $rc -eq 0"
check "detects non-empty content"  "grep -q 'PASS  non-empty content' <<<\"\$out\""
check "detects reasoning field"    "grep -q 'PASS  parsed reasoning' <<<\"\$out\""
check "detects tool calls"         "grep -q 'PASS  tool_calls emitted' <<<\"\$out\""

# empty-reply diagnosis: hit the mock with max effort and a tiny budget, bypassing the guard
python3 - "$PORT" <<'PY' && ok "diagnoses empty reply at max effort" || bad "diagnoses empty reply at max effort"
import json, sys, urllib.request
port = sys.argv[1]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 200}).encode(),
    headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=10).read())
ch = d["choices"][0]
assert ch["message"]["content"] == "", ch
assert ch["finish_reason"] == "length", ch
PY

echo
echo "  $pass passed, $fail failed"
[[ $fail -eq 0 ]]
