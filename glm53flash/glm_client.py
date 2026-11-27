#!/usr/bin/env python3
"""glm_client.py — minimal, dependency-free client for GLM 5.3 Flash Uncensored.

Encodes the serving gotchas the model cards document, so you don't rediscover them:

  * `reasoning_effort` accepts ONLY "low" and "high". Anything else (including
    omitting it) silently resolves to "max", the deepest mode.
  * At max effort a small `max_tokens` returns an EMPTY reply (finish_reason=length)
    because the whole budget is spent inside <think>. This client auto-raises
    max_tokens to >= 6000 at max effort and tells you when it happens.
  * Do NOT send `enable_thinking` — the template has no such variable. Passing it
    (or a top-level reasoning_effort="none") disables the parser while the model
    still thinks, dumping raw reasoning into content.
  * Reasoning comes back in `message.reasoning` (vLLM), not `reasoning_content`.
  * `clear_thinking` must be nested inside `chat_template_kwargs`.

Usage:
  ./glm_client.py --check                       # 5-probe smoke test against the server
  ./glm_client.py -p "explain MoE routing"
  ./glm_client.py -p "what's in this chart?" --image chart.png --effort high
  ./glm_client.py -p "..." --tools               # demo a function call
  ./glm_client.py --check --base-url https://your-host/v1 --api-key sk-...
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "glm-5.3-flash-uncensored"
MAX_TOKENS_AT_MAX_EFFORT = 6000


def post(base_url: str, path: str, payload: dict, api_key: str | None, timeout: int = 600) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:2000]
        raise SystemExit(f"HTTP {exc.code} from {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach {url}: {exc.reason}\n"
            "Is the server up?  ./serve.sh   (first start can take minutes)"
        ) from exc


def get(base_url: str, path: str, api_key: str | None) -> dict:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        raise SystemExit(f"cannot reach {url}: {exc}") from exc


def build_payload(
    prompt: str,
    *,
    model: str,
    effort: str | None,
    max_tokens: int,
    image: str | None,
    tools: bool,
    system: str | None,
) -> dict:
    if effort is not None and effort not in ("low", "high"):
        raise SystemExit(f'--effort must be "low" or "high" (got {effort!r}).')
    # No effort given => the template resolves to "max", so budget for thinking.
    effective_max = max(max_tokens, MAX_TOKENS_AT_MAX_EFFORT) if effort is None else max_tokens

    content: str | list[dict] = prompt
    if image:
        mime = mimetypes.guess_type(image)[0] or "image/png"
        b64 = base64.b64encode(open(image, "rb").read()).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    template_kwargs: dict = {"clear_thinking": True}
    if effort is not None:
        template_kwargs["reasoning_effort"] = effort

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": effective_max,
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": template_kwargs,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        payload["tool_choice"] = "auto"
    return payload


def reasoning_of(message: dict) -> str:
    """vLLM returns `reasoning`; older builds and OpenAI-compat layers use `reasoning_content`."""
    return message.get("reasoning") or message.get("reasoning_content") or ""


def one_shot(args, prompt: str, *, tools: bool = False, image: str | None = None) -> dict:
    payload = build_payload(
        prompt,
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
        image=image,
        tools=tools,
        system=args.system,
    )
    data = post(args.base_url, "/chat/completions", payload, args.api_key)
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": reasoning_of(msg).strip(),
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def check(args) -> int:
    failures = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            failures += 1

    print("1) server + model id")
    try:
        models = get(args.base_url, "/models", args.api_key)
        ids = [m.get("id", "") for m in models.get("data", [])]
        report("GET /models", bool(ids), ", ".join(ids[:4]) or "no models listed")
    except SystemExit as exc:
        print(f"  FAIL  GET /models  — {exc}")
        return 1

    print("2) plain completion, default (max) effort")
    out = one_shot(args, "Reply with exactly the word: ready")
    empty = not out["content"]
    report("non-empty content", not empty,
           "empty reply: budget spent inside <think> — raise --max-tokens"
           if empty else out["content"][:80].replace("\n", " "))
    report("parsed reasoning into .reasoning", bool(out["reasoning"]),
           f"{len(out['reasoning'])} chars" if out["reasoning"] else "no reasoning field (parser may be off)")
    report("finish_reason != length", out["finish_reason"] != "length", str(out["finish_reason"]))

    print("3) low effort (cheap, and the least-uncensored mode by design)")
    low = one_shot_with_effort(args, "Give one sentence on why MoE routing is cheap.", "low")
    report("responded", bool(low["content"]), f"{len(low['content'])} chars")

    print("4) tool calling")
    tooled = one_shot(args, "What's the weather in Bengaluru? Use the tool.", tools=True)
    report("tool_calls emitted", bool(tooled["tool_calls"]),
           json.dumps(tooled["tool_calls"])[:120] if tooled["tool_calls"] else
           "no tool call — check --tool-call-parser glm47 --enable-auto-tool-choice")

    print("5) vision (skipped without --image)")
    if args.image:
        vis = one_shot(args, "Describe this image in one sentence.", image=args.image)
        report("vision response", bool(vis["content"]), vis["content"][:80])
    else:
        print("  SKIP  pass --image path/to.png to test the vision tower")

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) failed'}"
          f" · usage last call: {out['usage']}")
    return 0 if failures == 0 else 1


def one_shot_with_effort(args, prompt: str, effort: str) -> dict:
    payload = build_payload(
        prompt, model=args.model, effort=effort, max_tokens=args.max_tokens,
        image=None, tools=False, system=args.system,
    )
    data = post(args.base_url, "/chat/completions", payload, args.api_key)
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": reasoning_of(msg).strip(),
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="client for GLM 5.3 Flash Uncensored")
    ap.add_argument("-p", "--prompt", help="prompt to send")
    ap.add_argument("--image", help="image path for a vision probe")
    ap.add_argument("--tools", action="store_true", help="advertise a demo weather tool")
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--effort", choices=["low", "high"], help="low/high; omit for max")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--check", action="store_true", help="run the 5-probe smoke test")
    ap.add_argument("--quiet-reasoning", action="store_true", help="hide the thinking trace")
    args = ap.parse_args()

    if args.check:
        return check(args)
    if not args.prompt:
        ap.error("need -p/--prompt or --check")

    out = one_shot(args, args.prompt, tools=args.tools, image=args.image)
    if out["reasoning"] and not args.quiet_reasoning:
        print(f"[reasoning · {len(out['reasoning'])} chars]\n{out['reasoning'][:1200]}\n")
    print(out["content"] or "(empty reply — raise --max-tokens)")
    if out["tool_calls"]:
        print("\n[tool_calls]\n" + json.dumps(out["tool_calls"], indent=2))
    if out["finish_reason"] == "length":
        print("\n[warning] hit max_tokens; at max effort use >= 6000", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
