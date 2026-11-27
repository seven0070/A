#!/usr/bin/env python3
"""tests/mock_vllm.py — a 60-line fake of the GLM 5.3 Flash vLLM endpoint.

Lets you exercise glm_client.py (and CI) with no GPU and no weights. It reproduces
the four behaviours that trip people up on the real server:

  1. reasoning text arrives in `message.reasoning`, not `content`
  2. omitting reasoning_effort => "max", and a small max_tokens => EMPTY content
     with finish_reason="length"
  3. tools produce `message.tool_calls`
  4. image parts are accepted and ignored

Run:  python3 tests/mock_vllm.py --port 8000
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "glm-5.3-flash-uncensored"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return

        kwargs = req.get("chat_template_kwargs") or {}
        effort = kwargs.get("reasoning_effort")  # None => template resolves to "max"
        max_tokens = int(req.get("max_tokens") or 0)
        messages = req.get("messages") or []
        last = messages[-1] if messages else {}
        parts = last.get("content")
        has_image = isinstance(parts, list) and any(p.get("type") == "image_url" for p in parts)
        prompt = (
            " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if isinstance(parts, list)
            else (parts or "")
        )

        if req.get("tools"):
            msg = {
                "role": "assistant",
                "content": "",
                "reasoning": "The user asks about weather; call the weather tool.",
                "tool_calls": [{
                    "id": "call_mock1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Bengaluru\"}"},
                }],
            }
            self._send(200, self._completion(msg, "tool_calls", req))
            return

        if effort is None and max_tokens < 6000:
            # exactly what the real template does: all budget consumed inside <think>
            msg = {"role": "assistant", "content": "", "reasoning": "thinking" * 50}
            self._send(200, self._completion(msg, "length", req))
            return

        if has_image:
            text = "A bar chart of monthly revenue, rising from left to right."
        elif "ready" in prompt.lower():
            text = "ready"
        elif "moe" in prompt.lower():
            text = "Only 8 of 288 routed experts fire per token, so decode cost tracks active FLOPs, not total parameters."
        elif "weather" in prompt.lower():
            text = "It is 24°C and clear in Bengaluru."
        else:
            text = f"(mock reply) {prompt[:120]}"

        msg = {"role": "assistant", "content": text, "reasoning": "short plan: answer directly."}
        self._send(200, self._completion(msg, "stop", req))

    @staticmethod
    def _completion(msg: dict, finish: str, req: dict) -> dict:
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": req.get("model", MODEL),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 32, "completion_tokens": 48, "total_tokens": 80},
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock vLLM on http://{args.host}:{args.port}/v1  (model: {MODEL})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
