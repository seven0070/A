"""OMEGA-GLM Unified OpenAI-Compatible & ASI Creator HTTP Server
================================================================
Provides:
1. `/v1/chat/completions` — OpenAI-compatible chat & tool-calling endpoint backed by
   GLM-5.3-Flash-Uncensored (auto-detects local GPU NVFP4 engine, llama.cpp GGUF+LoRA,
   remote API, or the Neuro-Symbolic Mathematical Synthesis Kernel).
2. `/v1/asi/create_agent` — Synthesizes, formally verifies (via PUCT MCTS + Process Reward
   Model sandbox execution), and spawns new specialized AGI/ASI sub-agents.
3. `/v1/asi/self_improve` — Triggers a Darwin-Gödel recursive self-improvement cycle.
4. `/v1/asi/status` — Returns real-time GLM-5.3-Flash-Uncensored architecture telemetry,
   4-wide mHC Birkhoff polytope matrices, and spawned agent registry.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from core_engine import OmegaAGIASICreatorEngine


ENGINE = OmegaAGIASICreatorEngine()


class OmegaASIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path in ("/", "/v1/asi/status", "/v1/models"):
            self._send_json(
                200,
                {
                    "engine": "OMEGA-GLM AGI/ASI Creator Engine",
                    "backbone": asdict(ENGINE.backbone.profile),
                    "generations_spawned": ENGINE.generation_counter,
                    "registered_agents": [
                        {
                            "agent_id": spec.agent_id,
                            "name": spec.name,
                            "domain": spec.domain,
                            "prm_score": spec.prm_verification["prm_score"],
                            "passed_execution": spec.prm_verification["passed_execution"],
                        }
                        for spec in ENGINE.registry.values()
                    ],
                },
            )
        else:
            self._send_json(404, {"error": f"Unknown endpoint {self.path}"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/v1/chat/completions":
            msgs = body.get("messages", [])
            user_prompt = msgs[-1].get("content", "") if msgs else ""
            sys_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "You are OMEGA-GLM.")
            res = ENGINE.backbone.generate(
                prompt=str(user_prompt),
                system_prompt=str(sys_prompt),
                temperature=float(body.get("temperature", 0.2)),
                max_tokens=int(body.get("max_tokens", 1024)),
            )
            self._send_json(
                200,
                {
                    "id": "chatcmpl-omega-glm53",
                    "object": "chat.completion",
                    "model": ENGINE.backbone.profile.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": res.get("reasoning", ""),
                                "content": res.get("content", ""),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"latency_s": res.get("latency_s", 0.0), "backend": res.get("backend", "")},
                },
            )
            return

        if self.path == "/v1/asi/create_agent":
            spec = ENGINE.create_specialized_agent(
                name=body.get("name", "Quantum Theorem Synthesizer"),
                domain=body.get("domain", "Theoretical Physics & Formal Verification"),
                objective=body.get("objective", "Synthesize and verify symplectic conservation invariants."),
            )
            self._send_json(200, asdict(spec))
            return

        if self.path == "/v1/asi/self_improve":
            cycles = int(body.get("cycles", 3))
            report = ENGINE.recursive_self_improve(cycles=cycles)
            self._send_json(200, report)
            return

        if self.path == "/v1/agi/build":
            from agi_builder import UniversalAGIBuilder
            builder = UniversalAGIBuilder()
            bp = builder.build_agi(
                name=body.get("name", "Atlas-GLM-AGI-v1"),
                specialization_profile=body.get("profile", "Universal General Intelligence"),
                test_prompt=body.get("prompt", "Unify causal do-calculus counterfactuals with symplectic conservation laws."),
            )
            self._send_json(200, asdict(bp))
            return

        if self.path == "/v1/swarm/execute":
            from swarm_agent import ThermodynamicSwarmOrchestrator
            swarm = ThermodynamicSwarmOrchestrator()
            rep = swarm.execute_swarm_mission(
                mission=body.get("mission", "Unify symplectic physics, Birkhoff spectral stability, causal do-calculus, and KDA fast-weight plasticity.")
            )
            self._send_json(200, rep)
            return

        self._send_json(404, {"error": f"Unknown POST endpoint {self.path}"})


def run_server(host: str = "127.0.0.1", port: int = 8090) -> None:
    srv = ThreadingHTTPServer((host, port), OmegaASIHandler)
    print(f"[OMEGA-GLM ASI Server] Listening on http://{host}:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("OMEGA_PORT", "8090"))
    run_server(port=port)
