# Executed inside the running daemon (variable `eng` in scope): starts an OpenAI-style HTTP API on :8000
import json, re, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def parse_output(text):
    reasoning, content = '', text
    if '</think>' in text:
        reasoning, content = text.split('</think>', 1)
        reasoning = reasoning.replace('<think>', '').strip()
    calls = []
    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.S):
        body = m.group(1)
        name = re.match(r'\s*([^<\s]+)', body).group(1)
        args = {}
        for k, v in re.findall(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', body, re.S):
            try:
                args[k.strip()] = json.loads(v)
            except Exception:
                args[k.strip()] = v
        calls.append({'id': 'call_' + uuid.uuid4().hex[:8], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.S).strip()
    return reasoning, content, calls


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {'status': 'ok', 'model': 'glm-5.3-flash-agentic-nvfp4'})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            t0 = time.time()
            with LOCK:
                raw, st, n_prompt = eng.chat(req['messages'], tools=req.get('tools'),
                                             max_new=int(req.get('max_tokens', 512)),
                                             temperature=float(req.get('temperature', 0.0)))
            reasoning, content, calls = parse_output(raw)
            msg = {'role': 'assistant', 'content': content}
            if reasoning:
                msg['reasoning_content'] = reasoning
            if calls:
                msg['tool_calls'] = calls
            self._send(200, {'id': 'chatcmpl-' + uuid.uuid4().hex[:8], 'object': 'chat.completion', 'model': 'glm-5.3-flash',
                             'choices': [{'index': 0, 'message': msg, 'finish_reason': 'tool_calls' if calls else 'stop'}],
                             'usage': {'prompt_tokens': n_prompt, 'tok_s': st.get('tok_s'), 'seconds': round(time.time() - t0, 1)}})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(500, {'error': str(e)})


srv = ThreadingHTTPServer(('127.0.0.1', 8000), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print('server up on 127.0.0.1:8000')
