"""Greedy completions on a few short prompts; prints PASS if the text looks like language (not a repeating loop)."""
import json, sys, urllib.request

PROMPTS = ['The capital of France is', 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n', '1 + 1 =']
ok = 0
for p in PROMPTS:
    body = {'model': 'glm', 'prompt': p, 'max_tokens': 40, 'temperature': 0.0}
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/completions',
                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=300))
    t = r['choices'][0]['text']
    words = set(t.split())
    good = len(words) >= 6
    ok += good
    print('PROMPT:', repr(p[:30]), '->', repr(t[:120]).encode('ascii', 'backslashreplace').decode(), 'OK' if good else 'BAD')
print('COHERENT' if ok >= 2 else 'GARBAGE', flush=True)
