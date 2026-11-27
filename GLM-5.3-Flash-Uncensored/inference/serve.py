"""Serve the pruned model fully on GPU with the custom engine (OpenAI-style API on :8000)."""
import sys, time

sys.path.insert(0, '/tmp/work')
from engine import Engine

eng = Engine(sys.argv[1] if len(sys.argv) > 1 else '/tmp/pruned', gpu_layers=99, log=lambda *a: print(*a, flush=True))
g = {'eng': eng}
exec(open('/tmp/work/server_cmd.py').read(), g)
print('SERVING', flush=True)
while True:
    time.sleep(3600)
