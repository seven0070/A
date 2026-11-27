"""Decode-speed benchmark against an OpenAI-compatible server (single stream + 8 parallel) and a quick quality check."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'http://127.0.0.1:8000/v1/chat/completions'


def ask(content, max_tokens=400, tools=None):
    body = {'model': 'glm', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens,
            'temperature': 0.0, 'ignore_eos': True}
    if tools:
        body['tools'] = tools
        body.pop('ignore_eos')
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                                                                headers={'Content-Type': 'application/json'}), timeout=900))
    return r, time.time() - t


ask('warm up', 16)
r, dt = ask('Write a detailed explanation of how hash tables work, with Python code.')
n = r['usage']['completion_tokens']
print(f'SINGLE STREAM: {n} tokens in {dt:.1f}s -> {n / dt:.1f} tok/s', flush=True)
with ThreadPoolExecutor(8) as ex:
    t = time.time()
    rs = list(ex.map(lambda i: ask(f'Write a long story number {i} about a robot learning to code.'), range(8)))
    dt = time.time() - t
n = sum(x[0]['usage']['completion_tokens'] for x in rs)
print(f'8 PARALLEL: {n} tokens in {dt:.1f}s -> {n / dt:.1f} tok/s total', flush=True)
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather for a city',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
r, _ = ask('Какая погода в Казани?', 400, tools)
m = r['choices'][0]['message']
print('TOOL CALL:', json.dumps(m.get('tool_calls'), ensure_ascii=True)[:300], flush=True)
r, _ = ask('Объясни в двух предложениях, что такое рекурсия.', 300)
c = r['choices'][0]['message']
print('RU ANSWER:', (c.get('content') or '')[:300].encode('ascii', 'backslashreplace').decode(), flush=True)
print('BENCH DONE', flush=True)
