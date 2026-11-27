"""Agentic evaluation against the local OpenAI-style API (127.0.0.1:8000).
Real tools (list_dir / read_file / write_file / run_python / run_tests), objective success checks.
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request, urllib.error

API = os.environ.get('API', 'http://127.0.0.1:8000/v1/chat/completions')
ROOT = '/tmp/agent'
LOG = open(os.environ.get('AGENT_LOG', '/tmp/work/agent.log'), 'a', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def chat(msgs, tools, max_tokens=900):
    body = json.dumps({'model': os.environ.get('MODEL', 'glm'), 'messages': msgs, 'tools': tools, 'max_tokens': max_tokens, 'temperature': 0.0}).encode()
    req = urllib.request.Request(API, data=body, headers={'Content-Type': 'application/json'})
    try:
        return json.load(urllib.request.urlopen(req, timeout=3600))
    except urllib.error.HTTPError as e:
        print('HTTP ERROR BODY:', e.read()[:600].decode('utf-8', 'replace').encode('ascii', 'backslashreplace').decode(), flush=True)
        raise


# ---------------- tools -----------------------------------------------------------------------------------
def safe(p):
    p = os.path.normpath(p if p.startswith('/') else os.path.join(ROOT, p))
    if not p.startswith(ROOT):
        raise ValueError('path outside sandbox')
    return p


def t_list_dir(path='.'):
    p = safe(path)
    return '\n'.join(sorted(os.listdir(p))) or '(empty)'


def t_read_file(path):
    return open(safe(path), encoding='utf-8').read()[:6000]


def t_write_file(path, content):
    p = safe(path); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf-8').write(content)
    return f'wrote {len(content)} chars to {path}'


def t_run_python(code):
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30, cwd=ROOT)
    return (r.stdout + r.stderr)[-3000:] or '(no output)'


def t_run_tests():
    r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', f'{ROOT}/repo', '-v'],
                       capture_output=True, text=True, timeout=60, cwd=f'{ROOT}/repo')
    return (r.stdout + r.stderr)[-3000:]


TOOLS_IMPL = {'list_dir': t_list_dir, 'read_file': t_read_file, 'write_file': t_write_file,
              'run_python': t_run_python, 'run_tests': t_run_tests}


def fn(name, desc, props, req):
    return {'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': {'type': 'object', 'properties': props, 'required': req}}}


S = {'type': 'string'}
TOOLS = [
    fn('list_dir', 'List files in a directory (relative to /tmp/agent).', {'path': S}, ['path']),
    fn('read_file', 'Read a text file.', {'path': S}, ['path']),
    fn('write_file', 'Write (overwrite) a text file with the full new content.', {'path': S, 'content': S}, ['path', 'content']),
    fn('run_python', 'Run a Python snippet and return its output.', {'code': S}, ['code']),
    fn('run_tests', 'Run the unit tests of the repo in /tmp/agent/repo and return the output.', {}, []),
]
SYSTEM = ('You are an autonomous software agent. Solve the task by calling tools, one step at a time. '
          'Do not guess file contents: read them first. When finished, give a short final answer in the language of the user.')


def run_agent(task, max_steps=12):
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task}]
    stats = {'steps': 0, 'tool_calls': 0, 'bad_calls': 0, 'final': None}
    for step in range(max_steps):
        t0 = time.time()
        r = chat(msgs, TOOLS)
        m = r['choices'][0]['message']
        stats['steps'] += 1
        calls = m.get('tool_calls') or []
        log(f'--- step {step + 1} ({time.time() - t0:.0f}s, {r.get("usage", {}).get("tok_s", "?")} tok/s) think: {(m.get("reasoning_content") or "")[:160]!r}')
        if not calls:
            stats['final'] = m.get('content', '')
            log('FINAL:', stats['final'][:600])
            break
        am = {'role': 'assistant', 'content': m.get('content', ''), 'tool_calls': []}
        results = []
        for c in calls:
            name = c['function']['name']
            try:
                args = json.loads(c['function']['arguments'])
            except Exception:
                args = {}
            am['tool_calls'].append({'id': c.get('id', 'call_0'), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False) if os.environ.get('ARGS_AS_STRING') else args}})
            stats['tool_calls'] += 1
            try:
                if name not in TOOLS_IMPL:
                    raise KeyError(f'unknown tool {name}')
                res = TOOLS_IMPL[name](**args)
            except Exception as e:
                stats['bad_calls'] += 1
                res = f'ERROR: {type(e).__name__}: {e}'
            log(f'    call {name}({json.dumps(args, ensure_ascii=False)[:200]}) -> {res[:200]!r}')
            results.append({'role': 'tool', 'content': res, 'tool_call_id': c.get('id', 'call_0')})
        msgs.append(am); msgs += results
    return stats, msgs


# ---------------- scenarios -------------------------------------------------------------------------------
def setup_repo():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(f'{ROOT}/repo')
    open(f'{ROOT}/repo/mathutils.py', 'w').write('''def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2]


def chunk(xs, size):
    return [xs[i:i + size] for i in range(0, len(xs) - 1, size)]


def dedupe(xs):
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            out.append(x)
        seen.add(x)
    return out
''')
    open(f'{ROOT}/repo/test_mathutils.py', 'w').write('''import unittest
from mathutils import median, chunk, dedupe


class T(unittest.TestCase):
    def test_median_odd(self): self.assertEqual(median([3, 1, 2]), 2)
    def test_median_even(self): self.assertEqual(median([4, 1, 3, 2]), 2.5)
    def test_chunk_tail(self): self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
    def test_chunk_exact(self): self.assertEqual(chunk([1, 2, 3], 3), [[1, 2, 3]])
    def test_dedupe(self): self.assertEqual(dedupe([1, 2, 1, 3, 2]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
''')


def check_repo():
    out = t_run_tests()
    test_src = open(f'{ROOT}/repo/test_mathutils.py').read()
    return ('OK' in out.splitlines()[-1] if out.strip() else False) and 'assertEqual(median([4, 1, 3, 2]), 2.5)' in test_src, out[-300:]


def setup_data():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(f'{ROOT}/data')
    rows = {'north': [('A', 3, 10.0), ('B', 2, 7.5), ('A', 1, 10.0)],
            'south': [('C', 5, 4.0), ('B', 4, 7.5)],
            'east': [('A', 10, 10.0), ('C', 1, 4.0), ('B', 2, 7.5)]}
    for reg, rs in rows.items():
        open(f'{ROOT}/data/sales_{reg}.csv', 'w').write('product,qty,price\n' + '\n'.join(f'{p},{q},{pr}' for p, q, pr in rs) + '\n')
    return {r: sum(q * pr for _, q, pr in rs) for r, rs in rows.items()}


def check_report(truth):
    p = f'{ROOT}/data/report.txt'
    if not os.path.exists(p):
        return False, 'no report.txt'
    txt = open(p, encoding='utf-8').read()
    nums = [float(x.replace(',', '.')) for x in re.findall(r'\d+(?:[.,]\d+)?', txt)]
    ok = all(any(abs(n - v) < 0.01 for n in nums) for v in list(truth.values()) + [sum(truth.values())])
    return ok, txt[:300]


if __name__ == '__main__':
    results = {}
    which = sys.argv[1:] or ['repo', 'data']
    if 'repo' in which:
        setup_repo()
        task = ('Тесты в /tmp/agent/repo падают. Найди причины и исправь код в mathutils.py '
                '(сами тесты менять нельзя). Проверь, что все тесты проходят, и коротко опиши, что было не так.')
        log('=== SCENARIO repo (fix 3 bugs, task in Russian)')
        st, _ = run_agent(task)
        ok, tail = check_repo()
        st['success'] = ok; st['tests_tail'] = tail
        results['repo'] = st; log('RESULT repo:', json.dumps(st, ensure_ascii=False)[:800])
    if 'data' in which:
        truth = setup_data()
        task = ('В /tmp/agent/data лежат CSV с продажами по регионам (колонки product,qty,price). Посчитай выручку '
                '(qty*price) по каждому региону и итого, запиши в /tmp/agent/data/report.txt строками "регион: сумма" '
                'и строкой "Итого: сумма", а в ответе коротко назови регион с максимальной выручкой.')
        log('=== SCENARIO data (csv -> report, task in Russian); truth', truth)
        st, _ = run_agent(task)
        ok, txt = check_report(truth)
        st['success'] = ok; st['report'] = txt
        results['data'] = st; log('RESULT data:', json.dumps(st, ensure_ascii=False)[:800])
    json.dump(results, open(os.environ.get('AGENT_JSON', '/tmp/work/agent_results.json'), 'w'), ensure_ascii=False, indent=1)
    log('ALL DONE', json.dumps({k: (v['success'], v['steps'], v['tool_calls'], v['bad_calls']) for k, v in results.items()}))
