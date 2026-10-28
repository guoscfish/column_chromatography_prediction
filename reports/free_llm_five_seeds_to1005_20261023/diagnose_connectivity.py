"""No-inference connectivity checks; never emit credentials or response bodies."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
import httpx
import parallel_resume as resume


def check(task):
    seed, route, operation = task
    start = time.monotonic()
    result = {'seed': seed, 'route': route, 'operation': operation}
    try:
        with httpx.Client(proxy=route, trust_env=False, timeout=12) as client:
            headers = {'Authorization': 'Bearer '+resume.KEYS[seed].read_text().strip()}
            if operation == 'models':
                response = client.get(resume.base.CONFIG['base_url']+'/models', headers=headers)
            else:
                response = client.post(resume.base.CONFIG['base_url']+'/responses', headers=headers, json={})
            result['http_status'] = response.status_code
            if response.status_code == 200 and operation == 'models':
                result['model_listed'] = any(x.get('id') == 'gpt-6.1-sol' for x in response.json().get('data', []))
    except Exception as error:
        result['error_type'] = type(error).__name__
    result['elapsed_seconds'] = round(time.monotonic()-start, 2)
    return result


if __name__ == '__main__':
    tasks = [(s, route, op) for s in resume.base.SEEDS
             for route in (None, 'http://127.0.0.1:3066') for op in ('models', 'empty_post')]
    with ThreadPoolExecutor(max_workers=12) as executor:
        checks = list(executor.map(check, tasks))
    record = {'time': resume.base.now(), 'checks': checks, 'inference_requests': 0}
    stamp = str(time.time_ns())
    resume.base.atomic_json(resume.OUT/f'connectivity_{stamp}.json', record)
    print(json.dumps(record))
