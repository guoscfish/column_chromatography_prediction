"""One small diagnostic inference, separate from all experiment transcripts."""
import json
import time
import httpx
import parallel_resume as resume


if __name__ == '__main__':
    seed = 887
    events = []
    resume.configure(seed)
    resume.base.event = lambda seed, action, **fields: events.append({'action': action, **fields})
    resume.base.client_for = lambda _proxy=None, timeout=60: httpx.Client(
        proxy=None, trust_env=False, timeout=60)
    started = time.monotonic()
    record = {'time': resume.base.now(), 'seed': seed, 'purpose': 'Connectivity diagnostic only',
              'experiment_packet_sent': False, 'attempts': 1, 'events': events}
    try:
        answer, provenance = resume.base.responses_call(seed,
            [{'role': 'user', 'content': 'Reply with OK.'}], resume.base.CONFIG)
        record.update(status='PASS', answer_nonempty=bool(answer.strip()), provenance=provenance)
    except BaseException as error:
        message = str(error)
        for key in resume.KEYS.values():
            message = message.replace(key.read_text().strip(), '[REDACTED]')
        record.update(status='FAIL', error_type=type(error).__name__, message=message[:400])
    record['elapsed_seconds'] = round(time.monotonic()-started, 2)
    resume.base.atomic_json(resume.OUT/f'inference_probe_{time.time_ns()}.json', record)
    print(json.dumps(record), flush=True)
