"""Write-once artifacts and append-only audit. Audit is never used as run state."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from .schema import dumps


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'immutable artifact drift: {path.name}')
        return
    with path.open('x') as stream:
        stream.write(dumps(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def operation_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.operation.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another V3 operation owns the study') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Audit:
    def __init__(self, directory, packet):
        self.path = Path(directory) / 'audit.jsonl'
        self.identity = {k: packet[k] for k in ('seed', 'method', 'round')}
        self.sequence = 0
        self.previous = None
        if self.path.exists():
            events = verify_audit(self.path)
            if events:
                self.sequence = len(events)
                self.previous = events[-1]['event_sha256']

    def append(self, event_type, payload, *, turn, request_sha256):
        event = {**self.identity, 'turn': turn, 'request_sha256': request_sha256,
            'event_type': event_type, 'sequence': self.sequence,
            'timestamp': datetime.now(timezone.utc).isoformat(), 'artifact_sha256': digest(payload),
            'previous_event_sha256': self.previous, 'payload': payload}
        event['event_sha256'] = digest(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(dumps(event)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.previous = event['event_sha256']
        self.sequence += 1


def verify_audit(path):
    events, previous = [], None
    for index, line in enumerate(Path(path).read_text().splitlines()):
        event = json.loads(line)
        content = {k: v for k, v in event.items() if k != 'event_sha256'}
        if (event['sequence'] != index or event['previous_event_sha256'] != previous or
                event['artifact_sha256'] != digest(event['payload']) or event['event_sha256'] != digest(content)):
            raise RuntimeError('audit hash chain drift')
        events.append(event)
        previous = event['event_sha256']
    return events
