"""Portable deterministic serialization and crash-safe receipt creation."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def dumps(value, *, sort_keys=False):
    return json.dumps(value, sort_keys=sort_keys, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def once(path: Path, value):
    """Publish a complete immutable file, allowing byte-equivalent replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (dumps(value, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            temp.write(encoded)
            temp.flush()
            os.fsync(temp.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise ValueError(f"immutable receipt differs: {path.name}") from None
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temp_path.unlink(missing_ok=True)
