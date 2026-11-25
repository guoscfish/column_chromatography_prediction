"""Resolve a provider key without exposing it in arguments, receipts or checks."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat

ENV_NAME = "OPENAI_API_KEY"
FILE_ENV = "PURE_LLM_API_KEY_FILE"


def _choice():
    if FILE_ENV in os.environ:
        name = os.environ[FILE_ENV]
        if not name:
            raise PermissionError("PURE_LLM_API_KEY_FILE is empty")
        return "explicit_file", Path(name).expanduser()
    if ENV_NAME in os.environ:
        return "environment", None
    return "default_file", Path.home() / ".config/purellm/api_key"


def _check_stat(info):
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & ~0o600 or info.st_size > 8192):
        raise PermissionError("key file must be owned by this user, regular and restricted to mode 0600")


def credential_status():
    """Metadata-only preflight; environment value and file content are never read."""
    source, path = _choice()
    if path is None:
        return {"source": source, "present": True, "permissions_ok": None, "content_checked": False}
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"source": source, "present": False, "permissions_ok": None, "content_checked": False}
    try:
        _check_stat(info)
    except PermissionError:
        return {"source": source, "present": True, "permissions_ok": False, "content_checked": False}
    return {"source": source, "present": True, "permissions_ok": True, "content_checked": False}


def private_file_status(path):
    try:
        _check_stat(path.lstat())
    except FileNotFoundError:
        return {"present": False, "permissions_ok": None}
    except PermissionError:
        return {"present": True, "permissions_ok": False}
    return {"present": True, "permissions_ok": True}


@contextmanager
def private_text(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), encoding="utf-8") as handle:
        _check_stat(os.fstat(handle.fileno()))
        try:
            value = handle.read(8193).strip()
        except UnicodeDecodeError:
            raise PermissionError("private configuration is not UTF-8") from None
    if not value or len(value) > 8192:
        raise PermissionError("private configuration is empty or malformed")
    yield value


@contextmanager
def key_text(path):
    with private_text(path) as value:
        if any(c.isspace() for c in value):
            raise PermissionError("provider credential is malformed")
        yield value


@contextmanager
def api_key():
    """Read credential contents only for an authorized transport call."""
    source, path = _choice()
    if path is None:
        value = os.environ.get(ENV_NAME, "").strip()
    else:
        with key_text(path) as key:
            value = key
    if not value or len(value) > 8192 or any(c.isspace() for c in value):
        raise PermissionError("provider credential is empty or malformed")
    yield value, source
