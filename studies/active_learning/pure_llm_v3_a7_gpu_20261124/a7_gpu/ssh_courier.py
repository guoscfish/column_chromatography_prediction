"""Authenticated, bounded SSH courier for the A7 bridge endpoint."""

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps


ENDPOINT = "studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_endpoint"
SCHEMA = "purellm_a7_ssh_bridge_v1"
GPU_PYTHON = "/1/column_chromatography_prediction/environment/venv/bin/python"
ATTEMPT_ID = re.compile(r"[0-9a-f]{32}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,254}\Z")
INFO_FIELDS = {"schema", "contract_sha256", "authorization_sha256", "contract", "mock", "paused"}
LIST_FIELDS = {"schema", "attempt_ids", "contract_sha256", "authorization_sha256", "mock", "paused"}


class CourierUnavailable(RuntimeError):
    """The SSH request did not return a trustworthy endpoint result."""


def _absolute_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != str(value) or ".." in path.parts:
        raise ValueError("bridge SSH paths must be canonical absolute POSIX paths")
    return str(path)


def _check_identity(contract_sha, auth_sha, mock, paused, *, expected_contract, expected_auth, expected_mock):
    if (not isinstance(contract_sha, str) or not SHA256.fullmatch(contract_sha)
            or not isinstance(auth_sha, str) or not SHA256.fullmatch(auth_sha)
            or type(mock) is not bool or type(paused) is not bool
            or contract_sha != expected_contract or mock is not expected_mock
            or (expected_auth is not None and auth_sha != expected_auth)):
        raise PermissionError("remote bridge identity or authorization differs from the approved run")


def verify_info(info, *, expected_contract, expected_auth=None, expected_mock=False):
    if (type(info) is not dict or set(info) != INFO_FIELDS or info["schema"] != SCHEMA
            or type(info["contract"]) is not dict or digest(info["contract"]) != info["contract_sha256"]):
        raise ValueError("remote bridge info does not bind an intact formal contract")
    _check_identity(info["contract_sha256"], info["authorization_sha256"], info["mock"], info["paused"],
                    expected_contract=expected_contract, expected_auth=expected_auth, expected_mock=expected_mock)
    return info


def verify_list(inventory, info):
    if type(inventory) is not dict or set(inventory) != LIST_FIELDS or inventory["schema"] != SCHEMA:
        raise ValueError("invalid remote bridge inventory")
    _check_identity(inventory["contract_sha256"], inventory["authorization_sha256"],
                    inventory["mock"], inventory["paused"], expected_contract=info["contract_sha256"],
                    expected_auth=info["authorization_sha256"], expected_mock=info["mock"])
    ids = inventory["attempt_ids"]
    if (not isinstance(ids, list) or any(not isinstance(item, str) or not ATTEMPT_ID.fullmatch(item) for item in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("invalid or duplicate remote attempt IDs")
    return inventory


@dataclass(frozen=True)
class SSHCourier:
    host: str
    remote_root: str
    remote_run_directory: str
    remote_python: str = GPU_PYTHON
    timeout: float = 20

    def __post_init__(self):
        if not HOST.fullmatch(self.host):
            raise ValueError("invalid SSH host or alias")
        for field in ("remote_root", "remote_run_directory", "remote_python"):
            object.__setattr__(self, field, _absolute_path(getattr(self, field)))
        if not 1 <= self.timeout <= 120:
            raise ValueError("SSH courier timeout is outside the supported range")

    def _exchange(self, action, *, attempt_id=None, dispatcher_id=None, result=None):
        if action not in {"info", "list", "get", "claim", "put"}:
            raise ValueError("invalid bridge action")
        if (action in {"get", "claim", "put"}) != (attempt_id is not None) or (
            attempt_id is not None and not ATTEMPT_ID.fullmatch(attempt_id)
        ):
            raise ValueError("invalid SSH attempt identifier")
        if (action in {"claim", "put"}) != (dispatcher_id is not None) or (
            dispatcher_id is not None and not ATTEMPT_ID.fullmatch(dispatcher_id)
        ):
            raise ValueError("invalid SSH dispatcher identifier")
        if (action == "put") != (result is not None):
            raise ValueError("only a result upload may carry a body")
        arguments = [self.remote_python, "-B", "-m", ENDPOINT, action, "--run-dir", self.remote_run_directory]
        if attempt_id is not None:
            arguments += ["--attempt-id", attempt_id]
        if dispatcher_id is not None:
            arguments += ["--dispatcher-id", dispatcher_id]
        remote_command = f"cd {shlex.quote(self.remote_root)} && exec {shlex.join(arguments)}"
        ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "StrictHostKeyChecking=yes", "--", self.host, remote_command]
        try:
            response = subprocess.run(ssh, input=(dumps(result) + "\n") if result is not None else "",
                                      capture_output=True, text=True, check=True, timeout=self.timeout)
        except subprocess.CalledProcessError as error:
            if error.returncode == 255:
                raise CourierUnavailable(f"SSH bridge {action} did not complete") from None
            raise PermissionError(f"authenticated GPU bridge refused {action}") from None
        except (OSError, subprocess.SubprocessError):
            # Endpoint stderr may contain payload excerpts. Keep it out of the coordinator logs.
            raise CourierUnavailable(f"SSH bridge {action} did not complete") from None
        try:
            value = json.loads(response.stdout)
        except (TypeError, json.JSONDecodeError):
            raise CourierUnavailable(f"SSH bridge {action} returned invalid JSON") from None
        if type(value) is not dict:
            raise CourierUnavailable(f"SSH bridge {action} returned an invalid record")
        return value

    def info(self):
        return self._exchange("info")

    def list(self):
        return self._exchange("list")

    def get(self, attempt_id):
        return self._exchange("get", attempt_id=attempt_id)

    def claim(self, attempt_id, dispatcher_id):
        return self._exchange("claim", attempt_id=attempt_id, dispatcher_id=dispatcher_id)

    def put(self, attempt_id, dispatcher_id, result):
        return self._exchange("put", attempt_id=attempt_id, dispatcher_id=dispatcher_id, result=result)
