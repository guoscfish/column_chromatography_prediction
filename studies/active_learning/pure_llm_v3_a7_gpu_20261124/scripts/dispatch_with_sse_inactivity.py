"""Apply an approved Mac-only SSE inactivity overlay to an existing drained V13 dispatcher."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import dispatcher, provider
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from .dispatch_prepared_ipv4 import _approved
from .dispatch_prompt_pilot import PromptDispatcher, SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool


SCHEMA = "a7_sse_inactivity_overlay_authorization_v1"
SCHEDULE_SCHEMA = "a7_ipv4_parallel_schedule_authorization_v1"
SSE_INACTIVITY_SECONDS = 900
MAIN = "a7-gpt62-astra-xhigh-ipv4-v13"
CONTINUATION = f"{MAIN}-2357-continuation-01"
RUN_IDENTITIES = {
    MAIN: ("b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85",
           "d43ca2ba5a3ac93173466bdb638735805d0092b321dd42e7d8aac862af922410"),
    CONTINUATION: ("99fd5844f05bca5d7d6be41bc82dbccc9dc77d90faba61101ff5a0c490403c2b",
                   "01dce8c6c2feef6f9b042858189fddcd95f77f0678867fc9f3f3731a39134f16"),
}
REPLACEMENTS = {(STUDY / "a7_gpu" / f"{module.__name__.split('.')[-1]}.py").as_posix(): module
                for module in (provider, dispatcher)}
SPOOL_FOLDERS = ("jobs", "slots", "attempts", "results", "delivered")


def _existing_spool(path, contract, authorization):
    dispatcher._private_directory(path)
    for folder in SPOOL_FOLDERS:
        dispatcher._private_directory(path / folder)
    binding = read_bridge_json(path / "run_binding.json")
    identity = read_bridge_json(path / "dispatcher_identity.json")
    dispatcher_id = identity.get("dispatcher_id") if type(identity) is dict else None
    if (binding != {"contract_sha256": contract, "authorization_sha256": authorization, "mock": False}
            or binding["mock"] is not False or type(identity) is not dict or set(identity) != {"dispatcher_id"}
            or not isinstance(dispatcher_id, str) or not ATTEMPT_ID.fullmatch(dispatcher_id)):
        raise PermissionError("the overlay requires the existing authorized Mac spool and dispatcher identity")
    return dispatcher_id


def _approved_fields(record, expected, approval_sha, label):
    if (type(record) is not dict or digest(record) != approval_sha
            or set(record) != {*expected, "authorized_by", "authorization_reference"}
            or any(type(record[key]) is not type(value) or record[key] != value for key, value in expected.items())
            or any(not isinstance(record[key], str) or not record[key].strip() or len(record[key]) > 300
                   for key in ("authorized_by", "authorization_reference"))):
        raise PermissionError(f"the exact {label} approval does not authorize this dispatcher")


def _verify_schedule(args, run, dispatcher_id):
    supplied = (args.schedule_authorization, args.schedule_authorization_sha256, args.schedule_launcher)
    if run == CONTINUATION:
        if any(value is not None for value in supplied):
            raise PermissionError("the 2357 continuation does not use the five-seed parallel scheduling addendum")
        return None
    if any(value is None for value in supplied):
        raise PermissionError("the five-seed overlay needs the existing parallel schedule approval and launcher")
    approval = read_bridge_json(args.schedule_authorization)
    expected = {"schema": SCHEDULE_SCHEMA, "contract_sha256": args.contract_sha256,
                "authorization_sha256": args.authorization_sha256, "dispatcher_id": dispatcher_id,
                "launcher_sha256": file_digest(args.schedule_launcher), "previous_api_parallelism": 1,
                "api_parallelism": 3, "authorize_paid_api_scheduling": True}
    _approved_fields(approval, expected, args.schedule_authorization_sha256, "parallel schedule")
    return args.schedule_authorization_sha256


def _source(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise PermissionError("a source manifest path is not release-relative")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError(f"source manifest escapes the release: {name}")
    return path


def _verify_sources(info, base):
    overlay = ROOT.resolve()
    if base == overlay:
        raise PermissionError("the overlay and unmodified base must be separate releases")
    original = info["contract"]["source_sha256"]
    if set(REPLACEMENTS) - original.keys():
        raise PermissionError("the original contract does not bind both transport modules")
    dispatcher.verify_local_source(info, root=base)
    replacements = {}
    for name, approved_sha in original.items():
        if file_digest(_source(base, name)) != approved_sha:
            raise PermissionError(f"the unmodified base release does not match its contract: {name}")
        overlay_source = _source(overlay, name)
        actual_sha = file_digest(overlay_source)
        if name in REPLACEMENTS:
            if actual_sha == approved_sha or Path(REPLACEMENTS[name].__file__).resolve() != overlay_source:
                raise PermissionError(f"the approved replacement is not installed and loaded: {name}")
            replacements[name] = actual_sha
        elif actual_sha != approved_sha:
            raise PermissionError(f"the overlay changes a source outside the approved transport: {name}")
    return replacements


def _authorized_scope(info, replacements, run, remote_run, spools, pilot_transport):
    record = info["contract"]
    # Scope validators check installed source digests. The dispatcher and courier get the original signed info.
    overlay_info = {**info, "contract": {**record, "source_sha256": {**record["source_sha256"], **replacements}}}
    if run == MAIN:
        transport, identity, prompt_sha = _approved(overlay_info, remote_run, pilot_transport)
        return PromptDispatcher, transport, identity, prompt_sha, 3
    from .dispatch_prepared_ipv4_continue_2357 import (
        EXTENSION_CONTRACT, TIMEOUTS, ContinuationDispatcher, _delivered_extension, _verified_scope,
    )

    scope = _verified_scope(overlay_info, remote_run)
    extension_transport, identity = _delivered_extension(spools / EXTENSION_CONTRACT, scope)
    transport = TransportConfig(**record["profile"]["transport"])
    expected = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if (transport != expected or extension_transport != replace(transport, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version,
                transport.logical_replies, transport.transport_attempts)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0", 3, 2)
            or not provider.sdk_status(transport)["matches"]):
        raise PermissionError("the continuation must retain its exact model, route, and attempt budget")
    return ContinuationDispatcher, transport, identity, _approved_prompt_sha(overlay_info), 1


def _unsealed_attempts(spool):
    return sum(not (spool / "results" / f"{marker.parent.name}.json").is_file()
               for marker in (spool / "attempts").glob("*/started.json"))


class DrainedPool(SharedKeyPool):
    def __init__(self, directory, config, spool):
        super().__init__(directory, config)
        self.spool = spool

    def prepare(self):
        # run_dispatcher calls prepare while holding the same original dispatcher.lock.
        if _unsealed_attempts(self.spool):
            raise PermissionError("drain and seal the original Mac attempts before enabling this overlay")
        super().prepare()


def run(args):
    run_name = Path(args.remote_run_dir).name
    if RUN_IDENTITIES.get(run_name) != (args.contract_sha256, args.authorization_sha256):
        raise PermissionError("the overlay is limited to the exact five-seed V13 and existing 2357 continuation contracts")
    spools = Path.home() / ".config/purellm/spool"
    spool = spools / args.contract_sha256
    dispatcher_id = _existing_spool(spool, args.contract_sha256, args.authorization_sha256)
    schedule_sha = _verify_schedule(args, run_name, dispatcher_id)
    approval = read_bridge_json(args.overlay_authorization)
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    replacements = _verify_sources(info, args.base_release.resolve(strict=True))
    expected = {"schema": SCHEMA, "contract_sha256": args.contract_sha256,
                "authorization_sha256": args.authorization_sha256, "dispatcher_id": dispatcher_id,
                "launcher_sha256": file_digest(Path(__file__)), "replacement_source_sha256": replacements,
                "sse_inactivity_seconds": SSE_INACTIVITY_SECONDS, "schedule_authorization_sha256": schedule_sha,
                "authorize_paid_api_scheduling": True}
    _approved_fields(approval, expected, args.overlay_authorization_sha256, "SSE inactivity overlay")
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    dispatcher_class, transport, model, prompt_sha, cap = _authorized_scope(
        info, replacements, run_name, args.remote_run_dir, spools, pilot_transport,
    )
    if read_bridge_json(spool / "provider_model_lock.json") != model:
        raise PermissionError("the original dispatcher has a different provider model lock")
    workers = min(cap, pilot.config.parallelism)
    if args.verify_only:
        return {"verified": True, "contract_sha256": args.contract_sha256, "api_workers": workers,
                "sse_inactivity_seconds": SSE_INACTIVITY_SECONDS, "unsealed_attempts": _unsealed_attempts(spool),
                "overlay_authorization_sha256": args.overlay_authorization_sha256}
    pool = DrainedPool(pilot.directory, pilot.config, spool)
    mac = dispatcher_class(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                           mock=False, pool=pool, sse_inactivity_seconds=SSE_INACTIVITY_SECONDS)
    if mac.dispatcher_id != dispatcher_id:
        raise PermissionError("the original dispatcher identity changed during startup")
    mac.prompt_sha = prompt_sha
    return dispatcher.run_dispatcher(mac, courier, watch=True, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-paid", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--overlay-authorization", type=Path, required=True)
    parser.add_argument("--overlay-authorization-sha256", required=True)
    parser.add_argument("--schedule-authorization", type=Path)
    parser.add_argument("--schedule-authorization-sha256")
    parser.add_argument("--schedule-launcher", type=Path)
    args = parser.parse_args()
    required = (args.contract_sha256, args.authorization_sha256, args.overlay_authorization_sha256)
    if (not all(SHA256.fullmatch(value) for value in required)
            or args.schedule_authorization_sha256 is not None and not SHA256.fullmatch(args.schedule_authorization_sha256)):
        parser.error("provide exact contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
