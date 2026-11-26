"""Continue 887 from the signed zero-POST L973 fit using the reviewed final-answer parser."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import OFFLINE, ROOT, STUDY, verify_authorization
from . import run_v13_887_r19_cached as cached
from . import run_v13_lane_continuation as continuation
from .run_v13_bounded_sse_recovery import RunIdentity, _archived


SEED, START, TARGET = 887, 20, 1485
RUN_NAME = f"{cached.PREFIX}-887-r20-continuation-01"
DISPATCHER = "dispatch_v13_887_r20_continuation"
LANE = cached.SimpleNamespace(seed=SEED, round=START - 1)
CACHED = RunIdentity("-887-r19-cached-01", "9cf634746ef96b162909e2e27f5e2bcdffcbe270b0435788b15e9dcd358a37d5",
                     "28840abdbf1577a8730a860aad0e69496da29c2c47296a0c72b1c61a173b0c20")
CACHED_RELEASE = "ipv4-v13-887-r19-cached-01-71fd5c7ad7af04e21c8f91132c841c4abd56993174551ea4533ad0487df3dfd9"
CACHED_MODULE = (STUDY / "scripts/run_v13_887_r19_cached.py").as_posix()
PARSER = (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()
OLD_PARSER = "25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739"
NEW_PARSER = "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"
BOUNDARY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
BUDGET = {"first_logical_requests": 16, "maximum_logical_requests": 46, "maximum_physical_attempts": 92}
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in ("run_v13_887_r20_continuation", DISPATCHER))
CACHED_HELPER = "689af1b0e8a5f7e818d3b64079d3d8d95c00f214ba34cf3ecbac184a5128ecac"
CUSTODY_FILE = "e4cc98e51ed137b83f4b1a939a9a25c6af5843ea2872da9e1515e3e9a54b89de"
NORMAL_PROMPT = "d656796e97dee76bd12bb7ce94bc3056d042e858e417759eb4108710744a594f"
HANDOFF = {
    'contract_sha256': '9cf634746ef96b162909e2e27f5e2bcdffcbe270b0435788b15e9dcd358a37d5',
    'authorization_sha256': '28840abdbf1577a8730a860aad0e69496da29c2c47296a0c72b1c61a173b0c20',
    'checkpoint_sha256': '4aff86682563b7f50d7f251605e661ae65897c2f2580d3d6d4ada9130ece5b5e',
    'transition_file_sha256': 'bed902db127e3592ae8ba249c587313761413e51194c120399a2033e04d44889',
    'source_file_sha256': '79f91d8382922a05f29a239176e7e9e20b48756ee3a3813e77f152cb640d3ae6',
    'state_sha256': '394fe3b0734e05e73e1392469dc90120636dfbcdf85557a8de76851b78b6d337',
    'scientific_memory_sha256': 'f9e9df2076dcab08989b4b8593533ac16824c020bc38b210cb04214d1af31a7e',
    'fit_audit_file_sha256': '6d1c856322ce5425c6ba342193f3c0eda831c54ae21cda5dc523be49372001fe',
    'freeze_sha256': 'a2bc6929408d4e1b8aa28bd2c43684067bdcf92beeb92f3a5561979088c2dc9f',
    'cached_selection_sha256': '79e2aec35cfb01139edd803f71c266d0e9a36f0f549c7e0c15ed3021a4863d82',
    'local_terminal_sha256': 'a009103826e344c35a1c22aea6bb44ced424c7ca1f36e0cff5b8ae40299e1e3f',
    'local_attempt_id': 'c58b67811ed748818110b4ace8109fba',
    'archived_provider_response_id': 'resp_ft_b3057b24d2204c009332156fc0aa411c',
    'candidate_order_sha256': 'bdc2cbebfce55b823eeeffa453dc0c2d15b185075ea50d8e2243a3aaaee12be8',
    'provider_post': False,
}
R20_REQUEST = {
    'payload_sha256': 'aeb44e5e8d85845e7bad1ea14bad6467eb64de837c4829f6ed1d9e75c1671ee4',
    'checkpoint_sha256': '4aff86682563b7f50d7f251605e661ae65897c2f2580d3d6d4ada9130ece5b5e',
    'packet_sha256': '4f15bf9e504d4252806ca9752d706c16653e7e25a77f3afac121affb08013367',
    'wire_packet_sha256': 'aa0013a09bfbc18d3342fccdef555a73f7687a092cabbfe5e3f327f6328b0c34',
    'base_messages_sha256': '43d9ed96925fbb215122285e587fd8d2aab2dba1e09ec2fb3d83ea009be6b392',
    'authoritative_base_messages_sha256': '664cb38d346e988a02535f5e70be9f8fdd6b51df53c18bd5b7206e8e9dc613cb',
    'prompt_sha256': '0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40',
    'mapping_sha256': '08ce40569025d9b259335705f5c6a4dcf16ba8ef1a19a8e372f5649191396e9c',
    'split_sha256': '0413fb5c356dc14b1d2c056379ef09f1925835e64fa41fa86dc12aa8703a1807',
    'state_sha256': '79881eaae89a1e01ea3f3bce861fea63a394289ecd9feed2b6a7a81796c07482',
}


def request_identity(request):
    return {key: request["payload_sha256"] if key == "payload_sha256" else request["binding"][key] for key in R20_REQUEST}


def _normal_sources(root, parent, imported):
    inherited = parent.source_sha256
    if inherited.get(PARSER) != OLD_PARSER or inherited[seed_runner.PROMPT_PATH.as_posix()] != NORMAL_PROMPT:
        raise PermissionError("the archived ordinary scientific release or parser identity changed")
    expected = {**inherited, PARSER: NEW_PARSER}
    actual = {name: file_digest(root / name) for name in expected}
    if actual != expected:
        raise PermissionError("only the exact reviewed parser may replace inherited normal source")
    if imported.source_sha256.get(CACHED_MODULE) != CACHED_HELPER or file_digest(root / CACHED_MODULE) != CACHED_HELPER:
        raise PermissionError("the old cached verifier source differs from its separately signed archive")
    actual[CACHED_MODULE] = CACHED_HELPER
    actual.update({path.as_posix(): file_digest(root / path) for path in SOURCES})
    return actual


def _verify_legacy_handoff(cached_root, old_root, old, imported_run, imported):
    if (imported.source_sha256.get(PARSER) != OLD_PARSER or file_digest(cached_root / PARSER) != OLD_PARSER
            or imported.source_sha256.get(CACHED_MODULE) != CACHED_HELPER
            or file_digest(cached_root / CACHED_MODULE) != CACHED_HELPER):
        raise PermissionError("the historical verifier must run only with its immutable old parser and signed code")
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONHASHSEED"] = "0"
    module = ".".join(Path(CACHED_MODULE).with_suffix("").parts)
    child = subprocess.run([
        sys.executable, "-B", "-s", "-m", module, "verify-handoff",
        "--old-release", str(old_root), "--old-run", str(old), "--run-dir", str(imported_run),
        "--contract-sha256", CACHED.contract, "--authorization-sha256", CACHED.approval,
    ], cwd=cached_root, env=env, capture_output=True, text=True, timeout=180, check=False)
    if child.returncode:
        raise PermissionError(f"the immutable archival verifier failed (exit {child.returncode})")
    proof = json.loads(child.stdout)
    if proof != HANDOFF:
        raise PermissionError("the signed R19 zero-POST handoff no longer reproduces all archived evidence")
    scheduler._assert_no_old_workers(imported_run, (SEED,))
    return proof


def construct(root, old_root, old, cached_root, imported_run, run):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    continuation._require_fixed_reconstruction(LANE)
    if (root in (old_root, cached_root) or old_root.name != cached.OLD_RELEASE or cached_root.name != CACHED_RELEASE
            or imported_run.name != CACHED.name or run.name != RUN_NAME
            or len({old.parent, imported_run.parent, run.parent}) != 1):
        raise PermissionError("use the exact halted archives and independent normal-Prompt sibling release")
    parent, parent_approval = _archived(old, old_root, cached.OLD)
    imported, _ = _archived(imported_run, cached_root, CACHED)
    sources = _normal_sources(root, parent, imported)
    handoff = _verify_legacy_handoff(cached_root, old_root, old, imported_run, imported)
    worker = imported_run / f"seed_{SEED}"
    active, following = (worker / f"round_{index:02d}" for index in (START - 1, START))
    source, transition = (following / name for name in ("source.json", "transition.json"))
    memory_file = active / "memory.json"
    state = seed_runner.FrozenState(**read(source))
    memory = read(memory_file)
    custody_file = imported_run / "cached_response_import.json"
    custody = read(custody_file)
    backend_cache = Path(parent.predecessor["cache_directory"]).resolve(strict=True)
    cache_sha = parent.predecessor["public_resource_receipt_sha256"]
    if ((state.seed, state.round, len(state.provenance["expected_observed"]), state.checkpoint)
            != (SEED, START, 973, handoff["checkpoint_sha256"])
            or file_digest(source) != handoff["source_file_sha256"]
            or file_digest(transition) != handoff["transition_file_sha256"]
            or digest(asdict(state)) != handoff["state_sha256"] or digest(memory) != handoff["scientific_memory_sha256"]
            or state.provenance["fit_audit_sha256"] != handoff["fit_audit_file_sha256"]
            or digest(custody) != handoff["cached_selection_sha256"] or custody.get("provider_post") is not False
            or file_digest(custody_file) != CUSTODY_FILE
            or file_digest(backend_cache / "resource_receipt.json") != cache_sha):
        raise PermissionError("the sealed L973 state, scientific memory, custody or CUDA cache changed")
    stage = seed_runner._stage(root, state, memory, seed_runner._scope(parent, parent_approval, SEED, False), parent)
    request = make_request(stage.binding, 0, stage.messages, parent.transport)
    if request_identity(request) != R20_REQUEST:
        raise PermissionError("the normal R20 request differs from the independently verified L973 boundary")
    descriptor = {"round": START, "state_sha256": handoff["state_sha256"], "memory_sha256": handoff["scientific_memory_sha256"]}
    profile = {**parent.profile, "resume_boundaries": {str(SEED): descriptor}, "replay_eligible_from_round": START}
    proof = {
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START, "r20_request": R20_REQUEST,
        "parent_run_directory": str(old), "parent_release_directory": str(old_root), "parent_contract_sha256": parent.digest,
        "parent_authorization_sha256": digest(parent_approval), "parent_seed_pause_sha256": cached.ARCHIVE_FILES["seed_887/pause.request"],
        "extension_run_directory": str(imported_run), "extension_release_directory": str(cached_root),
        "r19_handoff": handoff, "cached_selection": custody, "cached_response_import_file_sha256": CUSTODY_FILE,
        "checkpoint_sha256": state.checkpoint, "cache_directory": str(backend_cache), "public_resource_receipt_sha256": cache_sha,
        "model_identity": parent.predecessor["model_identity"], "bridge_wait_seconds": parent.predecessor["bridge_wait_seconds"],
        "recovery_budget": BOUNDARY_BUDGET, "execution_budget": BUDGET,
    }
    contract = replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, (state, memory, START), TrainingBackend(root, backend_cache, SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "cached-release", "cached-run", "run-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_root, old, cached_root, imported, run = (path.resolve() for path in (
        args.old_release, args.old_run, args.cached_release, args.cached_run, args.run_dir,
    ))
    contract, boundary, backend = construct(ROOT, old_root, old, cached_root, imported, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("provide the exact new paid and CUDA authorization")
    approval = verify_authorization(contract, read(args.authorization))
    with scheduler._lock(imported / "coordinator.lock", busy="the local zero-POST fit still owns its worker"):
        scheduler._assert_no_old_workers(imported, (SEED,))
        result = continuation.execute(ROOT, old, run, LANE, contract, approval, boundary, backend,
                                      resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
