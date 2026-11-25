"""Freeze deployment content and the one authorization covering all formal rounds."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path("studies/active_learning/pure_llm_v3_a7_gpu_20261124")
OFFLINE = Path("studies/active_learning/pure_llm_v3_20261124")
PROFILE = STUDY / "execution_gpt62_ultra_a7.json"
DIRECT_API = "direct_platform"
SSH_API = "mac_ssh_dispatch_v1"
SEEDS = (157, 887, 2357, 6101, 12203)
NUMERIC_STUDY = Path("studies/active_learning/qgeognn_v2_row_sequential_b32")
METADATA_STUDY = Path("studies/active_learning/qgeognn_v2_row_small_batch_benchmark")
E0 = Path("experiments/e0_4g_baseline")
LEASE_ROOT = Path("/1/column_chromatography_prediction/formal_cuda/row4g_b32_20261124")
ASSETS = (
    Path("dataset/dataset_4g.csv"), E0 / "canonical_4g.csv", E0 / "graph_cache_4g.pt",
    E0 / "scaler.json", E0 / "data_metadata.json", METADATA_STUDY / "protocol.json",
    NUMERIC_STUDY / "protocol.json", NUMERIC_STUDY / "splits/split_manifest.json",
    *(NUMERIC_STUDY / f"splits/row_seed_{seed}.csv" for seed in SEEDS),
)
AUTH_KEYS = {"scope", "contract_sha256", "authorized_by", "authorization_reference",
             "authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training"}


@dataclass(frozen=True)
class FormalContract:
    profile: dict
    transport: TransportConfig
    source_sha256: dict
    asset_sha256: dict
    anchor_root: str | None = None
    gpu_slots: tuple[int, ...] = ()
    lease_root: str = str(LEASE_ROOT)
    run_directory: str | None = None
    api_execution: str = DIRECT_API
    predecessor: dict | None = None

    @property
    def seed_ids(self):
        return tuple(self.profile["confirmation_seeds"])

    @property
    def initial_labels(self):
        return self.profile["initial_labels"]

    @property
    def target_labels(self):
        return self.profile["target_labels"]

    @property
    def batch_size(self):
        return self.profile["batch_size"]

    @property
    def rounds(self):
        return (self.target_labels - self.initial_labels) // self.batch_size

    @property
    def qualification_seed(self):
        return self.profile["qualification_seed"]

    @property
    def api_slots(self):
        return self.profile["api_parallelism"]

    @property
    def device(self):
        return self.profile["device"]

    def record(self):
        base = len(self.seed_ids) * self.rounds
        record = {"profile": self.profile, "source_sha256": self.source_sha256,
                "asset_sha256": self.asset_sha256, "initial_policy": (
                    "reuse_verified_cuda_random_l333" if self.anchor_root else "fresh_identical_cuda_l333"
                ), "anchor_root": self.anchor_root, "gpu_slots": list(self.gpu_slots),
                "lease_root": self.lease_root, "run_directory": self.run_directory,
                "budget": {"first_logical_requests": base,
                           "maximum_logical_requests": base * self.transport.logical_replies,
                           "maximum_physical_attempts": base * self.transport.logical_replies
                           * self.transport.transport_attempts}}
        if self.api_execution != DIRECT_API:
            record["api_execution"] = self.api_execution
            record["predecessor"] = self.predecessor
        return record

    @property
    def digest(self):
        return digest(self.record())


def _profile(path):
    profile = read(path)
    expected = {"schema": "pure_llm_v3_a7_formal_v1", "arm": "A7", "memory": "M2", "device": "cuda",
                "confirmation_seeds": list(SEEDS), "qualification_seed": 157,
                "initial_labels": 333, "target_labels": 1485, "initial_pool": 2997, "validation": 416,
                "test": 417, "batch_size": 32, "api_parallelism": 3, "store": False,
                "provider_capability": "unverified_until_formal_round0"}
    if set(profile) != {*expected, "transport", "verified_served_model_reference"} or any(
        type(profile[k]) is not type(v) or profile[k] != v for k, v in expected.items()
    ):
        raise ValueError("formal A7/M2 scientific contract differs from its registered values")
    if set(profile["transport"]) != set(TransportConfig.__dataclass_fields__):
        raise ValueError("formal transport configuration fields differ")
    transport = TransportConfig(**profile["transport"])
    wire = {"requested_model": "gpt-6.2-sol", "reasoning_effort": "ultra", "provider": "token4research",
            "base_url": "https://token4research.cn/v1", "max_output_tokens": 32768,
            "wire_mode": "responses_stream", "transport_attempts": 2, "logical_replies": 3,
            "rate_limit_threshold": 2}
    if any(getattr(transport, k) != v for k, v in wire.items()):
        raise ValueError("formal provider, reasoning and budget are fixed")
    verified = profile["verified_served_model_reference"]
    if transport.served_model != transport.requested_model and not (
        re.fullmatch(r"gpt-6\.2-sol-[A-Za-z0-9][A-Za-z0-9._-]{0,79}", transport.served_model)
        and isinstance(verified, str) and verified.strip() and len(verified) <= 300
    ):
        raise ValueError("a versioned served model requires an exact ID and verification reference")
    if verified is not None and (not isinstance(verified, str) or not verified.strip() or len(verified) > 300):
        raise ValueError("invalid served model verification reference")
    return profile, transport


def load_contract(root, config_path, *, anchor_root=None, gpu_slots=(), lease_root=LEASE_ROOT,
                  run_directory=None, api_execution=DIRECT_API, predecessor_run=None):
    root = Path(root)
    profile, transport = _profile(config_path)
    slots = tuple(gpu_slots)
    if len(slots) != len(set(slots)) or any(type(gpu) is not int or gpu < 0 for gpu in slots):
        raise ValueError("physical GPU slots must be distinct nonnegative indices")
    code = [
        *sorted((root / STUDY / "a7_gpu").glob("*.py")),
        *sorted((root / OFFLINE / "pure_llm_v3").glob("*.py")),
        *sorted((root / "src/qgeognn_al").rglob("*.py")),
        *(root / name for name in ("application/QGeoGNN.py", "application/utils.py",
                                   str(OFFLINE / "PROMPT_PURE_LLM_V3.md"))),
    ]
    sources = {path.relative_to(root).as_posix(): file_digest(path) for path in code}
    assets = {path.as_posix(): file_digest(root / path) for path in ASSETS}
    run_path = Path(run_directory).resolve() if run_directory is not None else None
    anchors = str(Path(anchor_root).resolve()) if anchor_root else None
    lease = str(Path(lease_root).resolve())
    predecessor = None
    if api_execution == SSH_API:
        from .supersession import inspect_predecessor

        if predecessor_run is None or run_path is None or anchors is None:
            raise ValueError("SSH execution needs a separate run and its original shared CUDA anchors")
        original = inspect_predecessor(predecessor_run)
        old_run = Path(original["run_directory"])
        if run_path == old_run or run_path.parent != old_run.parent:
            raise ValueError("SSH execution needs its own sibling run directory")
        expected = (original["profile"], original["asset_sha256"], original["anchor_root"],
                    tuple(original["gpu_slots"]), original["lease_root"])
        if (profile, assets, anchors, slots, lease) != expected:
            raise ValueError("SSH execution may change transport location only, not the original A7 profile or GPU inputs")
        predecessor = original
    elif api_execution != DIRECT_API or predecessor_run is not None:
        raise ValueError("unsupported API execution mode or predecessor binding")
    return FormalContract(profile, transport, sources, assets, anchors, slots, lease,
                          str(run_path) if run_path else None, api_execution, predecessor)


def verify_authorization(contract, authorization):
    if type(authorization) is not dict or set(authorization) != AUTH_KEYS or (
        authorization["scope"] != "pure_llm_v3_a7_formal"
        or authorization["contract_sha256"] != contract.digest
        or any(authorization[key] is not True for key in (
            "authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training"
        ))
        or any(not isinstance(authorization[key], str) or not authorization[key].strip()
               or len(authorization[key]) > 300 for key in ("authorized_by", "authorization_reference"))
    ):
        raise PermissionError("formal run needs the exact approved contract, including API, reveal and GPU fit")
    return dict(authorization)
