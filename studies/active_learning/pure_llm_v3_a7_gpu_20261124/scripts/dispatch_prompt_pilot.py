"""Dispatch the revised three-seed pilot through the already shared Mac pool."""
import argparse
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import KeyPool
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from .dispatch_prompt_successor import ROOT, _approved_prompt_sha, _pool


class SharedKeyPool(KeyPool):
    def prepare(self):
        status = self.status(readonly=True)
        if not status.get("initialized") or status["limits"]["cap"] != self.config.parallelism:
            raise PermissionError("the already shared Mac Key Pool is unavailable")


class PromptDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        binding = request["binding"]
        boundary = self.info["contract"]["profile"]["resume_boundaries"].get(str(binding["seed"]))
        if (boundary is None or binding["round"] < boundary["round"]
                or binding["prompt_sha256"] != self.prompt_sha):
            raise PermissionError("job is outside the revised prompt continuation boundary")
        return request, config, started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(157, 887, 2357), required=True)
    args = parser.parse_args()
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    if (info["contract"]["run_directory"] != args.remote_run_dir
            or set(info["contract"]["profile"].get("resume_boundaries", {})) != {str(args.seed)}):
        raise PermissionError("unexpected revised pilot continuation scope")
    verify_local_source(info, root=ROOT)
    pool_path, spool = args.pilot_spool.expanduser().resolve(strict=True), args.spool_dir.expanduser().resolve()
    if pool_path == spool or spool.is_relative_to(pool_path) or pool_path.is_relative_to(spool):
        raise PermissionError("revised pilot needs a separate private replay-safe spool")
    pool, _, pilot_transport = _pool(pool_path)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if transport != pilot_transport or not sdk_status(transport)["matches"]:
        raise PermissionError("revised pilot transport differs from the registered protocol")
    pool = SharedKeyPool(pool.directory, pool.config)
    dispatcher = PromptDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                  transport=transport, mock=False, pool=pool)
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", read_bridge_json(pool_path / "provider_model_lock.json"))
    print(dumps(run_dispatcher(dispatcher, courier, watch=True, workers=1)))


if __name__ == "__main__":
    main()
