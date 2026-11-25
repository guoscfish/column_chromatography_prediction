"""Device adapter matching the qualified Row B32 CUDA training loop."""

from contextlib import contextmanager
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


CUBLAS_WORKSPACE = ":4096:8"
SEQUENTIAL_PROTOCOL = Path("studies/active_learning/qgeognn_v2_row_sequential_b32/protocol.json")


def qualified_packages(asset_root):
    """Check the same scientific releases and CUDA build as the numeric suite."""
    import torch

    protocol = json.loads((Path(asset_root) / SEQUENTIAL_PROTOCOL).read_text())
    expected = protocol["packages"]
    packages = {name: version(name) for name in expected}
    scientific = {**packages, "torch": packages["torch"].split("+", 1)[0]}
    if scientific != expected:
        raise RuntimeError(f"scientific package versions drifted: {packages}")
    if packages["torch"] != f"{expected['torch']}+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("the qualified CUDA 12.8 PyTorch build is required")
    return packages


def configure_device():
    """Bind one child process to its single visible GPU before any actual work."""
    if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE) != CUBLAS_WORKSPACE:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the qualified CUDA suite")
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("assign exactly one visible CUDA GPU to each worker")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def cuda_contract(device):
    import torch

    return {
        "python": sys.version.split()[0],
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_properties(device).name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "float32_precision": torch.backends.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def move_model(model, device):
    from src.qgeognn_al.data import qg

    model.to(device)
    # Historical RBF centers are fixed tensors outside the named buffers.
    for module in model.modules():
        if isinstance(module, qg.RBF) and module.centers.device != device:
            module.centers = module.centers.detach().to(device)
    return model


@contextmanager
def cuda_adapter(device):
    """Patch only the existing fitter and inference device transfer in this child."""
    import torch

    from src.qgeognn_al.active_learning_v2 import runner
    from src.qgeognn_al.models import load_predictor_checkpoint
    from src.qgeognn_al.training import predictor

    cpu_loader_pair = predictor.loader_pair
    base_predict = predictor.predict
    base_train = runner.train_source

    def batches(atom, angle, indices, batch_size=2048):
        return tuple(
            (batch.to(device, *(key for key in batch.keys() if key != "canonical_position")) for batch in stream)
            for stream in cpu_loader_pair(atom, angle, indices, batch_size)
        )

    def checkpoint(path):
        return move_model(load_predictor_checkpoint(path), device)

    def timed_train(model, atom, angle, train_indices, validation_indices, preprocessing,
                    config, checkpoint_path, progress=None):
        model = move_model(model, device)
        parameter = next(model.parameters())
        initial = parameter.detach().clone()
        path = Path(checkpoint_path).parent
        torch.cuda.reset_peak_memory_stats(device)
        started = previous = time.perf_counter()
        seconds = []

        def timed_predict(*args, **kwargs):
            nonlocal previous
            predicted = base_predict(*args, **kwargs)
            now = time.perf_counter()
            seconds.append(now - previous)
            previous = now
            return predicted

        def record_progress(epoch):
            record = {**epoch, "fit": str(path), "training_seed": config["seed"],
                      "elapsed_seconds": time.perf_counter() - started}
            predictor.atomic_json(path / "training_progress.json", record)
            print(json.dumps(record), flush=True)
            if progress is not None:
                progress(epoch)

        with patch.object(predictor, "predict", timed_predict):
            history, best = base_train(
                model, atom, angle, train_indices, validation_indices, preprocessing,
                config, checkpoint_path, progress=record_progress,
            )
        update = (parameter.detach() - initial).abs().max().item()
        if not math.isfinite(update) or update == 0 or len(history) != len(seconds):
            raise RuntimeError("fit failed to update the sampled parameter or report every epoch")
        predictor.atomic_json(path / "timing.json", {
            "epochs": len(history),
            "best_epoch": best,
            "epoch_seconds_through_validation": seconds,
            "training_validation_checkpoint_seconds": time.perf_counter() - started,
            "first_parameter_max_abs_update": update,
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
        })
        return history, best

    with (patch.object(predictor, "loader_pair", batches),
          patch.object(runner, "loader_pair", batches),
          patch.object(runner, "load_predictor_checkpoint", checkpoint),
          patch.object(runner, "train_source", timed_train)):
        try:
            yield
        finally:
            # The seed can spend its next phase on a network request. Return unused
            # training blocks while the process still owns the physical GPU lease.
            torch.cuda.empty_cache()
