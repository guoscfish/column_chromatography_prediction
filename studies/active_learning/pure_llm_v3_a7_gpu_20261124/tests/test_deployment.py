import ast
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124 import deployment
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import formal_contract


@pytest.fixture
def reviewed_transport_bytes():
    root = Path(deployment.__file__).resolve().parents[3]
    return deployment.reviewed_transport(root)


def _write_bundle(path, inputs):
    records = {name: {"bytes": len(raw), "sha256": deployment.sha(raw),
                      "origin": deployment.source_origin(name)} for name, raw in inputs.items()}
    manifest = {"schema": deployment.RELEASE_SCHEMA, "published_git_commit": deployment.BASE,
                "files": records}
    with tarfile.open(path, mode="w:gz") as bundle:
        for name, raw in inputs.items():
            deployment.add_member(bundle, name, raw)
        deployment.add_member(bundle, deployment.RELEASE_MANIFEST, json.dumps(manifest).encode())


def test_release_name_boundary():
    assert deployment.safe_name(f"{deployment.FORMAL}/a7_gpu/backend.py")
    assert deployment.safe_name("dataset/dataset_4g.csv")
    for name in (
        "/etc/passwd", "../escape.py", "src/qgeognn_al/../../escape.py",
        ".git/config", "application/.ssh/config", "dataset/dataset_8g.csv",
        "src/qgeognn_al/__pycache__/candidate.pyc",
        "src/qgeognn_al/credentials.txt",
        "scripts/studies/run_qgeognn_cuda_three_methods.py",
        f"{deployment.FORMAL}/runtime/selection.json",
        f"{deployment.FORMAL}/key_pool.json",
        f"{deployment.PUBLISHED}/label_free_metadata.csv",
    ):
        assert not deployment.safe_name(name)


def test_release_sha_and_fresh_destination(tmp_path, reviewed_transport_bytes):
    inputs = {name: name.encode() for name in deployment.ASSETS | deployment.REQUIRED_FORMAL}
    inputs[deployment.REVIEWED_TRANSPORT] = reviewed_transport_bytes
    release = tmp_path / "release.tar.gz"
    _write_bundle(release, inputs)
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="uploaded archive SHA"):
        deployment.install(release, destination, "a" * 64)
    assert not destination.exists()
    digest = deployment.sha(release.read_bytes())
    receipt = deployment.install(release, destination, digest)
    assert receipt["files"] == len(inputs)
    assert (destination / deployment.RELEASE_MANIFEST).is_file()
    with pytest.raises(ValueError, match="fresh destination"):
        deployment.install(release, destination, digest)
    assert (destination / next(iter(inputs))).is_file()


def test_rejects_tar_symlink(tmp_path):
    release = tmp_path / "bad.tar.gz"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        sample = tarfile.TarInfo("dataset/dataset_4g.csv")
        sample.type = tarfile.SYMTYPE
        sample.linkname = "/outside"
        bundle.addfile(sample)
    release.write_bytes(gzip.compress(buffer.getvalue()))
    with pytest.raises(ValueError, match="invalid archive entry"):
        deployment.read_archive(release)


def test_release_covers_registered_formal_assets():
    assert {path.as_posix() for path in formal_contract.ASSETS} <= deployment.ASSETS


def test_only_reviewed_transport_overlays_the_published_runtime(tmp_path, monkeypatch, reviewed_transport_bytes):
    repo = tmp_path / "repo"
    name = deployment.REVIEWED_TRANSPORT
    sibling = str(PurePosixPath(name).with_name("runner.py"))
    model_file = "src/qgeognn_al/models/qgeognn_v2.py"
    base = {asset: asset.encode() for asset in deployment.ASSETS}
    base.update({name: b"old transport", sibling: b"published runner", model_file: b"published predictor"})
    monkeypatch.setattr(deployment, "committed_inputs", lambda repo: base)
    worktree = {file: file.encode() for file in deployment.REQUIRED_FORMAL}
    worktree.update({name: reviewed_transport_bytes, sibling: b"dirty runner", model_file: b"dirty predictor"})
    for file, raw in worktree.items():
        path = repo / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    release = tmp_path / "release.tar.gz"
    deployment.build(repo, release)
    extracted = deployment.read_archive(release)
    origins = json.loads(extracted[deployment.RELEASE_MANIFEST])["files"]
    assert [file for file, record in origins.items() if record["origin"] == "reviewed_transport_overlay"] == [name]
    for unchanged in (sibling, model_file):
        assert extracted[unchanged] == base[unchanged]
        assert origins[unchanged]["origin"] == "published_git"
    assert extracted[name] == reviewed_transport_bytes
    function = next(node for node in ast.parse(extracted[name]).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_request")
    assert "attempt_slot" in {arg.arg for arg in function.args.kwonlyargs}

    (repo / name).write_bytes(reviewed_transport_bytes + b"\n# unreviewed change\n")
    with pytest.raises(ValueError, match="reviewed transport overlay differs"):
        deployment.build(repo, tmp_path / "unreviewed.tar.gz")


def test_archive_rejects_legacy_transport_with_self_consistent_manifest(tmp_path):
    inputs = {name: name.encode() for name in deployment.ASSETS | deployment.REQUIRED_FORMAL}
    inputs[deployment.REVIEWED_TRANSPORT] = b"def run_request(directory, request, config, call): pass\n"
    release = tmp_path / "legacy.tar.gz"
    _write_bundle(release, inputs)
    with pytest.raises(ValueError, match="reviewed transport overlay differs"):
        deployment.read_archive(release)
