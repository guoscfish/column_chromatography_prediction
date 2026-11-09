"""Recovery is exercised against real temporary Git worktrees, never live data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('history_restore_under_test', ROOT / 'tools/history_restore.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


def command(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def legacy(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    command(root, 'init', '-b', 'main')
    command(root, 'config', 'user.email', 'test@example.invalid')
    command(root, 'config', 'user.name', 'History Test')
    code = """def code_hashes():
    files = list((ROOT / 'src/qgeognn_al').rglob('*.py'))
    files += list((ROOT / 'application').glob('*.py'))
    files += [ROOT / 'scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py']
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}
"""
    files = {'src/qgeognn_al/core.py': 'VALUE = 7\n',
             'src/qgeognn_al/active_learning_v2/benchmark_protocol.py': code,
             'application/primitives.py': 'UNCHANGED = True\n',
             'scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py': '# Never executed\n',
             'dataset/sealed.csv': 'do_not_materialize\n'}
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    command(root, 'add', '.')
    command(root, 'commit', '-qm', 'fixed source')
    commit = command(root, 'rev-parse', 'HEAD')
    hashes = {name: history.file_hash(root / name) for name in files if name.endswith('.py')}
    protocol = {'commit': commit, 'tag': 'archive/test-source', 'source_hashes': hashes,
                'src_application_hashes': history.source_universe(root),
                'benchmark_code_hashes': history.benchmark_hashes(root), 'environment': {}}
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'active_repository': str(root), 'protocols': {'scientific_source': protocol}}))
    return root, registry, protocol, tmp_path


def remove_worktree(root, path):
    command(root, 'worktree', 'remove', '--force', str(path))


def test_restores_exact_source_without_expanding_data_or_changing_main(legacy):
    root, registry, protocol, temp = legacy
    command(root, 'tag', protocol['tag'], protocol['commit'])
    main_head = command(root, 'rev-parse', 'HEAD')
    index_hash = history.file_hash(root / '.git/index')
    destination = temp / 'historical'
    try:
        result = history.restore(root, 'scientific_source', destination, registry)
        assert result['passed'] and result['tag_verified']
        assert result['source_hashes_verified'] == 4
        assert result['benchmark_hashes_verified'] == 4
        assert result['environment_matches']
        assert not (destination / 'dataset/sealed.csv').exists()
        assert not (destination / 'src/qgeognn_al/core.py').stat().st_mode & 0o222
        assert command(root, 'rev-parse', 'HEAD') == main_head
        assert history.file_hash(root / '.git/index') == index_hash
        assert command(destination, 'rev-parse', 'HEAD') == protocol['commit']
    finally:
        if destination.exists():
            remove_worktree(root, destination)


def test_restored_byte_tamper_and_extra_prototype_file_fail_closed(legacy):
    root, registry, protocol, temp = legacy
    destination = temp / 'historical'
    try:
        history.restore(root, 'scientific_source', destination, registry)
        extra = destination / 'src/qgeognn_al/new_prototype.py'
        extra.write_text('NEW = True\n')
        report = history.verify_restored(destination, protocol)
        assert not report['passed']
        assert any(r['reason'] == 'whole_src_application_hash_universe_changed' for r in report['failures'])
        extra.unlink()
        file = destination / 'src/qgeognn_al/core.py'
        file.chmod(0o644)
        file.write_text('VALUE = 99\n')
        assert not history.verify_restored(destination, protocol)['passed']
    finally:
        if destination.exists():
            remove_worktree(root, destination)


def test_moved_tag_is_rejected_before_restoration(legacy):
    root, registry, protocol, temp = legacy
    command(root, 'tag', protocol['tag'], protocol['commit'])
    (root / 'new.txt').write_text('new main commit')
    command(root, 'add', '.')
    command(root, 'commit', '-qm', 'different identity')
    command(root, 'tag', '-f', protocol['tag'], 'HEAD')
    destination = temp / 'should_not_exist'
    with pytest.raises(ValueError, match='tag moved'):
        history.restore(root, 'scientific_source', destination, registry)
    assert not destination.exists()


def test_active_nested_existing_and_symlink_destinations_rejected(legacy):
    root, registry, _, temp = legacy
    with pytest.raises(ValueError, match='outside the active'):
        history.restore(root, 'scientific_source', root / 'new_child', registry)
    existing = temp / 'existing'
    existing.mkdir()
    with pytest.raises(ValueError, match='new absent'):
        history.restore(root, 'scientific_source', existing, registry)
    link = temp / 'linked'
    link.symlink_to(root / 'missing')
    with pytest.raises(ValueError, match='new absent'):
        history.restore(root, 'scientific_source', link, registry)


def test_environment_mismatch_does_not_qualify_source_recovery(legacy):
    root, _, protocol, _ = legacy
    protocol = dict(protocol, environment={'codex_nonexistent_science_package': '999.0'})
    result = history.verify_restored(root, protocol)
    assert not result['passed']
    assert not result['environment_matches']
    assert result['environment_failures'] == ['codex_nonexistent_science_package']
