"""Recovery is exercised against real temporary Git worktrees, never live data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
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
             'scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py': "raise RuntimeError('controller must never execute')\n",
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



def archive_manifest(temp,protocol):
    path=temp/'archive.json';path.write_text(json.dumps({'archive_commit':protocol['commit'],'archive_tag':protocol['tag']}));return path


def test_generic_archive_reads_only_selected_bytes_and_never_executes_code(legacy):
    root,registry,protocol,temp=legacy
    destination=temp/'selected';before=command(root,'rev-parse','HEAD');index=history.file_hash(root/'.git/index')
    report=history.restore_archive(root,destination,paths=('scripts/studies',),manifest=archive_manifest(temp,protocol))
    assert report['passed'] and report['files_restored']==1 and report['controllers_or_training_or_provider_calls']==0
    restored=destination/'scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py'
    assert restored.read_bytes()==(root/'scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py').read_bytes()
    assert not restored.stat().st_mode&0o222 and not (destination/'dataset').exists()
    assert command(root,'rev-parse','HEAD')==before and history.file_hash(root/'.git/index')==index


def test_legacy_registry_captured_machine_path_is_not_a_live_source_fallback(legacy):
    root,registry,protocol,temp=legacy
    payload=json.loads(registry.read_text());payload['active_repository']='/nonexistent/captured/old/live/repository';registry.write_text(json.dumps(payload))
    destination=temp/'source'
    try:
        report=history.restore(root,'scientific_source',destination,registry)
        assert report['passed'] and report['active_HEAD_unchanged'] and report['active_index_unchanged']
    finally:
        if destination.exists():remove_worktree(root,destination)


def test_generic_archive_rejects_missing_paths_symlink_entries_and_moved_tag(legacy):
    root,_,protocol,temp=legacy;manifest=archive_manifest(temp,protocol)
    with pytest.raises(ValueError,match='no tracked|did not resolve'):
        history.restore_archive(root,temp/'missing',paths=('absent.md',),manifest=manifest)
    (root/'linked').symlink_to('dataset/sealed.csv');command(root,'add','linked');command(root,'commit','-qm','symlink fixture')
    protocol=dict(protocol,commit=command(root,'rev-parse','HEAD'));manifest=archive_manifest(temp,protocol)
    with pytest.raises(ValueError,match='symlinks'):
        history.restore_archive(root,temp/'refused-link',paths=('linked',),manifest=manifest)
    command(root,'tag',protocol['tag'],protocol['commit']);(root/'new.txt').write_text('next');command(root,'add','new.txt');command(root,'commit','-qm','different identity');command(root,'tag','-f',protocol['tag'],'HEAD')
    with pytest.raises(ValueError,match='tag moved'):
        history.restore_archive(root,temp/'refused-tag',paths=('src',),manifest=manifest)


def test_historical_oracle_missing_git_object_is_explicitly_blocked(monkeypatch):
    from tests.support import historical_reference
    from types import SimpleNamespace
    calls=[]
    def missing(arguments,**kwargs):calls.append(arguments);return SimpleNamespace(returncode=128,stdout=b'',stderr=b'not available')
    monkeypatch.setattr(historical_reference.subprocess,'run',missing)
    with pytest.raises(pytest.skip.Exception,match='BLOCKED: pinned historical Git source missing'):
        historical_reference.source('src/qgeognn_al/active_learning_v2/lcmd.py')
    assert calls and all('fetch' not in c for c in calls) and all('show' in c for c in calls)
