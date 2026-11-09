"""Main advances only through an explicit diagnostics-only commit receipt."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('runtime_isolation_under_test', ROOT / 'tools/verify_runtime_isolation.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def setup(tmp_path):
    active = tmp_path / 'active'
    active.mkdir()
    git(active, 'init', '-b', 'main')
    git(active, 'config', 'user.email', 'test@example.invalid')
    git(active, 'config', 'user.name', 'Isolation Test')
    source = active / 'src/core.py'
    source.parent.mkdir()
    source.write_text('SCIENCE = "fixed"\n')
    git(active, 'add', '.')
    git(active, 'commit', '-qm', 'protected scientific source')
    initial = git(active, 'rev-parse', 'HEAD')
    candidate = tmp_path / 'candidate'
    git(active, 'worktree', 'add', '-b', verify.REFACTOR_BRANCH, str(candidate), 'HEAD')
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'GATE0.json').write_text(json.dumps({'active_root': str(active), 'active_head': initial,
        'immutable_identity_checks': [], 'runtime_directory_identity_checks': []}))
    (snapshot / 'active_source_hashes.json').write_text(json.dumps({'hashes': {'src/core.py': verify.digest(source)}}))
    (snapshot / 'main_index_hash.json').write_text(json.dumps({'sha256': verify.digest(active / '.git/index')}))
    return active, candidate, snapshot, source, initial


def diagnostics_commit(active):
    report = active / verify.DIAGNOSTICS_PREFIX / 'REPORT.md'
    report.parent.mkdir(parents=True)
    report.write_text('Offline diagnostic; no scientific source was changed.\n')
    git(active, 'add', verify.DIAGNOSTICS_PREFIX)
    git(active, 'commit', '-qm', 'authorized offline diagnostics')
    return git(active, 'rev-parse', 'HEAD')


def test_unchanged_main_uses_independent_candidate_head_index(setup):
    active, candidate, snapshot, _, _ = setup
    result = verify.verify(snapshot, candidate)
    assert result['passed']
    assert result['main_index_binary_unchanged']
    assert result['worktree_isolation']['separate_index']
    assert result['worktree_isolation']['separate_HEAD']
    assert result['main_progression']['mode'] == 'unchanged_main'


def test_diagnostics_head_requires_exact_receipt_and_preserves_source(setup):
    active, candidate, snapshot, source, initial = setup
    before = verify.digest(source)
    head = diagnostics_commit(active)
    assert not verify.verify(snapshot, candidate)['passed']
    assert not verify.verify(snapshot, candidate, initial)['passed']
    result = verify.verify(snapshot, candidate, head)
    assert result['passed']
    assert result['main_progression']['mode'] == 'authorized_diagnostics_additions'
    assert not result['main_index_binary_unchanged']
    assert not result['main_staged_paths']
    assert verify.digest(source) == before


def test_receipt_cannot_authorize_other_main_paths_or_source_tampering(setup):
    active, candidate, snapshot, source, _ = setup
    head = diagnostics_commit(active)
    source.write_text('SCIENCE = "changed"\n')
    result = verify.verify(snapshot, candidate, head)
    assert not result['passed'] and result['source_failures'] == ['src/core.py']
    source.write_text('SCIENCE = "fixed"\n')
    (active / 'unrelated.txt').write_text('not authorized')
    git(active, 'add', '.')
    git(active, 'commit', '-qm', 'unrelated main change')
    unrelated = git(active, 'rev-parse', 'HEAD')
    assert not verify.verify(snapshot, candidate, unrelated)['passed']


def test_staged_main_and_wrong_candidate_branch_fail_closed(setup):
    active, candidate, snapshot, _, _ = setup
    extra = active / verify.DIAGNOSTICS_PREFIX / 'REPORT.md'
    extra.parent.mkdir(parents=True)
    extra.write_text('uncommitted diagnostics')
    git(active, 'add', verify.DIAGNOSTICS_PREFIX)
    assert not verify.verify(snapshot, candidate)['passed']
    git(active, 'reset', '-q', 'HEAD', '--', verify.DIAGNOSTICS_PREFIX)
    git(candidate, 'switch', '-c', 'unexpected-candidate')
    result = verify.verify(snapshot, candidate)
    assert not result['passed']
    assert not result['worktree_isolation']['authorized_branch']
