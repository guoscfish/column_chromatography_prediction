"""Retirement gates use isolated Git fixtures; no real label/model payloads."""
import importlib.util
import gzip
import csv
import json
from pathlib import Path
import subprocess

import pytest

TOOL = Path(__file__).resolve().parents[2] / 'tools/audit_research_artifacts.py'
spec = importlib.util.spec_from_file_location('artifact_audit', TOOL)
audit_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tool)


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments]).decode().strip()


def repository(tmp_path, files):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'Audit Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    default = {'docs/PROTECTED_ARTIFACTS.json': json.dumps({'artifacts': []})}
    for name, text in {**default, **files}.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    return root, git(root, 'rev-parse', 'HEAD')


def by_path(rows):
    return {r['path']: r for r in rows}


def test_raw_labels_models_and_sensitive_payloads_are_not_opened(tmp_path, monkeypatch):
    root, commit = repository(tmp_path, {
        'experiments/e0_4g_baseline/canonical_4g.csv': 'sample_id,truth\na,17\n',
        'experiments/e0_4g_baseline/checkpoints/best.pt': 'opaque scientific model',
        'studies/demo/raw_responses/message.json': '{"private":"value"}',
        'docs/model/INPUT_SCHEMA.md': 'Known public input contract',
    })
    observed = []
    original = audit_tool.BlobReader.read
    def spy(self, path, entry):
        observed.append(path)
        assert 'canonical_4g.csv' not in path and 'best.pt' not in path and 'raw_responses' not in path
        return original(self, path, entry)
    monkeypatch.setattr(audit_tool.BlobReader, 'read', spy)
    rows, _, _, receipt = audit_tool.audit(root, baseline=commit)
    row = by_path(rows)['experiments/e0_4g_baseline/canonical_4g.csv']
    assert row['hash'] == 'NOT_READ_PROTECTED_OR_UNREVIEWED'
    assert row['deletion_decision'] == 'KEEP_PROTECTED'
    assert 'docs/model/INPUT_SCHEMA.md' in observed
    assert receipt['safety']['data_checkpoint_payloads_decoded'] == 0


def test_stable_dynamic_reference_blocks_retirement(tmp_path):
    name = 'docs/model/CLEAN_QGEOGNN_DESIGN.md'
    root, commit = repository(tmp_path, {
        name: 'superseded historical design',
        'src/consumer.py': "CONFIG = 'docs/model/CLEAN_QGEOGNN_DESIGN.md'\n",
    })
    rows, candidates, graph, _ = audit_tool.audit(root, baseline=commit)
    row = by_path(rows)[name]
    assert row['deletion_decision'] == 'BLOCKED_MACHINE_REFERENCE'
    assert row['external_machine_reference_count'] >= 1
    assert any(e['to'] == name and e['kind'] == 'DYNAMIC_PATH_LITERAL' for e in graph['edges'])
    assert by_path(candidates)[name] == row


def test_pinned_git_links_do_not_become_live_relative_dependencies(tmp_path):
    name = 'docs/model/CLEAN_QGEOGNN_DESIGN.md'
    root, commit = repository(tmp_path, {
        name: 'historical design',
        'README.md': '[history](https://github.com/example/repo/blob/' + 'a'*40 + '/' + name + ')',
    })
    rows, _, _, _ = audit_tool.audit(root, baseline=commit)
    assert by_path(rows)[name]['deletion_decision'] == 'ELIGIBLE_AFTER_REMOTE_RECOVERY_GATE'
    (root / 'docs/README.md').write_text('[design](model/CLEAN_QGEOGNN_DESIGN.md)')
    rows, _, _, _ = audit_tool.audit(root, baseline=commit)
    assert by_path(rows)[name]['deletion_decision'] == 'REVIEW_DOCUMENT_REFERENCES'


def test_sparse_omission_is_not_falsely_reported_as_git_retirement(tmp_path):
    name = 'reports/llm_progress_20261006/snapshot.json'
    root, commit = repository(tmp_path, {name: '{"budget":100}'})
    git(root, 'sparse-checkout', 'set', '--no-cone', '/docs/')
    rows, candidates, _, _ = audit_tool.audit(root, baseline=commit)
    row = by_path(rows)[name]
    assert row['candidate_checkout_state'] == 'TRACKED_SPARSE_OMITTED'
    assert row['candidate_path_exists'] is False
    assert row['hash_algorithm'] == 'SHA256'
    assert row['git_blob'] == git(root, 'rev-parse', 'HEAD:' + name)


def test_worktree_removal_is_distinguished_from_sparse_omission(tmp_path):
    name = 'reports/llm_progress_20261006/snapshot.json'
    root, commit = repository(tmp_path, {name: '{"budget":100}'})
    (root / name).unlink()
    rows, _, _, _ = audit_tool.audit(root, baseline=commit)
    assert by_path(rows)[name]['candidate_checkout_state'] == 'REMOVED_FROM_CANDIDATE_WORKTREE_OR_INDEX'


def test_machine_locks_and_preregistration_override_candidate_categories(tmp_path):
    root, commit = repository(tmp_path, {
        'docs/PRUNE_MANIFEST.csv': 'path,hash\n',
        'docs/repository/phase2/HISTORICAL_PROTOCOLS.json': '{"protocols":{}}',
        'reports/llm_progress_20261006/freeze_manifest.json': '{}',
        'reports/llm_progress_20261006/README.md': 'progress snapshot',
    })
    rows, _, _, _ = audit_tool.audit(root, baseline=commit)
    rows = by_path(rows)
    for p in ('docs/PRUNE_MANIFEST.csv', 'docs/repository/phase2/HISTORICAL_PROTOCOLS.json',
              'reports/llm_progress_20261006/freeze_manifest.json'):
        assert rows[p]['deletion_decision'] == 'KEEP_PROTECTED'
    assert rows['reports/llm_progress_20261006/README.md']['deletion_decision'] == 'ELIGIBLE_AFTER_REMOTE_RECOVERY_GATE'


def test_separate_main_references_are_visible_without_writing_main(tmp_path):
    name = 'docs/model/CLEAN_QGEOGNN_DESIGN.md'
    root, baseline = repository(tmp_path, {name: 'historical design'})
    p = root / 'scripts/old_controller.py'
    p.parent.mkdir()
    p.write_text("DOCUMENT = 'docs/model/CLEAN_QGEOGNN_DESIGN.md'\n")
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'main observer')
    before = git(root, 'rev-parse', 'HEAD')
    rows, _, _, _ = audit_tool.audit(root, baseline=baseline, active_root=root)
    assert by_path(rows)[name]['separate_main_reference_count'] > 0
    assert git(root, 'rev-parse', 'HEAD') == before
    assert not git(root, 'status', '--porcelain')


def test_directory_and_ambiguous_basename_rules_do_not_expand_cartesian_edges():
    paths = {'studies/demo/r'+str(i)+'/selection.json' for i in range(1000)}
    edges = audit_tool.reference_edges('src/consumer.py', "ROOT = 'studies/demo/'\nNAME = 'selection.json'", paths, audit_tool.indexes(paths))
    assert len(edges) == 2
    assert {e['to'] for e in edges} == {'studies/demo/', '**/selection.json'}


def test_relative_imports_are_resolved_without_importing_modules():
    paths = {'studies/demo/consumer.py', 'studies/demo/helper.py'}
    edges = audit_tool.reference_edges('studies/demo/consumer.py', 'from .helper import contract\n', paths, audit_tool.indexes(paths))
    assert any(e['to']=='studies/demo/helper.py' and e['kind']=='DIRECT_IMPORT' for e in edges)


def test_deterministic_gzip_keeps_complete_machine_artifacts_without_duplicates(tmp_path):
    root, commit = repository(tmp_path, {'docs/model/INPUT_SCHEMA.md': 'public input contract'})
    results = audit_tool.audit(root, baseline=commit)
    output = tmp_path / 'output'
    audit_tool.write_results(output, *results, compress=True)
    assert not (output/'ARTIFACT_INVENTORY.csv').exists()
    assert not (output/'DEPENDENCY_REFERENCES.json').exists()
    with gzip.open(output/'ARTIFACT_INVENTORY.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    assert set(rows[0]) == set(results[0][0])
    assert {r['path'] for r in rows} == {r['path'] for r in results[0]}
    with gzip.open(output/'DEPENDENCY_REFERENCES.json.gz', 'rt') as stream:
        assert json.load(stream) == results[2]
    before = (output/'DEPENDENCY_REFERENCES.json.gz').read_bytes()
    audit_tool.write_results(output, *results, compress=True)
    assert (output/'DEPENDENCY_REFERENCES.json.gz').read_bytes() == before
