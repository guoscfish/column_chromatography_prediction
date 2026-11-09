"""Static retirement proof distinguishes documentation, runtime and hash edges."""
import importlib.util
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('migration_inventory_under_test', ROOT / 'tools/build_migration_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_graph_keeps_real_imports_and_documentary_mentions_distinct():
    old = 'src/qgeognn_al/active_learning_v2/kernel.py'
    source = {'src/qgeognn_al/al/new.py': b'"""Historical source is kernel.py."""\nVALUE = 1\n',
              'src/qgeognn_al/al/bad.py': b'from ..active_learning_v2.kernel import choose\n'}
    edges, _, errors = inventory.graph(source, [old])
    assert not errors
    assert {'from': 'src/qgeognn_al/al/new.py', 'to': old, 'kind': 'DOCSTRING_REFERENCE'} in edges
    assert {'from': 'src/qgeognn_al/al/bad.py', 'to': old, 'kind': 'IMPORT'} in edges
    assert not any(e['from'].endswith('/new.py') and e['kind'] == 'PATH_LITERAL_CONSERVATIVE' for e in edges)


def test_import_cycles_are_explicitly_collapsed_into_a_dag():
    nodes = ['a.py', 'b.py', 'c.py']
    edges = [{'from': 'a.py', 'to': 'b.py', 'kind': 'IMPORT'},
             {'from': 'b.py', 'to': 'a.py', 'kind': 'IMPORT'},
             {'from': 'c.py', 'to': 'a.py', 'kind': 'IMPORT'}]
    dag = inventory.condensed_dag(nodes, edges)
    assert dag['components'] == [['a.py', 'b.py'], ['c.py']]
    assert dag['edges'] == [(1, 0)]
    assert dag['cycle_component_count'] == 1


def test_retirement_is_a_group_and_remaining_stable_import_blocks(tmp_path):
    repo = tmp_path / 'source'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-b', 'main')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Inventory Test')
    first = repo / 'src/qgeognn_al/active_learning_v2/first.py'
    first.parent.mkdir(parents=True)
    first.write_text('from .second import choose\n')
    (first.parent / 'second.py').write_text('def choose():\n    return 7\n')
    git('add', '.')
    git('commit', '-qm', 'legacy group')
    commit = git('rev-parse', 'HEAD')
    candidate = tmp_path / 'candidate'
    target = candidate / 'src/qgeognn_al/al/new.py'
    target.parent.mkdir(parents=True)
    target.write_text('VALUE = 7\n')
    _, _, deleted, graph = inventory.audit(repo, candidate, commit)
    assert len(deleted) == 2
    assert all(row['decision'] == 'RECOVERABLE_CANDIDATE_RETIREMENT' for row in deleted)
    assert not graph['summary']['remaining_stable_runtime_references_to_retired_source']
    assert any(row['retired_callers_in_closed_group'] for row in deleted)
    target.write_text('from ..active_learning_v2.second import choose\n')
    _, _, deleted, graph = inventory.audit(repo, candidate, commit)
    assert graph['summary']['remaining_stable_runtime_references_to_retired_source']
    assert any(row['decision'] == 'BLOCKED_STABLE_REFERENCE' for row in deleted)
