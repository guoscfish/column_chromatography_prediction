"""Audit candidate source closure and recoverable group retirement from Git.

Only source bytes are inspected. Dataset, prediction, checkpoint and provider
payload contents are never opened. Retired modules are evaluated as a group:
callers retired with them are distinct from remaining stable-runtime callers.
"""
from __future__ import annotations
import argparse
import ast
import collections
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

BASELINE = '4ee480378b0293f1985e3ee4a09df2c78f77901e'
ARCHIVE_REF = 'archive/qgeognn-al-phase1-20261109'
ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
 'src/qgeognn_al/models/qgeognn_v2.py': 'src/qgeognn_al/predictor/model.py',
 'src/qgeognn_al/data.py': 'src/qgeognn_al/predictor/data.py',
 'src/qgeognn_al/input_schema.py': 'src/qgeognn_al/predictor/input_schema.py',
 'src/qgeognn_al/schemas/conditions.py': 'src/qgeognn_al/predictor/conditions.py',
 'src/qgeognn_al/training/predictor.py': 'src/qgeognn_al/predictor/training.py;src/qgeognn_al/predictor/inference.py',
 'src/qgeognn_al/evaluation/point.py': 'src/qgeognn_al/al/evaluation/point.py',
 'src/qgeognn_al/artifacts.py': 'src/qgeognn_al/infra/artifacts.py',
 'src/qgeognn_al/resources.py': 'src/qgeognn_al/infra/resources.py',
 'src/qgeognn_al/partitions.py': 'src/qgeognn_al/infra/partitions.py',
 'src/qgeognn_al/active_learning_v2/runner.py': 'src/qgeognn_al/predictor/preprocessing.py;src/qgeognn_al/predictor/inference.py;src/qgeognn_al/al/workflow/backends.py',
 'src/qgeognn_al/active_learning_v2/benchmark_reporting.py': 'src/qgeognn_al/al/evaluation/point.py (shared metric mathematics only; historical reporting protocol archived)',
 'src/qgeognn_al/active_learning_v2/efficiency_reporting.py': 'src/qgeognn_al/al/evaluation/curves.py (AULC/labels-to-target mathematics only; historical reporting protocol archived)',
 'src/qgeognn_al/active_learning_v2/protocol.py': 'src/qgeognn_al/infra/identity.py (identity hash helpers only; historical partition/label protocol archived)',
 'src/qgeognn_al/active_learning_v2/coverage.py': 'src/qgeognn_al/al/representations/latent.py',
 'src/qgeognn_al/active_learning_v2/gradient_features.py': 'src/qgeognn_al/al/representations/gradient.py',
 'src/qgeognn_al/active_learning_v2/gradient_transforms.py': 'src/qgeognn_al/al/representations/center_width.py',
 'src/qgeognn_al/active_learning_v2/fused_features.py': 'src/qgeognn_al/al/representations/fused.py',
 'src/qgeognn_al/active_learning_v2/lcmd.py': 'src/qgeognn_al/al/acquisitions/lcmd.py',
 'src/qgeognn_al/active_learning_v2/maxdet.py': 'src/qgeognn_al/al/acquisitions/maxdet.py',
 'src/qgeognn_al/active_learning_v2/ivr.py': 'src/qgeognn_al/al/acquisitions/ivr.py',
 'src/qgeognn_al/representations/base.py': 'src/qgeognn_al/al/representations/base.py',
 'src/qgeognn_al/representations/latent.py': 'src/qgeognn_al/al/representations/latent.py',
 'src/qgeognn_al/acquisitions/base.py': 'src/qgeognn_al/al/acquisitions/base.py',
 'src/qgeognn_al/acquisitions/lcmd.py': 'src/qgeognn_al/al/acquisitions/lcmd.py',
 'src/qgeognn_al/agents/plans.py': 'src/qgeognn_al/al/agents/plans.py',
 'src/qgeognn_al/workflows/selection.py': 'src/qgeognn_al/al/workflow/state.py',
 'scripts/plot_saved_validation.py': 'tools/plot_saved_validation.py',
}
OWNED_SCOPES = ('src/', 'scripts/', 'tests/', 'tools/')
LEGACY_PARTS = ('active_learning_v2', 'active_learning_v3', 'active_learning_v3_2')


def git(repository, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repository), *args])


def baseline_sources(repository, commit):
    entries = {}
    for raw in git(repository, 'ls-tree', '-r', '-z', commit).split(b'\0'):
        if raw:
            metadata, encoded = raw.split(b'\t', 1)
            mode, kind, oid = metadata.split()
            path = encoded.decode()
            if kind == b'blob' and path.endswith('.py'):
                entries[path] = {'blob_oid': oid.decode(), 'mode': mode.decode()}
    source = {}
    process = subprocess.Popen(['git', '--no-optional-locks', '-C', str(repository), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for path, item in entries.items():
            process.stdin.write((item['blob_oid'] + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            data = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            source[path] = data
    finally:
        process.stdin.close()
        process.wait()
    return entries, source


def aliases(path):
    name = path[:-3].replace('/', '.')
    if name.endswith('.__init__'):
        name = name[:-9]
    values = {name}
    if name.startswith('src.'):
        values.add(name[4:])
    if path.startswith('application/'):
        values.add(Path(path).stem)
    return values


def resolve_imports(path, tree, module_map):
    module = path[:-3].replace('/', '.')
    package = module[:-9] if module.endswith('.__init__') else module.rpartition('.')[0]
    targets = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [entry.name for entry in node.names]
        elif isinstance(node, ast.ImportFrom):
            try:
                base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else (node.module or '')
            except (ImportError, ValueError):
                continue
            names = [base] + [base + '.' + entry.name for entry in node.names if entry.name != '*']
        for name in names:
            targets.update(module_map.get(name, ()))
    targets.discard(path)
    return targets


def area(path):
    if path.startswith('src/'):
        return 'stable_package'
    if path.startswith('scripts/'):
        return 'entrypoint'
    if path.startswith('tools/'):
        return 'audit_or_history_tool'
    if path.startswith('tests/'):
        return 'test_contract'
    return 'historical_evidence'


def graph(source, all_known_paths=None):
    mapping = collections.defaultdict(set)
    by_basename = collections.defaultdict(set)
    for path in set(source) | set(all_known_paths or ()):
        for name in aliases(path):
            mapping[name].add(path)
        by_basename[Path(path).name].add(path)
    edges, parsed, failures = [], {}, {}
    for path, raw in source.items():
        try:
            parsed[path] = ast.parse(raw.decode('utf-8-sig'), filename=path)
        except (SyntaxError, UnicodeDecodeError) as error:
            failures[path] = str(error)
            continue
        tree = parsed[path]
        docstrings = set()
        for owner in ast.walk(tree):
            if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and owner.body:
                first = owner.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    docstrings.add(id(first.value))
        for target in sorted(resolve_imports(path, tree, mapping)):
            edges.append({'from': path, 'to': target, 'kind': 'IMPORT'})
        subprocess_targets, hash_targets = set(), set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = ast.unparse(node.func)
            literals = [value.value for value in ast.walk(node) if isinstance(value, ast.Constant) and isinstance(value.value, str)]
            for value in literals:
                for basename in re.findall(r'[A-Za-z_][A-Za-z_0-9]*\.py', value):
                    for target in by_basename.get(basename, ()):
                        if target == path:
                            continue
                        if function.startswith('subprocess.'):
                            subprocess_targets.add(target)
                        elif 'sha256' in function or 'hash' in function:
                            hash_targets.add(target)
        for target in sorted(subprocess_targets):
            edges.append({'from': path, 'to': target, 'kind': 'SUBPROCESS_LITERAL'})
        for target in sorted(hash_targets):
            edges.append({'from': path, 'to': target, 'kind': 'HASH_LITERAL'})
        literals = [value.value for value in ast.walk(tree) if isinstance(value, ast.Constant) and isinstance(value.value, str) and id(value) not in docstrings]
        targets = set()
        for literal in literals:
            for basename in re.findall(r'[A-Za-z_][A-Za-z_0-9]*\.py', literal):
                targets.update(by_basename.get(basename, ()))
            for token in re.findall(r'\b(?:src\.)?qgeognn_al(?:\.[A-Za-z_][A-Za-z_0-9]*)+|\bscripts(?:\.[A-Za-z_][A-Za-z_0-9]*)+', literal):
                while token:
                    targets.update(mapping.get(token, ()))
                    token = token.rpartition('.')[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and id(node) in docstrings:
                for basename in re.findall(r'[A-Za-z_][A-Za-z_0-9]*\.py', node.value):
                    for target in by_basename.get(basename, set()) - {path}:
                        edges.append({'from': path, 'to': target, 'kind': 'DOCSTRING_REFERENCE'})
        for target in sorted(targets - {path} - subprocess_targets - hash_targets):
            edges.append({'from': path, 'to': target, 'kind': 'PATH_LITERAL_CONSERVATIVE'})
    benchmark = 'src/qgeognn_al/active_learning_v2/benchmark_protocol.py'
    if benchmark in source:
        for target in sorted(source):
            if target != benchmark and target.startswith(('src/qgeognn_al/', 'application/')):
                edges.append({'from': benchmark, 'to': target, 'kind': 'WHOLE_SOURCE_HASH_SCOPE'})
    return edges, parsed, failures



def condensed_dag(nodes, edges):
    """Keep full edges and collapse import cycles into a genuine DAG."""
    adjacency = {name: set() for name in nodes}
    for edge in edges:
        if edge['kind'] == 'IMPORT' and edge['from'] in adjacency and edge['to'] in adjacency:
            adjacency[edge['from']].add(edge['to'])
    sequence, index, low, stack, stacked, components = 0, {}, {}, [], set(), []
    def visit(node):
        nonlocal sequence
        index[node] = low[node] = sequence
        sequence += 1
        stack.append(node)
        stacked.add(node)
        for child in sorted(adjacency[node]):
            if child not in index:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in stacked:
                low[node] = min(low[node], index[child])
        if low[node] == index[node]:
            group = []
            while True:
                child = stack.pop()
                stacked.remove(child)
                group.append(child)
                if child == node:
                    break
            components.append(sorted(group))
    for node in sorted(adjacency):
        if node not in index:
            visit(node)
    components.sort(key=lambda group: group[0])
    membership = {name: number for number, group in enumerate(components) for name in group}
    links = sorted({(membership[node], membership[child]) for node in adjacency for child in adjacency[node]
                    if membership[node] != membership[child]})
    return {'components': components, 'edges': links, 'cycle_component_count': sum(len(g) > 1 for g in components),
            'scope': 'IMPORT edges only; full typed source/runtime/hash/document edges are retained separately'}


def duplicate_functions(parsed):
    groups = collections.defaultdict(list)
    wrappers = []
    for path, tree in parsed.items():
        if not path.startswith('src/qgeognn_al/'):
            continue
        body = [item for item in tree.body if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
        if Path(path).name != '__init__.py' and body and all(isinstance(item, (ast.Import, ast.ImportFrom)) or
            (isinstance(item, ast.Assign) and all(isinstance(target, ast.Name) and target.id == '__all__' for target in item.targets)) for item in body):
            wrappers.append(path)
        for function in body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            normalized = copy.deepcopy(function)
            normalized.name = 'function'
            if normalized.body and isinstance(normalized.body[0], ast.Expr) and isinstance(normalized.body[0].value, ast.Constant) and isinstance(normalized.body[0].value.value, str):
                normalized.body.pop(0)
            identity = hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()
            groups[identity].append({'path': path, 'function': function.name, 'line': function.lineno})
    repeated = [locations for locations in groups.values() if len(locations) > 1]
    return {'duplicate_top_level_function_groups': len(repeated), 'excess_function_copies': sum(len(g) - 1 for g in repeated),
            'groups': repeated, 'noninit_reexport_modules': wrappers,
            'definition': 'Exact normalized AST of top-level functions, excluding docstring/function name but retaining argument names/constants; not semantic equivalence'}


def write_csv(path, columns, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for record in records:
            writer.writerow({name: json.dumps(record.get(name), sort_keys=True) if isinstance(record.get(name), (dict, list, tuple)) else record.get(name, '') for name in columns})


def audit(repository, candidate, commit=BASELINE):
    repository, candidate = Path(repository).resolve(), Path(candidate).resolve()
    entries, before = baseline_sources(repository, commit)
    current = {}
    for path, data in before.items():
        file = candidate / path
        if file.is_file():
            current[path] = file.read_bytes()
        elif not path.startswith(OWNED_SCOPES):
            # Partial source-only scratch trees do not imply removal of sealed
            # historical artifacts. Final worktree audit retains their Git IDs.
            current[path] = data
    for file in candidate.rglob('*.py'):
        name = file.relative_to(candidate).as_posix()
        if '__pycache__' not in file.parts and '.git' not in file.parts and not file.is_symlink():
            current[name] = file.read_bytes()
    before_edges, before_ast, before_errors = graph(before)
    after_edges, after_ast, after_errors = graph(current, before)
    retired = sorted(set(before) - set(current))
    retired_set = set(retired)
    incoming_before = collections.defaultdict(list)
    incoming_after = collections.defaultdict(list)
    for edge in before_edges:
        incoming_before[edge['to']].append(edge)
    for edge in after_edges:
        incoming_after[edge['to']].append(edge)
    stable_unresolved = [edge for edge in after_edges if edge['to'] in retired_set and area(edge['from']) in ('stable_package', 'entrypoint') and edge['kind'] != 'DOCSTRING_REFERENCE']
    records, deletions, mapping = [], [], []
    for name in sorted(set(before) | set(current)):
        old, new = before.get(name), current.get(name)
        replacement = REPLACEMENTS.get(name, '')
        if old is None:
            status = 'NEW_STABLE_OR_CONTRACT'
        elif new is None:
            status = 'MIGRATED_CORE_OLD_PATH_REMOVED' if replacement else 'ARCHIVED_PROTOCOL_SOURCE'
        elif old == new:
            status = 'UNCHANGED_HISTORICAL_EVIDENCE' if area(name) == 'historical_evidence' else 'RETAINED_BYTE_IDENTICAL'
        else:
            status = 'UPDATED_CANDIDATE_ONLY'
        tree = after_ast.get(name) or before_ast.get(name)
        purpose = (ast.get_docstring(tree) or '').splitlines()[0] if tree and ast.get_docstring(tree) else area(name)
        record = {'path': name, 'area': area(name), 'purpose': purpose[:220],
                  'before_lines': len(old.splitlines()) if old is not None else 0,
                  'after_lines': len(new.splitlines()) if new is not None else 0,
                  'status': status, 'replacement': replacement,
                  'dynamic_capabilities': [token for token in ['importlib', 'sys.path', 'subprocess', 'getattr(', 'pickle.load', 'torch.load'] if token in (new or old or b'').decode(errors='replace')],
                  'authoritative_output_methods': sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'savefig', 'to_csv', 'to_json', 'write_text', 'write_bytes'}}) if tree else [],
                  'source_commit': commit if old is not None else 'CANDIDATE_UNCOMMITTED',
                  'git_blob_oid': entries.get(name, {}).get('blob_oid'),
                  'before_sha256': hashlib.sha256(old).hexdigest() if old is not None else '',
                  'after_sha256': hashlib.sha256(new).hexdigest() if new is not None else '',
                  'before_callers': incoming_before[name], 'after_callers': incoming_after[name]}
        records.append(record)
        if old is not None:
            mapping.append({'old_path': name, 'new_path': replacement or (name if new is not None else ARCHIVE_REF + ':' + name),
                            'decision': status, 'scientific_math': 'UNCHANGED_OR_ARCHIVED',
                            'legacy_commit': commit, 'recovery_ref': ARCHIVE_REF,
                            'exit_condition': 'Candidate only. Active main retains its original paths until CW completion and explicit switching authorization.'})
        if new is None:
            remaining = [e for e in stable_unresolved if e['to'] == name]
            historical = [e for e in incoming_after[name] if area(e['from']) == 'historical_evidence']
            closed = [e for e in incoming_before[name] if e['from'] in retired_set]
            deletions.append({'path': name, 'source_commit': commit, 'git_blob_oid': entries[name]['blob_oid'],
                              'sha256': record['before_sha256'], 'removed_lines': record['before_lines'],
                              'recovery_ref': ARCHIVE_REF, 'replacement': replacement,
                              'retired_callers_in_closed_group': closed,
                              'remaining_stable_runtime_references': remaining,
                              'frozen_historical_evidence_references': historical,
                              'historical_reference_policy': 'HISTORICAL_ONLY_RESTORE_REF',
                              'file_kind': 'python_source',
                              'decision': 'BLOCKED_STABLE_REFERENCE' if remaining else 'RECOVERABLE_CANDIDATE_RETIREMENT',
                              'restore_command': f'git show {commit}:{name} > <independent-history-worktree>/{name}',
                              'main_action': 'NONE; original main source is separately hash-guarded'})
    python_deleted_count = len(deletions)
    python_deleted_lines = sum(row['removed_lines'] for row in deletions)
    plan_path = candidate / 'docs/repository/phase2/SOURCE_EXIT_PLAN.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get('baseline_commit') != commit:
            raise ValueError('exit plan source commit differs from the fixed inventory baseline')
        for name in plan['removed_scratch_paths']:
            if name.endswith('.py') or (candidate / name).exists():
                continue
            file = Path(name)
            if file.is_absolute() or '..' in file.parts or file.suffix not in {'.md', '.json'}:
                raise ValueError('non-Python retirement plan must name only source docs/configs/fixtures')
            oid = git(repository, 'rev-parse', commit + ':' + name).decode().strip()
            raw = git(repository, 'show', commit + ':' + name)
            deletions.append({'path': name, 'source_commit': commit, 'git_blob_oid': oid,
                              'sha256': hashlib.sha256(raw).hexdigest(), 'removed_lines': len(raw.splitlines()),
                              'recovery_ref': ARCHIVE_REF, 'replacement': '',
                              'retired_callers_in_closed_group': [], 'remaining_stable_runtime_references': [],
                              'frozen_historical_evidence_references': [], 'historical_reference_policy': 'HISTORICAL_ONLY_RESTORE_REF',
                              'file_kind': 'source_document_or_synthetic_fixture',
                              'decision': 'RECOVERABLE_CANDIDATE_RETIREMENT',
                              'restore_command': f'git show {commit}:{name} > <independent-history-worktree>/{name}',
                              'main_action': 'NONE; no original scientific result is removed'})
    before_dup, after_dup = duplicate_functions(before_ast), duplicate_functions(after_ast)
    versions = {name: {'before': sum(p.startswith('src/qgeognn_al/' + name + '/') for p in before),
                       'after': sum(p.startswith('src/qgeognn_al/' + name + '/') for p in current)} for name in LEGACY_PARTS}
    summary = {'schema_version': 1, 'source_commit': commit, 'before_python_files': len(before), 'after_inventory_python_files': len(current),
               'legacy_versions': versions, 'legacy_protocol_directories_before': sum(v['before'] > 0 for v in versions.values()),
               'legacy_protocol_directories_after': sum(v['after'] > 0 for v in versions.values()),
               'stable_src_python_before': sum(p.startswith('src/qgeognn_al/') for p in before),
               'stable_src_python_after': sum(p.startswith('src/qgeognn_al/') for p in current),
               'representation_acquisition_modules_after': len([p for p in current if p.startswith(('src/qgeognn_al/al/representations/', 'src/qgeognn_al/al/acquisitions/')) and Path(p).name != '__init__.py']),
               'active_algorithm_implementation_files_after': len([p for p in current if p.startswith(('src/qgeognn_al/al/representations/', 'src/qgeognn_al/al/acquisitions/')) and Path(p).name not in {'__init__.py', 'base.py'}]),
               'recoverable_deleted_python_files': python_deleted_count, 'recoverable_deleted_lines': python_deleted_lines,
               'recoverable_deleted_source_artifacts': len(deletions),
               'recoverable_deleted_text_artifact_lines_including_docs_fixtures': sum(r['removed_lines'] for r in deletions),
               'remaining_stable_runtime_references_to_retired_source': stable_unresolved,
               'duplicate_functions_before': before_dup, 'duplicate_functions_after': after_dup,
               'parse_errors_before': before_errors, 'parse_errors_after': after_errors,
               'safety': {'truth_tables_opened': 0, 'checkpoint_payloads_opened': 0, 'active_files_written': 0},
               'limits': ['Source-only graph; arbitrary reflection and external consumers cannot be proven absent',
                          'Path-literal edges conservatively overapproximate basename mentions',
                          'Frozen source outside stable roots remains evidence; restoring its formal runner requires its immutable Git source ref',
                          'Only fully overlaid src/scripts/tests/tools scopes are considered candidate retirements in partial scratch trees']}
    return records, mapping, deletions, {'before': {'nodes': sorted(before), 'edges': before_edges, 'condensed_import_dag': condensed_dag(before, before_edges)}, 'after': {'nodes': sorted(current), 'edges': after_edges, 'condensed_import_dag': condensed_dag(current, after_edges)}, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT)
    parser.add_argument('--source-commit', default=BASELINE)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/repository/phase2')
    args = parser.parse_args()
    records, mapping, deletions, graph_data = audit(args.repository, args.candidate, args.source_commit)
    active_marker = args.candidate / 'docs/repository/phase2/GATE0.json'
    if active_marker.exists():
        active = Path(json.loads(active_marker.read_text())['active_root']).resolve()
        destination = args.output.resolve()
        if destination == active or active in destination.parents:
            parser.error('audit outputs must not enter the active main directory')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'INVENTORY.csv', list(records[0]) if records else ['path'], records)
    write_csv(args.output / 'MIGRATION_MAP.csv', list(mapping[0]) if mapping else ['old_path'], mapping)
    write_csv(args.output / 'DELETION_PROOF.csv', list(deletions[0]) if deletions else ['path'], deletions)
    (args.output / 'DEPENDENCY_DAG.json').write_text(json.dumps(graph_data, separators=(',', ':')) + '\n')
    (args.output / 'STRUCTURAL_COUNTS.json').write_text(json.dumps(graph_data['summary'], indent=2) + '\n')
    summary = graph_data['summary']
    print(json.dumps({key: summary[key] for key in ['before_python_files', 'after_inventory_python_files',
          'legacy_versions', 'legacy_protocol_directories_after', 'stable_src_python_before', 'stable_src_python_after',
          'active_algorithm_implementation_files_after', 'recoverable_deleted_python_files', 'recoverable_deleted_lines']}, indent=2))
    return 1 if summary['parse_errors_after'] or summary['remaining_stable_runtime_references_to_retired_source'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
