"""Read-only artifact retirement audit; never executes scientific code.

Git is the source of the baseline universe, including sparse-omitted files.
Dataset/checkpoint/label/provider payload contents are not inspected. Eligible
documentation and code are decoded for conservative static references; only
eligible retirement artifacts additionally receive an opaque SHA-256 digest.
No candidate is deleted by this program and remote recovery is a separate gate.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import posixpath
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

BASELINE = '0e33a86d3dea3945acb0a638ca78a8f5065980a5'
ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOTS = ('docs/', 'reports/', 'experiments/', 'studies/')
RELEASE = 'docs/repository/releases/document-cleanup-20261110/'
AUDIT_SELF_SOURCES = {'tools/audit_research_artifacts.py', 'tests/document_cleanup/test_artifact_audit.py'}
MACHINE_CONTRACTS = {
    'docs/PROTECTED_ARTIFACTS.json', 'docs/data_consumption_register.json',
    'docs/PRUNE_MANIFEST.csv', 'docs/repository/RESEARCH_SCOPE.json',
    'docs/repository/RETIREMENTS.json',
}
PHASE1_NAMES = {
    'BRANCH_LAYOUT.md', 'CODE_INVENTORY.csv', 'DEPENDENCY_AUDIT.json',
    'LAB_DELIVERY.json', 'PLOT_EQUIVALENCE.json', 'REFACTOR_REPORT.md',
    'RETIREMENT_CANDIDATES.csv', 'RETIREMENT_MANIFEST.json',
    'RETIREMENT_PROOF.md', 'RUNTIME_AND_FROZEN_DEPENDENCIES.md',
}
OLD_DESIGNS = {
    'docs/QGEOGNN_IMPLEMENTATION_VARIANTS.md',
    'docs/model/CLEAN_QGEOGNN_DESIGN.md', 'docs/model/LEGACY_QGEOGNN_AUDIT.md',
    'docs/protocols/4G_PREDICTOR_BENCHMARK.md',
    'docs/research/4G_LLM_DUAL_EXPERT_ACTIVE_LEARNING_V1_2026-09-27.md',
    'docs/research/4G_LLM_SCIENTIST_ACTIVE_LEARNING_V2_2026-09-26.md',
    'docs/research/ACTIVE_LEARNING_TRANSFER_LITERATURE_REVIEW_2026-09-18.md',
}
OLD_PROGRESS = {
    'active_learning_latest', 'active_learning_status_20261005',
    'llm_progress_20261006', 'llm_progress_20261008',
    'llm_eight_to1165_20261014', 'llm_eight_to1293_20261015',
    'llm_eight_to1421_20261017', 'repository_cleanup_20261106',
}
STRICT_STUDIES = (
    'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/',
    'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/',
    'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/',
)
STRICT_REPORTS = (
    'reports/free_lcmd_five_seeds_to1485_20261026/',
    'reports/free_llm_five_seeds_to1485_20261024/',
    'reports/cw_retry_evidence_20261106_1445/',
    'reports/llm_eight_to1549_20261018/',
)
SENSITIVE_SEGMENTS = {
    'requests', 'responses', 'raw_requests', 'raw_responses', 'credentials',
    'keys', 'key', 'messages', 'transcripts', 'provider_payloads',
}
RAW_SUFFIXES = {'.pt', '.pth', '.pkl', '.pickle', '.npy', '.npz', '.parquet', '.gz', '.xz', '.sqlite', '.db'}
CODE_SUFFIXES = {'.py', '.sh', '.bash'}
CONFIG_NAMES = {'protocol.json', 'PROTOCOL.json', 'config.json', 'CONFIG.json',
                'manifest.json', 'source_manifest.json', 'frozen_manifest.json'}
PATH_RE = re.compile(r'(?<![\w/])(?:docs|reports|experiments|studies|src|scripts|tools|configs)/[^\s\"\'`<>\]\[(){};,]+')
MD_LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"\'][^)]*)?\)')


def git(repository, *arguments):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repository), *arguments])


def tree(repository, ref):
    files = {}
    for raw in git(repository, 'ls-tree', '-r', '-l', '-z', ref).split(b'\0'):
        if not raw:
            continue
        meta, name = raw.split(b'\t', 1)
        mode, kind, oid, size = meta.split()
        if kind == b'blob':
            files[name.decode()] = {'mode': mode.decode(), 'git_blob': oid.decode(), 'bytes': int(size)}
    return files


class BlobReader:
    """Only callers' explicitly selected blobs are opened, without checkout."""
    def __init__(self, repository):
        self.process = subprocess.Popen(['git', '--no-optional-locks', '-C', str(repository), 'cat-file', '--batch'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.read_paths = set()

    def read(self, path, entry):
        self.read_paths.add(path)
        self.process.stdin.write((entry['git_blob'] + '\n').encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        data = self.process.stdout.read(int(header[2]))
        if self.process.stdout.read(1) != b'\n':
            raise ValueError('invalid Git batch boundary')
        return data

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise ValueError('Git batch reader failed')


def sensitive(path):
    parts = set(Path(path).parts)
    stem = Path(path).stem.lower()
    return bool(parts & SENSITIVE_SEGMENTS or any(word in stem for word in
                ('credential', 'raw_request', 'raw_response', 'api_key', 'key_switch', 'transport_probe')))


def proposed_group(path):
    """Reviewed categories are proposals, never sufficient deletion authority."""
    p = Path(path)
    if path in OLD_DESIGNS:
        return 'superseded_design'
    if path.startswith('docs/repository/') and p.parent.as_posix() == 'docs/repository' and p.name in PHASE1_NAMES:
        return 'phase1_engineering_audit'
    if path.startswith(('docs/repository/validation/', 'docs/repository/runtime_snapshot_20261109/')):
        return 'phase1_engineering_audit'
    if path in ('docs/repository_inventory_before.json', 'docs/repository_inventory_after.json'):
        return 'early_inventory_snapshot'
    parts = path.split('/')
    if len(parts) > 2 and parts[0] == 'reports' and parts[1] in OLD_PROGRESS:
        # Controllers/source, authorization, raw log streams and their receipts
        # remain separately protected even if a human snapshot is superseded.
        if p.suffix in {'.md', '.png', '.svg', '.pdf', '.json', '.csv'}:
            return 'superseded_progress_snapshot'
    if path in ('experiments/DECISION_ARCHIVE.md', 'experiments/METHOD_DECISION_REGISTER.md',
                'experiments/RESEARCH_ROADMAP.md'):
        return 'historical_decision_documents'
    return ''


def protected_reason(path, registered):
    if path in MACHINE_CONTRACTS:
        return 'MACHINE_CONTRACT'
    if any(path == target or path.startswith(target.rstrip('/') + '/') for target in registered):
        return 'REGISTERED_PROTECTED_ARTIFACT'
    if path.startswith('docs/repository/phase2/'):
        return 'PHASE2_RESTORE_AND_VALIDATION_CONTRACT'
    if path.startswith(STRICT_STUDIES + STRICT_REPORTS):
        return 'ACTIVE_OR_RECOVERY_IDENTITY'
    if sensitive(path):
        return 'SENSITIVE_PAYLOAD_OR_CREDENTIAL_METADATA'
    lower = path.lower()
    parts = set(lower.split('/'))
    if Path(path).suffix.lower() in RAW_SUFFIXES:
        return 'OPAQUE_DATA_MODEL_OR_RUNTIME_PAYLOAD'
    if parts & {'frozen_source', 'source_snapshot', 'checkpoints', 'partitions', 'selections',
                'requests', 'responses', 'raw_responses', 'raw_requests', 'authorization', 'authorizations'}:
        return 'FROZEN_SOURCE_SELECTION_PARTITION_OR_AUTHORIZATION'
    if any(token in Path(path).name.lower() for token in
           ('protocol', 'preregistration', 'ledger', 'label_access', 'freeze', 'scaler', 'canonical', 'graph_cache')):
        # Superseded design draft is explicitly proposed, but never a live prereg.
        if path != 'docs/protocols/4G_PREDICTOR_BENCHMARK.md':
            return 'SCIENTIFIC_IDENTITY_OR_TRUTH_CONSUMPTION'
    if path.startswith(('experiments/', 'studies/')) and Path(path).suffix.lower() in {'.csv', '.jsonl', '.log'}:
        return 'RESEARCH_DATA_METRIC_SEQUENCE_OR_LOG'
    if path.startswith(('experiments/', 'studies/')) and Path(path).suffix.lower() == '.json':
        return 'RESEARCH_JSON_NOT_INTERPRETED'
    if path.startswith('reports/') and any(token in Path(path).name.lower() for token in
           ('credential', 'recovery', 'recoveries', 'receipt', 'heartbeat', 'launch', 'authorization', 'transport', 'events')):
        return 'RUNTIME_OR_RECOVERY_METADATA'
    return ''


def inspectable(path, registered):
    if sensitive(path):
        return False
    suffix = Path(path).suffix.lower()
    if suffix in CODE_SUFFIXES or suffix == '.md':
        return True
    if path.startswith('docs/') and suffix in {'.json', '.csv', '.xml', '.txt', '.log'}:
        return True
    if path.startswith('configs/') and suffix in {'.json', '.toml', '.yaml', '.yml', '.ini'}:
        return True
    # Named scientific configurations are decoded only when they are not a
    # protected payload; all other research JSON stays opaque metadata.
    return Path(path).name in CONFIG_NAMES and not protected_reason(path, registered)


def source_scope(path):
    if path.startswith(('src/', 'scripts/', 'application/')):
        return 'STABLE_RUNTIME'
    if path.startswith('tools/'):
        return 'AUDIT_TOOL'
    if path.startswith('tests/'):
        return 'TEST_CONTRACT'
    if path.startswith(('docs/',)) or path.endswith('.md'):
        return 'DOCUMENT'
    if '/frozen_source/' in path or '/source_snapshot/' in path:
        return 'FROZEN_HISTORICAL_SOURCE'
    if path.startswith(('reports/', 'studies/')) and Path(path).suffix in CODE_SUFFIXES:
        return 'HISTORICAL_CONTROLLER_OR_SOURCE'
    return 'CONFIG_OR_HISTORICAL_METADATA'


def reference_edges(path, text, known, by_name, *, scope=None):
    """Conservative static references, retaining kind and source role."""
    scope = scope or source_scope(path)
    result = set()
    directory = posixpath.dirname(path)
    suffix = Path(path).suffix.lower()
    if suffix == '.md':
        for match in MD_LINK.finditer(text):
            value = unquote(match.group(1).strip('<>'))
            split = urlsplit(value)
            if split.scheme or value.startswith(('#', '//')):
                # SHA links are provenance, not live filesystem dependencies.
                if re.search(r'/blob/[0-9a-f]{40}/|/tree/[0-9a-f]{40}/', value):
                    for candidate in PATH_RE.findall(unquote(value)):
                        if candidate in known:
                            result.add((candidate, 'PINNED_HISTORY_LINK'))
                continue
            target = posixpath.normpath(posixpath.join(directory, split.path))
            if target in known:
                result.add((target, 'DOCUMENT_RELATIVE_LINK'))
            elif target.rstrip('/') in by_name['directories']:
                result.add((target.rstrip('/') + '/', 'DOCUMENT_DIRECTORY_LINK'))
        # Inline code/backtick paths and shell examples are still conservative
        # documentation mentions; they do not become import edges.
        stripped = MD_LINK.sub('', text)
        for token in PATH_RE.findall(stripped):
            target = token.rstrip('.:')
            if target in known:
                result.add((target, 'DOCUMENT_PATH_MENTION'))
    elif suffix in CODE_SUFFIXES:
        literals = []
        if suffix == '.py':
            try:
                parsed = ast.parse(text)
                for node in ast.walk(parsed):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = [a.name for a in node.names]
                        if isinstance(node, ast.ImportFrom):
                            module = path[:-3].replace('/', '.')
                            package = module[:-9] if module.endswith('.__init__') else module.rpartition('.')[0]
                            try:
                                base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else (node.module or '')
                            except (ImportError, ValueError):
                                base = node.module or ''
                            names = [base] + [base + '.' + a.name for a in node.names if a.name != '*']
                        for name in names:
                            candidate = name.replace('.', '/') + '.py'
                            package_candidate = name.replace('.', '/') + '/__init__.py'
                            for target in (candidate, 'src/' + candidate, package_candidate, 'src/' + package_candidate):
                                if target in known:
                                    result.add((target, 'DIRECT_IMPORT'))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        literals.append(node.value)
                    if isinstance(node, ast.Call):
                        name = ast.unparse(node.func)
                        if name.startswith('subprocess.') or name in ('os.system', 'os.popen'):
                            for value in ast.walk(node):
                                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                    for target in PATH_RE.findall(value.value):
                                        if target in known:
                                            result.add((target, 'SUBPROCESS_LITERAL'))
                if any('sha256' in value or 'code_hash' in value for value in literals) or any(
                    isinstance(node, ast.Call) and 'hash' in ast.unparse(node.func) for node in ast.walk(parsed)):
                    literal_kind = 'HASH_OR_FROZEN_PATH_LITERAL'
                else:
                    literal_kind = 'DYNAMIC_PATH_LITERAL'
            except SyntaxError:
                literals = [text]
                literal_kind = 'UNPARSED_SOURCE_PATH'
        else:
            literals = [text]
            literal_kind = 'SHELL_OR_CLI_PATH'
        for literal in literals:
            for token in PATH_RE.findall(literal):
                target = token.rstrip('.:')
                if target in known:
                    result.add((target, literal_kind))
                elif target.rstrip('/') in by_name['directories'] and len(target.rstrip('/').split('/')) >= 2:
                    result.add((target.rstrip('/') + '/', literal_kind + '_DIRECTORY_SCOPE'))
            # Pure basename matches catch ROOT/'folder'/'name.ext'; ambiguous
            # generic README/config names are omitted to prevent false certainty.
            for name in re.findall(r'[\w.-]+\.(?:json|csv|md|py|pt|npz)', literal):
                if name not in {'README.md', 'config.json', 'manifest.json', 'PROTOCOL.md', 'REPORT.md'}:
                    matches = by_name['files'].get(name, ())
                    if len(matches) == 1:
                        result.add((next(iter(matches)), 'BASENAME_LITERAL_CONSERVATIVE'))
                    elif matches:
                        result.add(('**/' + name, 'AMBIGUOUS_BASENAME_RULE'))
    else:
        for token in PATH_RE.findall(text):
            target = token.rstrip('.:')
            if target in known:
                result.add((target, 'CONFIG_OR_HASH_METADATA_REFERENCE'))
    return [{'from': path, 'to': target, 'kind': kind, 'source_scope': scope}
            for target, kind in sorted(result) if target != path]


def indexes(paths):
    files, directories = collections.defaultdict(set), collections.defaultdict(set)
    for path in paths:
        files[Path(path).name].add(path)
        parent = posixpath.dirname(path)
        while parent:
            directories[parent].add(path)
            parent = posixpath.dirname(parent)
    return {'files': files, 'directories': directories}


def owner(path):
    parts = path.split('/')
    if path.startswith('studies/'):
        return '/'.join(parts[:3]) if len(parts) > 3 else '/'.join(parts[:2])
    if path.startswith(('experiments/', 'reports/')):
        return '/'.join(parts[:2])
    if path.startswith('docs/repository/'):
        return 'repository-engineering-audit'
    return 'repository-documentation'


def audit(repository, *, baseline=BASELINE, main_ref='refs/heads/main', active_root=None):
    repository = Path(repository).resolve()
    baseline = git(repository, 'rev-parse', baseline + '^{commit}').decode().strip()
    entries = tree(repository, baseline)
    head_entries = tree(repository, 'HEAD')
    sparse_omitted = {raw[2:].decode() for raw in git(repository, 'ls-files', '-v', '-z').split(b'\0') if raw.startswith(b'S ')}
    removed_worktree = {raw.decode() for raw in git(repository, 'diff', '--name-only', '--diff-filter=D', '-z').split(b'\0') if raw}
    removed_index = {raw.decode() for raw in git(repository, 'diff', '--cached', '--name-only', '--diff-filter=D', '-z').split(b'\0') if raw}
    research = {p: e for p, e in entries.items() if p.startswith(RESEARCH_ROOTS)}
    reader = BlobReader(repository)
    scan_count, texts, sha, edges = 0, {}, {}, []
    try:
        registry_path = 'docs/PROTECTED_ARTIFACTS.json'
        registry = json.loads(reader.read(registry_path, entries[registry_path])) if registry_path in entries else {'artifacts': []}
        registered = {item['path'] for item in registry['artifacts']}
        known = set(entries)
        index = indexes(known)
        proposed = {p for p in research if proposed_group(p) and not protected_reason(p, registered)}
        for path, entry in entries.items():
            if path.startswith(RELEASE) or path in AUDIT_SELF_SOURCES:
                continue
            if inspectable(path, registered):
                raw = reader.read(path, entry)
                sha[path] = hashlib.sha256(raw).hexdigest()
                text = raw.decode('utf-8-sig', errors='replace')
                # Current editable docs/code must be considered before deleting
                # baseline artifacts; never silently scan a stale Git blob.
                file = repository / path
                if file.is_file() and not file.is_symlink():
                    text = file.read_text(encoding='utf-8-sig', errors='replace')
                texts[path] = text
                # Historic machine inventories/hash receipts are retained as
                # file-level contracts. Their rows are not runtime consumers.
                historical_receipt = (path.startswith('docs/repository/') or
                                      path in {'docs/repository_inventory_before.json', 'docs/repository_inventory_after.json'}) and Path(path).suffix != '.md'
                if not historical_receipt:
                    edges.extend(reference_edges(path, text, known, index))
                scan_count += 1
            elif path in proposed:
                # Images and report snapshots are hashed opaquely, not parsed.
                sha[path] = hashlib.sha256(reader.read(path, entry)).hexdigest()
        # Newly written navigation/source files can reference old artifacts
        # before they are committed. Scan only safe extensions, never payloads.
        for scope in ('docs', 'reports', 'experiments', 'studies', 'src', 'scripts', 'tools', 'tests'):
            for file in (repository / scope).rglob('*'):
                if not file.is_file() or file.is_symlink() or '__pycache__' in file.parts:
                    continue
                path = file.relative_to(repository).as_posix()
                if path in entries or path.startswith(RELEASE) or path in AUDIT_SELF_SOURCES:
                    continue
                if Path(path).suffix.lower() in (CODE_SUFFIXES | {'.md'}) and not sensitive(path):
                    text = file.read_text(encoding='utf-8-sig', errors='replace')
                    edges.extend(reference_edges(path, text, known, index))
                    scan_count += 1
        # Main is scanned only as a separately rooted source/CLI dependency
        # observer. No main file or index is written, no module is imported.
        main_head = git(repository, 'rev-parse', main_ref).decode().strip()
        main_entries = tree(repository, main_head)
        for path, entry in main_entries.items():
            if Path(path).suffix not in CODE_SUFFIXES or sensitive(path):
                continue
            raw = reader.read('main:' + path, entry)
            text = raw.decode('utf-8-sig', errors='replace')
            if active_root:
                physical = Path(active_root) / path
                if physical.is_file() and not physical.is_symlink():
                    text = physical.read_text(encoding='utf-8-sig', errors='replace')
            for edge in reference_edges(path, text, known, index, scope='SEPARATE_MAIN_SOURCE'):
                edge['from'] = 'main:' + edge['from']
                edges.append(edge)
    finally:
        reader.close()
    # Deduplicate typed edges and keep directory/basename rules as one edge.
    # CSV cells reference these IDs instead of copying entire source graphs.
    edge_keys = sorted({(e['from'], e['to'], e['kind'], e['source_scope']) for e in edges})
    edges = [{'id': 'E' + str(i), 'from': f, 'to': t, 'kind': k, 'source_scope': s}
             for i, (f, t, k, s) in enumerate(edge_keys)]
    incoming, directory_rules, basename_rules = (collections.defaultdict(list) for _ in range(3))
    for number, edge in enumerate(edges):
        if edge['to'].startswith('**/'):
            basename_rules[edge['to'][3:]].append(number)
        elif edge['to'].endswith('/'):
            directory_rules[edge['to'].rstrip('/')].append(number)
        else:
            incoming[edge['to']].append(number)
    def refs_for(path):
        numbers = list(incoming[path]) + list(basename_rules[Path(path).name])
        parent = posixpath.dirname(path)
        while parent:
            numbers.extend(directory_rules[parent])
            parent = posixpath.dirname(parent)
        return [edges[i] for i in sorted(set(numbers))]
    reference_set_numbers, reference_sets = {}, {}
    def ref_cell(values):
        identities = tuple(e['id'] for e in values)
        if identities not in reference_set_numbers:
            key = 'S' + str(len(reference_set_numbers))
            reference_set_numbers[identities] = key
            reference_sets[key] = list(identities)
        return str(len(identities)) + '@' + reference_set_numbers[identities]
    rows, candidates = [], []
    for path, entry in sorted(research.items()):
        protection = protected_reason(path, registered)
        group = proposed_group(path)
        refs = refs_for(path)
        current_refs = [e for e in refs if e['source_scope'] != 'SEPARATE_MAIN_SOURCE' and e['kind'] != 'PINNED_HISTORY_LINK']
        # Documents inside the same archive proposal are a closed group; every
        # external reference remains visible for root's explicit repair review.
        external = [e for e in current_refs if e['from'] not in proposed]
        machine = [e for e in external if e['source_scope'] not in {'DOCUMENT', 'AUDIT_TOOL'}]
        main_refs = [e for e in refs if e['source_scope'] == 'SEPARATE_MAIN_SOURCE']
        if protection:
            decision = 'KEEP_PROTECTED'
        elif group and Path(path).suffix.lower() in {'.png', '.svg', '.pdf'}:
            decision = 'DEFERRED_DERIVED_REGENERATION_NOT_PROVEN'
        elif group and machine:
            decision = 'BLOCKED_MACHINE_REFERENCE'
        elif group and external:
            decision = 'REVIEW_DOCUMENT_REFERENCES'
        elif group:
            decision = 'ELIGIBLE_AFTER_REMOTE_RECOVERY_GATE'
        else:
            decision = 'KEEP_SCIENTIFIC_OR_UNREVIEWED'
        current = repository / path
        if path not in head_entries:
            checkout_state = 'ABSENT_FROM_CURRENT_HEAD'
        elif current.exists():
            checkout_state = 'EXPANDED_IN_CANDIDATE'
        elif path in removed_worktree or path in removed_index:
            checkout_state = 'REMOVED_FROM_CANDIDATE_WORKTREE_OR_INDEX'
        elif path in sparse_omitted:
            checkout_state = 'TRACKED_SPARSE_OMITTED'
        else:
            checkout_state = 'TRACKED_MISSING_UNEXPLAINED'
        row = {
            'path': path, 'bytes': entry['bytes'], 'hash': sha.get(path, 'NOT_READ_PROTECTED_OR_UNREVIEWED'),
            'hash_algorithm': 'SHA256' if path in sha else 'GIT_BLOB_ID_ONLY',
            'git_blob': entry['git_blob'], 'owner_study': owner(path),
            'active_import': ref_cell([e for e in refs if e['kind'] == 'DIRECT_IMPORT']),
            'dynamic_reference': ref_cell([e for e in refs if e['kind'] not in {'DIRECT_IMPORT', 'PINNED_HISTORY_LINK'} and e['source_scope'] != 'DOCUMENT']),
            'frozen_hash': ref_cell([e for e in refs if 'HASH' in e['kind'] or e['source_scope'] == 'FROZEN_HISTORICAL_SOURCE']),
            'report_reference': ref_cell([e for e in refs if e['source_scope'] == 'DOCUMENT']),
            'protected_artifact': protection, 'proposal_group': group,
            'derivation_gate': 'REGENERATION_NOT_PROVEN' if Path(path).suffix.lower() in {'.png', '.svg', '.pdf'} else 'NOT_APPLICABLE',
            'replacement': 'docs/repository/README.md' if path.startswith('docs/repository/') and group else
                           'reports/README.md' if path.startswith('reports/') and group else
                           'experiments/HISTORICAL_DECISIONS.md' if group == 'historical_decision_documents' else
                           'docs/README.md' if group else '',
            'recovery_ref': baseline, 'remote_recovery_status': 'ROOT_MUST_VERIFY_SEPARATELY',
            'deletion_decision': decision, 'external_machine_reference_count': len(machine),
            'external_document_reference_count': sum(e['source_scope'] == 'DOCUMENT' for e in external),
            'separate_main_reference_count': len(main_refs), 'candidate_checkout_state': checkout_state,
            'candidate_path_exists': current.exists(),
        }
        rows.append(row)
        if group:
            candidates.append(row)
    boundaries = {
        'schema_version': 1, 'baseline': baseline, 'observed_main_head': main_head,
        'tracked_artifact_files': len(research), 'tracked_artifact_bytes': sum(e['bytes'] for e in research.values()),
        'all_tracked_source_universe': len(entries), 'decoded_dependency_source_files': scan_count,
        'registered_protected_paths': sorted(registered), 'machine_contracts': sorted(MACHINE_CONTRACTS),
        'registered_protection_git_coverage': [
            {'path': p, 'tracked_files': sum(name == p or name.startswith(p.rstrip('/') + '/') for name in entries),
             'candidate_path_exists': (repository / p).exists(),
             'note': 'A missing/ignored artifact has no Git-payload recovery guarantee.'}
            for p in sorted(registered)],
        'strict_study_prefixes': list(STRICT_STUDIES), 'strict_report_prefixes': list(STRICT_REPORTS),
        'decisions': dict(collections.Counter(r['deletion_decision'] for r in rows)),
        'proposal_decisions': dict(collections.Counter(r['deletion_decision'] for r in candidates)),
        'safety': {'main_files_written': 0, 'scientific_modules_imported': 0, 'training_or_provider_calls': 0,
                   'labels_revealed': 0, 'data_checkpoint_payloads_decoded': 0, 'files_deleted': 0},
        'limits': [
            'Static reference scan is conservative; arbitrary reflection and external consumers are not proven absent.',
            'Main references describe a separate physical source root; no main cleanup is authorized.',
            'Nondecoded research payloads retain Git blob identity and size, never a fabricated SHA256.',
            'Ignored/untracked checkpoints, original labels and runtime are not recoverable merely from a Git tag.',
            'Remote availability and actual source restoration are a separate mandatory gate before deletion.',
            'Active process observations and runtime source hashes are supplied by the independent protection receipt.',
            'A proposed directory is not blanket deletion authority; per-file protections and references override it.',
            'The auditor policy and its isolated temporary-repository test fixtures are excluded from self-referential dependency inference.',
            'Closed-proposal references assume those callers retire together; the actual selected-subset deletion plan must recheck retained callers.',
            'Derived image/PDF regeneration has not been demonstrated; Git recoverability alone is not proof of derivation reproducibility.',
        ],
    }
    return rows, candidates, {
        'schema_version': 2, 'baseline': baseline, 'edges': edges,
        'reference_sets': reference_sets,
        'csv_reference_cell_format': 'COUNT@SET_ID; resolve SET_ID in reference_sets, then edge IDs in edges',
        'rule_semantics': 'A target ending / is one directory-prefix rule; **/name is one conservative ambiguous-basename rule, not an enumerated dependency graph.',
        'historical_receipt_semantics': 'Existing engineering inventory/manifest rows are provenance, not inferred live runtime consumers.',
    }, boundaries


def write_results(output, rows, candidates, references, boundaries, *, compress=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ['path']
    for name, data in [('ARTIFACT_INVENTORY.csv', rows), ('DELETION_CANDIDATES.csv', candidates)]:
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(data)
    for name, payload in [('DEPENDENCY_REFERENCES.json', references), ('PROTECTED_BOUNDARIES.json', boundaries)]:
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    # Deterministic gzip is the sole delivery of the large inventory/reference
    # artifacts when requested. No duplicate uncompressed snapshot is retained.
    large = ['ARTIFACT_INVENTORY.csv', 'DEPENDENCY_REFERENCES.json']
    for name in large:
        plain, packed = output / name, output / (name + '.gz')
        if compress:
            with plain.open('rb') as source, packed.open('wb') as destination:
                with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as stream:
                    shutil.copyfileobj(source, stream)
            plain.unlink()
        elif packed.exists():
            packed.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--main-ref', default='refs/heads/main')
    parser.add_argument('--active-root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / RELEASE)
    parser.add_argument('--compress', action='store_true', help='Deliver only deterministic .gz versions of the two large machine artifacts')
    args = parser.parse_args()
    if args.active_root:
        active = args.active_root.resolve()
        target = args.output.resolve()
        if target == active or active in target.parents:
            parser.error('audit reports must be outside the running main worktree')
    results = audit(args.repository, baseline=args.baseline, main_ref=args.main_ref, active_root=args.active_root)
    write_results(args.output, *results, compress=args.compress)
    print(json.dumps(results[-1], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
