#!/usr/bin/env python3
"""Verify an isolated engineering candidate without loading scientific code or labels.

All content reads are limited to Python source and repository audit/policy JSON.
Data/result/checkpoint preservation is checked with Git object IDs and object
existence. The active repository is strictly read-only; only --output is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

BASELINE_COMMIT = '16eca74769f664bc378a58ccbf380a806723f587'
DEFAULT_ACTIVE = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
DEFAULT_CANDIDATE = Path(__file__).resolve().parents[1]
REFACTOR_BRANCH = 'codex/repository-architecture-20261109'
PYTHON_ROOTS = ('src/', 'application/', 'scripts/')
SEALED_ROOTS = ('experiments/', 'studies/', 'reports/')
SOURCE_SCOPES = ('src', 'application', 'scripts',
                 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101')
APPROVED_RETIREMENT_PATHS = {
    'scripts/studies/plot_free_cw_snapshot_20261102.py',
    'scripts/studies/plot_free_rerun_audit_20261103.py',
}


def git(repository, *args, input_text=None, binary=False):
    result = subprocess.run(['git', '--no-optional-locks', *args], cwd=repository,
                            input=input_text, capture_output=True, text=not binary,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if result.returncode:
        stderr = result.stderr.decode(errors='replace') if binary else result.stderr
        raise RuntimeError(f'git {args[0]} failed: {stderr[:500]}')
    return result.stdout


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree(repository, revision):
    records = {}
    for entry in git(repository, 'ls-tree', '-r', '-z', revision).split('\0'):
        if entry:
            meta, path = entry.split('\t', 1)
            mode, kind, oid = meta.split()
            records[path] = {'oid': oid, 'mode': mode, 'type': kind}
    return records


def index(repository):
    records = {}
    for entry in git(repository, 'ls-files', '--stage', '-z').split('\0'):
        if entry:
            meta, path = entry.split('\t', 1)
            mode, oid, stage = meta.split()
            if stage != '0':
                raise RuntimeError(f'unmerged index entry: {path}')
            records[path] = {'oid': oid, 'mode': mode}
    skipped = {entry[2:] for entry in git(repository, 'ls-files', '-v', '-z').split('\0')
               if entry.startswith('S ')}
    return records, skipped


def read_metadata_blob(repository, revision, path):
    # Explicitly used only for audit/policy JSON, never data or result tables.
    return json.loads(git(repository, 'show', f'{revision}:{path}'))


def choose_artifact(candidate, filename, supplied=None):
    if supplied:
        return Path(supplied)
    options = [candidate / 'docs/repository' / filename,
               candidate / 'docs/repository/runtime_snapshot_20261109' / filename,
               Path('/private/tmp/column_refactor_runtime_audit') / filename]
    return next((path for path in options if path.is_file()), options[0])


def approved_retirements(path, baseline, candidate, baseline_commit):
    if not path or not Path(path).is_file():
        return {}, []
    payload = json.loads(Path(path).read_text())
    entries = payload if isinstance(payload, list) else payload.get('retired', payload.get('files', payload.get('retirements', [])))
    approved, errors = {}, []
    for entry in entries:
        action = str(entry.get('action', entry.get('status', ''))).upper()
        if action not in {'RETIRED', 'DELETED', 'REMOVED', 'DELETED_IN_CANDIDATE'}:
            continue
        name = entry.get('path', entry.get('original_path'))
        if name not in APPROVED_RETIREMENT_PATHS:
            errors.append({'path': name, 'reason': 'path is outside the explicitly approved retirement candidates'})
            continue
        original = baseline.get(name)
        if not original:
            errors.append({'path': name, 'reason': 'retirement path absent from baseline'})
            continue
        oid = entry.get('git_blob_oid', entry.get('blob_oid', entry.get('blob_sha', entry.get('original_blob'))))
        digest = entry.get('sha256', entry.get('original_sha256'))
        original_bytes = git(candidate, 'show', f'{baseline_commit}:{name}', binary=True)
        if oid != original['oid'] or digest != hashlib.sha256(original_bytes).hexdigest():
            errors.append({'path': name, 'reason': 'retirement requires matching original Git blob and SHA-256'})
            continue
        if not entry.get('evidence') or not entry.get('recovery', entry.get('restore_command', entry.get('recoverability'))):
            errors.append({'path': name, 'reason': 'retirement requires dependency evidence and recovery instruction'})
            continue
        if entry.get('source_commit') != baseline_commit or not entry.get('replacement'):
            errors.append({'path': name, 'reason': 'retirement requires the exact baseline source_commit and replacement'})
            continue
        approved[name] = entry
    return approved, errors


def check_preserved_source(candidate, baseline, current_index, skipped, approved, baseline_commit):
    failures, retired, checked = [], [], 0
    for name, original in baseline.items():
        if not (name.startswith(PYTHON_ROOTS) and name.endswith('.py')):
            continue
        checked += 1
        path = candidate / name
        if name in approved and not path.exists():
            retired.append(name)
            continue
        if not path.is_file():
            if name in skipped and current_index.get(name, {}).get('oid') == original['oid']:
                continue
            failures.append({'path': name, 'reason': 'source missing without approved retirement or sparse object'})
            continue
        expected = hashlib.sha256(git(candidate, 'show', f'{baseline_commit}:{name}', binary=True)).hexdigest()
        if sha(path) != expected:
            failures.append({'path': name, 'reason': 'existing scientific/runner source bytes changed'})
    return {'passed': not failures, 'baseline_python_files_checked': checked,
            'approved_retired_paths': retired, 'failures': failures}


def check_frozen(candidate, baseline, current_index, snapshot, compatibility, baseline_commit):
    failures, paths = [], set(compatibility)
    items = snapshot.get('cw_frozen_code', [])
    expected_sha = {item['path']: item.get('expected_sha256', item.get('sha256')) for item in items}
    paths.update(expected_sha)
    for name in sorted(paths):
        original = baseline.get(name)
        if not original or current_index.get(name, {}).get('oid') != original['oid']:
            failures.append({'path': name, 'reason': 'frozen/compatibility indexed object changed or missing'})
            continue
        path = candidate / name
        actual = sha(path) if path.is_file() else hashlib.sha256(git(candidate, 'show', f'{baseline_commit}:{name}', binary=True)).hexdigest()
        expected = expected_sha.get(name)
        if expected and actual != expected:
            failures.append({'path': name, 'reason': 'frozen Python SHA-256 differs from protocol'})
        if path.is_file() and hashlib.sha256(git(candidate, 'show', f'{baseline_commit}:{name}', binary=True)).hexdigest() != actual:
            failures.append({'path': name, 'reason': 'materialized compatibility bytes differ from baseline'})
    counts_ok = len(compatibility) == 28 and len(items) == 263
    return {'passed': counts_ok and not failures, 'historical_compatibility_paths': len(compatibility),
            'cw_frozen_code_paths': len(items), 'unique_paths_checked': len(paths), 'failures': failures}


def check_recoverability(candidate, baseline, current_index, approved, protection, baseline_commit):
    # --batch-check never emits blob contents, including test truth and checkpoints.
    objects = sorted({entry['oid'] for entry in baseline.values()})
    lines = git(candidate, 'cat-file', '--batch-check=%(objectname) %(objecttype)',
                input_text='\n'.join(objects) + '\n').splitlines()
    missing = [line.split()[0] for line in lines if line.endswith(' missing')]
    sealed_changed = []
    for name, entry in baseline.items():
        if name.startswith(SEALED_ROOTS) and name not in approved:
            if current_index.get(name, {}).get('oid') != entry['oid']:
                sealed_changed.append(name)
            # Catch unstaged deletions/changes in materialized Python only; do not
            # read data/result worktree content, even for preservation checking.
    protected = []
    for entry in protection.get('artifacts', []):
        name = entry['path']
        try:
            oid = git(candidate, 'rev-parse', f'{baseline_commit}:{name}').strip()
            kind = git(candidate, 'cat-file', '-t', oid).strip()
            protected.append({'path': name, 'oid': oid, 'type': kind, 'recoverable': True})
        except RuntimeError:
            protected.append({'path': name, 'recoverable': False})
    return {'passed': not missing and not sealed_changed and all(x['recoverable'] for x in protected),
            'baseline_objects_checked_without_contents': len(objects), 'missing_objects': missing,
            'sealed_index_changes': sealed_changed, 'protected_artifacts': protected,
            'truth_and_checkpoint_contents_opened': 0}


def repository_path(repository, *revision_args):
    value = Path(git(repository, 'rev-parse', *revision_args).strip())
    return (value if value.is_absolute() else repository / value).resolve()


def check_workspace_isolation(candidate, active, expected_active_head=BASELINE_COMMIT):
    """Allow either a private clone or a controlled linked worktree of active.

    Linked worktrees intentionally share objects and refs. Their source tree,
    Git directory, HEAD and index must be separate from the running main tree.
    Every Git query is read-only and uses GIT_OPTIONAL_LOCKS=0 through git().
    """
    candidate, active = Path(candidate).resolve(), Path(active).resolve()
    failures, samples = [], []
    if candidate == active or active in candidate.parents or candidate in active.parents:
        return {'passed': False, 'mode': 'invalid',
                'failures': ['candidate and active source directories must be separate and not nested']}
    candidate_top = repository_path(candidate, '--show-toplevel')
    active_top = repository_path(active, '--show-toplevel')
    if candidate_top != candidate or active_top != active:
        failures.append('supplied source directories must be their own Git worktree roots')
    candidate_git = repository_path(candidate, '--git-dir')
    active_git = repository_path(active, '--git-dir')
    common = repository_path(candidate, '--git-common-dir')
    active_common = repository_path(active, '--git-common-dir')
    candidate_head = repository_path(candidate, '--git-path', 'HEAD')
    active_head = repository_path(active, '--git-path', 'HEAD')
    candidate_index = repository_path(candidate, '--git-path', 'index')
    active_index = repository_path(active, '--git-path', 'index')
    separate_metadata = {}
    for label, left, right in [('HEAD', candidate_head, active_head),
                                ('index', candidate_index, active_index)]:
        independent = (left != right and left.is_file() and right.is_file()
                       and not os.path.samefile(left, right))
        separate_metadata[label] = independent
        if not independent:
            failures.append(f'candidate {label} must be a separate existing file from active {label}')
        if left.parent != candidate_git:
            failures.append(f'candidate {label} must belong to the candidate Git directory')
    if candidate_git == active_git:
        failures.append('candidate Git directory must be distinct from active Git directory')
    if os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES'):
        failures.append('Git alternate object environment variable is set')
    alternates = common / 'objects/info/alternates'
    if alternates.exists() and alternates.read_text().strip():
        failures.append('candidate has object alternates')
    shared = common == active_common
    mode = 'linked_worktree' if shared else 'independent_clone'
    details = {}
    if shared:
        pointer = candidate / '.git'
        if not pointer.is_file() or pointer.is_symlink():
            failures.append('linked candidate must have a regular .git worktree pointer')
        if not candidate_git.is_relative_to(active_common / 'worktrees'):
            failures.append('linked Git directory must belong to active repository worktrees')
        candidate_branch = git(candidate, 'symbolic-ref', '--short', 'HEAD').strip()
        active_branch = git(active, 'symbolic-ref', '--short', 'HEAD').strip()
        active_commit = git(active, 'rev-parse', 'HEAD').strip()
        main_commit = git(active, 'rev-parse', 'refs/heads/main').strip()
        if candidate_branch != REFACTOR_BRANCH:
            failures.append('linked candidate must use the authorized refactor branch')
        if active_branch != 'main' or active_commit != expected_active_head or main_commit != expected_active_head:
            failures.append('active main and HEAD must remain at the protected baseline commit')
        details.update(candidate_branch=candidate_branch, active_branch=active_branch,
                       active_head=active_commit, main_commit=main_commit,
                       hardlink_check_scope='objects intentionally shared; HEAD/index independence checked')
    else:
        if not (candidate / '.git').is_dir() or common != (candidate / '.git').resolve():
            failures.append('independent candidate must own its .git/common directory')
        original_objects = active_common / 'objects'
        candidate_objects = common / 'objects'
        relative_samples = []
        for path in sorted(candidate_objects.glob('pack/*')):
            if path.is_file():
                relative_samples.append(path.relative_to(candidate_objects))
        for prefix in sorted(candidate_objects.iterdir()):
            if prefix.is_dir() and len(prefix.name) == 2:
                relative_samples.extend(path.relative_to(candidate_objects) for path in sorted(prefix.iterdir())[:2] if path.is_file())
            if len(relative_samples) >= 24:
                break
        for name in relative_samples[:24]:
            left, right = original_objects / name, candidate_objects / name
            sample = {'object_file': str(name), 'candidate_link_count': right.stat().st_nlink,
                      'same_inode': False, 'matching_source_object_file': left.is_file()}
            if right.stat().st_nlink != 1:
                failures.append(f'candidate Git object has multiple hard links: {name}')
            if left.is_file():
                linked = os.path.samefile(left, right)
                sample['same_inode'] = linked
                if linked:
                    failures.append(f'hardlinked object file: {name}')
            samples.append(sample)
        details['hardlink_check_scope'] = 'up to 24 matching pack/loose object files'
    return {'passed': not failures, 'mode': mode, 'objects_shared': shared,
            'candidate_git_directory': str(candidate_git), 'active_git_directory': str(active_git),
            'common_git_directory': str(common), 'active_common_git_directory': str(active_common),
            'separate_metadata': separate_metadata, 'shared_object_file_samples': samples,
            **details, 'failures': failures}


def check_active(active, manifest, snapshot, expected_head):
    failures, changed, missing = [], [], []
    head = git(active, 'rev-parse', 'HEAD').strip()
    branch = git(active, 'symbolic-ref', '--short', 'HEAD').strip()
    if head != expected_head or branch != 'main':
        failures.append('active repository HEAD/main differs from baseline')
    for name, expected in manifest['hashes'].items():
        path = active / name
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            changed.append(name)
    present = {str(path.relative_to(active)) for dirname in SOURCE_SCOPES
               for path in (active / dirname).rglob('*.py') if '__pycache__' not in path.parts}
    added = sorted(present - set(manifest['hashes']))
    identity_failures = []
    for entry in snapshot.get('identity_files', []):
        name = entry['path']
        # active_runtime is mutable supervisor metadata. It is deliberately not
        # constrained; immutable controller/auth/launch/protocol identities are.
        if name.endswith('/active_runtime.json'):
            continue
        path = active / name
        if not path.is_file() or sha(path) != entry.get('sha256'):
            identity_failures.append({'path': name, 'reason': 'identity bytes changed/missing'})
        elif path.stat().st_ino != entry.get('inode'):
            identity_failures.append({'path': name, 'reason': 'identity inode changed'})
    for entry in snapshot.get('identity_directories', []):
        path = active / entry['path']
        if not path.is_dir():
            identity_failures.append({'path': entry['path'], 'reason': 'runtime directory missing'})
            continue
        current = path.stat()
        if ((current.st_dev, current.st_ino) != (entry['device'], entry['inode'])
                or stat.filemode(current.st_mode) != entry['mode']):
            identity_failures.append({'path': entry['path'], 'reason': 'runtime directory identity or permissions changed'})
    if changed or missing or added:
        failures.append('active source tree differs from initial SHA-256 manifest')
    if identity_failures:
        failures.append('active frozen runtime identity binding changed')
    return {'passed': not failures, 'head': head, 'branch': branch,
            'initial_hash_capture_utc': manifest.get('captured_at_utc'),
            'source_files_checked': len(manifest['hashes']), 'changed_source': changed,
            'missing_source': missing, 'added_source': added, 'identity_failures': identity_failures,
            'runtime_mutable_outputs_compared': False, 'processes_signaled': 0,
            'identity_directories_checked': len(snapshot.get('identity_directories', [])),
            'failures': failures}


def audit(candidate=DEFAULT_CANDIDATE, active=DEFAULT_ACTIVE, baseline_commit=BASELINE_COMMIT,
          active_manifest=None, runtime_snapshot=None, retirement_manifest=None):
    candidate, active = Path(candidate).resolve(), Path(active).resolve()
    if candidate == active:
        raise ValueError('candidate must be outside the active repository')
    if active in candidate.parents or candidate in active.parents:
        raise ValueError('candidate and active repository must be separate trees')
    source_manifest_path = choose_artifact(candidate, 'active_source_hashes.json', active_manifest)
    snapshot_path = choose_artifact(candidate, 'RUNTIME_SNAPSHOT.json', runtime_snapshot)
    manifest, snapshot = json.loads(source_manifest_path.read_text()), json.loads(snapshot_path.read_text())
    baseline = tree(candidate, baseline_commit)
    current_index, skipped = index(candidate)
    scope = read_metadata_blob(candidate, baseline_commit, 'docs/repository/RESEARCH_SCOPE.json')
    protection = read_metadata_blob(candidate, baseline_commit, 'docs/PROTECTED_ARTIFACTS.json')
    retirement = Path(retirement_manifest) if retirement_manifest else candidate / 'docs/repository/RETIREMENT_MANIFEST.json'
    approved, retirement_errors = approved_retirements(retirement, baseline, candidate, baseline_commit)
    checks = {
        'retirement_authorization': {'passed': not retirement_errors, 'approved_count': len(approved), 'errors': retirement_errors},
        'baseline_python_bytes': check_preserved_source(candidate, baseline, current_index, skipped, approved, baseline_commit),
        'frozen_and_compatibility': check_frozen(candidate, baseline, current_index, snapshot, scope['frozen_llm_compatibility_paths'], baseline_commit),
        'object_recoverability_and_protection': check_recoverability(candidate, baseline, current_index, approved, protection, baseline_commit),
        'workspace_isolation': check_workspace_isolation(candidate, active, baseline_commit),
        'active_repository_unchanged': check_active(active, manifest, snapshot, snapshot['head']),
    }
    return {'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'candidate': str(candidate), 'active': str(active), 'baseline_commit': baseline_commit,
            'active_manifest': str(source_manifest_path), 'runtime_snapshot': str(snapshot_path),
            'passed': all(check['passed'] for check in checks.values()), 'checks': checks,
            'safety': {'scientific_modules_imported': 0, 'controllers_imported': 0,
                       'truth_tables_parsed': 0, 'active_repository_files_written': 0}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument('--active-root', type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument('--baseline-commit', default=BASELINE_COMMIT)
    parser.add_argument('--active-manifest', type=Path)
    parser.add_argument('--runtime-snapshot', type=Path)
    parser.add_argument('--retirement-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == args.active_root.resolve() or args.active_root.resolve() in output.parents:
        parser.error('--output must not be inside the active repository')
    report = audit(args.repository, args.active_root, args.baseline_commit,
                   args.active_manifest, args.runtime_snapshot, args.retirement_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': {name: check['passed'] for name, check in report['checks'].items()},
                      'output': str(output)}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
