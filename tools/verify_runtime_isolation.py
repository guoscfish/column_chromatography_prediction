"""Read-only proof that candidate cleanup leaves the running experiment intact.

An explicitly supplied diagnostics commit may advance main only by additions
under the separately authorized diagnostics directory. Scientific source,
controllers, identities and protected directories remain byte-bound to Gate 0.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_PREFIX = 'studies/diagnostics/quantile_loss_audit_20261109/'
REFACTOR_BRANCH = 'codex/repository-architecture-20261109'
SOURCE_SCOPES = ('src', 'application', 'scripts',
                 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101')


def git(repository, *arguments):
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(repository), *arguments],
                            capture_output=True, text=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if result.returncode:
        raise RuntimeError(f'git {arguments[0]} failed: {result.stderr[:400]}')
    return result.stdout.strip()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def git_path(repository, *arguments):
    path = Path(git(repository, 'rev-parse', *arguments))
    return (path if path.is_absolute() else repository / path).resolve()


def authorized_main_progression(active, initial, current, allowed_commit):
    if current == initial:
        return {'passed': True, 'mode': 'unchanged_main', 'initial': initial, 'current': current, 'changes': []}
    failures, changes = [], []
    if not isinstance(allowed_commit, str) or not re.fullmatch('[0-9a-f]{40}', allowed_commit):
        failures.append('main moved without an explicit full diagnostics commit receipt')
        return {'passed': False, 'initial': initial, 'current': current, 'failures': failures, 'changes': []}
    if allowed_commit != current:
        failures.append('diagnostics receipt must identify the current main commit exactly')
    ancestry = subprocess.run(['git', '--no-optional-locks', '-C', str(active), 'merge-base', '--is-ancestor', initial, current],
                              capture_output=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if ancestry.returncode:
        failures.append('initial main is not an ancestor of current main')
    for line in git(active, 'diff', '--name-status', initial + '..' + current).splitlines():
        status, path = line.split('\t', 1)
        changes.append({'status': status, 'path': path})
        if status != 'A' or not path.startswith(DIAGNOSTICS_PREFIX):
            failures.append('main net change is not a new authorized diagnostics file: ' + path)
    commits = git(active, 'rev-list', '--reverse', initial + '..' + current).splitlines()
    for commit in commits:
        parents = git(active, 'rev-list', '--parents', '-n', '1', commit).split()[1:]
        if len(parents) != 1:
            failures.append('diagnostics progression must not merge unrelated history: ' + commit)
            continue
        for line in git(active, 'diff-tree', '--no-commit-id', '--name-status', '-r', commit).splitlines():
            status, path = line.split('\t', 1)
            if status not in ('A', 'M') or not path.startswith(DIAGNOSTICS_PREFIX):
                failures.append('intermediate main commit touches outside authorized diagnostics: ' + path)
    if not changes:
        failures.append('moved main receipt has no reviewable diagnostics additions')
    return {'passed': not failures, 'mode': 'authorized_diagnostics_additions', 'initial': initial,
            'current': current, 'receipt_commit': allowed_commit, 'allowed_prefix': DIAGNOSTICS_PREFIX,
            'changes': changes, 'commits': commits, 'failures': failures}


def verify(snapshot_directory, candidate=ROOT, allowed_main_diagnostics_commit=None):
    snapshot_directory, candidate = Path(snapshot_directory).resolve(), Path(candidate).resolve()
    gate = json.loads((snapshot_directory / 'GATE0.json').read_text())
    manifest = json.loads((snapshot_directory / 'active_source_hashes.json').read_text())
    initial_index = json.loads((snapshot_directory / 'main_index_hash.json').read_text())
    active = Path(gate['active_root']).resolve()
    if active == candidate or active in candidate.parents or candidate in active.parents:
        raise ValueError('candidate must be a separate, nonnested worktree')
    failures = []
    branch = git(active, 'symbolic-ref', '--short', 'HEAD')
    head = git(active, 'rev-parse', 'HEAD')
    main = git(active, 'rev-parse', 'refs/heads/main')
    if branch != 'main' or main != head:
        failures.append('active directory must remain on main')
    progression = authorized_main_progression(active, gate['active_head'], head, allowed_main_diagnostics_commit)
    if not progression['passed']:
        failures.append('main progression does not match the explicit diagnostics receipt')
    source_failures = []
    for name, expected in manifest['hashes'].items():
        path = active / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            source_failures.append(name)
    if not manifest['hashes'] or source_failures:
        failures.append('protected scientific/controller source changed')
    present = {p.relative_to(active).as_posix() for folder in SOURCE_SCOPES
               for p in (active / folder).rglob('*.py') if '__pycache__' not in p.parts}
    added = sorted(present - set(manifest['hashes']))
    if added:
        failures.append('unapproved Python source appeared in protected hash scopes')
    identities = []
    for item in gate.get('immutable_identity_checks', []):
        path = active / item['path']
        matched = path.is_file() and not path.is_symlink() and digest(path) == item['sha256']
        if matched:
            metadata = path.stat()
            matched = metadata.st_ino == item['inode'] and stat.filemode(metadata.st_mode) == item['mode']
        if not matched:
            identities.append(item['path'])
    directory_failures = []
    for item in gate.get('runtime_directory_identity_checks', []):
        path = active / item['path']
        if not path.is_dir() or path.is_symlink():
            directory_failures.append(item['path'])
            continue
        info = path.stat()
        if (info.st_dev, info.st_ino, stat.filemode(info.st_mode)) != (item['device'], item['inode'], item['mode']):
            directory_failures.append(item['path'])
    if identities or directory_failures:
        failures.append('protected frozen files or runtime directory identity changed')
    candidate_git = git_path(candidate, '--git-dir')
    active_git = git_path(active, '--git-dir')
    candidate_common = git_path(candidate, '--git-common-dir')
    active_common = git_path(active, '--git-common-dir')
    candidate_index = git_path(candidate, '--git-path', 'index')
    active_index = git_path(active, '--git-path', 'index')
    candidate_head_file = git_path(candidate, '--git-path', 'HEAD')
    active_head_file = git_path(active, '--git-path', 'HEAD')
    isolation = {'same_common_git': candidate_common == active_common,
                 'separate_git_directory': candidate_git != active_git,
                 'separate_index': candidate_index != active_index and not os.path.samefile(candidate_index, active_index),
                 'separate_HEAD': candidate_head_file != active_head_file and not os.path.samefile(candidate_head_file, active_head_file),
                 'authorized_branch': git(candidate, 'symbolic-ref', '--short', 'HEAD') == REFACTOR_BRANCH,
                 'symlinks': [p.relative_to(candidate).as_posix() for p in candidate.rglob('*') if p.is_symlink()]}
    if not all(value is True for key, value in isolation.items() if key != 'symlinks') or isolation['symlinks']:
        failures.append('candidate worktree shares editable metadata/source paths or has an unexpected branch')
    staged = git(active, 'diff', '--cached', '--name-only').splitlines()
    if staged:
        failures.append('active main has staged files; completed diagnostics receipt must leave staging empty')
    result = {'schema_version': 1, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'passed': not failures, 'active_root': str(active), 'candidate_root': str(candidate),
              'active_branch': branch, 'active_head': head, 'initial_main': gate['active_head'],
              'main_progression': progression, 'protected_source_files': len(manifest['hashes']),
              'source_failures': source_failures, 'unapproved_source_additions': added,
              'immutable_identity_failures': identities, 'runtime_directory_identity_failures': directory_failures,
              'worktree_isolation': isolation, 'main_staged_paths': staged,
              'main_index_binary_unchanged': digest(active_index) == initial_index['sha256'],
              'index_note': 'Authorized committed diagnostics may change index bytes; no staged paths and source/identity preservation are enforced',
              'failures': failures, 'safety': {'main_files_written': 0, 'process_signals_sent': 0,
                                             'controllers_imported': 0, 'truth_tables_opened': 0}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-directory', type=Path, default=ROOT / 'docs/repository/phase2')
    parser.add_argument('--candidate', type=Path, default=ROOT)
    parser.add_argument('--allowed-main-diagnostics-commit', help='Explicit full-SHA receipt for the separately authorized diagnostics-only main commit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads((args.snapshot_directory / 'GATE0.json').read_text())
    active = Path(gate['active_root']).resolve()
    output = args.output.resolve()
    if output == active or active in output.parents:
        parser.error('verification report must not be written inside the active worktree')
    result = verify(args.snapshot_directory, args.candidate, args.allowed_main_diagnostics_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'active_head': result['active_head'],
                      'main_progression': result['main_progression']['mode'] if 'mode' in result['main_progression'] else 'REJECTED',
                      'protected_source_files': result['protected_source_files'], 'failures': result['failures']}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
