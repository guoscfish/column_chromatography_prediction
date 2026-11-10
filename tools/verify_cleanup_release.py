"""Verify a documentation-only cleanup against its immutable release receipts.

Never changes Git refs, source, processes, runtime resources or environments.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'docs/repository/releases/document-cleanup-20261110'
BRANCHES = {'main', 'codex/archive-transfer-20261028', 'codex/repository-architecture-20261109'}
MACHINE_CONTRACTS = ('docs/PROTECTED_ARTIFACTS.json', 'docs/data_consumption_register.json',
                     'docs/PRUNE_MANIFEST.csv', 'docs/repository/RESEARCH_SCOPE.json',
                     'docs/repository/RETIREMENTS.json',
                     'docs/repository/phase2/HISTORICAL_PROTOCOLS.json',
                     'docs/repository/phase2/DELETION_PROOF.csv',
                     'docs/repository/phase2/SOURCE_EXIT_PLAN.json')


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_path(root, name):
    value = Path(git(root, 'rev-parse', name).decode().strip())
    return (value if value.is_absolute() else root/value).resolve()


def verify(root, release):
    root, release = Path(root).resolve(), Path(release).resolve()
    baseline = json.loads((release/'MAIN_BASELINE.json').read_text())
    remote = json.loads((release/'REMOTE_RECOVERY.json').read_text())
    deletions = list(csv.DictReader((release/'DELETED_FILES.csv').open()))
    main = Path(baseline['main_root']).resolve()
    failures = []
    if root == main or root in main.parents or main in root.parents:
        failures.append('candidate and main must have separate, non-nested source roots')
    if git_path(root, '--git-common-dir') != git_path(main, '--git-common-dir'):
        failures.append('candidate must belong to the original common Git repository')
    if git_path(root, '--git-dir') == git_path(main, '--git-dir'):
        failures.append('candidate Git/index metadata must be separate from main')
    main_head = git(main, 'rev-parse', 'HEAD').decode().strip()
    if main_head != baseline['main_head'] or git(main, 'branch', '--show-current').decode().strip() != 'main':
        failures.append('main HEAD/branch differs from this release baseline')
    index = Path(git(main, 'rev-parse', '--git-path', 'index').decode().strip())
    index = index if index.is_absolute() else main/index
    if sha(index.read_bytes()) != baseline['main_index_sha256']:
        failures.append('main index bytes changed')
    staged = git(main, 'diff', '--cached', '--name-only').decode().splitlines()
    if staged:
        failures.append('main has staged paths')
    source_failures = [name for name, digest in baseline['protected_source_hashes'].items()
                       if not (main/name).is_file() or (main/name).is_symlink() or sha((main/name).read_bytes()) != digest]
    present_sources = {p.relative_to(main).as_posix() for scope in baseline['source_scopes']
                       for p in (main/scope).rglob('*.py') if '__pycache__' not in p.parts}
    new_main_sources = sorted(present_sources - set(baseline['protected_source_hashes']))
    if source_failures or new_main_sources or not baseline['protected_source_hashes']:
        failures.append('main protected scientific/controller source changed')
    changed_code = git(root, 'diff', baseline['candidate_baseline'], '--name-only', '--',
                       'src', 'application', 'scripts', 'configs').decode().splitlines()
    untracked_code = [name for name in git(root, 'ls-files', '--others', '--exclude-standard').decode().splitlines()
                      if name.startswith(('src/', 'application/', 'scripts/', 'configs/'))]
    changed_code = sorted(set(changed_code) | set(untracked_code))
    if changed_code:
        failures.append('scientific implementation/config changed during documentation cleanup')
    changed_contracts = []
    for name in MACHINE_CONTRACTS:
        original = git(root, 'show', baseline['candidate_baseline'] + ':' + name)
        if not (root/name).is_file() or sha((root/name).read_bytes()) != sha(original):
            changed_contracts.append(name)
    if changed_contracts:
        failures.append('retained machine contracts changed or disappeared')
    recovery = {row['path']: row for row in remote.get('files', [])}
    actual_deletions = set(git(root, 'diff', '--no-renames', baseline['candidate_baseline'], '--diff-filter=D', '--name-only').decode().splitlines())
    proof_paths = [row['path'] for row in deletions]
    if actual_deletions != set(proof_paths) or len(set(proof_paths)) != len(proof_paths):
        failures.append('actual deleted path set must exactly match unique deletion proof rows')
    if remote.get('remote_url') != 'https://github.com/guoscfish/column_chromatography_prediction.git' or remote.get('fetched_commit') != baseline['candidate_baseline']:
        failures.append('remote receipt must bind this repository and baseline commit')
    deletion_failures = []
    for row in deletions:
        name = row['path']
        if not re.fullmatch('[0-9a-f]{40}', row['recovery_ref']) or row['recovery_ref'] != baseline['candidate_baseline']:
            deletion_failures.append({'path': name, 'reason': 'unregistered or mutable recovery ref'})
            continue
        if Path(name).is_absolute() or '..' in Path(name).parts or (root/name).exists():
            deletion_failures.append({'path': name, 'reason': 'unsafe path or still present'})
            continue
        original = git(root, 'show', row['recovery_ref'] + ':' + name)
        independent = recovery.get(name, {})
        if sha(original) != row['sha256'] or independent.get('sha256') != row['sha256'] or independent.get('git_blob') != row['git_blob']:
            deletion_failures.append({'path': name, 'reason': 'local/independent remote source differs'})
    if deletion_failures or remote.get('passed') is not True:
        failures.append('independent remote recovery or deletion proof failed')
    branches = set(git(root, 'for-each-ref', '--format=%(refname:short)', 'refs/heads').decode().splitlines())
    if branches != BRANCHES or git(root, 'branch', '--show-current').decode().strip() != 'codex/repository-architecture-20261109':
        failures.append('unexpected branch set or active candidate branch')
    return {'schema_version': 1, 'passed': not failures, 'failures': failures,
            'main_head': main_head, 'protected_source_files': len(baseline['protected_source_hashes']),
            'source_failures': source_failures, 'new_main_source_files': new_main_sources, 'main_staged_paths': staged,
            'changed_scientific_code_or_configs': changed_code, 'changed_machine_contracts': changed_contracts,
            'deleted_files_verified': len(deletions), 'deletion_failures': deletion_failures,
            'actual_deleted_paths_count': len(actual_deletions), 'unlisted_deletions': sorted(actual_deletions - set(proof_paths)),
            'independent_remote_recovery_passed': remote.get('passed') is True,
            'local_branches': sorted(branches),
            'safety': {'main_files_written': 0, 'signals_sent': 0, 'training_or_llm_calls': 0},
            'limits': ['Runtime progress is allowed to evolve; source/index identity is checked.',
                       'Remote evidence is a fixed independent-fetch receipt, not a fresh network verification on every invocation.',
                       'Remote evidence covers listed tracked bytes only, not ignored payloads or credentials.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release = args.release or args.repository/RELEASE
    result = verify(args.repository, release)
    active = Path(json.loads((release/'MAIN_BASELINE.json').read_text())['main_root']).resolve()
    if args.output.resolve() == active or active in args.output.resolve().parents:
        parser.error('report output must stay outside main')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['passed', 'protected_source_files', 'deleted_files_verified', 'failures']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
