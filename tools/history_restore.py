"""Restore pinned historical bytes into an independent, read-only snapshot.

The default archive is declared in configs/archive_manifest.json. Explicit paths
or source-only mode never execute restored code. A fixed legacy source profile
also retains its whole-source/hash/environment verification. No live fallback,
implicit fetch, training or provider invocation is performed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / 'configs/historical_protocols.json'
DEFAULT_ARCHIVE_MANIFEST = ROOT / 'configs/archive_manifest.json'


def git(repository, *arguments, binary=False):
    completed = subprocess.run(['git', '--no-optional-locks', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
                                '-C', str(repository), *arguments], capture_output=True,
                               text=not binary, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'}, check=False)
    if completed.returncode:
        error = completed.stderr.decode(errors='replace') if binary else completed.stderr
        raise RuntimeError(f'git {arguments[0]} failed: {error[:400]}')
    return completed.stdout


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value):
    if not isinstance(value, str):
        raise ValueError('historical source paths must be strings')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.as_posix() != value:
        raise ValueError('historical source paths must be normalized relative paths')
    return path


def protected_worktrees(repository):
    """Discover this Git repository's source roots, never a captured local path."""
    repository = Path(repository).resolve()
    roots, current, primary = {repository}, None, repository
    for line in git(repository, 'worktree', 'list', '--porcelain').splitlines():
        if line.startswith('worktree '):
            current = Path(line[9:]).resolve()
            roots.add(current)
        elif line == 'branch refs/heads/main' and current is not None:
            primary = current
    return roots, primary


def independent_destination(repository, destination, extra_protected=()):
    requested = Path(destination).absolute()
    if requested.exists() or requested.is_symlink():
        raise ValueError('restoration requires a new absent destination')
    for parent in requested.parents:
        if parent.exists() and parent.is_symlink():
            raise ValueError('restoration destination parents must not be symlinks')
    destination = requested.resolve()
    roots, primary = protected_worktrees(repository)
    roots.update(Path(p).resolve() for p in extra_protected)
    if any(p == destination or p in destination.parents or destination in p.parents for p in roots):
        raise ValueError('historical source must be outside the active and candidate source directories')
    return destination, primary


def archive_identity(repository, manifest=DEFAULT_ARCHIVE_MANIFEST):
    payload = json.loads(Path(manifest).read_text())
    commit, tag = payload['archive_commit'], payload['archive_tag']
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('archive must pin a full commit SHA')
    try:
        resolved, tag_verified = resolve_protocol(repository, {'commit': commit, 'tag': tag})
    except RuntimeError as error:
        raise ValueError('Pinned archive is unavailable locally. Explicitly fetch its declared tag before restoring.') from error
    return resolved, tag, tag_verified


def restore_archive(repository, destination, *, paths=(), source_only=False,
                    manifest=DEFAULT_ARCHIVE_MANIFEST, extra_protected=()):
    """Restore selected tracked bytes without a checkout/index mutation or IO hooks."""
    repository = Path(repository).resolve()
    destination, _ = independent_destination(repository, destination, extra_protected)
    if bool(paths) == bool(source_only):
        raise ValueError('choose explicit archive paths or source-only mode')
    normalized = [normalize_path(value).as_posix() for value in paths]
    commit, tag, tag_verified = archive_identity(repository, manifest)
    args = ['ls-tree', '-r', '-z', commit]
    if normalized:
        args += ['--', *normalized]
    files = []
    for item in git(repository, *args, binary=True).split(b'\0'):
        if not item:
            continue
        meta, raw_name = item.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        name = raw_name.decode()
        normalize_path(name)
        if source_only and not (name.startswith(('src/', 'application/', 'scripts/', 'tests/', 'configs/'))
                                or (name.startswith('requirements-') and '/' not in name)
                                or name in ('pytest.ini', '.gitignore', '.gitattributes')):
            continue
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('archive restore refuses symlinks/submodules/non-regular entries: ' + name)
        files.append((name, oid, mode))
    if not files:
        raise ValueError('selected archive paths contain no tracked regular files')
    if any(not any(name == path or name.startswith(path + '/') for name, _, _ in files) for path in normalized):
        raise ValueError('an explicit archive path did not resolve')
    destination.mkdir(parents=True)
    process = subprocess.Popen(['git', '--no-optional-locks', '-C', str(repository), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})
    records = []
    completed = False
    try:
        for name, oid, mode in files:
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[0].decode() != oid or header[1] != b'blob':
                raise ValueError('invalid archive blob boundary')
            remaining = int(header[2])
            size, digest = remaining, hashlib.sha256()
            output = destination/name
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                while remaining:
                    chunk = process.stdout.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('truncated archive blob')
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            if process.stdout.read(1) != b'\n':
                raise ValueError('invalid archive blob trailer')
            output.chmod(0o555 if mode == '100755' else 0o444)
            records.append({'path': name, 'git_blob': oid, 'bytes': size, 'sha256': digest.hexdigest()})
        completed = True
    finally:
        if not completed:
            process.stdout.close()
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = process.wait()
        if completed:
            process.stdout.close()
            if returncode != 0:
                raise ValueError('archive batch reader failed')
    return {'passed': True, 'archive_commit': commit, 'archive_tag': tag, 'tag_verified': tag_verified,
            'destination': str(destination), 'source_only': bool(source_only), 'files_restored': len(records),
            'files': records, 'scope': 'TRACKED_GIT_BYTES_ONLY',
            'controllers_or_training_or_provider_calls': 0,
            'limits': ['Ignored/private/uncommitted resources are not restored.',
                       'No restored code was imported or executed; this is not full experiment/environment reproduction.']}


def load_protocol(registry, protocol_id):
    payload = json.loads(Path(registry).read_text())
    protocol = payload['protocols'][protocol_id]
    if not re.fullmatch('[0-9a-f]{40}', protocol['commit']):
        raise ValueError('historical protocol must pin a full commit SHA')
    for name, digest in protocol['source_hashes'].items():
        normalize_path(name)
        if not name.endswith('.py') or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('historical source manifest must pin Python bytes by SHA-256')
    return protocol


def resolve_protocol(repository, protocol):
    commit = git(repository, 'rev-parse', protocol['commit'] + '^{commit}').strip()
    if commit != protocol['commit']:
        raise ValueError('historical commit did not resolve to the fixed identity')
    reference = protocol.get('tag')
    tag_verified = False
    if reference:
        result = subprocess.run(['git', '--no-optional-locks', '-C', str(repository), 'rev-parse',
                                 '--verify', '--quiet', reference + '^{commit}'], capture_output=True,
                                text=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})
        if result.returncode == 0:
            if result.stdout.strip() != commit:
                raise ValueError('historical tag moved away from its pinned commit')
            tag_verified = True
    return commit, tag_verified


def source_universe(path):
    path = Path(path)
    files = list((path / 'src/qgeognn_al').rglob('*.py')) + list((path / 'application').glob('*.py'))
    return {file.relative_to(path).as_posix(): file_hash(file) for file in sorted(files)}


def benchmark_hashes(path):
    """Execute only the old pure code_hashes AST, not its importing module."""
    path = Path(path)
    file = path / 'src/qgeognn_al/active_learning_v2/benchmark_protocol.py'
    if not file.is_file():
        return None
    tree = ast.parse(file.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'code_hashes')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {'ROOT': path, 'sha256_file': file_hash}
    exec(compile(module, str(file), 'exec'), namespace)
    return namespace['code_hashes']()


def verify_restored(path, protocol):
    path = Path(path).resolve()
    failures = []
    for name, expected in protocol['source_hashes'].items():
        file = path / name
        if not file.is_file() or file.is_symlink() or file_hash(file) != expected:
            failures.append({'path': name, 'reason': 'source_missing_symlink_or_hash_changed'})
    actual = source_universe(path)
    expected_universe = protocol.get('src_application_hashes', {})
    if expected_universe and actual != expected_universe:
        failures.append({'reason': 'whole_src_application_hash_universe_changed',
                         'added': sorted(set(actual) - set(expected_universe)),
                         'missing': sorted(set(expected_universe) - set(actual)),
                         'different': sorted(k for k in set(actual) & set(expected_universe) if actual[k] != expected_universe[k])})
    # Never execute even the extracted historical function after source/universe tamper.
    observed_benchmark = benchmark_hashes(path) if not failures else None
    expected_benchmark = protocol.get('benchmark_code_hashes')
    if expected_benchmark is not None and observed_benchmark != expected_benchmark:
        failures.append({'reason': 'historical_benchmark_code_hashes_changed'})
    declared = protocol.get('environment', {})
    environment, environment_failures = {}, []
    for name, version in declared.items():
        try:
            current = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            current = None
        environment[name] = {'expected': version, 'observed': current, 'matches': current == version}
        if current != version:
            environment_failures.append(name)
    return {'passed': not failures and not environment_failures, 'source_hashes_verified': len(protocol['source_hashes']) - len([x for x in failures if 'path' in x]),
            'whole_src_application_files': len(actual), 'benchmark_hashes_verified': len(observed_benchmark or {}),
            'environment_matches': not environment_failures, 'environment': environment,
            'environment_failures': environment_failures, 'failures': failures,
            'controllers_imported': 0, 'data_or_truth_tables_opened': 0, 'training_or_provider_calls': 0}


def restore(repository, protocol_id, destination, registry=DEFAULT_REGISTRY):
    repository = Path(repository).resolve()
    destination, active = independent_destination(repository, destination)
    protocol = load_protocol(registry, protocol_id)
    commit, tag_verified = resolve_protocol(repository, protocol)
    before_head = git(active, 'rev-parse', 'HEAD').strip()
    active_index = Path(git(active, 'rev-parse', '--git-path', 'index').strip())
    if not active_index.is_absolute():
        active_index = active / active_index
    before_index = file_hash(active_index)
    destination.parent.mkdir(parents=True, exist_ok=True)
    git(repository, 'worktree', 'add', '--no-checkout', '--detach', str(destination), commit)
    # Settings and index belong to the new historical worktree only. Scientific
    # data, result tables and checkpoints are deliberately absent.
    git(destination, 'sparse-checkout', 'set', '--no-cone', '/src/', '/application/', '/scripts/', '/tests/',
        '/README.md', '/pytest.ini', '/.gitignore', '/.gitattributes')
    git(destination, 'read-tree', '-mu', 'HEAD')
    for folder in ('src', 'application', 'scripts', 'tests'):
        for file in (destination / folder).rglob('*.py'):
            file.chmod(file.stat().st_mode & ~0o222)
    result = verify_restored(destination, protocol)
    result.update(protocol_id=protocol_id, pinned_commit=commit, tag=protocol.get('tag'), tag_verified=tag_verified,
                  destination=str(destination), restored_head=git(destination, 'rev-parse', 'HEAD').strip(),
                  active_HEAD_unchanged=git(active, 'rev-parse', 'HEAD').strip() == before_head,
                  active_index_unchanged=file_hash(active_index) == before_index)
    result['passed'] = result['passed'] and result['active_HEAD_unchanged'] and result['active_index_unchanged']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--archive-manifest', type=Path, default=DEFAULT_ARCHIVE_MANIFEST)
    parser.add_argument('--protocol', choices=['scientific_source', 'phase1_candidate'])
    parser.add_argument('--path', action='append', default=[], help='Exact archived file or directory; repeat for multiple paths')
    parser.add_argument('--source-only', action='store_true', help='Restore only archived code/config/tests, not research payloads')
    parser.add_argument('--protected-root', type=Path, action='append', default=[])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    roots, _ = protected_worktrees(args.repository)
    roots.update(p.resolve() for p in args.protected_root)
    if any(output == root or root in output.parents for root in roots):
        parser.error('report output must be outside active source directories')
    if args.protocol:
        if args.path or args.source_only:
            parser.error('--protocol cannot be combined with --path/--source-only')
        result = restore(args.repository, args.protocol, args.destination, args.registry)
    else:
        result = restore_archive(args.repository, args.destination, paths=args.path, source_only=args.source_only,
                                 manifest=args.archive_manifest, extra_protected=args.protected_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('environment', 'failures', 'files')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
