"""Restore immutable historical source in an isolated detached Git worktree.

No scientific module/controller is imported and no data/checkpoint is expanded.
The original main worktree, its HEAD, index and running resources are untouched.
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
DEFAULT_REGISTRY = ROOT / 'docs/repository/phase2/HISTORICAL_PROTOCOLS.json'


def git(repository, *arguments, binary=False):
    completed = subprocess.run(['git', '--no-optional-locks', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
                                '-C', str(repository), *arguments], capture_output=True,
                               text=not binary, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}, check=False)
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
                                text=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
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
    requested_destination = Path(destination).absolute()
    if requested_destination.exists() or requested_destination.is_symlink():
        raise ValueError('restoration requires a new absent destination')
    destination = requested_destination.resolve()
    active = Path(json.loads(Path(registry).read_text())['active_repository']).resolve()
    if active == destination or active in destination.parents or destination in active.parents:
        raise ValueError('historical source must be outside the active source directory')
    for parent in destination.parents:
        if parent.exists() and parent.is_symlink():
            raise ValueError('restoration destination parents must not be symlinks')
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
        '/docs/', '/README.md', '/pytest.ini', '/.gitignore', '/.gitattributes')
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
    parser.add_argument('--protocol', choices=['scientific_source', 'phase1_candidate'], default='scientific_source')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    active = Path(json.loads(args.registry.read_text())['active_repository']).resolve()
    if output == active or active in output.parents:
        parser.error('report output must be outside the active source directory')
    result = restore(args.repository, args.protocol, args.destination, args.registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('environment', 'failures')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
