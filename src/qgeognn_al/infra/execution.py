"""Explicit run permissions and protected output roots; no secrets are loaded."""
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from .contracts import content_hash
from .resources import SnapshotFile

ROOT = Path(__file__).resolve().parents[3]


def protected_worktrees():
    result = subprocess.run(['git','--no-optional-locks','-C',str(ROOT),'worktree','list','--porcelain'],capture_output=True,text=True,check=True)
    return tuple(Path(line[9:]).resolve() for line in result.stdout.splitlines() if line.startswith('worktree '))


def runtime_path(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path,*path.parents]):
        raise ValueError('runtime path may not traverse symlinks')
    if any((parent/'.git').exists() for parent in [path,*path.parents]) or any(path.resolve().is_relative_to(p) for p in protected_worktrees()):
        raise ValueError('runtime output cannot be inside a protected Git worktree')
    return path


def snapshot_resource(value):
    if set(value) != {'path','sha256','root'}:
        raise ValueError('resource requires explicit path, sha256 and authorized snapshot root')
    item=SnapshotFile(Path(value['path']),value['sha256'],Path(value['root']))
    if any((parent/'.git').exists() for parent in item.path.absolute().parents) or any(item.path.resolve().is_relative_to(p) for p in protected_worktrees()):
        raise ValueError('real run resources require independent snapshots, never live worktree files')
    item.verify()
    return item


@dataclass(frozen=True)
class ExecutionGrant:
    protocol_hash: str
    output: str
    actions: frozenset[str]

    @classmethod
    def read(cls,path,*,config,output):
        if path is None:
            raise PermissionError('separate explicit authorization file is required before real execution')
        file=Path(path)
        if file.is_symlink():raise ValueError('authorization may not be a symlink')
        value=json.loads(file.read_text())
        if set(value)!={'schema_version','protocol_hash','output','actions'} or value['schema_version']!=1:
            raise ValueError('invalid execution authorization schema')
        actions=frozenset(value['actions'])
        if not actions <= {'training','provider','reveal'}:
            raise ValueError('unknown execution authorization action')
        if value['protocol_hash']!=content_hash(config) or Path(value['output']).resolve()!=runtime_path(output).resolve():
            raise PermissionError('authorization is for another config or output directory')
        return cls(value['protocol_hash'],str(Path(output).resolve()),actions)

    def require(self,*actions):
        if not set(actions)<=self.actions:
            raise PermissionError('separate authorization missing actions: '+','.join(sorted(set(actions)-self.actions)))


def scientific_source_identity(repository=ROOT):
    """Require a committed scientific tree before real actions or resume.

    Documentation/tests may change independently. Untracked executable source
    cannot be hidden behind an unchanged HEAD or ignored by the run manifest.
    """
    import hashlib
    repository=Path(repository)
    scopes=('src','application','scripts','configs/method_registry.json')
    def git(*args):
        return subprocess.check_output(['git','--no-optional-locks','-C',str(repository),*args])
    if git('diff','--name-only','HEAD','--',*scopes).strip():
        raise ValueError('real execution requires committed scientific source; dirty code is not HEAD')
    others=[p for p in git('ls-files','--others','--exclude-standard','-z','--',*scopes).decode().split('\0') if p]
    if any(p.endswith('.py') or p.startswith('configs/') for p in others):
        raise ValueError('real execution rejects untracked scientific source')
    digest=hashlib.sha256()
    for row in git('ls-files','-s','-z','--',*scopes).split(b'\0'):
        if not row:continue
        info,path=row.split(b'\t',1);file=repository/path.decode()
        if file.is_symlink() or not file.is_file():raise ValueError('scientific source must be regular and materialized')
        digest.update(info+b'\t'+path+b'\0'+hashlib.sha256(file.read_bytes()).digest())
    return {'commit':git('rev-parse','HEAD').decode().strip(),'scientific_tree_sha256':digest.hexdigest()}
