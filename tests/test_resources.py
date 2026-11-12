"""Candidate maintenance surface and explicit resource ownership boundaries."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import pytest
from src.qgeognn_al.infra.resources import SnapshotFile

ROOT=Path(__file__).resolve().parents[1]


def test_one_stable_tree_and_one_user_runner():
    package=ROOT/'src/qgeognn_al'
    assert {p.name for p in package.iterdir() if p.is_dir() and p.name!='__pycache__'}=={'predictor','al','infra'}
    assert {p.name for p in (ROOT/'scripts').glob('*.py')}=={'__init__.py','run_al.py'}
    assert not list(package.rglob('legacy'))


def test_stable_imports_have_no_historical_protocol_or_old_prototype_dependencies():
    forbidden={'active_learning_v2','active_learning_v3','active_learning_v3_2','models','training','schemas',
               'representations','acquisitions','agents','workflows','evaluation','transfer','historical','data','t1_formal','t1b1'}
    violations=[]
    for file in (ROOT/'src/qgeognn_al').rglob('*.py'):
        relative=file.relative_to(ROOT).with_suffix('');package='.'.join(relative.parts[:-1])
        for node in ast.walk(ast.parse(file.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                modules=[importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else (node.module or '')]
            for module in modules:
                parts=module.split('.')
                if parts[:2]==['src','qgeognn_al'] and len(parts)>2 and parts[2] in forbidden:violations.append((str(relative),module))
    assert not violations,violations


def test_snapshot_resource_rejects_writes_symlinks_and_hardlinks(tmp_path):
    file=tmp_path/'resource';file.write_text('synthetic fixture')
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='read-only'):SnapshotFile(file,digest,tmp_path).verify()
    file.chmod(0o444);assert SnapshotFile(file,digest,tmp_path).verify()==file
    link=tmp_path/'link';link.symlink_to(file)
    with pytest.raises(ValueError,match='symlink'):SnapshotFile(link,digest,tmp_path).verify()
    import os
    hard=tmp_path/'hard';os.link(file,hard)
    with pytest.raises(ValueError,match='independent'):SnapshotFile(hard,digest,tmp_path).verify()
