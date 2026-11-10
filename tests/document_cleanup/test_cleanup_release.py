import hashlib
import json
import subprocess

from tools import verify_cleanup_release as guard


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def fixture(tmp_path, monkeypatch):
    main = tmp_path/'main';main.mkdir()
    git(main, 'init', '-q', '-b', 'main')
    (main/'src').mkdir();(main/'src/unchanged.py').write_text('x = 1\n')
    (main/'obsolete.md').write_text('historical report\n')
    git(main, 'add', '.')
    git(main, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline')
    ref = git(main, 'rev-parse', 'HEAD')
    git(main, 'branch', 'codex/archive-transfer-20261028')
    candidate = tmp_path/'candidate'
    git(main, 'worktree', 'add', '-qb', 'codex/repository-architecture-20261109', str(candidate), ref)
    release = candidate/'release';release.mkdir()
    index = main/'.git/index'
    baseline = {'main_root':str(main), 'main_head':ref, 'main_index_sha256':guard.sha(index.read_bytes()),'source_scopes':['src'],
                'candidate_baseline':ref, 'protected_source_hashes':{'src/unchanged.py':guard.sha((main/'src/unchanged.py').read_bytes())}}
    (release/'MAIN_BASELINE.json').write_text(json.dumps(baseline))
    original = (candidate/'obsolete.md').read_bytes();blob = git(main, 'rev-parse', ref+':obsolete.md')
    row = {'path':'obsolete.md','bytes':len(original),'sha256':guard.sha(original),'git_blob':blob,'recovery_ref':ref}
    import csv
    with (release/'DELETED_FILES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    (release/'REMOTE_RECOVERY.json').write_text(json.dumps({'passed':True,'files':[row],
        'remote_url':'https://github.com/guoscfish/column_chromatography_prediction.git','fetched_commit':ref}))
    (candidate/'obsolete.md').unlink()
    monkeypatch.setattr(guard,'MACHINE_CONTRACTS',())
    return main,candidate,release


def test_valid_cleanup_preserves_main_and_original_branch_set(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    result=guard.verify(candidate,release)
    assert result['passed'] and result['deleted_files_verified']==1
    assert (main/'obsolete.md').exists()


def test_changed_scientific_source_is_rejected(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    (candidate/'src/unchanged.py').write_text('x = 2\n')
    result=guard.verify(candidate,release)
    assert not result['passed'] and result['changed_scientific_code_or_configs']==['src/unchanged.py']
    (main/'src/unchanged.py').write_text('x = 3\n')
    assert guard.verify(candidate,release)['source_failures']==['src/unchanged.py']


def test_remote_hash_mismatch_or_extra_branch_fails(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    remote=json.loads((release/'REMOTE_RECOVERY.json').read_text());remote['files'][0]['sha256']='0'*64
    (release/'REMOTE_RECOVERY.json').write_text(json.dumps(remote))
    git(candidate,'branch','unwanted-branch')
    result=guard.verify(candidate,release)
    assert not result['passed'] and result['deletion_failures']
    assert 'unexpected branch set or active candidate branch' in result['failures']


def test_untracked_scientific_files_in_main_or_candidate_fail(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    (main/'src/new_controller.py').write_text('new = 1\n')
    (candidate/'src/new_model.py').write_text('new = 1\n')
    result=guard.verify(candidate,release)
    assert not result['passed']
    assert result['new_main_source_files']==['src/new_controller.py']
    assert result['changed_scientific_code_or_configs']==['src/new_model.py']


def test_unlisted_protocol_deletion_is_rejected(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    git(candidate,'rm','-q','src/unchanged.py')
    result=guard.verify(candidate,release)
    assert not result['passed']
    assert result['unlisted_deletions']==['src/unchanged.py']


def test_rename_similarity_does_not_hide_an_actual_removed_path(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    (candidate/'new-report.md').write_text('historical report\n')
    git(candidate,'add','new-report.md')
    assert guard.verify(candidate,release)['passed']


def compact_receipt(release):
    import gzip
    baseline=json.loads((release/'MAIN_BASELINE.json').read_text())
    remote=json.loads((release/'REMOTE_RECOVERY.json').read_text())
    (release/'TEST_RESULTS.json').write_text(json.dumps({'schema_version':2,'protection_baseline':baseline,'remote_recovery':remote}))
    with gzip.open(release/'ACTUAL_REMOVALS.csv.gz','wb') as f:f.write((release/'DELETED_FILES.csv').read_bytes())
    for name in ['MAIN_BASELINE.json','REMOTE_RECOVERY.json','DELETED_FILES.csv']:(release/name).unlink()


def test_compact_release_keeps_the_same_strict_guards(tmp_path, monkeypatch):
    main,candidate,release=fixture(tmp_path,monkeypatch)
    compact_receipt(release)
    assert guard.verify(candidate,release)['passed']
    result=json.loads((release/'TEST_RESULTS.json').read_text())
    result['remote_recovery']['files'][0]['sha256']='0'*64
    (release/'TEST_RESULTS.json').write_text(json.dumps(result))
    assert not guard.verify(candidate,release)['passed']


def test_current_default_does_not_require_old_receipt_copies(tmp_path, monkeypatch):
    current=tmp_path/guard.CURRENT_RELEASE;current.mkdir(parents=True)
    assert guard.default_release(tmp_path)==tmp_path/guard.RELEASE
    (current/'TEST_RESULTS.json').write_text('{}')
    assert guard.default_release(tmp_path)==current
