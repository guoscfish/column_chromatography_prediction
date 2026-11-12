"""Portable current Markdown checks; archive URLs never trigger an implicit fetch."""
import subprocess
from pathlib import Path
from tools.check_documentation_links import check, markdown_links


def command(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()



def repository(tmp_path):
    command(tmp_path, 'init', '-q')
    return tmp_path



def commit(root):
    command(root, 'add', '.')
    command(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    return command(root, 'rev-parse', 'HEAD')



def test_parser_ignores_examples_and_handles_encoded_and_nested_paths():
    text = '```md\n[example](missing.md)\n```\n`[inline](absent.md)`\n[one](a%20b.md)\n[two](x(y).md)\n[ref]: <z z.md>\n'
    assert [value for value, _ in markdown_links(text)] == ['a%20b.md', 'x(y).md', 'z z.md']



def test_sparse_tracked_payload_is_not_reported_missing(tmp_path):
    root = repository(tmp_path)
    (root/'README.md').write_text('[resource](payload.png)\n')
    (root/'payload.png').write_bytes(b'not opened as an image')
    commit(root)
    command(root, 'update-index', '--skip-worktree', 'payload.png')
    (root/'payload.png').unlink()
    result = check(root)
    assert result['navigation_passed']
    assert result['records'][0]['status'] == 'GIT_TRACKED_SPARSE_TARGET'



def test_deleted_target_and_invalid_anchor_fail_current_navigation(tmp_path):
    root = repository(tmp_path)
    (root/'README.md').write_text('[target](target.md#missing)\n')
    (root/'target.md').write_text('# Actual heading\n')
    commit(root)
    assert check(root)['navigation_failures'][0]['status'] == 'BROKEN_ANCHOR'
    (root/'target.md').unlink()
    assert check(root)['navigation_failures'][0]['status'] == 'BROKEN_LOCAL_TARGET'



def test_own_pinned_git_links_and_repository_boundary_are_verified(tmp_path):
    root = repository(tmp_path)
    (root/'data.md').write_text('evidence\n')
    ref = commit(root)
    url = 'https://github.com/guoscfish/column_chromatography_prediction/blob/' + ref
    (root/'README.md').write_text(f'[ok]({url}/data.md)\n[bad]({url}/absent.md)\n[escape](../private.md)\n')
    result = check(root)
    assert [r['status'] for r in result['records']] == ['PINNED_GIT_TARGET', 'BROKEN_PINNED_TARGET', 'OUTSIDE_REPOSITORY']
    assert not result['navigation_passed']



def test_heading_inline_code_keeps_rendered_content(tmp_path):
    root = repository(tmp_path)
    (root/'README.md').write_text('[valid](target.md#use-python-now)\n')
    (root/'target.md').write_text('# Use `python` now\n')
    commit(root)
    assert check(root)['navigation_passed']



def test_pinned_external_heading_is_also_verified(tmp_path):
    root = repository(tmp_path)
    (root/'target.md').write_text('# Original\n')
    ref = commit(root)
    url = 'https://github.com/guoscfish/column_chromatography_prediction/blob/' + ref
    (root/'README.md').write_text(f'[bad]({url}/target.md#missing)\n')
    assert not check(root)['navigation_passed']




def test_all_current_markdown_has_strict_local_paths_without_history_fallback(tmp_path):
    root=repository(tmp_path);(root/'docs').mkdir();(root/'docs/guide.md').write_text('[missing](../retired.md)\n')
    assert not check(root)['navigation_passed']
    assert check(root)['navigation_failures'][0]['status']=='BROKEN_LOCAL_TARGET'


def test_missing_archive_object_is_explicit_not_an_implicit_fetch(tmp_path):
    root=repository(tmp_path);(root/'README.md').write_text('[archive](https://github.com/guoscfish/column_chromatography_prediction/blob/'+('9'*40)+'/old.md)\n');commit(root)
    result=check(root);assert result['navigation_passed'] and result['records'][0]['status']=='EXTERNAL_PINNED_UNVERIFIED'
    strict=check(root,require_archive_objects=True)
    assert not strict['navigation_passed'] and strict['records'][0]['status']=='MISSING_ARCHIVE_OBJECT'


def test_document_symlinks_and_absolute_local_links_fail_without_read(tmp_path):
    root=repository(tmp_path);(root/'README.md').write_text('[outside](/private/synthetic.md)\n')
    (root/'linked.md').symlink_to(root/'README.md')
    result=check(root)
    assert not result['navigation_passed']
    assert {r['status'] for r in result['records']}=={'OUTSIDE_REPOSITORY','UNSAFE_DOCUMENT_SYMLINK'}
