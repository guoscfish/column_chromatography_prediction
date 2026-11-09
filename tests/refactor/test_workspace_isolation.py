"""Exercise the Git workspace boundary using tiny temporary repositories only."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/verify_refactor_integrity.py'
spec = importlib.util.spec_from_file_location('workspace_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def git(root, *args):
    return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null',
                                    '-c', 'commit.gpgsign=false', *args],
                                   cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def metadata(root):
    result = {}
    for name in ('HEAD', 'index'):
        value = Path(git(root, 'rev-parse', '--git-path', name))
        path = value if value.is_absolute() else root / value
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class WorkspaceIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='column-workspace-isolation-')
        self.base = Path(self.temp.name).resolve()
        self.active = self.base / 'active'
        self.active.mkdir()
        git(self.active, 'init', '--quiet', '--initial-branch=main')
        git(self.active, 'config', 'user.name', 'Synthetic workspace audit')
        git(self.active, 'config', 'user.email', 'audit@example.invalid')
        (self.active / 'source.py').write_text('VALUE = 1\n')
        git(self.active, 'add', 'source.py')
        git(self.active, 'commit', '--quiet', '-m', 'Synthetic baseline')
        self.baseline = git(self.active, 'rev-parse', 'HEAD')

    def tearDown(self):
        self.temp.cleanup()

    def linked(self):
        candidate = self.base / 'candidate'
        git(self.active, 'worktree', 'add', '--quiet', '-b', verifier.REFACTOR_BRANCH,
            str(candidate), self.baseline)
        return candidate

    def test_linked_worktree_shares_objects_but_not_head_or_index(self):
        candidate = self.linked()
        (candidate / 'candidate_only.py').write_text('VALUE = 2\n')
        git(candidate, 'add', 'candidate_only.py')
        before = metadata(self.active)
        result = verifier.check_workspace_isolation(candidate, self.active, self.baseline)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['mode'], 'linked_worktree')
        self.assertTrue(result['objects_shared'])
        self.assertEqual(result['separate_metadata'], {'HEAD': True, 'index': True})
        self.assertEqual(result['shared_object_file_samples'], [])
        self.assertEqual(metadata(self.active), before)
        self.assertEqual(git(self.active, 'status', '--porcelain'), '')

    def test_linked_worktree_rejects_foreign_branch_and_changed_main(self):
        candidate = self.linked()
        git(candidate, 'branch', '-m', 'wrong-refactor')
        result = verifier.check_workspace_isolation(candidate, self.active, self.baseline)
        self.assertFalse(result['passed'])
        self.assertIn('linked candidate must use the authorized refactor branch', result['failures'])
        git(candidate, 'branch', '-m', verifier.REFACTOR_BRANCH)
        git(self.active, 'commit', '--quiet', '--allow-empty', '-m', 'Synthetic moved main')
        result = verifier.check_workspace_isolation(candidate, self.active, self.baseline)
        self.assertFalse(result['passed'])
        self.assertIn('active main and HEAD must remain at the protected baseline commit', result['failures'])

    def test_independent_clone_remains_supported_without_shared_objects(self):
        candidate = self.base / 'private-clone'
        git(self.base, 'clone', '--quiet', '--no-hardlinks', str(self.active), str(candidate))
        result = verifier.check_workspace_isolation(candidate, self.active, self.baseline)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['mode'], 'independent_clone')
        self.assertFalse(result['objects_shared'])
        self.assertTrue(result['shared_object_file_samples'])
        self.assertFalse(any(item['same_inode'] for item in result['shared_object_file_samples']))

    def test_same_or_nested_source_tree_is_never_allowed(self):
        for candidate in (self.active, self.active / 'nested'):
            result = verifier.check_workspace_isolation(candidate, self.active, self.baseline)
            self.assertFalse(result['passed'])
            self.assertEqual(result['mode'], 'invalid')


if __name__ == '__main__':
    unittest.main()
