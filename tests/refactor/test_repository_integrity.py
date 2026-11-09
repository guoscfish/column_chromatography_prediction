"""Integration checks for the actual candidate and untouched running repository.

Set REFACTOR_REPOSITORY/REFACTOR_ACTIVE_MANIFEST/REFACTOR_RUNTIME_SNAPSHOT when
the candidate or audit artifacts are located outside the standard sibling paths.
No scientific runner, controller, label table or model is imported.
"""
import importlib.util
import os
from pathlib import Path
import unittest


def load_verifier():
    script = Path(__file__).resolve().parents[2] / 'scripts/verify_refactor_integrity.py'
    spec = importlib.util.spec_from_file_location('verify_refactor_integrity', script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


class RepositoryIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verifier = load_verifier()
        cls.report = verifier.audit(
            candidate=Path(os.environ.get('REFACTOR_REPOSITORY', str(verifier.DEFAULT_CANDIDATE))),
            active=Path(os.environ.get('REFACTOR_ACTIVE_ROOT', str(verifier.DEFAULT_ACTIVE))),
            active_manifest=os.environ.get('REFACTOR_ACTIVE_MANIFEST'),
            runtime_snapshot=os.environ.get('REFACTOR_RUNTIME_SNAPSHOT'),
            retirement_manifest=os.environ.get('REFACTOR_RETIREMENT_MANIFEST'))

    def assert_check(self, name):
        check = self.report['checks'][name]
        self.assertTrue(check['passed'], f'{name}: {check}')

    def test_baseline_scientific_sources_keep_exact_bytes(self):
        self.assert_check('baseline_python_bytes')

    def test_frozen_263_files_and_28_compatibility_paths_remain_recoverable(self):
        self.assert_check('frozen_and_compatibility')

    def test_sparse_data_and_protected_artifacts_have_git_objects(self):
        self.assert_check('object_recoverability_and_protection')

    def test_candidate_git_storage_is_independent(self):
        self.assert_check('independent_clone')

    def test_running_main_sources_and_frozen_identities_are_untouched(self):
        self.assert_check('active_repository_unchanged')

    def test_retirement_records_identify_original_bytes_and_restore_path(self):
        self.assert_check('retirement_authorization')


if __name__ == '__main__':
    unittest.main()
