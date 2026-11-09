"""Safety guards for the source-only audit and retirement proof gate."""
import ast
import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[2] / 'scripts/audit_refactor_dependencies.py'
spec = importlib.util.spec_from_file_location('audit_refactor_dependencies', FILE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DependencyAuditTests(unittest.TestCase):
    def test_metadata_allowlist_excludes_truth_and_provider_payloads(self):
        for path in ('studies/x/protocol.json', 'docs/repository/RESEARCH_SCOPE.json', 'src/qgeognn_al/data.py', 'scripts/check.sh'):
            self.assertTrue(audit.safe_source(path), path)
        for path in ('dataset/test.csv', 'runtime/model/best.pt', 'runtime/model/predictions.csv.gz', 'selections/round_1/request.json', 'selections/round_1/response_manifest.json', 'runtime/catalogue_manifest.json', 'private/.env'):
            self.assertFalse(audit.safe_source(path), path)

    def test_relative_and_from_import_module_edges(self):
        mapping = {'src.qgeognn_al.models': {'src/qgeognn_al/models/__init__.py'}, 'src.qgeognn_al.models.network': {'src/qgeognn_al/models/network.py'}}
        tree = ast.parse('from ..models import network\nfrom src.qgeognn_al.models.network import Model')
        actual = audit.resolve_imports('src/qgeognn_al/training/fit.py', tree, mapping)
        self.assertEqual(actual, {'src/qgeognn_al/models/__init__.py', 'src/qgeognn_al/models/network.py'})

    def test_init_alias_and_installed_package_alias(self):
        self.assertEqual(audit.module_aliases('src/qgeognn_al/models/__init__.py'), {'src.qgeognn_al.models', 'qgeognn_al.models'})
        self.assertIn('QGeoGNN', audit.module_aliases('application/QGeoGNN.py'))

    def test_absent_imports_do_not_authorize_deletion(self):
        row = {'evidence': {'git_blob_oid': 'abc', 'sha256': 'def'}}
        result = audit.retirement_gate(row)
        self.assertFalse(result['safe'])
        self.assertIn('live_process_review_missing', result['blockers'])
        self.assertIn('replacement_equivalence_missing', result['blockers'])

    def test_frozen_paths_block_even_with_replacement(self):
        row = {'frozen_hash_dependents': ['benchmark_protocol.py:rglob'], 'evidence': {'git_blob_oid': 'abc', 'sha256': 'def'}}
        result = audit.retirement_gate(row, replacement_verified=True, live_reviewed=True, output_reproduced=True)
        self.assertFalse(result['safe'])
        self.assertEqual(result['blockers'], ['frozen_hash_dependents'])

    def test_output_reproduction_and_restore_identity_required(self):
        result = audit.retirement_gate({'authoritative_outputs': ['figure.png']}, replacement_verified=True, live_reviewed=True)
        self.assertFalse(result['safe'])
        self.assertIn('unique_output_reproduction_missing', result['blockers'])
        self.assertIn('recoverable_baseline_identity_missing', result['blockers'])

    def test_documentary_review_does_not_bypass_executable_callers(self):
        row = {'dynamic_references': ['reports/snapshot/README.md'], 'importers': ['scripts/caller.py'], 'evidence': {'git_blob_oid': 'abc', 'sha256': 'def'}}
        result = audit.retirement_gate(row, replacement_verified=True, live_reviewed=True, output_reproduced=True, reviewed_nonruntime_references=['reports/snapshot/README.md'])
        self.assertFalse(result['safe'])
        self.assertEqual(result['blockers'], ['importers'])

    def test_fully_evidenced_record_passes_only_explicit_reviews(self):
        row = {'evidence': {'git_blob_oid': 'abc', 'sha256': 'def'}, 'authoritative_outputs': ['figure.png']}
        self.assertTrue(audit.retirement_gate(row, replacement_verified=True, live_reviewed=True, output_reproduced=True)['safe'])


if __name__ == '__main__':
    unittest.main()
