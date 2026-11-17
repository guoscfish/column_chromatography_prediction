"""Adversarial cache/identity checks without mutating any historical artifact."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_boundary as BOUNDARY

SOURCE_HASH = hashlib.sha256((HERE / "data_boundary.py").read_bytes()).hexdigest()


class DataPermissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The historical audit loads only identities/checkpoint byte hashes.
        cls.state = BOUNDARY.audit_lcmd_state(BOUNDARY.ROOT, 157, 333)
        cls.receipt = BOUNDARY.LEGACY / "cache/seed_157_L333.json"

    def test_prepared_cache_hash_drift_rejected_before_deserialization(self):
        with patch.object(BOUNDARY, "audit_lcmd_state", return_value=self.state), \
             patch.object(BOUNDARY, "file_hash", return_value="tampered"), \
             patch.object(BOUNDARY.np, "load", side_effect=AssertionError("drifted cache was deserialized")) as load:
            with self.assertRaisesRegex(ValueError, "historical state/feature identity drift"):
                BOUNDARY.load_state(157, 333)
            load.assert_not_called()

    def test_checkpoint_state_drift_rejected_before_deserialization(self):
        changed = copy.deepcopy(self.state.audit)
        changed["checkpoint_sha256"] = "tampered"
        with patch.object(BOUNDARY, "audit_lcmd_state", return_value=SimpleNamespace(audit=changed)), \
             patch.object(BOUNDARY.np, "load", side_effect=AssertionError("drifted state was deserialized")) as load:
            with self.assertRaisesRegex(ValueError, "historical state/feature identity drift"):
                BOUNDARY.load_state(157, 333)
            load.assert_not_called()

    def test_source_identity_drift_rejected_before_deserialization(self):
        actual_hash = BOUNDARY.file_hash

        def changed_hash(path):
            return "tampered" if Path(path).name == "canonical_4g.csv" else actual_hash(path)

        with patch.object(BOUNDARY, "audit_lcmd_state", return_value=self.state), \
             patch.object(BOUNDARY, "file_hash", side_effect=changed_hash), \
             patch.object(BOUNDARY.np, "load", side_effect=AssertionError("source drift bypassed")) as load:
            with self.assertRaisesRegex(ValueError, "source hash drift"):
                BOUNDARY.load_state(157, 333)
            load.assert_not_called()

    def test_language_cache_drift_rejected_before_language_deserialization(self):
        actual_hash, actual_load = BOUNDARY.file_hash, BOUNDARY.np.load
        loaded = []

        def changed_hash(path):
            return "tampered" if Path(path).name == "llm.npz" else actual_hash(path)

        def guarded_load(path, *args, **kwargs):
            loaded.append(Path(path).name)
            if Path(path).name == "llm.npz":
                raise AssertionError("drifted language cache was deserialized")
            return actual_load(path, *args, **kwargs)

        with patch.object(BOUNDARY, "audit_lcmd_state", return_value=self.state), \
             patch.object(BOUNDARY, "file_hash", side_effect=changed_hash), \
             patch.object(BOUNDARY.np, "load", side_effect=guarded_load):
            with self.assertRaisesRegex(ValueError, "LLM feature cache drift"):
                BOUNDARY.load_state(157, 333)
        self.assertEqual(loaded, ["seed_157_L333.npz"])

    def test_test_id_injected_as_candidate_is_rejected(self):
        original_read = Path.read_text
        meta = json.loads(original_read(self.receipt))
        forbidden = str(self.state.partition.loc[self.state.partition.role.eq("test"), "sample_id"].iloc[0])
        meta["candidate_ids"][0] = forbidden

        def changed_read(path, *args, **kwargs):
            return json.dumps(meta) if path == self.receipt else original_read(path, *args, **kwargs)

        with patch.object(BOUNDARY, "audit_lcmd_state", return_value=self.state), \
             patch.object(Path, "read_text", changed_read):
            with self.assertRaisesRegex(PermissionError, "candidates outside frozen U_t"):
                BOUNDARY.load_state(157, 333)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataPermissionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"scope": "In-memory adversarial metadata/hash substitutions; no historical writes; no test truths",
               "test_count": result.testsRun, "success": result.wasSuccessful(),
               "failures": len(result.failures), "errors": len(result.errors),
               "source_sha256": SOURCE_HASH, "source_unchanged": SOURCE_HASH == hashlib.sha256((HERE / "data_boundary.py").read_bytes()).hexdigest(),
               "test_truth_access_count": 0, "future_U_response_reads": 0,
               "authorized_cache_note": "The final two tests may deserialize the existing authorized Lt/validation NPZ; no response table or forbidden label is read."}
    (HERE / "data_permission_results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
