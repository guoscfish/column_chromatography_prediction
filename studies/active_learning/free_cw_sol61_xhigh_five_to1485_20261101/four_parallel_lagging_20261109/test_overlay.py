"""Offline resource exclusion tests: no API calls or training."""
import fcntl
import importlib.util
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('four_overlay', Path(__file__).with_name('controller_overlay.py'))
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


class SlotsFull(Exception):
    pass


class ResourceTests(unittest.TestCase):
    def test_lagging_then_seed_priority_and_finished_exclusion(self):
        counts = {157: 23, 887: 32, 2357: 36, 6101: 31, 12203: 31}
        self.assertEqual(overlay.priority(counts), [157, 6101, 12203, 887])
        counts[157] = 32
        self.assertEqual(overlay.priority(counts), [6101, 12203, 157, 887])

    def test_four_leases_include_adopted_old_compute_and_release(self):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            study = Path(temp)
            (study / 'locks').mkdir()
            priority_wait, status = Mock(), Mock()
            adapter = SimpleNamespace(load_base=lambda: None, validate=lambda: {},
                a=SimpleNamespace(), m=SimpleNamespace(training_priority_wait=priority_wait,
                    base=SimpleNamespace(status_record=status)))
            stack.enter_context(patch.object(overlay, 'STUDY', study))
            stack.enter_context(patch.object(overlay, 'round_counts', return_value={s: 0 for s in overlay.SEEDS}))
            overlay.bind(adapter)
            adapter.load_base()
            # An old worker already owns the original lease before the new overlay loads.
            old = stack.enter_context((study / 'locks/new_free_resource.lock').open('a+'))
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with ExitStack() as active:
                for seed in (887, 6101, 12203):
                    active.enter_context(adapter.a.compute_slot(seed))
                with patch.object(overlay.time, 'sleep', side_effect=SlotsFull):
                    with self.assertRaises(SlotsFull):
                        with adapter.a.compute_slot(2357):
                            self.fail('fifth computation acquired a lease')
                status.assert_called_once_with(2357, 'WAITING_COMPUTE_SLOT',
                    completed_rounds=0, maximum_expensive_compute=4)
                self.assertGreaterEqual(priority_wait.call_count, 7)
            # Released new slots are usable while the adopted original lease remains held.
            with adapter.a.compute_slot(157):
                self.assertEqual(priority_wait.call_count, 9)


if __name__ == '__main__':
    unittest.main()
