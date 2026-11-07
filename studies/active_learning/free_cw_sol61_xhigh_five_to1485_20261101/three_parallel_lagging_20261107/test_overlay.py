"""Offline resource and scheduling checks; never calls selection or training."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('three',Path(__file__).with_name('controller_overlay.py'))
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)

class Checks(unittest.TestCase):
 def test_lagging_priority(self):
  self.assertEqual(x.priority(dict(zip(x.SEEDS,[15,29,27,27,24])))[:3],[157,12203,2357])
 def test_round_completion_reorders(self):
  self.assertEqual(x.priority(dict(zip(x.SEEDS,[15,29,28,27,24])))[:3],[157,12203,6101])
 def test_completed_excluded(self):
  self.assertEqual(x.priority({s:36 for s in x.SEEDS}),[])
 def test_real_completed_trajectory(self):
  counts=x.round_counts()
  self.assertEqual(set(counts),set(x.SEEDS))
  self.assertTrue(all(0 <= n <= 36 for n in counts.values()))
 def test_three_compute_leases_and_fourth_waits(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'locks').mkdir()
   fake=SimpleNamespace(load_base=lambda:None,validate=lambda:{},a=SimpleNamespace(),
    m=SimpleNamespace(training_priority_wait=lambda:None,base=SimpleNamespace(status_record=lambda *a,**k:None)))
   with patch.object(x,'STUDY',root):
    x.bind(fake);fake.load_base()
    a,b,c=(fake.a.compute_slot(s) for s in (157,12203,2357))
    a.__enter__();b.__enter__();c.__enter__()
    try:
     with patch.object(x,'round_counts',return_value={s:0 for s in x.SEEDS}),patch.object(x.time,'sleep',side_effect=RuntimeError('waiting')):
      with self.assertRaisesRegex(RuntimeError,'waiting'):fake.a.compute_slot(887).__enter__()
     b.__exit__(None,None,None)
     with fake.a.compute_slot(6101):pass
    finally:
     a.__exit__(None,None,None);c.__exit__(None,None,None)
 def test_existing_compute_counts_toward_three(self):
  import fcntl
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'locks').mkdir()
   fake=SimpleNamespace(load_base=lambda:None,validate=lambda:{},a=SimpleNamespace(),
    m=SimpleNamespace(training_priority_wait=lambda:None,base=SimpleNamespace(status_record=lambda *a,**k:None)))
   with patch.object(x,'STUDY',root):
    x.bind(fake);fake.load_base()
    with (root/'locks/new_free_resource.lock').open('a+') as old:
     fcntl.flock(old,fcntl.LOCK_EX)
     with fake.a.compute_slot(157),fake.a.compute_slot(2357):
      with patch.object(x,'round_counts',return_value={s:0 for s in x.SEEDS}),patch.object(x.time,'sleep',side_effect=RuntimeError('waiting')):
       with self.assertRaisesRegex(RuntimeError,'waiting'):fake.a.compute_slot(887).__enter__()

if __name__=='__main__':unittest.main()
