"""Offline tests: permit exact historical failures, never new authentication failures."""
import importlib.util
from pathlib import Path
import json
import unittest
spec=importlib.util.spec_from_file_location('resume',Path(__file__).with_name('controller_overlay.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
policy=json.loads((m.HERE/'authorization.json').read_text())
class Checks(unittest.TestCase):
 def test_all_five_pinned_receipts(self):
  self.assertEqual(len(policy['preserved_fatal_receipts']),5)
  for name in policy['preserved_fatal_receipts']:
   self.assertTrue(m.historical_401(json.loads((m.STUDY/name).read_text()),policy))
 def test_new_401_fails_closed(self):
  record=json.loads((m.STUDY/next(iter(policy['preserved_fatal_receipts']))).read_text())
  record['ended_utc']='new-request'
  self.assertFalse(m.historical_401(record,policy))
 def test_other_fatal_codes_not_allowed(self):
  for reason in ('provider HTTP 402','provider HTTP 400','returned model changed'):
   self.assertFalse(m.historical_401({'fatal_protocol_error':True,'reason':reason},policy))
 def test_only_pinned_state_can_dispatch(self):
  state={'phase':'FAILED_CLOSED','reason':'provider HTTP 401','pid':123}
  local={**policy,'historical_state_fingerprints':[m.fingerprint(state)]}
  self.assertTrue(m.resume_state(state,local))
  state['pid']=124
  self.assertFalse(m.resume_state(state,local))
 def test_resource_policy_retained(self):
  self.assertEqual((policy['maximum_seed_workers'],policy['maximum_expensive_compute']),(3,3))
  self.assertFalse(policy['new_http401_retry'])
if __name__=='__main__':unittest.main()
