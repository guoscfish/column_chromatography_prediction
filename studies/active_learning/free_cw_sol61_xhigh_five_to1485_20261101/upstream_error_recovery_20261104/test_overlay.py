import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import httpx

spec = importlib.util.spec_from_file_location('overlay', Path(__file__).with_name('controller_overlay.py'))
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
class Fatal(Exception): pass

class RecoveryTests(unittest.TestCase):
    def adapter(self, reason=None):
        def stream(*a, **k):
            if reason: raise Fatal(reason)
            return ('unchanged', a, k)
        return SimpleNamespace(legacy_transient=lambda r: r.get('prior') is True,
                               m=SimpleNamespace(live_stream=stream, FatalProtocolError=Fatal))
    def test_exact_stream_retry(self):
        a=self.adapter(o.REASON);o.install(a)
        with self.assertRaises(httpx.RemoteProtocolError):a.m.live_stream()
    def test_other_fatal_preserved(self):
        for reason in ['provider HTTP 401','provider HTTP 402','wrong model','artifact drift','provider terminal code: upstream_error_extra']:
            with self.subTest(reason=reason):
                a=self.adapter(reason);o.install(a)
                with self.assertRaises(Fatal):a.m.live_stream()
    def test_success_unchanged(self):
        a=self.adapter();o.install(a)
        self.assertEqual(a.m.live_stream('full', config='unchanged'),('unchanged',('full',),{'config':'unchanged'}))
    def test_receipt_exact_only(self):
        a=self.adapter();o.install(a)
        self.assertTrue(a.legacy_transient({'fatal_protocol_error':True,'reason':o.REASON}))
        self.assertFalse(a.legacy_transient({'reason':o.REASON}))
        self.assertFalse(a.legacy_transient({'fatal_protocol_error':True,'reason':'provider HTTP 401'}))
        self.assertTrue(a.legacy_transient({'prior':True}))
    def test_supervisor_scope(self):
        a=SimpleNamespace(load_base=lambda:None,validate=lambda:'valid',transient_stop=lambda s:s.get('old') is True,
                          retire_worker=lambda *a,**k:False)
        o.bind(a)
        self.assertTrue(a.transient_stop({'reason':o.REASON}))
        self.assertFalse(a.transient_stop({'reason':'wrong model'}))
        self.assertTrue(a.transient_stop({'old':True}))
        self.assertEqual(a.ENTRY,o.ENTRY)
if __name__=='__main__':unittest.main()
