"""Pixel/file regression against committed generators, using synthetic evidence only."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault('SOURCE_DATE_EPOCH', '946684800')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/column_refactor_plot_test_cache')
ROOT = Path(__file__).resolve().parents[2]
COMMIT = '16eca74769f664bc378a58ccbf380a806723f587'
BASELINE = Path(os.environ.get('REFACTOR_BASELINE_REPO', str(ROOT)))
spec = importlib.util.spec_from_file_location('plot_saved_validation', ROOT/'scripts/plot_saved_validation.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def original(path):
    source = subprocess.check_output(['git', '--no-optional-locks', '-C', str(BASELINE), 'show', COMMIT+':'+path], text=True)
    module = types.ModuleType('historical_plot')
    module.__dict__['__file__'] = str(ROOT/path)
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def reset_plot_state():
    plots.plt.close('all')
    plots.plt.rcdefaults()
    plots.plt.rcParams['svg.hashsalt'] = 'synthetic-equivalence-fixture'


def hashes(folder, names):
    return {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}


def snapshot_fixture(folder):
    folder.mkdir(parents=True)
    (folder/'snapshot.json').write_text(json.dumps({'seeds':[{'seed':157},{'seed':6101}]}))
    (folder/'snapshot_manifest.json').write_text('{}')
    (folder/'REPORT.md').write_text('# Synthetic validation snapshot\n')
    methods = ['new_free_cw_sol61_xhigh','historical_free_mixed_model','nocw','numerical_hybrid32','gradient_lcmd32']
    with (folder/'validation_curves.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['method','seed','budget','validation_selection_score']);writer.writeheader()
        for mi,method in enumerate(methods):
            for si,seed in enumerate([157,6101]):
                for rnd,budget in enumerate([333,365,397]):
                    writer.writerow({'method':method,'seed':seed,'budget':budget,'validation_selection_score':.8+.02*mi+.01*si-.05*rnd})


def audit_fixture(root):
    folder=root/'reports/free_rerun_audit_20261103';folder.mkdir(parents=True)
    pair={}
    for kind in ('historical_free_mixed_model','new_free_cw_sol61_xhigh'):
        model=root/'models'/kind;model.mkdir(parents=True)
        payload=b'synthetic-checkpoint-bytes-no-model-or-data'
        (model/'best.pt').write_bytes(payload)
        (model/'fit_audit.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(payload).hexdigest(),'best_epoch':3,'best_validation_combined_normalized_rmse':.5}))
        with (model/'history.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['epoch','validation_selection_score']);writer.writeheader()
            for epoch in range(5):writer.writerow({'epoch':epoch,'validation_selection_score':.8-.1*epoch})
        pair[kind]=[{'L':333+32*i,'metric':.8-.02*i,'model':str(model.relative_to(root))} for i in range(16)]
    evidence={'summary':[{'seed':seed,'r1_overlap':i+1,'L813_delta':.01*i} for i,seed in enumerate([157,887,2357,6101,12203])], 'rows':[{'seed':157,'pair':pair}]}
    (folder/'trajectory_audit.json').write_text(json.dumps(evidence))
    return folder


class PlotEquivalenceTests(unittest.TestCase):
    def test_snapshot_png_svg_pdf_manifest_and_report_match_committed_generator(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'old',Path(temp)/'new';snapshot_fixture(a);snapshot_fixture(b)
            legacy=original('scripts/studies/plot_free_cw_snapshot_20261102.py')
            reset_plot_state();legacy.plot(a)
            reset_plot_state();plots.plot_snapshot(b)
            names=['validation_comparison.png','validation_comparison.svg','validation_comparison.pdf','REPORT.md','snapshot_manifest.json']
            self.assertEqual(hashes(a,names),hashes(b,names))
            plots.plt.close('all')

    def test_rerun_audit_png_svg_pdf_match_committed_generator(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'old',Path(temp)/'new';fa=audit_fixture(a);fb=audit_fixture(b)
            legacy=original('scripts/studies/plot_free_rerun_audit_20261103.py');legacy.ROOT=a;legacy.OUT=fa
            reset_plot_state();legacy.main()
            reset_plot_state();plots.plot_rerun_audit(fb,b)
            names=['audit_evidence.png','audit_evidence.svg','audit_evidence.pdf']
            self.assertEqual(hashes(fa,names),hashes(fb,names))
            plots.plt.close('all')

    def test_config_dispatch_rejects_unrecognized_kind_and_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'plot.json'; config.write_text(json.dumps({'kind':'unknown','snapshot_directory':temp}))
            with self.assertRaises(ValueError):plots.main(['--config',str(config)])
            config.write_text(json.dumps({'kind':'validation-snapshot','snapshot_directory':temp,'run_training':True}))
            with self.assertRaises(ValueError):plots.main(['--config',str(config)])


if __name__=='__main__':
    unittest.main()
