"""Frozen design, historical provenance, and read-only source guards."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import TRAINING_CONFIG, package_versions
from .maxdet_study import BASELINE, historical_round0_paths, _validate_split_and_reuse
from .lcmd_to_ivr_study import make_context
from .protocol import stable_hash

STUDY = ROOT / 'studies/active_learning/qgeognn_v2_row_cw_ivr_portfolio_b32'
METHOD = 'cw_ivr_portfolio'
METHODS = (METHOD,)
SEEDS = (157, 6101)
BATCH_SIZE = 32
SKETCH_DIMENSION = 512
ACQUISITION_ROUNDS = 10
ACTIVE_LABEL_BUDGETS = tuple(range(333,654,32))
FINAL_ACTIVE_LABELS = 653
TOLERANCE = .01
CW_STUDY = STUDY.parent / 'qgeognn_v2_row_cw_lcmd_to_653'
IVR_STUDY = STUDY.parent / 'qgeognn_v2_row_kernel_ivr_b32'
MAXDET_STUDY = STUDY.parent / 'qgeognn_v2_row_maxdet_b32'
COMPARATORS = ('center_width_lcmd','kernel_ivr','lcmd','hybrid','gradient_maxdet')


def read_json(path):
    return json.loads(Path(path).read_text())


def split_path(seed, study=STUDY):
    return Path(study) / f'splits/row_seed_{seed}.csv'


def comparator_sources():
    return {m: p / 'results/learning_curve_metrics.csv' for m,p in
            [('center_width_lcmd',CW_STUDY),('kernel_ivr',IVR_STUDY),('lcmd',BASELINE),('hybrid',BASELINE),('gradient_maxdet',MAXDET_STUDY)]}


def protocol_record():
    return {'study': STUDY.name, 'evidence': 'development / exposed cohort; not independent confirmation',
            'seeds': list(SEEDS), 'method': METHOD, 'budgets': list(ACTIVE_LABEL_BUDGETS),
            'batch_size': 32, 'cw_turns': 16, 'ivr_turns': 16, 'first_expert': 'cw',
            'new_fits': 20, 'anchor_fits': 0, 'maximum_active_labels':653,
            'training':TRAINING_CONFIG, 'tolerance':TOLERANCE, 'packages':package_versions(),
            'source_sha256':sha256_file(SOURCE_DATA), 'graph_cache_sha256':sha256_file(SOURCE_GRAPH_CACHE),
            'protocol_markdown_sha256':sha256_file(STUDY / 'PROTOCOL.md')}


def audit_reuse():
    anchors, comparisons, protected = [], [], set()
    for seed in SEEDS:
        audit = _validate_split_and_reuse(seed)
        paths = historical_round0_paths(seed)
        old = read_json(paths['context'])['contract']
        with tempfile.TemporaryDirectory(prefix='portfolio_lineage_') as d:
            context = make_context(seed, Path(d))
            fit = read_json(paths['member0_audit'])
            inputs = read_json(paths['member0_model'].parent.parent / 'input_contract.json')
            if inputs['training_config'] != context.training_config(0):
                raise RuntimeError('anchor training configuration mismatch')
            from ..models import build_predictor, load_predictor_checkpoint
            from ..training.predictor import seed_everything
            from .gradient_features import state_dict_hash
            seed_everything(fit['initialization_seed'])
            if state_dict_hash(build_predictor(context.normalization)) != fit['initialization_hash']:
                raise RuntimeError('frozen initialization mismatch')
            if state_dict_hash(load_predictor_checkpoint(paths['member0_model'])) != fit['checkpoint_state_hash']:
                raise RuntimeError('checkpoint state mismatch')
        if fit['checkpoint_sha256'] != sha256_file(paths['member0_model']) or fit['prediction_sha256'] != sha256_file(paths['member0_predictions']):
            raise RuntimeError('anchor model/prediction byte mismatch')
        source_round = paths['member0_model'].parent.parent
        source_record = read_json(source_round / 'contract.json')
        global_entry = read_json(BASELINE / 'global_pre_test_freeze.json')['entries'][f'seed_{seed}/lcmd/round_00']
        if sha256_file(source_round/'contract.json') != global_entry['round_contract_sha256']:
            raise RuntimeError('anchor global freeze mismatch')
        for name, digest in source_record['files'].items():
            if sha256_file(source_round/name) != digest:
                raise RuntimeError(f'anchor nested artifact mismatch: {name}')
            protected.add(source_round/name)
        partition = pd.read_csv(BASELINE / f'splits/row_seed_{seed}.csv')
        for role, prefix in [('l0','labeled'),('u0','unlabeled')]:
            if np.load(source_round/f'{prefix}_ids.npy').astype(str).tolist() != partition.loc[partition.role.eq(role),'sample_id'].astype(str).tolist():
                raise RuntimeError('ordered L333/U333 lineage mismatch')
        audit.update(initialization_hash=fit['initialization_hash'], preprocessing_hash=stable_hash(old['preprocessing']),
                     graph_cache_sha256=old['source_graph_cache_sha256'], checkpoint_reused=True,
                     gradient_reused=False, new_anchor_fits=0,
                     gradient_reason='one shared multi-transform pass; old endpoint-only compressed bank is insufficient for CW')
        anchors.append(audit)
        protected.update(paths.values())
        protected.update([source_round/'contract.json', BASELINE/'global_pre_test_freeze.json', BASELINE/f'splits/row_seed_{seed}.csv',BASELINE/f'runtime/seed_{seed}/scaler.json'])
        for method, source in comparator_sources().items():
            table = pd.read_csv(source)
            arm = table.loc[table.outer_seed.eq(seed)&table.method.eq(method)&table.active_label_count.isin(ACTIVE_LABEL_BUDGETS)]
            if sorted(arm.active_label_count.tolist()) != list(ACTIVE_LABEL_BUDGETS):
                raise RuntimeError('incomplete comparator grid')
            root = source.parent.parent
            # IVR inherits exactly the baseline splits/context in its frozen seal.
            split = (BASELINE if method=='kernel_ivr' else root)/f'splits/row_seed_{seed}.csv'
            identity = ['canonical_index','sample_id','role']
            if not pd.read_csv(split)[identity].equals(partition[identity]):
                raise RuntimeError('comparator split mismatch')
            ctxpath = (BASELINE if method=='kernel_ivr' else root)/f'runtime/seed_{seed}/context.json'
            ctx = read_json(ctxpath)
            ctx = ctx.get('contract',ctx)
            if ctx['preprocessing'] != old['preprocessing']:
                raise RuntimeError('comparator preprocessing/target-scale mismatch')
            # The published CW653 table already integrates exact five-arm sources.
            published = pd.read_csv(CW_STUDY/'results/learning_curve_metrics.csv')
            other = published.loc[published.outer_seed.eq(seed)&published.method.eq(method)].sort_values('active_label_count')
            cols = ['combined_normalized_RMSE','V1_RMSE','V2_RMSE','V1_R2','V2_R2']
            if not np.allclose(arm.sort_values('active_label_count')[cols], other[cols],rtol=1e-12,atol=1e-12):
                raise RuntimeError('published comparator metric mismatch')
            comparisons.append({'outer_seed':seed,'method':method,'source':str(source.relative_to(ROOT)),
                                'budgets':list(ACTIVE_LABEL_BUDGETS),'preprocessing_hash':stable_hash(ctx['preprocessing']),
                                'metric':'benchmark_reporting.metric_row; normalized trapezoidal AULC'})
            protected.update([source,split,ctxpath,root/'global_pre_test_freeze.json'])
    return {'status':'PASS','anchors':anchors,'comparators':comparisons,'test_truth_access_count':0,
            'protected_files':{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(protected)}}


def prepare(study=STUDY):
    study=Path(study)
    if (study/'seal.json').exists():
        return validate_prepared(study)
    if (study/'runtime').exists():
        raise RuntimeError('cannot prepare after runtime exists')
    tests=ET.parse(study/'preflight_tests.xml').getroot()
    suites=list(tests.iter('testsuite'))
    if not suites or sum(int(s.get('tests',0)) for s in suites)<13 or any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites):
        raise RuntimeError('passing non-skipped preflight tests required')
    reuse=audit_reuse()
    (study/'splits').mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        shutil.copy2(BASELINE/f'splits/row_seed_{seed}.csv',split_path(seed,study))
    atomic_json(study/'protocol.json',protocol_record())
    atomic_json(study/'reuse_audit.json',reuse)
    # Protect relevant imported implementation, not unrelated concurrently developed studies.
    paths=list((ROOT/'src/qgeognn_al').rglob('*.py'))+list((ROOT/'application').glob('*.py'))
    paths=[p for p in paths if 'cw_lcmd_to_1005' not in p.name]
    paths += [ROOT/'scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py',
              ROOT/'tests/active_learning_v2/test_cw_ivr_portfolio.py',study/'protocol.json',study/'PROTOCOL.md',study/'reuse_audit.json',study/'preflight_tests.xml']
    paths += list((study/'splits').glob('*.csv'))
    atomic_json(study/'seal.json',{'status':'SEALED_BEFORE_NEW_FITS','head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in paths},'test_truth_access_count':0})
    atomic_json(study/'global_pre_test_freeze.json',{'status':'PENDING_TRAJECTORIES','test_truth_access_count':0})
    atomic_json(study/'decision.json',{'status':'PENDING_RESULTS','automatic_followup':False})
    return {'status':'PREPARED','new_fits':0,'anchors_reused':2}


def validate_prepared(study=STUDY):
    study=Path(study)
    for manifest in (read_json(study/'seal.json')['files'],read_json(study/'reuse_audit.json')['protected_files']):
        for name,digest in manifest.items():
            if sha256_file(ROOT/name)!=digest:
                raise RuntimeError(f'frozen source/artifact changed: {name}')
    if protocol_record()!=read_json(study/'protocol.json'):
        raise RuntimeError('protocol/environment drift')
    return {'status':'VALID','protocol_hash':stable_hash(read_json(study/'protocol.json'))}
