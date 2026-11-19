#!/usr/bin/env python3
"""Predict only nonquarantined formal train/validation X with zero graph labels."""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from hashlib import sha256
from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT))
import checkpoint_audit
from checkpoint_audit import load_formal
from common import infer
from data_boundary import FEATURE_COLUMNS, QUARANTINED_IDS, load_formal_split

SOURCES = {}
OUTPUT_COLUMNS = ['sample_id','source_role','V1_q10','V1_q50','V1_q90','V2_q10','V2_q50','V2_q90']


def record(path: Path) -> str:
    path = Path(path)
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    item = dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)
    if item['path'] in SOURCES and SOURCES[item['path']] != item:
        raise ValueError(f'Inference source changed: {path}')
    SOURCES[item['path']] = item
    return item['sha256']


def ordered_hash(ids) -> str:
    return sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def main() -> None:
    started = time.time()
    for name in ['LABEL_BOUNDARY_INCIDENT.json','data_boundary.py','data_boundary_inputs.json',
                 'data_boundary_verification.json','source_metadata.csv','common.py','checkpoint_audit.py']:
        record(OUT/name)
    incident = json.loads((OUT/'LABEL_BOUNDARY_INCIDENT.json').read_text())
    quarantine = set(incident['quarantined_ids'])
    assert quarantine == set(QUARANTINED_IDS) and len(quarantine) == 3
    metadata_path = OUT/'source_metadata.csv'
    metadata_columns = ['canonical_index', *FEATURE_COLUMNS, 'global_quarantine']
    assert set(pd.read_csv(metadata_path, nrows=0).columns) == set(metadata_columns)
    metadata = pd.read_csv(metadata_path, usecols=metadata_columns, dtype={'sample_id': str})
    assert len(metadata) == metadata.sample_id.nunique() == 4163
    assert np.array_equal(metadata.canonical_index.to_numpy(), np.arange(4163))
    assert set(metadata.loc[metadata.global_quarantine, 'sample_id']) == quarantine
    assert not any(c in metadata.columns for c in ['V1_ml','V2_ml','V1_true','V2_true'])
    protocol_path = ROOT/'studies/predictor/final_4g_qualification/protocol.json'
    record(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    cache_path = ROOT/'experiments/e0_4g_baseline/graph_cache_4g.pt'
    cache_sha = record(cache_path)
    assert cache_sha == protocol['graph_cache_sha256']
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    assert len(cache) == 217 and all(set(entry) == {'descriptor','graph'} for entry in cache.values())
    assert all(not ({'y','V1_ml','V2_ml','V1_true','V2_true'} & set(entry['graph'])) for entry in cache.values())
    assert set(metadata.canonical_smiles) == set(cache)
    runs, outputs = [], []
    for seed in [42, 525, 1101]:
        began = time.time()
        split = load_formal_split(seed, metadata)
        split_path = ROOT/f'studies/predictor/final_4g_qualification/splits/row_seed_{seed}.csv'
        record(split_path)
        selected = split.source_role.isin(['train','validation']) & ~split.sample_id.isin(quarantine)
        ids = split.loc[selected, 'sample_id'].tolist()
        role_map = dict(zip(split.sample_id, split.source_role))
        frame = metadata.set_index('sample_id').loc[ids].reset_index()
        assert frame.sample_id.tolist() == ids
        assert not set(ids) & quarantine
        assert all(role_map[sid] in ['train','validation'] for sid in ids)
        assert not set(ids) & set(split.loc[split.source_role.eq('test'),'sample_id'])
        model, identity = load_formal(seed)
        assert identity['split_sha256'] == SOURCES[str(split_path.relative_to(ROOT))]['sha256']
        prediction = infer(model, frame, cache, identity['preprocessing']['scaler'])
        assert prediction.sample_id.tolist() == ids and prediction.sample_id.is_unique
        prediction.insert(1, 'source_role', prediction.sample_id.map(role_map))
        assert prediction.columns.tolist() == OUTPUT_COLUMNS
        assert prediction.source_role.isin(['train','validation']).all()
        assert np.isfinite(prediction.iloc[:,2:].to_numpy()).all()
        output = OUT/f'formal_predictions_seed_{seed}.csv'
        temp = output.with_suffix('.csv.tmp')
        prediction.to_csv(temp, index=False)
        temp.replace(output)
        output_sha = sha256(output.read_bytes()).hexdigest()
        outputs.append(dict(path=str(output.relative_to(ROOT)),sha256=output_sha,rows=len(prediction),columns=OUTPUT_COLUMNS))
        excluded = split.loc[split.source_role.isin(['train','validation']) & split.sample_id.isin(quarantine)]
        run = dict(row_seed=seed,checkpoint_path=identity['checkpoint_path'],checkpoint_sha256=identity['checkpoint_sha256'],
            checkpoint_schema_version=identity['checkpoint_schema_version'],official_loader=identity['official_loader'],
            input_schema_hash=identity['input_schema_hash'],preprocessing=identity['preprocessing'],normalization=identity['normalization'],
            source_split_sha256=identity['split_sha256'],original_role_counts=identity['role_counts'],
            inference_role_counts=prediction.source_role.value_counts().to_dict(),inference_rows=len(prediction),
            excluded_quarantine_ids=excluded.sample_id.tolist(),ordered_sample_ids_sha256=ordered_hash(ids),
            ids_unique=True,canonical_order_preserved=True,graph_y_sentinel=0,all_graph_targets_zero_asserted_by='common.infer',
            test_graph_rows_built=0,test_prediction_rows=0,model_forward_batches=int(np.ceil(len(ids)/128)),
            all_predictions_finite=True,output_unit='mL',seconds=time.time()-began)
        runs.append(run)
        print(json.dumps(dict(row_seed=seed,rows=len(prediction),role_counts=run['inference_role_counts'],
                              excluded_quarantine=len(excluded),seconds=run['seconds'])),flush=True)
        del model, prediction, frame
        gc.collect()
    for source in checkpoint_audit.SOURCES.values():
        record(ROOT/source['path'])
    # The complete inputs are stable, including the common graph builder.
    for source in list(SOURCES.values()):
        assert record(ROOT/source['path']) == source['sha256']
    manifest = dict(status='PASS',training_runs=0,llm_api_calls=0,new_label_reveals=0,
        target_csv_files_read_by_this_program=[],numerical_prediction_caches_read=[],label_arrays_read=[],
        quarantine_incident_path=str((OUT/'LABEL_BOUNDARY_INCIDENT.json').relative_to(ROOT)),quarantined_ids=sorted(quarantine),
        quarantine_policy='Exclude every quarantined ID in all model seeds before graph construction; source-test graphs never built.',
        metadata_source=str(metadata_path.relative_to(ROOT)),metadata_target_columns_read=[],
        graph_cache_path=str(cache_path.relative_to(ROOT)),graph_cache_sha256=cache_sha,
        graph_cache_schema='217 molecule-keyed records, each containing graph and descriptor only; no response keys',
        original_training_identity_unchanged=True,zero_label_sentinel=True,runs=runs,outputs=outputs,
        inputs=sorted(SOURCES.values(),key=lambda item:item['path']),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.time()-started)
    (OUT/'FORMAL_PREDICTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',models=3,total_prediction_rows=sum(r['inference_rows'] for r in runs),
                         input_files=len(SOURCES),elapsed_seconds=manifest['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
