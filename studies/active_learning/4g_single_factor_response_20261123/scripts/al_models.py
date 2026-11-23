"""Replay frozen predictions and historical label receipts without fitting."""
import itertools
import json
import lzma
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from run import (ROOT, AL, HIST, INPUTS, SEEDS, BUDGETS, KEYS, design, enrich, join_predictions, load,
                 project_truth, stable, table, track, verified_feedback)

METHOD = 'new_free_cw_sol61_xhigh'
STUDY = AL / 'qgeognn_v2_row_sequential_b32'


def states():
    path = AL / 'llm_decision_behavior_audit_20261116/states.json.xz'
    snapshot = json.loads(lzma.decompress(track(path).read_bytes()))
    return [state for state in snapshot['states'] if state['method'] == METHOD], snapshot['metadata']


def require_validation_ids(partition, labeled, unlabeled):
    ids = partition.loc[partition.role.eq('validation'), 'sample_id']
    if ids.empty or not ids.is_unique or set(ids) & (set(labeled) | set(unlabeled)):
        raise ValueError('validation role is missing, duplicated, trained on, or in the pool')
    return ids


def al_models(meta):
    import torch

    sys.path.insert(0, str(ROOT / 'src'))
    from qgeognn_al.evaluation.point import point_metrics
    from qgeognn_al.models import load_predictor_checkpoint
    from qgeognn_al.training.predictor import predict
    from qgeognn_al.data import qg

    torch.set_num_threads(2)
    qg.device = torch.device('cpu')
    records, _ = states()
    curves = pd.read_csv(track(AL / 'llm_decision_behavior_audit_20261116/row_curves.csv'))
    curves = curves[curves.method.eq(METHOD)].set_index(['seed','budget']).nrmse
    outputs, metrics, coverage = [], [], []
    for seed in SEEDS:
        selection = [state for state in records if state['seed'] == seed and state['L_before'] in BUDGETS]
        assert sorted(state['L_before'] for state in selection) == list(BUDGETS)
        partition = pd.read_csv(track(STUDY / f'splits/row_seed_{seed}.csv'))
        graph_path = STUDY / f'runtime/seed_{seed}/scrubbed_graphs.pt'
        contract = load(graph_path.with_suffix('.pt.contract.json'))
        track(graph_path)
        assert INPUTS[str(graph_path)] == contract['sha256']
        assert stable(partition.to_dict('list')) == contract['contract']['partition_hash']
        assert partition.canonical_index.tolist() == list(range(len(partition)))
        assert partition.sample_id.tolist() == meta.sample_id.tolist()
        atom, angle = torch.load(ROOT / graph_path, map_location='cpu', weights_only=False)
        assert all(torch.count_nonzero(row.y) == 0 for row in atom)
        validation = require_validation_ids(partition, selection[-1]['labeled_ids'], selection[-1]['unlabeled_ids'])
        assert stable(sorted(validation)) == contract['contract']['validation_ids_hash']
        truth = enrich(project_truth(validation))
        positions = partition.loc[partition.role.eq('validation'), 'canonical_index'].to_numpy()
        for state in selection:
            budget = state['L_before']
            require_validation_ids(partition, state['labeled_ids'], state['unlabeled_ids'])
            freeze = Path(state['source'])
            assert load(freeze)['checkpoint_sha256'] == state['checkpoint_sha256']
            assert INPUTS[str(freeze)] == state['source_sha256']
            receipt = load(Path(str(freeze.parent).replace('/selections/', '/runtime/')) / 'prediction_freeze.json')
            assert receipt['checkpoint_sha256'] == state['checkpoint_sha256']
            assert len(state['labeled_ids']) == budget and receipt['L_t_ids_hash'] == stable(sorted(state['labeled_ids']))
            checkpoint = Path(receipt['checkpoint_path']).relative_to(ROOT)
            track(checkpoint)
            assert INPUTS[str(checkpoint)] == state['checkpoint_sha256']
            payload = torch.load(ROOT/checkpoint, map_location='cpu', weights_only=False)
            assert payload['preprocessing']['scaler'] == contract['contract']['preprocessing']['scaler']
            scaler = contract['contract']['preprocessing']['target_scales']
            network = load_predictor_checkpoint(ROOT/checkpoint)
            masked, predictions, indices = predict(network, atom, angle, positions)
            np.testing.assert_array_equal(indices, positions)
            assert not np.any(masked)
            pred = enrich(pd.DataFrame(predictions[:,[1,4]].astype(float), index=validation, columns=['V1','V2']))
            pred['checkpoint'] = state['checkpoint_sha256']
            frame = meta.join(join_predictions(truth,pred), on='sample_id', how='inner')
            scope = dict(cohort='al_new_free',seed=seed,budget=budget,role='validation_matched')
            numbers = point_metrics(truth[['V1','V2']].to_numpy(), predictions.astype(float), scaler)
            score = numbers['combined_normalized_rmse']
            original = float(curves.loc[(seed,budget)])
            np.testing.assert_allclose(score, original, rtol=1e-4, atol=2e-5)
            coverage.append({**scope,'checkpoint':state['checkpoint_sha256'],'checkpoint_path':str(checkpoint),
                             'validation_rows':len(validation),'validation_ids_hash':stable(sorted(validation)),
                             'freeze':str(freeze),'historical_nrmse':original,'replay_nrmse':score})
            metrics.append({**scope, **numbers})
            outputs.append(design(frame,scope))
            print(f'AL frozen validation replay: seed {seed}, L{budget}',flush=True)
        del atom, angle
    table('al_state_coverage',coverage)
    return outputs, metrics


def historical(meta):
    records, old_meta = states()
    old_path = AL / 'nocw_free_selection_diagnostics_20261101'
    sys.path.insert(0, str(ROOT / old_path))
    track(old_path / 'condition_metrics.py')
    from condition_metrics import _canonical_conditions

    outputs, legacy = [], []
    for state in records:
        path = Path(state['source'])
        freeze = load(path)
        assert INPUTS[str(path)] == state['source_sha256']
        assert freeze['checkpoint_sha256'] == state['checkpoint_sha256']
        assert freeze['batch_ids'] == state['selected_ids']
        feedback = verified_feedback(state, freeze, load(path.parent/'feedback.json'), load(path.parent/'label_access_receipt.json'))
        truth = enrich(feedback[[f'true_{name}_ml' for name in ('V1','V2')]].astype(float).set_axis(['V1','V2'],axis=1))
        pred = enrich(feedback[[f'pred_{name}_ml' for name in ('V1','V2')]].astype(float).set_axis(['V1','V2'],axis=1))
        pred['checkpoint'] = state['checkpoint_sha256']
        frame = meta.join(join_predictions(truth,pred), on='sample_id', how='inner')
        assert len(frame) == len(feedback) and set(frame.sample_id) == set(feedback.index)
        budget = state['L_before']
        phase = '333-525' if budget < 525 else '525-813' if budget < 813 else '813-1005' if budget < 1005 else '1005-1485'
        scope = dict(cohort='al_accepted_feedback',seed=state['seed'],budget=budget,role='both_untrained_then_revealed', phase=phase,round=state['round'])
        outputs.append(design(frame,scope))
        volume_index = 2
        for left, right in itertools.combinations(state['selected_ids'],2):
            left_meta, right_meta = old_meta[left], old_meta[right]
            if left_meta['smiles'] != right_meta['smiles']:
                continue
            left_cond, right_cond = map(_canonical_conditions, (left_meta,right_meta))
            if left_cond is None or right_cond is None or [idx for idx,(a,b) in enumerate(zip(left_cond,right_cond)) if a != b] != [volume_index]:
                continue
            if left_cond[volume_index] > right_cond[volume_index]:
                left,right = right,left
            for endpoint in ('V1','V2'):
                legacy.append({**scope,'target':endpoint,'low_id':left,'high_id':right,'checkpoint':state['checkpoint_sha256'],
                               'actual':truth.loc[right,endpoint]-truth.loc[left,endpoint], 'predicted':pred.loc[right,endpoint]-pred.loc[left,endpoint]})
    legacy = pd.DataFrame(legacy)
    assert len(legacy) == 330 and len(records) == 180
    table('legacy_165_pairs',legacy)
    numbers = legacy.assign(actual_abs_delta_ml=legacy.actual.abs(),pred_abs_delta_ml=legacy.predicted.abs())
    result = numbers.groupby(['seed','phase','target'])[['actual_abs_delta_ml','pred_abs_delta_ml']].mean().reset_index()
    original = pd.read_csv(track(AL / 'comprehensive_history_diagnostics_20261112/tables/model_same_batch_single_factor_response.csv'))
    comparison = result.merge(original[original.axis.eq('V/ul')],on=['seed','phase','target'],validate='one_to_one',suffixes=('_replay','_original'))
    assert len(comparison) == len(result)
    for metric in ('actual_abs_delta_ml','pred_abs_delta_ml'):
        np.testing.assert_allclose(comparison[f'{metric}_replay'],comparison[f'{metric}_original'],atol=1e-10,rtol=0)
    table('legacy_seed_phase_check',comparison)
    table('legacy_seed_weighting',pd.concat([numbers.groupby('target')[['actual_abs_delta_ml','pred_abs_delta_ml']].mean().assign(weight='pair'),
          numbers.groupby(['seed','target'])[['actual_abs_delta_ml','pred_abs_delta_ml']].mean().groupby('target').mean().assign(weight='equal_seed')]).reset_index())
    return outputs
