"""Full-budget entry point; no target-reading capability in acquisition callback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/epig_v2_matplotlib')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exploratory_runner as runner
from acquisition_v2 import IndependentStreamingCache, select_streaming_batch, batch_target_information
import numpy as np
from qgeognn_al.active_learning_v2.epig_models import GaussianProcessSurrogate
from qgeognn_al.active_learning_v2.protocol import ids_hash


def formal_callback(seed, arm, round_index, labeled_indices, unlabeled_indices, labeled_truth,
                    feature_matrix_X, feature_frame, target_scales, previous_checkpoint_path=None):
    """Only X_outer and authorized current-L labels enter this probability fit."""
    if int(seed) not in runner.SEEDS or arm not in runner.ARMS or not 0 <= round_index < 21:
        raise ValueError('unregistered seed, arm or round')
    if previous_checkpoint_path is not None:
        raise ValueError('primary surrogate does not use a supervised encoder')
    labeled = np.asarray(labeled_indices, dtype=int)
    unlabeled = np.asarray(unlabeled_indices, dtype=int)
    outer = np.r_[labeled, unlabeled]
    if len(outer) != 3330 or len(np.unique(outer)) != 3330 or len(labeled) != 333 + 32*round_index:
        raise ValueError('L/U identity or row-budget drift')
    x = np.asarray(feature_matrix_X, dtype=np.float64)
    truth = np.asarray(labeled_truth, dtype=np.float64)
    if x.shape != (4163, 29) or truth.shape != (len(labeled), 2) or not np.isfinite(x[outer]).all() or not np.isfinite(truth).all():
        raise ValueError('invalid primary features or current training labels')
    forbidden = np.setdiff1d(np.arange(len(x)), outer)
    if not np.isnan(x[forbidden]).all() or not feature_frame.iloc[forbidden].isna().all().all():
        raise PermissionError('acquisition capability must mask validation/test inputs')
    scales = np.asarray([target_scales[t] for t in ('V1', 'V2')], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('positive fixed L333 endpoint scales required')
    ids = feature_frame.iloc[outer].sample_id.to_numpy(str)
    if len(set(ids.tolist())) != 3330:
        raise ValueError('duplicate outer row ID')
    started = time.perf_counter()
    model = GaussianProcessSurrogate(kernel='matern32', max_components=16, maxiter=60, restarts=1)
    model.fit(x[labeled], truth/scales)
    posterior = model.posterior(x[outer], row_ids=ids)
    candidates = np.arange(len(labeled), len(outer), dtype=int)
    references = np.arange(len(outer), dtype=int)
    objective = 'epig' if arm == runner.ARMS[0] else 'ivr'
    # Every attempt is retained independently, including one interrupted before
    # selection persistence. No timing-dependent artifact is reused by accident.
    directory = HERE/f'runtime/seed_{seed}/{arm}/round_{round_index:02d}/auxiliary_attempts'/str(time.time_ns())
    directory.mkdir(parents=True)
    with (directory/'model.pkl').open('wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    runner.support.dump(directory/'fit_audit.json', {
        **model.audit_, 'fixed_L333_scales_ml':scales.tolist(),
        'acquisition_units':'fixed L333 endpoint standard deviations',
        'outer_ids_hash':ids_hash(ids), 'L_ids_hash':ids_hash(ids[:len(labeled)]),
        'U_ids_hash':ids_hash(ids[len(labeled):]),
        'ordered_outer_features_sha256':hashlib.sha256(x[outer].tobytes()).hexdigest(),
        'authorized_training_truth_float64_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),
        'validation_labels_used':0,'future_U_labels_used':0,'test_labels_used':0,
        'original_gate_satisfied_for_all_seeds':False})
    with tempfile.TemporaryDirectory(prefix='covariance_', dir=directory) as temp:
        cache = IndependentStreamingCache(posterior, candidates, temp, target_chunk=64, candidate_chunk=256)
        result = select_streaming_batch(cache, references, objective=objective, batch_size=32)
        selected = np.asarray(result['selected_positions'], dtype=int)
        information = batch_target_information(posterior, references, selected)
        if objective == 'epig' and not np.isclose(information, sum(t['conditional_score'] for t in result['trace']), rtol=1e-7, atol=1e-10):
            raise RuntimeError('selected batch MI differs from chain-rule gains')
        result['audit'] = {'reference_rows':3330,'target_distribution':'P1_full',
            'target_semantics':'independent_future_noisy_replicate',
            'batch_target_information_nats':information,'disk_covariance_bytes':cache.bytes,
            'max_cross_request_shape':cache.max_covariance_shape,'cache_build_seconds':cache.build_seconds,
            'same_posterior_rule':'F0_Matern32_PCA16_60iter_1start',
            'total_auxiliary_seconds':time.perf_counter()-started}
        del cache
    runner.support.dump(directory/'acquisition.json', result)
    receipt = {'model_path':str((directory/'model.pkl').relative_to(runner.ROOT)),
        'files':{name:runner.support.digest(directory/name) for name in ('model.pkl','fit_audit.json','acquisition.json')},
        'posterior_model_audit':model.audit_,'original_gate_satisfied_for_all_seeds':False}
    runner.support.write_once(directory/'complete.json', receipt)
    return {'selected_ids':ids[selected].tolist(), 'diagnostics':result,
        'posterior':receipt,'units':'fixed L333 endpoint standard deviations'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze','verify','run'])
    args = parser.parse_args()
    if args.phase == 'freeze':
        result = runner.freeze_study()
    elif args.phase == 'verify':
        result = runner.verify_freeze()
    else:
        result = runner.run_exploratory(formal_callback)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
