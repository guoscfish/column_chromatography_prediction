"""Feature-only checkpoint replay: never reads observed endpoint or time cells."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1 as r


def verify():
    torch.set_num_threads(1)
    checkpoint = r.old.source_path()
    scaler = torch.load(checkpoint, map_location='cpu', weights_only=False)['preprocessing']['scaler']
    source_cache = torch.load(r.old.SOURCE_GRAPH_CACHE, map_location='cpu', weights_only=False)
    schedule = r.row_schedule()
    data = {}
    for c in r.TARGETS:
        allowed = schedule.loc[schedule.column.eq(c) & schedule.outer_seed.eq(r.PILOT_SEED) & schedule.role.eq('gradient_train'), 'sample_id']
        frame = pd.read_csv(r.old.PARENT/f'filtered_features_{c}.csv', usecols=r.old.FEATURES)
        frame = frame.loc[frame.sample_id.isin(allowed)].reset_index(drop=True)
        frame['V1_ml'], frame['V2_ml'] = 0., 0.
        cache = dict(source_cache)
        cache.update(torch.load(r.ROOT/f'studies/transfer/cross_column/data_audit/graph_cache_{c}_only.pt', map_location='cpu', weights_only=False))
        atoms, angles = r.build_model_data(frame, cache, None, scaler)
        assert all(torch.count_nonzero(a.y) == 0 for a in atoms)
        r.old.cache_frozen_prefix(r.load_predictor_checkpoint(checkpoint), atoms, angles)
        data[c] = dict(frame=frame, atoms=atoms, angles=angles, lookup={sid:i for i,sid in enumerate(frame.sample_id)})
    rows = []
    for fold in range(1, 6):
        for arm in r.old.ARMS:
            run = r.STUDY/f'runtime/inner/seed_{r.PILOT_SEED}/fold_{fold}/{arm}'
            saved = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            assert saved['protocol_sha256'] == r.old.sha(r.STUDY/'protocol.json')
            model = r.old.build_multitask_model(checkpoint, r.old.ARMS[arm])
            model.load_state_dict(saved['model_state_dict'])
            frozen_pred = pd.read_csv(run/'predictions.csv')
            for c in r.TARGETS:
                frame = frozen_pred.loc[frozen_pred.column.eq(c)]
                positions = [data[c]['lookup'][sid] for sid in frame.sample_id]
                actual = r.old.predict(model, data, c, positions)
                expected = frame[r.old.QUANTILES].to_numpy()
                difference = float(np.max(np.abs(actual-expected)))
                if not np.allclose(actual, expected, rtol=1e-6, atol=1e-5):
                    raise RuntimeError(f'checkpoint replay mismatch: fold {fold} {arm} {c}: {difference}')
                rows.append(dict(inner_fold=fold, arm=arm, column=c, rows=len(frame),
                                 max_abs_difference_ml=difference, replay_passed=True))
    pd.DataFrame(rows).to_csv(r.STUDY/'CHECKPOINT_REPLAY.csv', index=False)
    result = dict(contexts_checked=len(rows), target_prediction_rows=sum(row['rows'] for row in rows),
                  max_abs_difference_ml=max(row['max_abs_difference_ml'] for row in rows),
                  label_cells_parsed=0, all_passed=True)
    r.save_json('CHECKPOINT_REPLAY.json', result)
    return result


if __name__ == '__main__':
    print(verify())
