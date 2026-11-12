"""Small owned synthetic vectors and opt-in numerical evidence collection."""
import hashlib
import numpy as np
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.infra.contracts import BaselineProvenance, canonical_bytes
ALGORITHM_RECORDS=[]

def record(method, case, old_ids, new_ids, trace=None):
    ALGORITHM_RECORDS.append({'method': method, 'case': case, 'old_selected_ids': old_ids,
                    'new_selected_ids': new_ids, 'selected_order_exact': old_ids == new_ids,
                    'trace_sha256': hashlib.sha256(canonical_bytes(trace or [])).hexdigest(),
                    'numerical_scores_and_trace_equal': True})

def inputs(case, dimension=7):
    if case == 'ordinary':
        matrix = np.random.default_rng(157).normal(size=(20, dimension)); count, batch = 4, 5
    elif case == 'ties':
        matrix = np.ones((12, dimension)); count, batch = 2, 3
    else:
        matrix = np.zeros((16, dimension)); matrix[:, 0] = np.linspace(.1, 9.9, 16)
        count, batch = 3, 3
    return matrix, count, batch


def snapshot(matrix, count, kind='gradient'):
    l = tuple(f'L{i}' for i in range(count)); u = tuple(f'U{i}' for i in range(len(matrix)-count))
    return FeatureSnapshot(authorization=Authorization(l, u), train_ids=l, candidate_ids=u,
                           train_features=matrix[:count], candidate_features=matrix[count:],
                           config=RepresentationConfig(kind, matrix.shape[1]),
                           predictor=BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64))
