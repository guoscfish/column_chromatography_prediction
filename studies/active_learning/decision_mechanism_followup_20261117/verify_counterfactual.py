#!/usr/bin/env python3
"""Independent scalar metadata checks for second-half counterfactual tables."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
from itertools import combinations
import json
import lzma
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116'


def main():
    snapshot = json.loads(lzma.decompress((ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz').read_bytes()))
    originals = json.loads((PRIOR / 'hybrid_batches.json').read_text())
    originals = {(b['seed'], b['round']): b for b in originals}
    states = json.loads((OUT / 'counterfactual_selected_ids.json').read_text())
    meta = snapshot['metadata']

    def factors(sid):
        value = meta[sid]
        c = value['conditions']
        return (value['smiles'], Fraction(c['PE/EA']), Decimal(str(c['Density g/ml'])),
                Decimal(str(c['V/ul'])), c['loading solvent'], Decimal(str(c['Volume of loading solvent/ul'])))

    f = {sid: factors(sid) for sid in meta}

    def sample_pair(a, b):
        return f[a][0] == f[b][0] and [j for j in range(1, 6) if f[a][j] != f[b][j]] == [3]

    rows = []
    for state in states:
        assert state['included']
        old = originals[(state['seed'], state['round'])]
        h, n = old['labeled_ids'], old['numerical_ids']
        assert state['reconstructed_first16'] == n
        for group, selected in [('llm16', state['actual_llm_ids']), ('lcmd_back16', state['reconstructed_back16'])]:
            assert len(set(selected)) == 16 and not set(selected) & set(h + n)
            combined = n + selected
            pairs = [(a, b) for a, b in combinations(combined, 2) if sample_pair(a, b)]
            participants = {a for a, b in pairs} | {b for a, b in pairs}
            rows.append({'seed': state['seed'], 'round': state['round'], 'group': group,
                         'sample_only_H_fraction': sum(any(sample_pair(a, b) for b in h) for a in selected) / 16,
                         'sample_only_HN_fraction': sum(any(sample_pair(a, b) for b in h + n) for a in selected) / 16,
                         'N_plus_group_sample_pairs': len(pairs),
                         'N_plus_group_sample_selected_fraction': len(participants) / 32})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'counterfactual_independent_metadata.csv', index=False)
    batches = pd.read_csv(OUT / 'counterfactual_batches.csv').set_index(['seed', 'round'])
    largest_error = 0
    for row in d.itertuples():
        b = batches.loc[(row.seed, row.round)]
        pairs = [(row.sample_only_H_fraction, b[row.group + '_single_sample_volume_history_fraction']),
                 (row.sample_only_HN_fraction, b[row.group + '_single_sample_volume_hn_fraction']),
                 (row.N_plus_group_sample_pairs, b[row.group + '_full32_single_sample_volume_pairs']),
                 (row.N_plus_group_sample_selected_fraction, b[row.group + '_full32_single_sample_volume_within_fraction'])]
        largest_error = max(largest_error, max(abs(a - b) for a, b in pairs))
    assert largest_error < 1e-12
    trace = pd.read_csv(OUT / 'counterfactual_lcmd_trace.csv')
    assert len(trace) == 71 * 32
    assert trace.groupby(['seed', 'round']).selection_order.apply(lambda x: x.tolist() == list(range(32))).all()
    candidates = pd.read_csv(OUT / 'counterfactual_candidates.csv')
    assert len(candidates) == 71 * 32
    assert not candidates.true_labels_read.any()
    result = {'status': 'PASS_INDEPENDENT_METADATA_AND_IDENTITIES', 'states': len(states),
              'comparison_groups': len(d), 'scalar_metadata_max_error': largest_error,
              'lc_first16_ordered_equal': True, 'trace_rows': len(trace),
              'counterfactual_candidate_truth_read': False}
    (OUT / 'counterfactual_independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
