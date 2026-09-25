#!/usr/bin/env python3
"""Verify all frozen selections without opening observed outcomes or metrics."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.llm_screen import (
    STUDY, METHODS, SEEDS, PROMPT, read, validate, stable_hash, sha256_file, atomic_json,
)


def main():
    validate()
    protocol_hash = stable_hash(read(STUDY / 'protocol.json'))
    rows = []
    card_keys = {'id', 'smiles', 'conditions', 'descriptors', 'pred_V1_V2',
                 'cw_distance_percentile', 'max_Tanimoto_to_Lt', 'same_molecule_rows_in_Lt'}
    for seed in SEEDS:
        for method in METHODS:
            runtime = STUDY / 'runtime' / f'seed_{seed}' / method
            original = read(runtime / 'round_00/state.json')
            selected = []
            for r in range(3):
                state = read(runtime / f'round_{r:02d}/state.json')
                assert state['labeled'] == original['labeled'] + selected
                assert state['unlabeled'] == [i for i in original['unlabeled'] if i not in set(selected)]
                current = state['labeled'] + state['unlabeled']
                path = STUDY / 'selections' / f'seed_{seed}' / method / f'round_{r:02d}'
                packet, proposal, batch, response = [read(path / name) for name in (
                    'packet.json', 'proposal.json', 'batch_freeze.json', 'validated_response.json')]
                prediction = read(runtime / f'round_{r:02d}/prediction_freeze.json')
                contract = {'protocol': protocol_hash, 'checkpoint': prediction['checkpoint_sha256'],
                            'labeled': state['labeled'], 'unlabeled': state['unlabeled']}
                assert batch['input_hash'] == proposal['input_hash'] == stable_hash(contract)
                for name, digest in batch['artifact_hashes'].items():
                    assert sha256_file(path / name) == digest
                assert packet['prompt'] == PROMPT
                assert packet['labeled_count'] == 333 + 32*r
                assert packet['observed_outcome_values_in_packet'] == 0
                assert packet['validation_or_test_records_in_packet'] == 0
                for section, key, count in (('pending_CW16', 'pending_positions', 16),
                                             ('candidates', 'candidate_positions', 128)):
                    cards = packet[section]
                    assert len(cards) == count and all(set(c) == card_keys for c in cards)
                    assert [c['id'] for c in cards] == [current[p] for p in proposal[key]]
                pending = [c['id'] for c in packet['pending_CW16']]
                candidates = [c['id'] for c in packet['candidates']]
                assert len(set(pending + candidates)) == 144
                assert set(pending + candidates) <= set(state['unlabeled'])
                supplement = [c['id'] for c in response['choices']]
                assert len(set(supplement)) == 16 and set(supplement) <= set(candidates)
                assert batch['batch_ids'] == pending + supplement
                if method == 'cw16_random16':
                    assert supplement == [current[p] for p in proposal['random16_positions']]
                else:
                    assert response['packet_sha256'] == sha256_file(path / 'packet.json')
                selected += batch['batch_ids']
                assert len(selected) == len(set(selected)) == 32*(r+1)
                rows.append({'seed': seed, 'method': method, 'round': r,
                             'valid_batch_size': 32, 'eligible_candidate_count': 128,
                             'response_failed': response['response_failed'],
                             'fallback_count': response['fallback_count']})
            final = read(runtime / 'round_03/state.json')
            assert final['labeled'] == original['labeled'] + selected
            assert len(final['labeled']) == 429
        first = STUDY / 'selections' / f'seed_{seed}'
        assert (first / 'cw16_llm16/round_00/packet.json').read_bytes() == (
            first / 'cw16_random16/round_00/packet.json').read_bytes()
    result = {'status': 'PASSED', 'protocol_hash': protocol_hash,
              'frozen_batches_checked': len(rows), 'observed_outcomes_read_by_audit': 0,
              'round0_packets_identical_across_supplement_arms': True, 'batches': rows}
    atomic_json(STUDY / 'selection_integrity_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
