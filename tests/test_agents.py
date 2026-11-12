from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from src.qgeognn_al.al.acquisitions import select
from src.qgeognn_al.al.acquisitions.lcmd import lcmd_tp_select
from src.qgeognn_al.al.acquisitions.maxdet import conditional_gradient_maxdet
from src.qgeognn_al.al.acquisitions.ivr import conditional_batch_ivr
from src.qgeognn_al.al.acquisitions.hybrid import hybrid_select
from src.qgeognn_al.al.acquisitions.random import random_control_positions, random_control_seed
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.infra.contracts import BaselineProvenance, canonical_bytes
from tests.support import historical_reference as legacy
from tests.support.fixtures import inputs, snapshot, record


@pytest.mark.historical_archive
@pytest.mark.parametrize('bad_case', ['duplicate', 'foreign', 'wrong_count', 'wrong_packet'])
def test_free_candidate_identity_contract_matches_historical_direct_delivery(bad_case):
    import copy
    import types
    from src.qgeognn_al.al.agents import make_packet, validate_bound_plan
    ids = tuple(f'U{i}' for i in range(64))
    rows = [{'sample_id': identity, 'rank': i, 'nearest_center_sq_distance': float(i)} for i, identity in enumerate(ids)]
    packet = make_packet(rows, candidate_ids=ids, seed=157, round_index=0, budget=32, protocol_hash='a'*64)
    response = {'packet_sha256': packet['packet_sha256'], 'selected_ids': list(ids[:32])}
    old = legacy.load('src/qgeognn_al/active_learning_v2/scientist_full_delivery.py', ['validate_direct'],
                      injected={'copy': copy, 'MAX_HYPOTHESES': 8,
                                'decode_table': lambda packet, table: [{'id': identity} for identity in ids]})
    catalog = types.SimpleNamespace(candidates={identity: {} for identity in ids}, observed={})
    old_packet = {'packet_sha256': packet['packet_sha256'], 'selection_count': 32, 'round': 0}
    old_response = {'type': 'selection', 'packet_sha256': packet['packet_sha256'],
                    'choices': [{'id': identity, 'reason': 'synthetic reason', 'hypothesis_id': None} for identity in ids[:32]],
                    'hypotheses': [], 'batch_strategy': 'synthetic', 'batch_rationale': 'synthetic', 'feedback_interpretation': ''}
    assert validate_bound_plan(response, packet) == ids[:32]
    assert [choice['id'] for choice in old.validate_direct(old_response, catalog, old_packet)['choices']] == list(ids[:32])
    if bad_case == 'duplicate':
        response['selected_ids'][-1] = response['selected_ids'][0]
        old_response['choices'][-1]['id'] = old_response['choices'][0]['id']
    elif bad_case == 'foreign':
        response['selected_ids'][-1] = 'test_ID'
        old_response['choices'][-1]['id'] = 'test_ID'
    elif bad_case == 'wrong_count':
        response['selected_ids'].pop(); old_response['choices'].pop()
    else:
        response['packet_sha256'] = 'b'*64; old_response['packet_sha256'] = 'b'*64
    with pytest.raises(ValueError): validate_bound_plan(response, packet)
    with pytest.raises(ValueError): old.validate_direct(old_response, catalog, old_packet)



def test_provided_packet_has_no_truth_channel_and_cannot_cross_seed_or_round():
    from src.qgeognn_al.al.agents import make_packet, validate_bound_plan
    rows = [{'sample_id': 'U0', 'rank': 0}, {'sample_id': 'U1', 'rank': 1}]
    kwargs = dict(candidate_ids=('U0', 'U1'), seed=157, round_index=0, budget=1, protocol_hash='a'*64)
    packet = make_packet(rows, **kwargs)
    assert packet['candidate_labels_available'] is False and packet['provider_called'] is False
    for field in ['true_V1_ml', 'true_V2_ml', 'y', 'feedback', 'test_truth']:
        bad = [dict(rows[0], **{field: 999}), rows[1]]
        with pytest.raises(ValueError, match='only approved label-free'):
            make_packet(bad, **kwargs)
    response = {'packet_sha256': packet['packet_sha256'], 'selected_ids': ['U0']}
    for field, value in [('seed', 887), ('round_index', 1)]:
        modified = dict(kwargs); modified[field] = value
        with pytest.raises(ValueError, match='binding mismatch'):
            validate_bound_plan(response, make_packet(rows, **modified))
    registry = json.loads((Path(__file__).resolve().parents[1]/'configs/method_registry.json').read_text())
    assert registry['provider_execution_supported'] is True
    assert registry['provider_authorization_required'] is True and registry['full_ready'] is False
    assert all(p['runner_adapter_implemented'] and p['trajectory_equivalence']=='NOT_VERIFIED' for p in registry['agent_profiles'].values())
    assert all(profile['new_runner_adapter'] is False and profile['status'] == 'HISTORICAL_FROZEN'
               for profile in registry['archived_agent_protocols'].values())
