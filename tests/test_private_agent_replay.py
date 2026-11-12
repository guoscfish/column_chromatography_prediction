"""External frozen source-visible snapshots; no automatic live-main fallback."""
import json
import os
from pathlib import Path
import pytest
from src.qgeognn_al.al.agents.recorded import replay_fixture, _private_case
from src.qgeognn_al.al.agents.replay_packets import rebuild_recorded_packet

pytestmark=pytest.mark.frozen_regression


@pytest.fixture(scope='module')
def private_manifests():
    configured=os.environ.get('QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST')
    if not configured:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: set QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST to independent readonly snapshot manifest(s); never borrow live main')
    paths=[Path(name) for name in configured.split(os.pathsep) if name]
    if not paths:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: no external private replay manifest supplied')
    for path in paths:
        if not path.is_file():
            pytest.skip('BLOCKED_EXTERNAL_NOT_FOUND: explicitly configured private snapshot manifest unavailable')
        value=json.loads(path.read_text())
        assert value['schema_version']==2 and value['source_kind']=='private_snapshot'
    return paths


@pytest.mark.parametrize('profile',['free32_cwaware','free32_nocw',
    'cw16_llm16_scientist_v2','free16_cwgradient_lcmd16'])
def test_private_recorded_profiles_exact_selection_and_correction(private_manifests,profile):
    results=[replay_fixture(path,profile) for path in private_manifests]
    included=[r for r in results if r['cases_replayed']]
    if not included:
        pytest.skip('BLOCKED_PRIVATE_PROFILE_NOT_SUPPLIED: this explicit manifest set contains no matching profile; not a claim of physical absence')
    for result in included:
        assert result['status']=='EXACT_REPLAY_PASS' and result['source_kind']=='private_snapshot'
        assert result['provider_calls']==result['new_real_labels']==result['new_fits']==0
        assert all(r['status']=='EXACT_REPLAY_PASS' and r['id_diff_count']==0 for r in result['results'])
        assert all(r['messages']=='EXACT_MESSAGES_CONFIG_REQUEST_HASH_MATCH' for r in result['results'])
        assert result['fit_reveal_metrics']=='NOT_VERIFIED_IN_SELECTION_REPLAY'


def test_private_full_pool_packet_reconstruction(private_manifests):
    seen=[]
    for path in private_manifests:
        manifest=json.loads(path.read_text())
        for item in manifest['cases']:
            result=rebuild_recorded_packet(_private_case(item,path.parent))
            assert result['status']=='EXACT_REPLAY_PASS'
            assert result['id_order_and_numeric_values_exact'] and result['message_serialization_exact']
            seen.append((item['profile'],item['seed'],item['round']))
    assert seen and all(round_index in (0,1) for _,_,round_index in seen)
