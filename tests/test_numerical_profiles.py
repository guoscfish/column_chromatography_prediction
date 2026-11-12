"""Scientific profile identity and same-round ensemble supplier contracts."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.qgeognn_al.al.numerical_profiles import (
    NUMERICAL_PROFILES, numerical_profile, select_numerical_profile,
)
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.al.representations.ensemble import (
    FrozenEnsembleMember, K3EnsembleSnapshot, collect_checkpoint_ensemble, member_seeds,
)
from src.qgeognn_al.infra.contracts import BaselineProvenance, canonical_bytes, content_hash
from src.qgeognn_al.infra.resources import SnapshotFile


def _snapshot(kind='latent', dimension=128, *, round_index=0, candidate_ids=None, checkpoint='4'*64):
    train = tuple('L'+str(i) for i in range(333+32*round_index))
    candidates = candidate_ids or tuple('U'+str(i) for i in range(160))
    matrix = np.zeros((len(train)+len(candidates), dimension))
    matrix[len(train):, 0] = np.arange(1, len(candidates)+1)
    return FeatureSnapshot(authorization=Authorization(train, candidates), train_ids=train, candidate_ids=candidates,
        train_features=matrix[:len(train)], candidate_features=matrix[len(train):],
        config=RepresentationConfig(kind, dimension, canonical_bytes({'sketch_seed':157+4_000_037}).decode()),
        predictor=BaselineProvenance('1'*40,'2'*64,'3'*64,checkpoint,'5'*64))


def _members(tmp_path, snapshot, outer_seed=157):
    result=[]
    for member in range(3):
        file=tmp_path/f'member_{member}.json';file.write_text(json.dumps({'member':member}))
        digest=hashlib.sha256(file.read_bytes()).hexdigest();file.chmod(0o444)
        init, train=member_seeds(outer_seed,member)
        result.append(FrozenEnsembleMember(member=member,outer_seed=outer_seed,
            resource=SnapshotFile(file,digest,tmp_path),checkpoint_state_hash=hashlib.sha256(str(member).encode()).hexdigest(),
            train_ids=snapshot.train_ids,candidate_ids=snapshot.candidate_ids,dataset_hash='2'*64,
            split_hash='3'*64,protocol_hash='5'*64,shared_fit_config_hash='6'*64,
            initialization_seed=init,training_seed=train,source_kind='synthetic'))
    return tuple(result)


@pytest.mark.parametrize('seed',[157,887,2357,6101,12203])
def test_random32_three_rounds_are_nested_slices_of_one_u0_permutation(seed):
    initial=tuple('U'+str(i) for i in range(96));current=initial
    expected=np.random.default_rng(seed*1_000_003+900_001).permutation(len(initial))
    for r in range(3):
        snapshot=_snapshot(candidate_ids=current,round_index=r)
        actual=select_numerical_profile(snapshot,profile='random32',outer_seed=seed,round_index=r,
                                        original_candidate_ids=initial)
        assert actual.selected_ids==tuple(initial[i] for i in expected[32*r:32*(r+1)])
        assert actual.seed==seed*1_000_003+900_001
        current=tuple(identity for identity in current if identity not in actual.selected_ids)


def test_profiles_preserve_independent_scientific_identities_and_reject_unknown_names():
    assert set(NUMERICAL_PROFILES)=={'random32','gradient_lcmd32','cw_gradient_lcmd32',
        'gradient_maxdet32','kernel_ivr32','numerical_hybrid32'}
    assert numerical_profile('cw_gradient_lcmd32').representation=='center_width'
    assert numerical_profile('numerical_hybrid32').acquisition=='hybrid'
    with pytest.raises(ValueError,match='unknown'):numerical_profile('free')


def test_profile_rejects_wrong_round_geometry_mapping_and_unbound_ivr_reference():
    with pytest.raises(ValueError,match='L size'):
        select_numerical_profile(_snapshot(),profile='random32',outer_seed=157,round_index=1)
    with pytest.raises(ValueError,match='representation'):
        select_numerical_profile(_snapshot(),profile='cw_gradient_lcmd32',outer_seed=157,round_index=0)
    with pytest.raises(ValueError,match='CountSketch'):
        select_numerical_profile(_snapshot('gradient',512),profile='gradient_lcmd32',outer_seed=887,round_index=0)
    with pytest.raises(ValueError,match='reference'):
        select_numerical_profile(_snapshot('gradient',512),profile='kernel_ivr32',outer_seed=157,round_index=0)


def test_member_seeds_match_frozen_initialization_and_training_streams():
    assert [member_seeds(157,i) for i in range(3)]==[
        (2_000_160,3_000_174),(3_000_193,4_000_207),(4_000_226,5_000_240)]
    with pytest.raises(ValueError):member_seeds(157,3)
    with pytest.raises(ValueError):member_seeds(True,0)


@pytest.mark.parametrize('field,value',[
    ('outer_seed',887),('train_ids',('DIFFERENT',)),('candidate_ids',('DIFFERENT',)),
    ('split_hash','a'*64),('shared_fit_config_hash','b'*64),('source_kind','authorized_snapshot')])
def test_k3_rejects_cross_seed_budget_order_split_and_training_protocol(tmp_path,field,value):
    snapshot=_snapshot();members=list(_members(tmp_path,snapshot))
    changes={field:value}
    if field=='outer_seed':
        changes.update(zip(('initialization_seed','training_seed'),member_seeds(value,1)))
    members[1]=replace(members[1],**changes)
    with pytest.raises(ValueError,match='same outer seed'):
        K3EnsembleSnapshot(members,np.zeros((3,len(snapshot.candidate_ids),6)))


def test_k3_rejects_one_checkpoint_reused_three_times(tmp_path):
    snapshot=_snapshot();members=list(_members(tmp_path,snapshot))
    members[1]=replace(members[1],resource=members[0].resource)
    with pytest.raises(ValueError,match='distinct'):
        K3EnsembleSnapshot(members,np.zeros((3,len(snapshot.candidate_ids),6)))


def test_checkpoint_supplier_does_real_inference_in_member_order_and_binds_latent_identity(tmp_path):
    snapshot=_snapshot();members=_members(tmp_path,snapshot);calls=[]
    def load(path):return json.loads(path.read_text())
    def state(model):return hashlib.sha256(str(model['member']).encode()).hexdigest()
    def predict(model,ids):
        calls.append((model['member'],ids));return np.full((len(ids),6),float(model['member'])),ids
    result=collect_checkpoint_ensemble(members,load_predictor=load,predict=predict,state_hash=state)
    assert [member for member,ids in calls]==[0,1,2]
    assert np.array_equal(result.predictions[:,0,0],[0,1,2])
    with pytest.raises(ValueError,match='member 0'):
        result.numerical_signals(snapshot,outer_seed=157,target_scales=(1,2))
    matching=_snapshot(checkpoint=members[0].resource.sha256)
    signals=result.numerical_signals(matching,outer_seed=157,target_scales=(1,2))
    assert signals['candidate_ids']==snapshot.candidate_ids
    assert signals['ensemble_predictions'] is result.predictions


@pytest.mark.parametrize('corruption',['order','model_state','checkpoint_bytes'])
def test_checkpoint_supplier_fails_closed_on_inference_or_resource_mutation(tmp_path,corruption):
    snapshot=_snapshot();members=_members(tmp_path,snapshot)
    def load(path):return json.loads(path.read_text())
    def state(model):return hashlib.sha256(str(model['member']).encode()).hexdigest()
    def predict(model,ids):
        if corruption=='order':return np.zeros((len(ids),6)),ids[::-1]
        if corruption=='model_state':model['member']=99
        if corruption=='checkpoint_bytes':
            p=members[0].resource.path;p.chmod(0o644);p.write_text('CHANGED');p.chmod(0o444)
        return np.zeros((len(ids),6)),ids
    with pytest.raises(ValueError):collect_checkpoint_ensemble(members,load_predictor=load,predict=predict,state_hash=state)


def test_hybrid_profile_matches_independent_top_quarter_then_farthest_first(tmp_path):
    snapshot=_snapshot();members=_members(tmp_path,snapshot)
    snapshot=_snapshot(checkpoint=members[0].resource.sha256)
    values=np.zeros((3,len(snapshot.candidate_ids),6))
    values[0,:,1]=-np.arange(1,len(snapshot.candidate_ids)+1)
    values[2,:,1]=np.arange(1,len(snapshot.candidate_ids)+1)
    ensemble=K3EnsembleSnapshot(members,values)
    result=select_numerical_profile(snapshot,profile='numerical_hybrid32',outer_seed=157,
                                    round_index=0,ensemble=ensemble,target_scales=(2.,3.))
    scores=values[:,:,[1,4]].std(axis=0,ddof=0)/np.array([2.,3.])
    shortlist=np.argsort(-np.sqrt(np.sum(scores*scores,axis=1)),kind='stable')[:40]
    centers=list(snapshot.train);expected=[]
    for _ in range(32):
        d=[min(float(np.sum((snapshot.candidates[i]-c)**2)) for c in centers) if i not in expected else -1 for i in shortlist]
        chosen=int(shortlist[int(np.argmax(d))]);expected.append(chosen);centers.append(snapshot.candidates[chosen])
    assert result.selected_ids==tuple(snapshot.candidate_ids[i] for i in expected)
    assert result.audit['shortlist_size']==40
    assert result.trace[0]['sample_id']=='U159'


# Public tracked pre-test freeze identity at SOURCE_COMMIT b1a6ed7a. Private
# inputs must bind through this immutable receipt, not just self-declare PASS.
K3_FREEZE_SHA256='9c2c4729d025d84368f17009818d3a1d2a27e2db412bda02d3483763d9c966f5'


@pytest.fixture(scope='module')
def k3_cases():
    manifest=os.getenv('QGEOGNN_K3_REGRESSION_MANIFEST')
    if not manifest or not Path(manifest).is_file():
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: explicit readonly same-seed/current-L real K3 fixture manifest required; no main fallback')
    path=Path(manifest).absolute();root=path.parent
    SnapshotFile(path,hashlib.sha256(path.read_bytes()).hexdigest(),root).verify()
    collection=json.loads(path.read_text())
    if collection['schema']!='same_round_k3_collection_v1':
        raise ValueError('real K3 regression requires the registered collection schema')
    frozen=collection['tracked_freeze']
    assert frozen['sha256']==K3_FREEZE_SHA256
    freeze=json.loads(SnapshotFile(root/frozen['path'],frozen['sha256'],root).verify().read_text())
    assert freeze['status']=='FROZEN_BEFORE_TEST_TRUTH' and freeze['test_truth_access_count']==0
    result=[]
    for item in collection['cases']:
        case_path=SnapshotFile(root/item['path'],item['sha256'],root).verify()
        record=json.loads(case_path.read_text())
        assert record['schema']=='same_round_k3_selection_v1'
        assert record['source_code_commit']=='b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'
        binding=record['source_case_bindings'];entry=freeze['entries'][f'seed_{record["seed"]}/hybrid/round_{record["round_index"]:02d}']
        assert binding['global_entry']==entry
        assert binding['round_contract']['sha256']==entry['round_contract_sha256']
        def resource(spec):return SnapshotFile(root/spec['path'],spec['sha256'],root)
        original=json.loads(resource(binding['round_contract']).verify().read_text())
        assert binding['acquisition_contract']['sha256']==original['files']['acquisition_artifacts/contract.json']
        acquisition=json.loads(resource(binding['acquisition_contract']).verify().read_text())
        files={**original['files'],**acquisition['files']}
        assert binding['historical_selected_ids']['sha256']==files['acquisition_artifacts/selected_next_batch.csv']
        assert binding['historical_latent']['sha256']==files['acquisition_artifacts/current_representations.npz']
        assert original['input']['active_label_count']==record['budget']==333+32*record['round_index']
        assert original['input']['test_truth_access_count']==0
        assert original['input']['endpoint_scales']==dict(zip(('V1','V2'),record['target_scales']))
        for member in record['members']:
            prefix='model' if member['member']==0 else f'acquisition_artifacts/ensemble_member_{member["member"]}'
            assert member['resource']['sha256']==files[prefix+'/best.pt']
            resource(member['resource']).verify()
            audit_path=(root/member['resource']['path']).with_name('fit_audit.json')
            audit=json.loads(SnapshotFile(audit_path,files[prefix+'/fit_audit.json'],root).verify().read_text())
            assert audit['train_rows']==record['budget']
            assert audit['train_ids_hash']==original['input']['L_t_ids_hash']
            assert audit['initialization_seed']==member['initialization_seed']
            assert audit['checkpoint_sha256']==member['resource']['sha256']
        result.append((root,record))
    assert result, 'registered real K3 collection must contain cases'
    assert len({(record['seed'],record['round_index']) for _,record in result})==len(result)
    return result


def _k3_case(root,record):
    def resource(item):return SnapshotFile(root/item['path'],item['sha256'],root)
    members=[]
    for spec in record['members']:
        values={key:value for key,value in spec.items() if key!='resource'}
        members.append(FrozenEnsembleMember(resource=resource(spec['resource']),**values))
    with np.load(resource(record['inputs']).verify(),allow_pickle=False) as saved:
        assert set(saved.files)=={'latent','ensemble_predictions'}
        latent=saved['latent'];predictions=saved['ensemble_predictions']
    with np.load(resource(record['source_case_bindings']['historical_latent']).verify(),allow_pickle=False) as saved:
        np.testing.assert_array_equal(latent,saved['features'])
    import pandas as pd
    historical=pd.read_csv(resource(record['source_case_bindings']['historical_selected_ids']).verify(),usecols=['selection_order','sample_id']).sort_values('selection_order')
    assert historical.selection_order.tolist()==list(range(32))
    assert record['expected_selected_ids']==historical.sample_id.astype(str).tolist()
    first=members[0];authorization=Authorization(first.train_ids,first.candidate_ids);cut=len(first.train_ids)
    provenance=BaselineProvenance(record['source_code_commit'],first.dataset_hash,first.split_hash,
                                 first.resource.sha256,first.protocol_hash,'authorized_snapshot')
    snapshot=FeatureSnapshot(authorization=authorization,train_ids=first.train_ids,candidate_ids=first.candidate_ids,
        train_features=latent[:cut],candidate_features=latent[cut:],config=RepresentationConfig('latent',128),predictor=provenance)
    return members,snapshot,predictions,provenance


@pytest.mark.frozen_regression
def test_real_saved_k3_selection_inputs_match_registered_original_frozen_batches(k3_cases):
    for root,record in k3_cases:
        members,snapshot,predictions,_=_k3_case(root,record)
        result=select_numerical_profile(snapshot,profile='numerical_hybrid32',outer_seed=record['seed'],
            round_index=record['round_index'],ensemble=K3EnsembleSnapshot(members,predictions),target_scales=record['target_scales'])
        assert result.selected_ids==tuple(record['expected_selected_ids'])
        assert result.audit['uncertainty_ranking']==record['expected_uncertainty_ranking']
        assert result.audit['shortlist_positions']==record['expected_shortlist_positions']


@pytest.mark.frozen_regression
@pytest.mark.parametrize('seed',[157,887,2357,6101,12203])
@pytest.mark.parametrize('round_index',[0,1])
def test_real_same_seed_current_l_k3_weights_latent_predictions_and_selection(k3_cases,seed,round_index):
    import torch
    from src.qgeognn_al.al.representations.ensemble import infer_frozen_graph_ensemble
    from src.qgeognn_al.al.representations.latent import GraphRow
    available=next(((root,record) for root,record in k3_cases if (record['seed'],record['round_index'])==(seed,round_index)),None)
    if available is None:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: this explicit seed/round is not registered in the readonly K3 collection')
    root,record=available;members,expected,predictions,provenance=_k3_case(root,record)
    graph=record['graphs'];path=SnapshotFile(root/graph['path'],graph['sha256'],root).verify()
    atom,angle=torch.load(path,map_location='cpu',weights_only=False)
    # Source recipe serializes constants V1_ml=V2_ml=0 before graph building;
    # this is the dedicated zero-sentinel cache, never a truth-bearing graph.
    assert len(atom)==len(angle)==4163
    assert all('y' in a and not bool(torch.count_nonzero(a.y)) and 'y' not in b for a,b in zip(atom,angle))
    indices=[]
    for key in ('labeled','candidates'):
        item=record['graph_indices'][key]
        indices.extend(np.load(SnapshotFile(root/item['path'],item['sha256'],root).verify(),allow_pickle=False).tolist())
    assert len(indices)==3330 and len(set(indices))==3330
    rows=[]
    for identity,index in zip(expected.train_ids+expected.candidate_ids,indices):
        a=atom[index].clone();del a.y
        rows.append(GraphRow(identity,a,angle[index].clone()))
    torch.set_num_threads(2)
    ensemble,latent=infer_frozen_graph_ensemble(members,authorization=expected.authorization,rows=rows,provenance=provenance)
    np.testing.assert_array_equal(latent.train,expected.train)
    np.testing.assert_array_equal(latent.candidates,expected.candidates)
    np.testing.assert_allclose(ensemble.predictions,predictions,rtol=1e-12,atol=1e-8)
    np.testing.assert_array_equal(ensemble.predictions.astype(np.float32),predictions.astype(np.float32))
    result=select_numerical_profile(latent,profile='numerical_hybrid32',outer_seed=seed,round_index=round_index,
        ensemble=ensemble,target_scales=record['target_scales'])
    assert result.selected_ids==tuple(record['expected_selected_ids'])
    assert result.audit['uncertainty_ranking']==record['expected_uncertainty_ranking']
    assert result.audit['shortlist_positions']==record['expected_shortlist_positions']


@pytest.mark.frozen_regression
def test_real_k3_validation_metrics_and_matched_budget_two_point_aulc(k3_cases):
    """Replay already fitted models; targets come only from preapproved L0/val."""
    import pandas as pd
    import torch
    from src.qgeognn_al.predictor.model import load_predictor_checkpoint
    from src.qgeognn_al.predictor.inference import predict_outputs
    from src.qgeognn_al.al.representations.latent import predictor_state_hash
    from src.qgeognn_al.al.evaluation.point import point_metrics
    from src.qgeognn_al.al.evaluation.curves import normalized_aulc
    path=Path(os.environ['QGEOGNN_K3_REGRESSION_MANIFEST']).absolute();root=path.parent
    collection=json.loads(path.read_text())
    if 'validation_manifest' not in collection:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: preapproved L0/validation-only metric fixture missing')
    def resource(item):return SnapshotFile(root/item['path'],item['sha256'],root)
    record=json.loads(resource(collection['validation_manifest']).verify().read_text())
    assert record['schema']=='frozen_k3_validation_v1'
    policy=json.loads(resource(record['source_role_manifest']).verify().read_text())
    assert set(policy['allowed_roles'])==set(record['allowed_target_roles'])=={'l0','validation'}
    assert set(policy['forbidden_target_roles'])==set(record['forbidden_target_roles'])=={'u0','test'}
    assert policy['label_access']['u0_values_converted']==policy['label_access']['test_values_converted']==0
    assert {item['seed'] for item in record['records']}=={157,887,2357,6101,12203}
    torch.set_num_threads(2)
    for item in record['records']:
        cases=sorted([(case_root,case) for case_root,case in k3_cases if case['seed']==item['seed']],key=lambda c:c[1]['round_index'])
        assert [case['round_index'] for _,case in cases]==[0,1]
        target_path=resource(item['authorized_rows']).verify()
        # Metadata-only pass must reject any U/test row before target parsing.
        identities=pd.read_csv(target_path,usecols=['sample_id','canonical_index','role'])
        assert len(identities)==749 and not identities.sample_id.duplicated().any()
        assert identities.role.value_counts().to_dict()=={'validation':416,'l0':333}
        validation=identities.loc[identities.role.eq('validation')].sort_values('canonical_index')
        assert content_hash(sorted(validation.sample_id.astype(str)))==item['validation_ids_hash']
        labels=pd.read_csv(target_path,usecols=['sample_id','V1_ml','V2_ml']).set_index('sample_id')
        truth=labels.loc[validation.sample_id,['V1_ml','V2_ml']].to_numpy(np.float32)
        graph=cases[0][1]['graphs'];atom,angle=torch.load(resource(graph).verify(),map_location='cpu',weights_only=False)
        assert all('y' in a and not bool(torch.count_nonzero(a.y)) and 'y' not in b for a,b in zip(atom,angle))
        clean=[]
        for a in atom:
            a=a.clone();del a.y;clean.append(a)
        old_scores=[];new_scores=[]
        for (_,case),point in zip(cases,item['points']):
            first=case['members'][0];checkpoint=resource(first['resource']).verify()
            assert point['checkpoint_sha256']==first['resource']['sha256']
            audit_path=checkpoint.with_name('fit_audit.json')
            assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==point['fit_audit_sha256']
            audit=json.loads(audit_path.read_text())
            assert audit['validation_ids_hash']==item['validation_ids_hash']
            assert point['old_validation_combined_row_nrmse']==audit['best_validation_combined_normalized_rmse']
            model=load_predictor_checkpoint(checkpoint);before=predictor_state_hash(model)
            assert before==first['checkpoint_state_hash']
            prediction=predict_outputs(model,clean,angle,validation.canonical_index.to_numpy(int).tolist())
            assert predictor_state_hash(model)==before
            score=point_metrics(truth,prediction,dict(zip(('V1','V2'),case['target_scales'])))['combined_normalized_rmse']
            np.testing.assert_allclose(score,point['old_validation_combined_row_nrmse'],rtol=0,atol=1e-12)
            old_scores.append(point['old_validation_combined_row_nrmse']);new_scores.append(score)
        old_raw,old_norm=normalized_aulc([333,365],old_scores,expected_budgets=[333,365])
        new_raw,new_norm=normalized_aulc([333,365],new_scores,expected_budgets=[333,365])
        assert old_raw==new_raw and old_norm==new_norm
        assert old_norm==item['aulc333_365']['old_normalized']
