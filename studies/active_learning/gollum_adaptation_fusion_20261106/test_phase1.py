"""Engineering and data-boundary tests, never benchmark effectiveness evidence."""
import numpy as np
import pytest
from data_boundary import assert_disjoint, group_partition, load_state, representations
from gp_training import ControlledProjectionGP, variance_temperature
from gp_models import ProjectedMaternGP


def test_group_disjointness_and_response_blindness():
    ids=[str(i) for i in range(150)]
    smiles=[f'molecule_{i//3}' for i in range(150)]
    a=group_partition(ids,smiles,157)
    assert np.array_equal(a,group_partition(ids,smiles,157))
    for smi in set(smiles):
        assert len(set(a[np.array(smiles)==smi]))==1
    assert set(a)=={'fit','calibration','check'}
    with pytest.raises(PermissionError):
        assert_disjoint(['train'],['train'])
    with pytest.raises(PermissionError):
        assert_disjoint(['a','a'],['b'])


def test_temperature_has_closed_form_and_checks_inputs():
    y=np.array([2.,-2.,2.,-2.]);v=np.ones(4)
    assert variance_temperature(y,np.zeros(4),v)==4
    with pytest.raises(ValueError):
        variance_temperature(y,np.zeros(4),-v)


def test_original_replays_actual_legacy_fit_and_variance():
    rng=np.random.default_rng(3);x=rng.normal(size=(22,5));y=np.sin(x[:,0])+rng.normal(size=22)*.05
    old=ProjectedMaternGP(projection_dim=3,max_iter=12,seed=17).fit(x,y)
    new=ControlledProjectionGP(method='original',projection_dim=3,max_iter=12,seed=17).fit(x,y)
    for a,b in zip(old.predict(x),new.predict(x)):
        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(old.projection_weight,new.projection_weight,rtol=1e-9,atol=1e-10)
    mu,latent,obs=new.predict_observation(x)
    np.testing.assert_allclose(obs-latent,new.noise)
    assert new.audit['covariance_condition_number']>=1


@pytest.mark.parametrize('method',['regularized','noise_floor','staged'])
def test_controlled_training_finite(method):
    rng=np.random.default_rng(4);x=rng.normal(size=(18,4));y=x[:,0]+rng.normal(size=18)*.1
    gp=ControlledProjectionGP(method=method,projection_dim=3,max_iter=10,stage_iterations=10).fit(x,y)
    mu,var=gp.predict(x,full_covariance=True)
    assert np.isfinite(mu).all() and np.linalg.eigvalsh(var).min()>-1e-9
    assert np.isfinite(gp.audit['optimizer']['final_projected_gradient_inf'])
    if method=='noise_floor':
        assert gp.noise/gp.target_scale**2>=.02-1e-12
    if method=='regularized':
        assert gp.audit['optimizer']['objective_evaluations'][-1]['penalty']>0
    if method=='staged':
        assert gp.audit['optimizer']['first_stage']['iterations']>0


def test_sealed_state_identity_permissions_and_train_transform():
    state,data,meta,subsets=load_state(157,333)
    assert set(np.concatenate(list(subsets.values())))==set(range(333))
    assert set(meta['candidate_ids'])<=set(state.unlabeled_ids)
    testindex=int(state.partition.loc[state.partition.role=='test','canonical_index'].iloc[0])
    with pytest.raises(PermissionError):
        state.validate_indices([testindex])
    reps,_=representations(data,subsets['fit'])
    edited={k:v.copy() for k,v in data.items()}
    outside=[i for i in range(len(data['indices'])) if i not in set(subsets['fit'])]
    edited['gnn'][outside]+=1000;edited['llm_conditions'][outside]-=1000
    changed,_=representations(edited,subsets['fit'])
    np.testing.assert_allclose(reps['fusion'][subsets['fit']],changed['fusion'][subsets['fit']],atol=1e-12)
