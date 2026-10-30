"""Small numerical tests of geometry definitions; no historical experiment runs."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('geometry_analysis',Path(__file__).with_name('geometry_analysis.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def test_within_state_normalization_is_scale_invariant():
    rng=np.random.default_rng(73)
    x=rng.normal(size=(89,6));p=np.arange(0,64,2)
    a,_=g.geometry(x,25,p);b,_=g.geometry(x*17,25,p)
    names=['selected_distance_percentile_mean','coverage_distance_relative_drop',
           'coverage_squared_distance_relative_drop','remaining_U_coverage_distance_relative_drop',
           'batch_nn_to_pool_coverage_ratio','centered_bank_effective_rank',
           'pool_coverage_mean_to_bank_RMS','coverage_gain_to_bank_RMS']
    for name in names:assert np.isclose(a[name],b[name],rtol=1e-10,atol=1e-10)
    assert np.isclose(17*a['selected_to_L_distance_mean'],b['selected_to_L_distance_mean'])


def test_coverage_excludes_selected_rows_for_sensitivity():
    x=np.arange(66,dtype=float)[:,None]
    p=np.arange(0,64,2)
    metrics,points=g.geometry(x,2,p)
    before=np.arange(1,65,dtype=float)
    # Pool 2..65, centers before 0/1, new centers 2,4,...,64.
    after=np.array([0 if i%2==0 else 1 for i in range(2,66)],dtype=float)
    keep=np.ones(64,dtype=bool);keep[p]=False
    assert np.isclose(metrics['coverage_distance_relative_drop'],1-after.mean()/before.mean())
    assert np.isclose(metrics['remaining_U_coverage_distance_relative_drop'],1-after[keep].mean()/before[keep].mean())
    assert np.allclose(points['distance_percentile'],(p+1)/64)
    assert metrics['coverage_distance_relative_drop']>=metrics['remaining_U_coverage_distance_relative_drop']


def test_effective_rank_and_tied_distance_midrank():
    er,pr,top=g.effective_rank(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]))
    assert np.allclose([er,pr,top],[2,2,.5])
    x=np.r_[np.array([[0.,0.]]),np.tile([[1.,0.]],(64,1))]
    metrics,points=g.geometry(x,1,np.arange(32))
    assert np.allclose(points['distance_percentile'],32.5/64)
    assert metrics['batch_exact_feature_duplicate_fraction']==31/32
    assert metrics['batch_nn_less_than_half_L_fraction']==1


def test_matched_condition_pairs_hold_other_recorded_conditions_fixed():
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    import condition_geometry_probe as probe
    c={'PE/EA':'20/1','Density g/ml':1.,'V/ul':50.,'loading solvent':'PE','Volume of loading solvent/ul':200.}
    rows=[{'smiles':'CCO','conditions':c},
          {'smiles':'CCO','conditions':{**c,'V/ul':100.}},
          {'smiles':'CCO','conditions':{**c,'V/ul':100.,'PE/EA':'10/1'}},
          {'smiles':'CCN','conditions':{**c,'V/ul':100.}}]
    assert probe.matched_pairs(rows,'V/ul').tolist()==[[0,1]]
    assert probe.matched_pairs(rows,'PE/EA').tolist()==[[1,2]]


def test_lcmd_contract_requires_all_fields_even_when_checkpoint_matches():
    import copy
    import pytest
    acquisition={'outer_seed':157,'method':'lcmd','source_round':10,'acquisition_round':11,
                 'L_t_ids_hash':'l'*64,'U_t_ids_hash':'u'*64,'evaluation_checkpoint_sha256':'a'*64,
                 'evaluation_checkpoint_state_hash':'b'*64,'endpoint_scales':{'V1':2.,'V2':3.},
                 'batch_size':32,'feature_recomputed_from_current_model':True,
                 'centers_are_current_L_t':True,'test_truth_access_count':0}
    bank={**copy.deepcopy(acquisition),'dimension':512,'sketch_seed':4_000_194,
          'kind':'current_full_network_q50_gradient_CountSketch','scales':[2.,3.]}
    g.validate_lcmd_bank_contract(bank,acquisition,157,10)
    for key,value in [('source_round',9),('L_t_ids_hash','wrong'),('endpoint_scales',{'V1':4.,'V2':3.})]:
        bad=copy.deepcopy(bank);bad[key]=value
        with pytest.raises(AssertionError):g.validate_lcmd_bank_contract(bad,acquisition,157,10)
