"""Frozen AL scaler extraction uses X and only explicit L0 target values."""
import pytest
import json
import numpy as np
import pandas as pd
from src.qgeognn_al.predictor.preprocessing import fit_al_preprocessing
from src.qgeognn_al.predictor import preprocessing as migrated
from src.qgeognn_al.infra.identity import ids_hash
from tests.support import historical_reference as legacy_reference


@pytest.mark.historical_archive
def test_legacy_and_migrated_scalers_are_exact_with_known_synthetic_L0(tmp_path):
    old=legacy_reference.load('src/qgeognn_al/active_learning_v2/runner.py',['fit_al_preprocessing'],
        injected={'pd':pd,'eluent_descriptor':migrated.eluent_descriptor,
                  'ConditionNormalization':migrated.ConditionNormalization,'ids_hash':ids_hash,
                  'sha256_file':migrated.sha256_file,'atomic_json':migrated.atomic_json})
    frame=pd.DataFrame({'sample_id':['a','b','c','val'],'canonical_smiles':['C','CC','CCC','O'],
                        'PE/EA':['1/0','1/1','0/1','1/0'],'Density g/ml':[1.,2.,3.,999.],
                        'V/ul':[2.,3.,4.,999.],'Volume of loading solvent/ul':[1.,2.,3.,999.]})
    cache={name:{'descriptor':np.arange(16,dtype=np.float32)+i} for i,name in enumerate(frame.canonical_smiles)}
    args=(frame,cache,[0,1,2],['a','b'],np.array([[1.,2.],[3.,6.]]))
    path=tmp_path/'scaler.json'
    before=old.fit_al_preprocessing(*args,path)
    after=fit_al_preprocessing(*args,path)
    assert before==after and after[1]['target_scales']=={'V1':1.,'V2':2.}
    assert after[1]['covariate_fit_rows']==3 and after[1]['target_scale_fit_rows']==2
    frame.loc[3,['Density g/ml','V/ul','Volume of loading solvent/ul']]=-999.
    assert fit_al_preprocessing(*args,path)==after



def test_covariate_outer_train_and_l0_target_permissions(tmp_path):
    frame=pd.DataFrame({'sample_id':['a','b','c','val'],'canonical_smiles':['C','CC','CCC','O'],
        'PE/EA':['1/0','1/1','0/1','1/0'],'Density g/ml':[1.,2.,3.,999.],
        'V/ul':[2.,3.,4.,999.],'Volume of loading solvent/ul':[1.,2.,3.,999.]})
    cache={name:{'descriptor':np.arange(16,dtype=np.float32)+i} for i,name in enumerate(frame.canonical_smiles)}
    args=(frame,cache,[0,1,2],['a','b'],np.array([[1.,2.],[3.,6.]]),tmp_path/'scaler.json')
    condition,preprocessing=fit_al_preprocessing(*args)
    assert (condition.loading_amount_min,condition.loading_amount_max)==(2.,12.)
    assert (condition.loading_volume_min,condition.loading_volume_max)==(1.,3.)
    assert preprocessing['target_scales']=={'V1':1.,'V2':2.}
    assert preprocessing['covariate_fit_role']=='outer_train_X_only' and preprocessing['covariate_fit_rows']==3
    assert preprocessing['target_scale_fit_role']=='l0_labels_only' and preprocessing['target_scale_fit_rows']==2
    frame.loc[3,['Density g/ml','V/ul','Volume of loading solvent/ul']]=-999.
    assert fit_al_preprocessing(*args)==(condition,preprocessing)
