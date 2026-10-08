import copy
import numpy as np
import pytest
from src.qgeognn_al.active_learning_v2.cw_factorial import select, transforms
from src.qgeognn_al.active_learning_v2.feedback_ablation import packet, fixture, freeze_batch, VARIANTS


def test_transform_isometry_and_selectors_preserve_input():
    rng=np.random.default_rng(19)
    y=rng.normal(size=(333,2)); y[:,1]+=2*y[:,0]
    t,_=transforms(y,tuple(y.std(axis=0)))
    g=rng.normal(size=(20,2,13))
    f={k:np.einsum('ab,nbp->nap',v,g).reshape(20,-1) for k,v in t.items()}
    for a,b in [('raw','rotation'),('cw','matched')]:
        np.testing.assert_allclose(f[a]@f[a].T,f[b]@f[b].T,atol=1e-12)
    for selector in ('lcmd','maxdet','ivr'):
        saved=f['cw'].copy(); positions,_=select(f['cw'],4,selector,5)
        assert len(set(positions))==5
        np.testing.assert_array_equal(saved,f['cw'])
        tied,_=select(np.ones((9,4)),3,selector,6)
        assert tied.tolist()==list(range(6))


def test_ablation_numeric_feedback_and_identity_stability(tmp_path):
    source=fixture(); saved=copy.deepcopy(source); key=b'a-stable-trajectory-key'
    packets={v:packet(source,v,key) for v in VARIANTS}
    assert source==saved
    for p in packets.values():
        assert [r['id'] for r in p['candidates']]==[r['id'] for r in packets['full']['candidates']]
        assert p['observed'][0]['compound_group']==p['history'][0]['compound_group']
    assert packets['no_narrative']['history'][0]['premeasurement_error_V1_ml']==-997.8
    assert packets['no_narrative']['history'][0]['elapsed_seconds']==42
    assert 'memory' not in packets['no_narrative']
    assert 'V1_ml' not in packets['no_direct_response']['history'][0]
    assert packets['no_direct_response']['observed'][0]['pred_V1_ml']==1.2
    assert 'smiles' not in packets['no_chemistry']['history'][0]
    p=packets['no_chemistry']; ids=[r['id'] for r in p['candidates']]
    freeze_batch(p,ids[:16],tmp_path/'freeze.json')
    freeze_batch(p,ids[:16],tmp_path/'freeze.json')
    for bad in (ids[:15],ids[:15]+[ids[0]],ids[:15]+[p['pending'][0]['id']],ids[:15]+[p['observed'][0]['id']]):
        with pytest.raises(ValueError): freeze_batch(p,bad,tmp_path/'bad.json')
    with pytest.raises(ValueError): freeze_batch(p,ids[1:17],tmp_path/'freeze.json')
    p['round']=99
    with pytest.raises(ValueError): freeze_batch(p,ids[:16],tmp_path/'other.json')


@pytest.mark.parametrize('variant',VARIANTS)
def test_fail_closed_unknown_fields_and_numeric_smuggling(variant):
    key=b'a-stable-trajectory-key'
    source=fixture(); source['observed'][0]['pred_V1_ml']='SMILES: secret; measured 10'
    with pytest.raises(ValueError): packet(source,variant,key)
    source=fixture(); source['observed'][0]['hidden_response_summary']=999
    with pytest.raises(ValueError): packet(source,variant,key)
    source=fixture(); source['candidates'][0]['V1_ml']=999
    with pytest.raises(ValueError): packet(source,variant,key)
    source=fixture(); source['pending'][0]['id']=source['observed'][0]['id']
    with pytest.raises(ValueError): packet(source,variant,key)
