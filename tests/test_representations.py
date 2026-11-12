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
@pytest.mark.parametrize('sketch_seed', [17, 157, 887])
def test_raw_and_cw_countsketch_match_every_component_and_authorized_extractor(sketch_seed):
    import torch
    from torch_geometric.data import Data
    from src.qgeognn_al.al.representations.gradient import (
        extract_q50_gradient_sketches, extract_linear_output_gradient_sketches, extract_gradient,
    )
    from src.qgeognn_al.al.representations.center_width import center_width_transform
    from src.qgeognn_al.al.representations.latent import GraphRow, predictor_state_hash
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(1, 6)
        def forward(self, atom, angle):
            return self.linear(atom.x.reshape(-1, 1).float())
    torch.set_num_threads(1); torch.manual_seed(31)
    model = Toy().eval()
    atom = [Data(x=torch.tensor([[float(i+1)]]), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, 13)), canonical_position=torch.tensor(i), num_nodes=1) for i in range(8)]
    angle = [Data(edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 17)), num_nodes=0) for _ in atom]
    ref = legacy.gradient()
    old = ref.extract_q50_gradient_sketches(model, atom, angle, list(range(8)), (2., 3.), dimension=512, sketch_seed=sketch_seed)
    new = extract_q50_gradient_sketches(model, atom, angle, list(range(8)), (2., 3.), dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new.features, old.features)
    assert new.parameter_audit == old.parameter_audit
    assert {k:v for k,v in new.audit.items() if k != 'elapsed_seconds'} == {k:v for k,v in old.audit.items() if k != 'elapsed_seconds'}
    l0 = np.array([[1., 3.], [3., 5.], [2., 9.]])
    transforms = legacy.load('src/qgeognn_al/active_learning_v2/gradient_transforms.py', ['center_width_transform', '_validated_targets'])
    old_transform, old_audit = transforms.center_width_transform(l0)
    transform, audit = center_width_transform(l0)
    np.testing.assert_array_equal(transform, old_transform); assert audit == old_audit
    old_cw = ref.extract_linear_output_gradient_sketches(model, atom, angle, list(range(8)), old_transform, dimension=512, sketch_seed=sketch_seed)
    new_cw = extract_linear_output_gradient_sketches(model, atom, angle, list(range(8)), transform, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_cw.features, old_cw.features)
    ids = tuple(f'G{i}' for i in range(8)); auth = Authorization(ids[:3], ids[3:])
    rows = [GraphRow(identity, a, b) for identity, a, b in zip(ids, atom, angle)]
    provenance = BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64)
    kwargs = dict(expected_state_hash=predictor_state_hash(model), authorization=auth, rows=rows,
                  provenance=provenance, sketch_seed=sketch_seed)
    public = extract_gradient(model, scales=(2., 3.), **kwargs)
    public_cw = extract_gradient(model, kind='center_width', l0_targets=l0, l0_ids=auth.train_ids, **kwargs)
    np.testing.assert_array_equal(np.vstack([public.train, public.candidates]), old.features)
    np.testing.assert_array_equal(np.vstack([public_cw.train, public_cw.candidates]), old_cw.features)
    assert public.config.dimension == 512 and public_cw.config.dimension == 512
    assert all(parameter.grad is None for parameter in model.parameters())
    assert public.extraction.predictor_state_hash == public.extraction.predictor_state_after_hash
    for kwargs_bad in [dict(l0_ids=('future_U',), l0_targets=[[1., 2.]]),
                       dict(l0_ids=auth.train_ids, l0_targets=[[1., 2.]])]:
        with pytest.raises(ValueError, match='identity-aligned'):
            extract_gradient(model, kind='center_width', **kwargs, **kwargs_bad)
    record('q50/CW CountSketch512', f'sketch_seed={sketch_seed}',
           [old.audit['sketch_mapping_sha256']], [new.audit['sketch_mapping_sha256']],
           {'parameter_count': new.audit['parameter_count'], 'raw_exact': True, 'cw_exact': True})



@pytest.mark.historical_archive
@pytest.mark.parametrize('case,alpha', [('ordinary', .5), ('ties', .8), ('rank_deficient', 0.)])
def test_fused_math_and_snapshot_composition_match_frozen_block_scaling(case, alpha):
    from src.qgeognn_al.al.representations.fused import fuse_gradient_latent_features, fuse_snapshots
    old = legacy.load('src/qgeognn_al/active_learning_v2/fused_features.py',
                      ['_array_hash', 'FusionResult', 'fuse_gradient_latent_features'],
                      injected={'hashlib': hashlib, 'GRADIENT_DIMENSION': 512,
                                'LATENT_DIMENSION': 128, 'FUSED_DIMENSION': 640, 'ALPHA': .5})
    g, count, _ = inputs(case, 512); h, _, _ = inputs(case, 128)
    before = old.fuse_gradient_latent_features(g, h, count, alpha=alpha)
    after = fuse_gradient_latent_features(g, h, count, alpha=alpha)
    np.testing.assert_array_equal(after.features, before.features)
    assert after.audit == before.audit
    composed = fuse_snapshots(snapshot(g, count), snapshot(h, count, 'latent'), alpha=alpha)
    np.testing.assert_array_equal(np.vstack([composed.train, composed.candidates]), before.features)
    params = json.loads(composed.config.parameters_json)
    assert params['alpha'] == alpha
    assert params['normalization'] == 'sqrt(mean_L_t(||block_i||_2^2)); L_t prefix only'
    assert composed.config.dimension == 640
    changed = h.copy(); changed[count:] *= 77
    # Candidate changes may affect vectors but must never re-estimate L block scales.
    check = fuse_gradient_latent_features(g, changed, count, alpha=alpha)
    assert check.gradient_scale == before.gradient_scale and check.latent_scale == before.latent_scale
    record('fused640', case, [before.audit['fused_feature_sha256']], [after.audit['fused_feature_sha256']])




def test_snapshot_owns_arrays_binds_order_and_detects_tampering():
    matrix=np.arange(12,dtype=np.float64).reshape(4,3);value=snapshot(matrix,2)
    before=value.sha256;matrix[:]=999.;value.verify();assert value.sha256==before
    assert not value.train.flags.writeable and not value.candidates.flags.writeable
    with pytest.raises(ValueError,match='exactly match'):
        FeatureSnapshot(authorization=value.authorization,train_ids=tuple(reversed(value.train_ids)),candidate_ids=value.candidate_ids,
                        train_features=np.ones((2,3)),candidate_features=np.ones((2,3)),config=value.config,predictor=value.predictor)
    value.train.setflags(write=True);value.train[0,0]=42.
    with pytest.raises(ValueError,match='hash changed'):value.verify()


def test_center_width_and_fusion_use_only_declared_labeled_scales():
    from src.qgeognn_al.al.representations.center_width import center_width_transform
    from src.qgeognn_al.al.representations.fused import fuse_gradient_latent_features
    targets=np.array([[1.,3.],[3.,7.],[5.,5.]])
    transform,audit=center_width_transform(targets)
    transformed=targets@transform.T
    np.testing.assert_allclose(transformed.std(axis=0),[1.,1.],rtol=1e-14)
    assert audit['target_source']=='L0_labels_only' and audit['scale_ddof']==0
    with pytest.raises(ValueError,match='scales'):
        center_width_transform(np.array([[1.,3.],[2.,4.]]))
    g=np.ones((4,512));h=np.ones((4,128));g[2:]*=10;h[2:]*=7
    first=fuse_gradient_latent_features(g,h,2,alpha=.25)
    np.testing.assert_allclose(np.square(first.features[:2]).sum(axis=1),[1.,1.],rtol=1e-6)
    g[2:]*=123;h[2:]*=456;after=fuse_gradient_latent_features(g,h,2,alpha=.25)
    assert (first.gradient_scale,first.latent_scale)==(after.gradient_scale,after.latent_scale)
    assert first.features.shape==(4,640) and first.features.dtype==np.float32


def test_public_latent_and_gradient_extractors_preserve_state_and_reject_labels():
    import torch
    from torch_geometric.data import Data
    from src.qgeognn_al.al.representations.latent import GraphRow,extract_latent,predictor_state_hash
    from src.qgeognn_al.al.representations.gradient import extract_gradient
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(1,6)
        def forward(self,atom,angle):return self.head(atom.x.float())
        def extract_representation(self,atom,angle):return atom.x.float().repeat(1,128)
    torch.manual_seed(17);model=Toy().eval();fingerprint=predictor_state_hash(model)
    rows=[]
    for i in range(4):
        atom=Data(x=torch.tensor([[float(i+1)]]),edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,13)),canonical_position=torch.tensor(i),num_nodes=1)
        angle=Data(edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,17)),num_nodes=0)
        rows.append(GraphRow(f'G{i}',atom,angle))
    auth=Authorization(('G0','G1'),('G2','G3'));provenance=BaselineProvenance('1'*40,'2'*64,'3'*64,'4'*64,'5'*64)
    kwargs=dict(expected_state_hash=fingerprint,authorization=auth,rows=rows,provenance=provenance,batch_size=2)
    latent=extract_latent(model,**kwargs)
    np.testing.assert_array_equal(np.vstack([latent.train,latent.candidates]),np.repeat(np.arange(1,5)[:,None],128,axis=1))
    first=extract_gradient(model,scales=(2.,3.),sketch_seed=157,**{k:v for k,v in kwargs.items() if k!='batch_size'})
    second=extract_gradient(model,scales=(2.,3.),sketch_seed=157,**{k:v for k,v in kwargs.items() if k!='batch_size'})
    np.testing.assert_array_equal(first.train,second.train);np.testing.assert_array_equal(first.candidates,second.candidates)
    assert first.config.dimension==512 and predictor_state_hash(model)==fingerprint
    assert all(p.grad is None for p in model.parameters())
    rows[0].atom.y=torch.tensor([[999.,999.]])
    with pytest.raises(ValueError,match='label'):extract_latent(model,**kwargs)
