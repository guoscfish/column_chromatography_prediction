"""Strict real frozen-checkpoint/cache regression, only authorized L0/validation."""
import ast
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader
from src.qgeognn_al.predictor import load_predictor_checkpoint
from src.qgeognn_al.predictor.data import build_model_data
from src.qgeognn_al.infra.resources import SnapshotFile
from tests.support import historical_reference as legacy_reference

ROOT=Path(__file__).resolve().parents[1]
RECORDS=[]



@pytest.mark.historical_archive
@pytest.mark.parametrize('old,new,names',[
 ('src/qgeognn_al/training/predictor.py','src/qgeognn_al/predictor/training.py',['target_loss','train_source','fit_source_preprocessing']),
 ('src/qgeognn_al/data.py','src/qgeognn_al/predictor/data.py',['build_model_data','minmax_transform','fit_standardizer']),
 ('src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/predictor/model.py',['GeometryBackbone','QGeoGNNV2','build_predictor','predictor_checkpoint','load_predictor_checkpoint'])])
def test_moved_scientific_bodies_are_ast_identical_to_frozen_commit(old,new,names):
    before=ast.parse(legacy_reference.source(old));after=ast.parse((ROOT/new).read_text())
    a={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in names:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)




def normalization():
    from src.qgeognn_al.predictor.conditions import ConditionNormalization
    return ConditionNormalization(0.,10.,0.,10.,'4g','source_train',2,'1'*64,'2'*64)


def test_model_checkpoint_schema_parameter_count_and_raw_quantile_order(tmp_path):
    from src.qgeognn_al.predictor.model import build_predictor,predictor_checkpoint,validate_predictor_checkpoint,load_predictor_checkpoint
    model=build_predictor(normalization()).eval()
    assert sum(p.numel() for p in model.parameters())==458952
    with torch.no_grad():
        model.head[0].weight.zero_();model.head[0].bias.copy_(torch.tensor([3.,1.,2.,8.,4.,6.]))
    model.extract_representation=lambda atom,angle:torch.zeros((1,128))
    np.testing.assert_array_equal(model(None,None).detach().numpy(),[[3.,1.,2.,8.,4.,6.]])
    checkpoint=predictor_checkpoint(model,preprocessing={'fit_role':'source_train'},training_config={},provenance={})
    validate_predictor_checkpoint(checkpoint)
    path=tmp_path/'synthetic_checkpoint.pt';torch.save(checkpoint,path);restored=load_predictor_checkpoint(path)
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in checkpoint['model_state_dict'].items())
    checkpoint['output_order']=list(reversed(checkpoint['output_order']))
    with pytest.raises(ValueError,match='output contract'):validate_predictor_checkpoint(checkpoint)


def test_target_loss_keeps_pinball_mse_and_soft_crossing_definition():
    from src.qgeognn_al.predictor.training import target_loss
    truth=torch.tensor([2.,4.]);prediction=torch.tensor([[3.,1.,0.],[2.,4.,6.]],requires_grad=True)
    loss=target_loss(truth,prediction)
    assert loss.item()==pytest.approx(3.55,abs=1e-6)
    gradient=torch.autograd.grad(loss,prediction)[0]
    assert torch.isfinite(gradient).all() and prediction.grad is None
