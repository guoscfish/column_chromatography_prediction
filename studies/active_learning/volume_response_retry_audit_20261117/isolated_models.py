"""Isolated two-parameter skips; the repository predictor contract is unchanged."""
from dataclasses import asdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch import nn
from qgeognn_al.models.qgeognn_v2 import QGeoGNNV2
from qgeognn_al.schemas.conditions import ConditionNormalization
from qgeognn_al.data import qg

VARIANTS={'A_original':'volume_control_original_v1','B_amount_skip':'volume_control_amount_skip_v1','C_volume_skip':'volume_control_volume_skip_v1'}

class IsolatedSkip(QGeoGNNV2):
    def __init__(self,normalization,kind,volume_min,volume_max):
        super().__init__(normalization)
        if kind not in ('B_amount_skip','C_volume_skip'):raise ValueError('Unknown skip variant')
        self.study_kind=kind;self.model_variant=VARIANTS[kind]
        self.skip_weight=nn.Parameter(torch.zeros(2))
        self.register_buffer('sample_volume_min',torch.tensor(float(volume_min),dtype=torch.float32))
        self.register_buffer('sample_volume_range',torch.tensor(float(volume_max-volume_min),dtype=torch.float32))
        if volume_max<=volume_min:raise ValueError('Invalid volume scale')
    def skip_input(self,atom):
        if self.study_kind=='B_amount_skip':return self.condition_branch.typed_inputs(atom)[:,2]
        if not hasattr(atom,'sample_volume_ul') or not hasattr(atom,'sample_id'):raise ValueError('Raw sample-volume identity missing')
        return (atom.sample_volume_ul.reshape(-1)-self.sample_volume_min)/self.sample_volume_range
    def forward(self,atom,angle):
        z=self.head[0](self.extract_representation(atom,angle))
        x=self.skip_input(atom)
        if len(x)!=len(z):raise ValueError('Volume batch identity/shape mismatch')
        z=z+x[:,None]*self.skip_weight.repeat_interleave(3)[None,:]
        y=self.head[1](z)
        return y if self.training else torch.clamp(y,min=0,max=1e8)

def build_control(kind,normalization,volume_min,volume_max):
    qg.device=torch.device('cpu')
    if kind=='A_original':return QGeoGNNV2(normalization)
    return IsolatedSkip(normalization,kind,volume_min,volume_max)

def checkpoint(model,kind,normalization,preprocessing,config,volume_scale,provenance):
    return dict(study_schema='volume_response_control_v1',model_variant=VARIANTS[kind],kind=kind,
        normalization=asdict(normalization),preprocessing=preprocessing,training_config=config,volume_scale=volume_scale,
        output_order=['V1_q10','V1_q50','V1_q90','V2_q10','V2_q50','V2_q90'],
        provenance=provenance,parameter_count=sum(p.numel() for p in model.parameters()),state_dict=model.state_dict())

def load_control(path):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d.get('study_schema')!='volume_response_control_v1' or d.get('kind') not in VARIANTS:raise ValueError('Not an isolated control checkpoint')
    if d['model_variant']!=VARIANTS[d['kind']]:raise ValueError('Variant contract mismatch')
    v=d['volume_scale'];model=build_control(d['kind'],ConditionNormalization(**d['normalization']),v['min'],v['max'])
    model.load_state_dict(d['state_dict'],strict=True);model.eval();return model,d
