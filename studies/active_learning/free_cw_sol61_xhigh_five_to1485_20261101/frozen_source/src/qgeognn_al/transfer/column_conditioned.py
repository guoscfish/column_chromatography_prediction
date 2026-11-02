"""Single shared V2 head with one small physical context branch."""
import torch
from torch import nn
from ..models import build_predictor


class ColumnConditionedQGeoGNN(nn.Module):
    def __init__(self, source_model, mass_normalized=False):
        super().__init__()
        self.predictor = build_predictor(source_model.condition_branch.normalization)
        self.predictor.load_state_dict(source_model.state_dict(), strict=True)
        self.column_branch = nn.Sequential(nn.Linear(4,16), nn.ReLU(), nn.Linear(16,128))
        nn.init.zeros_(self.column_branch[-1].weight)
        nn.init.zeros_(self.column_branch[-1].bias)
        self.mass_normalized = mass_normalized
        if mass_normalized:
            # Function-preserving source initialization in mL/g; shared head remains the only head.
            with torch.no_grad():
                self.predictor.head[0].weight.div_(4.)
                self.predictor.head[0].bias.div_(4.)

    def forward(self, atom, angle):
        if not hasattr(atom, 'column_context'):
            raise ValueError('missing column context')
        x = atom.column_context
        if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
            raise ValueError('invalid column context')
        representation = self.predictor.extract_representation(atom, angle)
        output = self.predictor.head(representation + self.column_branch(x))
        return output if self.training else output.clamp(0,1e8)

    def training_joint(self):
        self.train()
        # Fixed qualified BN statistics for both tasks, avoiding ordering-dependent running buffers.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
