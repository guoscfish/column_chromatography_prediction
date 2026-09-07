"""Explicit, unit-bearing engineering context; no inferred column geometry."""
import re
import numpy as np

FEATURE_NAMES = ('packing_mass_g', 'flow_ml_min', 'loading_mass_mg_per_g', 'loading_solvent_ul_per_g')
FEATURE_UNITS = ('g', 'mL/min', 'mg/g', 'uL/g')


def packing_mass(spec):
    if not isinstance(spec, str) or not re.fullmatch(r'Silica-CS \d+(?:\.\d+)?g(?:\+\d+(?:\.\d+)?g)*', spec.strip()):
        raise ValueError(f'missing or unsupported column metadata: {spec!r}')
    mass = sum(float(x) for x in re.findall(r'(\d+(?:\.\d+)?)g', spec))
    if mass <= 0:
        raise ValueError('packing mass must be positive')
    return mass


def context_matrix(frame):
    required = ['column_specs', 'Flow mL/min', 'Density g/ml', 'V/ul', 'Volume of loading solvent/ul']
    if set(required)-set(frame):
        raise ValueError('missing column metadata: '+str(sorted(set(required)-set(frame))))
    mass = frame.column_specs.map(packing_mass).to_numpy(float)
    flow = frame['Flow mL/min'].to_numpy(float)
    loading = frame['Density g/ml'].to_numpy(float)*frame['V/ul'].to_numpy(float)
    solvent = frame['Volume of loading solvent/ul'].to_numpy(float)
    x = np.column_stack([mass, flow, loading/mass, solvent/mass])
    if not np.isfinite(x).all() or (x[:, :2] <= 0).any() or (x[:, 2:] < 0).any():
        raise ValueError('nonfinite or invalid physical context')
    return x


class TrainingNormalizer:
    def fit(self, frame, allowed_train_ids):
        ids = frame.sample_id.astype(str).tolist()
        if len(set(ids)) != len(ids) or set(ids) != set(allowed_train_ids):
            raise ValueError('normalization rows must exactly equal authorized training IDs')
        x = context_matrix(frame)
        self.ids = sorted(ids)
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1.
        return self

    def transform(self, frame):
        return ((context_matrix(frame)-self.mean)/self.scale).astype(np.float32)
