"""Single scientific predictor: fixed QGeoGNN-V2 architecture and loss."""
from .model import (QGeoGNNV2, build_predictor, extract_representation, load_predictor_checkpoint,
                    predictor_checkpoint, validate_predictor_checkpoint)
from .conditions import ConditionNormalization
