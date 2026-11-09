"""The sole round lifecycle: fit, predict/acquire, freeze, reveal, retrain, feedback."""
from .state import RunSpec, RoundWorkflow, LabelBroker, AmbiguousReveal, AmbiguousFit
