"""Pure registered acquisition rules."""
from .base import Selection, select
from .random import random_control_seed
__all__ = ['Selection', 'select', 'random_control_seed']
