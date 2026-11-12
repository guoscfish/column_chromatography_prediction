"""Frozen scientific profiles selecting the single migrated numerical kernels.

These names identify historical B32 protocols, not interchangeable aliases.
Every representation and K3 signal is supplied explicitly by its authorized
owner. This dispatcher does not train, load models, reveal labels or fetch Git.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from types import MappingProxyType

from .acquisitions import select
from .acquisitions.random import historical_random_trajectory_seed
from .representations import FeatureSnapshot
from .representations.ensemble import K3EnsembleSnapshot

SOURCE_COMMIT = 'b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'
CONFIRMATION_SEEDS = (157, 887, 2357, 6101, 12203)


@dataclass(frozen=True)
class NumericalProfile:
    name: str
    acquisition: str
    representation: str | None
    dimension: int | None
    source_path: str
    version: str
    batch_size: int = 32
    initial_labels: int = 333

    def to_dict(self):
        return {**asdict(self), 'source_commit': SOURCE_COMMIT,
                'sketch_seed': 'outer_seed + 4_000_037' if self.dimension == 512 else None,
                'tie_break': 'stable canonical current-U input order',
                'training': 'same initialization from scratch each round; fixed original-L0 normalization'}


_PROFILES = (
    NumericalProfile('random32', 'random', None, None,
                     'src/qgeognn_al/active_learning_v2/sequential_acquisition.py', 'sequential-fixed-U0-permutation-b32'),
    NumericalProfile('gradient_lcmd32', 'lcmd', 'gradient', 512,
                     'src/qgeognn_al/active_learning_v2/sequential_acquisition.py', 'full-network-q50-countsketch512-current-L'),
    NumericalProfile('cw_gradient_lcmd32', 'lcmd', 'center_width', 512,
                     'src/qgeognn_al/active_learning_v2/short_sequential_runner.py', 'linear-output-autograd-CW-L0-ddof0'),
    NumericalProfile('gradient_maxdet32', 'maxdet', 'gradient', 512,
                     'src/qgeognn_al/active_learning_v2/maxdet_study.py', 'gradient-maxdet-all-current-U-ridge1'),
    NumericalProfile('kernel_ivr32', 'ivr', 'gradient', 512,
                     'src/qgeognn_al/active_learning_v2/ivr_study.py', 'fixed-reference-universe-kernel-ivr'),
    NumericalProfile('numerical_hybrid32', 'hybrid', 'latent', 128,
                     'src/qgeognn_al/active_learning_v2/sequential_acquisition.py', 'current-K3-q50-top25-latent-kcenter'),
)
NUMERICAL_PROFILES = MappingProxyType({p.name: p for p in _PROFILES})


def numerical_profile(name: str) -> NumericalProfile:
    try:
        return NUMERICAL_PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError('unknown numerical profile: ' + str(name)) from None


def select_numerical_profile(
    snapshot: FeatureSnapshot, *, profile: str, outer_seed: int, round_index: int,
    original_candidate_ids=None, reference_ids=None, ensemble: K3EnsembleSnapshot | None = None,
    target_scales=None,
):
    """Run one frozen B32 selection, with zero-based round and explicit inputs.

    The complete real trajectory still requires matching per-round model fits
    and independently frozen fixtures; passing this dispatcher alone is not a
    claim of five-seed real-AL equivalence.
    """
    current = numerical_profile(profile)
    if type(outer_seed) is not int or outer_seed < 0 or type(round_index) is not int or round_index < 0:
        raise ValueError('nonnegative outer seed and zero-based round required')
    snapshot.verify()
    if len(snapshot.train_ids) != current.initial_labels + current.batch_size * round_index:
        raise ValueError('current L size differs from historical L333 + 32*round')
    if current.representation is not None and (snapshot.config.kind != current.representation or snapshot.config.dimension != current.dimension):
        raise ValueError('representation differs from the named scientific profile')
    if current.dimension == 512:
        details = json.loads(snapshot.extraction.details_json) if snapshot.extraction else json.loads(snapshot.config.parameters_json)
        if details.get('sketch_seed') != outer_seed + 4_000_037:
            raise ValueError('named gradient profile requires its pinned outer-seed CountSketch mapping')
    signals = None
    effective_seed = outer_seed
    if current.acquisition == 'random':
        if original_candidate_ids is None:
            raise ValueError('random32 requires the original U0, never a fresh current-U draw')
        effective_seed = historical_random_trajectory_seed(outer_seed)
        signals = {'profile': 'fixed_trajectory', 'original_candidate_ids': original_candidate_ids,
                   'round_index': round_index, 'trajectory_batch_size': 32}
    elif original_candidate_ids is not None:
        raise ValueError('original U0 permutation input belongs only to random32')
    if current.acquisition == 'ivr':
        if reference_ids is None:
            raise ValueError('kernel_ivr32 requires the exact fixed L0+U0 reference universe')
        signals = {'reference_ids': reference_ids}
    elif reference_ids is not None:
        raise ValueError('fixed reference input belongs only to kernel_ivr32')
    if current.acquisition == 'hybrid':
        if not isinstance(ensemble, K3EnsembleSnapshot) or target_scales is None:
            raise ValueError('numerical Hybrid32 requires current same-seed/same-L K3 and fixed L0 scales')
        signals = ensemble.numerical_signals(snapshot, outer_seed=outer_seed, target_scales=target_scales)
    elif ensemble is not None or target_scales is not None:
        raise ValueError('K3 numerical signals belong only to numerical_hybrid32')
    return select(snapshot, acquisition=current.acquisition, budget=current.batch_size,
                  seed=effective_seed, numerical_signals=signals)
