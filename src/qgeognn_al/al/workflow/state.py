"""One seed-scoped AL state machine; owns labels, persistence and backend calls."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
from typing import Callable, Mapping, Protocol

import numpy as np
from ...infra.contracts import canonical_bytes, content_hash, sample_ids, budget_value
from ..representations import Authorization, FeatureSnapshot
from ..acquisitions import select
from ..agents import validate_plan


def write_envelope(path: Path, payload, *, exclusive=False):
    if payload.get('schema_version', 2) != 2:
        raise ValueError('unsupported workflow artifact schema')
    payload = {'schema_version': 2, **payload}
    envelope = {'sha256': content_hash(payload), 'payload': payload}
    path = Path(path)
    if exclusive:
        with path.open('xb') as stream:
            stream.write(canonical_bytes(envelope) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            stream.write(canonical_bytes(envelope) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    return envelope


def read_envelope(path: Path):
    envelope = json.loads(Path(path).read_bytes())
    if (set(envelope) != {'sha256', 'payload'} or envelope['payload'].get('schema_version') != 2
            or content_hash(envelope['payload']) != envelope['sha256']):
        raise ValueError('persisted artifact hash mismatch')
    return envelope


def label_values(values: Mapping[str, object], expected):
    if set(values) != set(expected):
        raise ValueError('label receipt must contain exactly the authorized IDs')
    result = {}
    for identity in expected:
        row = np.asarray(values[identity], dtype=float)
        if row.shape != (2,) or not np.isfinite(row).all():
            raise ValueError('two finite target values are required')
        result[identity] = row.tolist()
    return result


@dataclass(frozen=True)
class RunSpec:
    seed: int
    acquisition_seed: int
    acquisition: str
    representation: str
    batch_size: int
    target_budget: int
    initial_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    dataset_hash: str
    split_hash: str
    initial_state_hash: str
    protocol_hash: str
    random_profile: str = 'per_round_choice'
    reference_ids: tuple[str, ...] = ()
    execution_mode: str = 'offline_synthetic'

    def __post_init__(self):
        for key in ['seed', 'acquisition_seed']:
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError('seed must be a nonnegative integer')
        for key in ['initial_ids', 'candidate_ids', 'validation_ids']:
            object.__setattr__(self, key, sample_ids(getattr(self, key), name=key))
        groups = [set(self.initial_ids), set(self.candidate_ids), set(self.validation_ids)]
        if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('L0, U0 and validation must be nonempty and disjoint')
        reference = sample_ids(self.reference_ids or self.initial_ids + self.candidate_ids, name='fixed reference universe')
        if set(reference) != groups[0] | groups[1]:
            raise ValueError('fixed reference universe must equal L0 union U0')
        object.__setattr__(self, 'reference_ids', reference)
        budget_value(self.batch_size, len(self.candidate_ids))
        if type(self.target_budget) is not int or not len(self.initial_ids) < self.target_budget <= len(self.initial_ids) + len(self.candidate_ids):
            raise ValueError('invalid target label budget')
        for name in ['dataset_hash', 'split_hash', 'initial_state_hash', 'protocol_hash']:
            if len(getattr(self, name)) != 64 or any(c not in '0123456789abcdef' for c in getattr(self, name)):
                raise ValueError(name + ' must be SHA-256')
        if self.random_profile not in {'per_round_choice', 'fixed_trajectory'}:
            raise ValueError('random profile must be explicit')
        if self.execution_mode not in {'offline_synthetic', 'frozen_replay', 'authorized_scientific_run'}:
            raise ValueError('unknown execution mode')

    @property
    def scope(self):
        return content_hash(asdict(self))


class Backend(Protocol):
    """A workflow capability, never passed to an acquisition or agent."""
    kind: str
    initial_state_hash: str
    contract_hash: str
    def fit(self, *, seed, labeled_ids, labels, validation_ids, validation_labels): ...
    def represent(self, *, fit, authorization: Authorization) -> FeatureSnapshot: ...
    def validate(self, *, fit, validation_ids, validation_labels): ...


class LabelBroker:
    def __init__(self, *, spec: RunSpec, initial_labels, validation_labels, reveal: Callable | None):
        self.scope = spec.scope
        self.kind = spec.execution_mode
        self.initial = label_values(initial_labels, spec.initial_ids)
        self.validation = label_values(validation_labels, spec.validation_ids)
        self.candidates = set(spec.candidate_ids)
        self.callback = reveal
        self._contract = self.contract_hash

    @property
    def contract_hash(self):
        return content_hash({'scope': self.scope, 'initial': self.initial, 'validation': self.validation})

    def verify(self):
        if self.contract_hash != self._contract:
            raise ValueError('initial/validation label capability changed')

    def reveal(self, *, ids, scope, freeze_hash):
        self.verify()
        if scope != self.scope or len(set(ids)) != len(ids) or not set(ids) <= self.candidates:
            raise ValueError('reveal escaped seed-scoped candidate authorization')
        if self.callback is None:
            raise PermissionError('no label reveal capability was authorized')
        return label_values(self.callback(tuple(ids), freeze_hash), ids)


class AmbiguousFit(RuntimeError):
    """A fit started without a receipt; never silently retrain on resume."""


class AmbiguousReveal(RuntimeError):
    """A reveal started without a durable receipt; automatic repetition is unsafe."""


class RoundWorkflow:
    def __init__(self, directory: Path, *, spec: RunSpec, backend: Backend, labels: LabelBroker, forbidden_output_roots=()):
        self.root, self.spec, self.backend, self.labels = Path(directory), spec, backend, labels
        absolute = self.root.absolute()
        if any(p.is_symlink() for p in [absolute, *absolute.parents]):
            raise ValueError('workflow output symlinks are forbidden')
        resolved = absolute.resolve()
        if any(resolved == Path(p).resolve() or Path(p).resolve() in resolved.parents for p in forbidden_output_roots):
            raise ValueError('workflow cannot write into a protected runtime tree')
        if labels.scope != spec.scope or backend.initial_state_hash != spec.initial_state_hash:
            raise ValueError('backend or labels belong to a different scientific identity')
        allowed = {'offline_synthetic': 'synthetic', 'frozen_replay': 'replay', 'authorized_scientific_run': 'qgeognn'}
        if backend.kind != allowed[spec.execution_mode]:
            raise PermissionError('backend capability is not allowed for this execution mode')
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'state.json'
        if path.exists():
            self.state = self._load(path)
        else:
            self.state = {'schema_version': 2, 'scope': spec.scope, 'backend_contract_hash': backend.contract_hash,
                          'label_contract_hash': labels.contract_hash, 'round': 0, 'phase': 'fit_pending',
                          'labeled_ids': list(spec.initial_ids), 'candidate_ids': list(spec.candidate_ids),
                          'labels': label_values(labels.initial, spec.initial_ids), 'fit': None, 'validation': None}
            self._save()
        self._validate_state()

    def _load(self, path):
        payload = read_envelope(path)['payload']
        if payload.get('scope') != self.spec.scope:
            raise ValueError('artifact belongs to another seed, split or protocol')
        return payload

    def _save(self):
        write_envelope(self.root / 'state.json', self.state)

    def _validate_state(self):
        self.labels.verify()
        if self.state.get('backend_contract_hash') != self.backend.contract_hash or self.state.get('label_contract_hash') != self.labels.contract_hash:
            raise ValueError('backend recipe or initial/validation labels changed on resume')
        labeled = sample_ids(self.state['labeled_ids'], name='state L')
        candidates = sample_ids(self.state['candidate_ids'], name='state U')
        universe = set(self.spec.initial_ids) | set(self.spec.candidate_ids)
        if set(labeled) & set(candidates) or set(labeled) | set(candidates) != universe:
            raise ValueError('persisted L/U partition drift')
        if not set(self.spec.initial_ids) <= set(labeled) or len(labeled) > self.spec.target_budget:
            raise ValueError('persisted label budget drift')
        label_values(self.state['labels'], labeled)
        if set(self.spec.validation_ids) & universe:
            raise ValueError('validation identity leaked into acquisition universe')
        # Reconstruct every authorized label transition from the durable ledger.
        # A rehashed state alone cannot move an unobserved U row into L.
        expected_l = list(self.spec.initial_ids)
        expected_u = list(self.spec.candidate_ids)
        expected_labels = label_values(self.labels.initial, self.spec.initial_ids)
        if type(self.state['round']) is not int or not 0 <= self.state['round'] <= len(self.spec.candidate_ids):
            raise ValueError('invalid persisted round')
        for number in range(self.state['round']):
            frozen_path = self.root / f'round_{number:04d}.freeze.json'
            receipt_path = self.root / f'round_{number:04d}.labels.json'
            if not frozen_path.is_file() or not receipt_path.is_file():
                raise ValueError('label lineage lacks frozen selection or reveal receipt')
            frozen_envelope = read_envelope(frozen_path); frozen = self._load(frozen_path)
            if (frozen['round'] != number or frozen['outer_seed'] != self.spec.seed
                    or frozen['labeled_ids_hash'] != content_hash(expected_l)
                    or frozen['candidate_ids_hash'] != content_hash(expected_u)
                    or frozen['budget'] != min(self.spec.batch_size, self.spec.target_budget - len(expected_l))):
                raise ValueError('frozen label lineage identity drift')
            chosen = validate_plan({'selected_ids': frozen['selected_ids']}, candidate_ids=expected_u, budget=frozen['budget'])
            receipt = self._load(receipt_path)
            if receipt['freeze_hash'] != frozen_envelope['sha256']:
                raise ValueError('revealed label lineage is not bound to its freeze')
            expected_labels.update(label_values(receipt['labels'], chosen))
            expected_l += list(chosen)
            expected_u = [value for value in expected_u if value not in set(chosen)]
        if list(labeled) != expected_l or list(candidates) != expected_u or self.state['labels'] != expected_labels:
            raise ValueError('state L/U/labels do not match the freeze/reveal lineage')

    def _fit_context_hash(self):
        return content_hash({'scope': self.spec.scope, 'seed': self.spec.seed,
                             'backend': self.backend.contract_hash, 'labeled_ids': self.state['labeled_ids'],
                             'labels': self.state['labels'], 'validation_ids': self.spec.validation_ids,
                             'validation_labels': self.labels.validation})

    def _fit(self):
        self._validate_state()
        receipt_path = self.root / f'fit_{self.state["round"]:04d}.json'
        marker = self.root / f'fit_{self.state["round"]:04d}.started.json'
        if receipt_path.exists():
            receipt = self._load(receipt_path)
            if receipt['labeled_ids_hash'] != content_hash(self.state['labeled_ids']) or receipt['fit_context_hash'] != self._fit_context_hash():
                raise ValueError('fit receipt label identity drift')
            fit = receipt['fit']
        else:
            if marker.exists():
                raise AmbiguousFit('fit started without receipt; explicit reconciliation required')
            write_envelope(marker, {'scope': self.spec.scope, 'labeled_ids_hash': content_hash(self.state['labeled_ids'])}, exclusive=True)
            fit = self.backend.fit(seed=self.spec.seed, labeled_ids=tuple(self.state['labeled_ids']),
                                   labels=label_values(self.state['labels'], self.state['labeled_ids']), validation_ids=self.spec.validation_ids,
                                   validation_labels=label_values(self.labels.validation, self.spec.validation_ids))
            write_envelope(receipt_path, {'scope': self.spec.scope, 'labeled_ids_hash': content_hash(self.state['labeled_ids']), 'fit_context_hash': self._fit_context_hash(), 'fit': fit}, exclusive=True)
        required = {'checkpoint_hash', 'initial_state_hash', 'labeled_ids_hash', 'seed'}
        if not required <= set(fit):
            raise ValueError('fit receipt missing scientific identity')
        if (fit['seed'] != self.spec.seed or fit['initial_state_hash'] != self.spec.initial_state_hash
                or fit['labeled_ids_hash'] != content_hash(self.state['labeled_ids'])):
            raise ValueError('fit receipt escaped initialization, labels or seed')
        self.state.update(fit=fit, phase='validate_pending')
        self._save()

    def _validate(self):
        values = self.backend.validate(fit=self.state['fit'], validation_ids=self.spec.validation_ids,
                                       validation_labels=label_values(self.labels.validation, self.spec.validation_ids))
        if not isinstance(values, Mapping) or not values or any(not np.isfinite(float(v)) for v in values.values()):
            raise ValueError('validation feedback must be finite numeric aggregates')
        self.state.update(validation=dict(values), phase='feedback_pending')
        self._save()

    def _feedback(self):
        payload = {'schema_version': 2, 'scope': self.spec.scope, 'round': self.state['round'], 'event': 'validated_feedback',
                   'seed': self.spec.seed, 'budget': len(self.state['labeled_ids']),
                   'checkpoint_hash': self.state['fit']['checkpoint_hash'], 'validation': self.state['validation']}
        path = self.root / f'feedback_{self.state["round"]:04d}.json'
        if path.exists():
            if self._load(path) != payload:
                raise ValueError('feedback identity changed on resume')
        else:
            write_envelope(path, payload, exclusive=True)
        self.state['phase'] = 'complete' if len(self.state['labeled_ids']) == self.spec.target_budget else 'ready'
        self._save()

    def _freeze(self, *, numerical_signals=None, agent_plan=None):
        count = min(self.spec.batch_size, self.spec.target_budget - len(self.state['labeled_ids']))
        authorization = Authorization(tuple(self.state['labeled_ids']), tuple(self.state['candidate_ids']))
        snapshot = self.backend.represent(fit=self.state['fit'], authorization=authorization)
        if (snapshot.authorization != authorization or snapshot.predictor.model_checkpoint_hash != self.state['fit']['checkpoint_hash']
                or snapshot.config.kind != self.spec.representation or snapshot.predictor.dataset_hash != self.spec.dataset_hash
                or snapshot.predictor.split_hash != self.spec.split_hash or snapshot.predictor.protocol_hash != self.spec.protocol_hash):
            raise ValueError('representation is not bound to current fit and L/U identities')
        if numerical_signals is None and hasattr(self.backend, 'numerical_signals'):
            numerical_signals = self.backend.numerical_signals(fit=self.state['fit'], authorization=authorization, acquisition=self.spec.acquisition)
        if self.spec.acquisition == 'random' and self.spec.random_profile == 'fixed_trajectory':
            numerical_signals = {'profile': 'fixed_trajectory', 'original_candidate_ids': self.spec.candidate_ids,
                                 'round_index': self.state['round'], 'trajectory_batch_size': self.spec.batch_size}
        if self.spec.acquisition == 'ivr':
            numerical_signals = dict(numerical_signals or {})
            if 'reference_ids' in numerical_signals and tuple(numerical_signals['reference_ids']) != self.spec.reference_ids:
                raise ValueError('fixed reference universe order changed')
            numerical_signals['reference_ids'] = self.spec.reference_ids
        if agent_plan is None:
            selection = select(snapshot, acquisition=self.spec.acquisition, budget=count,
                               seed=self.spec.acquisition_seed, numerical_signals=numerical_signals).to_dict()
        else:
            ids = validate_plan(agent_plan, candidate_ids=authorization.candidate_ids, budget=count)
            selection = {'selected_ids': list(ids), 'acquisition': 'recorded_agent_plan', 'seed': self.spec.acquisition_seed,
                         'budget': count, 'feature_hash': snapshot.sha256,
                         'trace': [{'sample_id': value, 'selection_order': i} for i, value in enumerate(ids)],
                         'audit': {'provider_invoked': False}}
        chosen = selection['selected_ids']
        if len(set(chosen)) != count or not set(chosen) <= set(self.state['candidate_ids']):
            raise ValueError('acquisition violated the authorized budget or candidate set')
        payload = {'schema_version': 2, 'scope': self.spec.scope, 'event': 'selection_frozen', 'round': self.state['round'],
                   'outer_seed': self.spec.seed, 'checkpoint_hash': self.state['fit']['checkpoint_hash'],
                   'labeled_ids_hash': content_hash(self.state['labeled_ids']),
                   'candidate_ids_hash': content_hash(self.state['candidate_ids']),
                   'representation_provenance': snapshot.provenance(), **selection}
        path = self.root / f'round_{self.state["round"]:04d}.freeze.json'
        if path.exists():
            if self._load(path) != payload:
                raise ValueError('selection changed after durable freeze')
        else:
            write_envelope(path, payload, exclusive=True)
        self.state['freeze_hash'] = content_hash(payload)
        self.state['phase'] = 'frozen'; self._save()

    def _reveal(self):
        path = self.root / f'round_{self.state["round"]:04d}.freeze.json'
        envelope = read_envelope(path); frozen = self._load(path)
        if envelope['sha256'] != self.state['freeze_hash']:
            raise ValueError('durable freeze changed after selection')
        if frozen['budget'] != min(self.spec.batch_size, self.spec.target_budget - len(self.state['labeled_ids'])):
            raise ValueError('frozen budget drift')
        if frozen['checkpoint_hash'] != self.state['fit']['checkpoint_hash'] or frozen['labeled_ids_hash'] != content_hash(self.state['labeled_ids']) or frozen['candidate_ids_hash'] != content_hash(self.state['candidate_ids']):
            raise ValueError('frozen selection identity drift')
        ids = validate_plan({'selected_ids': frozen['selected_ids']}, candidate_ids=self.state['candidate_ids'], budget=frozen['budget'])
        marker = self.root / f'round_{self.state["round"]:04d}.reveal_started.json'
        receipt_path = self.root / f'round_{self.state["round"]:04d}.labels.json'
        if receipt_path.exists():
            receipt = self._load(receipt_path)
            if receipt['freeze_hash'] != envelope['sha256']:
                raise ValueError('label receipt does not match frozen IDs')
            values = label_values(receipt['labels'], ids)
        else:
            if marker.exists():
                raise AmbiguousReveal('reveal started without a receipt; explicit reconciliation required')
            write_envelope(marker, {'scope': self.spec.scope, 'freeze_hash': envelope['sha256']}, exclusive=True)
            values = self.labels.reveal(ids=ids, scope=self.spec.scope, freeze_hash=envelope['sha256'])
            write_envelope(receipt_path, {'scope': self.spec.scope, 'freeze_hash': envelope['sha256'], 'labels': values}, exclusive=True)
        labeled = self.state['labeled_ids'] + list(ids)
        candidates = [value for value in self.state['candidate_ids'] if value not in set(ids)]
        self.state.update(labeled_ids=labeled, candidate_ids=candidates,
                          labels={**self.state['labels'], **values}, round=self.state['round'] + 1,
                          fit=None, validation=None, phase='fit_pending')
        self._validate_state(); self._save()

    def advance(self, *, numerical_signals=None, agent_plan=None):
        self._validate_state()
        phase = self.state['phase']
        if phase == 'fit_pending': self._fit()
        elif phase == 'validate_pending': self._validate()
        elif phase == 'feedback_pending': self._feedback()
        elif phase == 'ready': self._freeze(numerical_signals=numerical_signals, agent_plan=agent_plan)
        elif phase == 'frozen': self._reveal()
        elif phase != 'complete': raise ValueError('unknown workflow phase')
        return self.state['phase']

    def run(self, *, maximum_steps=10000):
        for _ in range(maximum_steps):
            if self.state['phase'] == 'complete':
                self._validate_state()
                return self.state
            self.advance()
        raise RuntimeError('workflow step budget exhausted')
