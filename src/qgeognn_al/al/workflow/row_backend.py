"""Real Row adapter using the same model and same-initialization fit recipe.

Execution requires the caller's separate training/label/provider permissions.
This path is implemented, but new real training and K3 trajectories have not
been measured in the migration task.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .backends import QGeoGNNBackend
from ...infra.contracts import content_hash
from ...infra.resources import SnapshotFile
from ...predictor.row_fit import (initialize_row_model, initialization_seed,
    row_training_config, fit_row_from_initialization)


class QGeoGNNRowBackend(QGeoGNNBackend):
    def __init__(self, *, prepared, representation, provenance, output_root,
                 l0_targets, allow_training=False, forbidden_output_roots=(), source_guard=None):
        if provenance.dataset_hash != prepared.dataset_hash or provenance.split_hash != prepared.split_hash:
            raise ValueError('prepared Row data/split provenance mismatch')
        targets=np.asarray(l0_targets,dtype=float)
        if targets.shape != (len(prepared.l0_ids),2) or not np.isfinite(targets).all():
            raise ValueError('Row L0 targets must align with prepared L0 IDs')
        model, initial_hash = initialize_row_model(prepared.normalization, prepared.outer_seed)
        config = row_training_config(prepared.outer_seed, prepared.split_hash)
        super().__init__(normalization=prepared.normalization,initial_state=model.state_dict(),
            initial_state_hash=initial_hash,features=prepared.rows,preprocessing=prepared.preprocessing,
            training_config=config,representation=representation,provenance=provenance,output_root=output_root,
            allow_training=allow_training,l0_ids=prepared.l0_ids,l0_targets=l0_targets,
            forbidden_output_roots=forbidden_output_roots)
        self.outer_seed = prepared.outer_seed
        self.validation_ids = tuple(prepared.validation_ids)
        self.graph_cache_hash = prepared.graph_cache_hash
        self._frozen_graph_hash = self._graph_identity()
        self.source_guard=source_guard
        self._source_identity=None if source_guard is None else content_hash(source_guard())
        self.forbidden_output_roots = tuple(forbidden_output_roots)
        self._ordered_ids = tuple(self.features)
        self._positions = {i:n for n,i in enumerate(self._ordered_ids)}

    def _graph_identity(self):
        from ..representations.latent import _graph_hash
        rows = list(self.features.values())
        return _graph_hash([r.atom for r in rows],[r.angle for r in rows])

    @property
    def contract_hash(self):
        if self.source_guard is not None and content_hash(self.source_guard())!=self._source_identity:
            raise ValueError('scientific source identity changed during run')
        if self._graph_identity() != self._frozen_graph_hash:
            raise ValueError('prepared graph content changed after Row backend creation')
        return content_hash({'recipe':super().contract_hash,'graph_cache_hash':self.graph_cache_hash,
                             'prepared_graph_hash':self._frozen_graph_hash,'validation_ids':self.validation_ids,'source_identity':self._source_identity})

    def _fit_member(self, *, member, seed, labeled_ids, labels, validation_ids, validation_labels, allow_new_fit=None):
        if tuple(validation_ids) != self.validation_ids:
            raise ValueError('Row validation identity/order differs from prepared split')
        if seed != self.outer_seed or set(labels) != set(labeled_ids) or set(validation_labels) != set(validation_ids):
            raise ValueError('Row fit identity or label alignment drift')
        if set(labeled_ids) & set(validation_ids):
            raise ValueError('training and validation overlap')
        config = row_training_config(seed,self.provenance.split_hash,member)
        runtime = self.root/f'member_{member}'/('L'+str(len(labeled_ids))+'_'+content_hash(list(labeled_ids))[:16])
        # Preserve canonical graph indices; only the fitter injects L/validation y.
        atom = [self.features[i].atom for i in self._ordered_ids]
        angle = [self.features[i].angle for i in self._ordered_ids]
        prediction_ids = tuple(labeled_ids)+tuple(validation_ids)
        audit = fit_row_from_initialization(atom_base=atom,angle=angle,normalization=self.normalization,
            preprocessing=self.preprocessing,train_indices=[self._positions[i] for i in labeled_ids],
            train_truth=np.asarray([labels[i] for i in labeled_ids]),
            validation_indices=[self._positions[i] for i in validation_ids],
            validation_truth=np.asarray([validation_labels[i] for i in validation_ids]),
            prediction_indices=[self._positions[i] for i in prediction_ids],prediction_sample_ids=prediction_ids,
            initialization_seed=initialization_seed(seed,member),training_config=config,
            contract={'train_sample_ids':list(labeled_ids),'validation_sample_ids':list(validation_ids),
                      'dataset_hash':self.provenance.dataset_hash,'protocol_hash':self.provenance.protocol_hash,
                      'graph_cache_and_feature_contract':self.contract_hash,'member':member},
            runtime=runtime,allow_training=self.allow_training if allow_new_fit is None else allow_new_fit,forbidden_output_roots=self.forbidden_output_roots)
        return {'seed':seed,'initial_state_hash':audit.get('public_initialization_hash',audit['initialization_hash']),
            'labeled_ids_hash':content_hash(list(labeled_ids)),'checkpoint_hash':audit['checkpoint_sha256'],
            'checkpoint_path':str(runtime/'best.pt'),'checkpoint_state_hash':audit.get('public_checkpoint_state_hash',audit['checkpoint_state_hash']),
            'effective_training_seed':config['seed'],'training_performed':True,'backend':'qgeognn_v2_row',
            'fit_audit_path':str(runtime/'fit_audit.json')}

    def fit(self, **context):
        return self._fit_member(member=0,**context)

    def recover_fit(self, **context):
        from .state import AmbiguousFit
        runtime=self.root/'member_0'/('L'+str(len(context['labeled_ids']))+'_'+content_hash(list(context['labeled_ids']))[:16])
        if not all((runtime/name).is_file() for name in ('best.pt','predictions.csv.gz','fit_audit.json')):
            raise AmbiguousFit('backend completion is not proven; no automatic retraining')
        # The fitter recomputes and verifies its complete input/output contract.
        # This call can only reuse a completed fit: it cannot optimize weights.
        return self._fit_member(member=0,allow_new_fit=False,**context)

    def ensemble(self, *, fit, state, validation_ids, validation_labels):
        """Explicit Hybrid profile stage: reuse member0, fit/reuse members1/2.

        No other profile invokes this. Each fit has its own permission gate,
        same current ordered L and validation, and immutable complete receipt.
        """
        from ..representations.ensemble import FrozenEnsembleMember, collect_checkpoint_ensemble, member_seeds
        from ..representations.latent import predictor_state_hash
        from ...predictor.model import load_predictor_checkpoint
        from ...predictor.inference import predict_outputs
        context = {'seed':self.outer_seed,'labeled_ids':tuple(state['labeled_ids']),'labels':state['labels'],
                   'validation_ids':validation_ids,'validation_labels':validation_labels}
        fits = [fit]+[self._fit_member(member=m,**context) for m in (1,2)]
        shared = content_hash({k:v for k,v in self.config.items() if k!='seed'})
        members = []
        for member,receipt in enumerate(fits):
            path = Path(receipt['checkpoint_path'])
            path.chmod(0o444)
            init_seed,train_seed = member_seeds(self.outer_seed,member)
            members.append(FrozenEnsembleMember(member=member,outer_seed=self.outer_seed,
                resource=SnapshotFile(path,receipt['checkpoint_hash'],self.root),
                checkpoint_state_hash=receipt['checkpoint_state_hash'],train_ids=tuple(state['labeled_ids']),
                candidate_ids=tuple(state['candidate_ids']),dataset_hash=self.provenance.dataset_hash,
                split_hash=self.provenance.split_hash,protocol_hash=self.provenance.protocol_hash,
                shared_fit_config_hash=shared,initialization_seed=init_seed,training_seed=train_seed,
                source_kind='authorized_snapshot'))
        def predict(model,ids):
            atom,angle = self._graphs(ids)
            return predict_outputs(model,atom,angle,list(range(len(ids)))),ids
        return collect_checkpoint_ensemble(members,load_predictor=load_predictor_checkpoint,
                                           predict=predict,state_hash=predictor_state_hash)
