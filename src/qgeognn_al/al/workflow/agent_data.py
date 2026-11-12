"""Row predictor to frozen LLM protocol packets and measured-feedback memory.

Migrates main b1a6ed7a scientist_study.make_catalog, dialog_study._card/_observed,
scientist_study.advance, and llm_screen.make_packet. This builder does inference
and label-free feature extraction only. It never fits a model, calls a provider,
opens a source target table or exposes validation/test labels. Complete-history
replay qualification remains separate from implementation of this adapter.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math

import numpy as np

from ...infra.contracts import content_hash, sample_ids
from ...predictor.row import FEATURE_COLUMNS
from ..agents.catalog import CONDITIONS
from ..agents.profiles import get_profile
from ..agents.prompts import prompt
from ..agents.replay_packets import build_full_packet
from ..agents.screen import shortlist
from ..agents.validation import validate_packet
from ..acquisitions.lcmd import _squared_distances, lcmd_tp_select
from .state import read_envelope

SOURCE_COMMIT = 'b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'


class RowAgentPacketBuilder:
    def __init__(self, profile, dataset, backend, run_directory):
        self.profile, self.dataset, self.backend = get_profile(profile), dataset, backend
        self.root = Path(run_directory).absolute()
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError('agent history must not traverse symlinks')
        if set(dataset.feature_frame.columns) != set(FEATURE_COLUMNS):
            raise ValueError('agent data source must contain only the seven label-free feature columns')
        self._features = dataset.feature_frame.set_index('sample_id', drop=False)
        if self._features.index.duplicated().any():
            raise ValueError('agent feature IDs must be unique')
        self.initial_ids = sample_ids(dataset.l0_ids, name='agent original L0')
        self.original_candidates = sample_ids(dataset.u0_ids, name='agent original U0')
        if len(self.initial_ids) != 333 or set(self.initial_ids) & set(self.original_candidates):
            raise ValueError('named Row agent protocols require the frozen disjoint L333/U0 split')
        self._data_hash = content_hash(dataset.feature_frame.to_dict('records'))
        self._initial_contract = self.contract_hash

    @property
    def contract_hash(self):
        return content_hash({'builder':'main-row-card-feedback-v1','source_commit':SOURCE_COMMIT,
            'profile':asdict(self.profile),'dataset':self.dataset.source_sha256,
            'split':self.dataset.partition_hash,'graph_cache':self.dataset.graph_cache_sha256,
            'feature_rows':content_hash(self.dataset.feature_frame.to_dict('records')),
            'initial_ids':self.initial_ids,'original_U0':self.original_candidates,
            'backend':self.backend.contract_hash})

    def _read(self, name):
        path = self.root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('BLOCKED_MISSING_FIXTURE: missing or symlinked run-owned ledger: '+name)
        return read_envelope(path)

    def _card(self, identity, pred=None, percentile=None):
        row = self._features.loc[identity]
        result = {'id':identity,'smiles':str(row['canonical_smiles']),
            'conditions':{key:row[key].item() if hasattr(row[key],'item') else row[key] for key in CONDITIONS}}
        if pred is not None:
            result.update(pred_V1_ml=float(pred[1]),pred_V2_ml=float(pred[4]))
            result.update({f'pred_{target}_{quantile}_ml':float(pred[j]) for j,(target,quantile) in enumerate(
                (t,q) for t in ('V1','V2') for q in ('q10','q50','q90'))})
        if percentile is not None:
            result['cw_distance_percentile'] = round(float(percentile),5)
        return result

    def _history(self, public, scope):
        """Reconstruct only this run's already frozen-and-revealed training rows."""
        expected_l, expected_u = list(self.initial_ids), list(self.original_candidates)
        history, acquired = [], {}
        for r in range(public['round']):
            frozen_envelope = self._read(f'round_{r:04d}.freeze.json')
            frozen = frozen_envelope['payload']
            receipt = self._read(f'round_{r:04d}.labels.json')['payload']
            if (frozen.get('scope') != scope or receipt.get('scope') != scope
                    or frozen.get('outer_seed') != public['seed'] or frozen.get('round') != r
                    or frozen.get('acquisition') != self.profile.name
                    or receipt.get('freeze_hash') != frozen_envelope['sha256']
                    or frozen.get('labeled_ids_hash') != content_hash(expected_l)
                    or frozen.get('candidate_ids_hash') != content_hash(expected_u)):
                raise ValueError('agent feedback lineage identity/hash drift')
            ids = sample_ids(frozen['selected_ids'],name='historical agent batch')
            if len(ids)!=32 or not set(ids)<=set(expected_u) or set(receipt['labels'])!=set(ids):
                raise ValueError('agent history does not contain one complete authorized B32 reveal')
            for identity in ids:
                values = np.asarray(receipt['labels'][identity],dtype=float)
                if values.shape!=(2,) or not np.isfinite(values).all() or list(values)!=list(public['observed_labels'][identity]):
                    raise ValueError('agent history labels differ from current authorized L values')
            if self.profile.delivery!='screened_128':
                record = frozen.get('audit',{}).get('agent_record')
                if not isinstance(record,dict) or record.get('profile')!=self.profile.name:
                    raise ValueError('BLOCKED_MISSING_FIXTURE: frozen accepted response and premeasurement predictions absent')
                pending = sample_ids(record['pending_ids'],name='frozen pending')
                if len(pending)!=self.profile.pending_count or tuple(ids[:len(pending)])!=pending:
                    raise ValueError('historical pending-ID order differs from frozen profile')
                plan, predictions = record['accepted_response'],record['premeasurement_predictions']
                if [p['candidate_id'] for p in predictions]!=list(ids):
                    raise ValueError('premeasurement predictions are not aligned with the frozen batch')
                choices = {c['id']:c for c in plan['choices']}
                if tuple(c['id'] for c in plan['choices'])!=tuple(ids[len(pending):]):
                    raise ValueError('accepted response order differs from frozen LLM choices')
                records=[]
                for identity,pred in zip(ids,predictions):
                    if set(pred)!={'candidate_id','pred_V1_ml','pred_V2_ml'}:
                        raise ValueError('unauthorized premeasurement prediction fields')
                    true = receipt['labels'][identity]
                    if not all(math.isfinite(float(pred[f'pred_{t}_ml'])) for t in ('V1','V2')):
                        raise ValueError('nonfinite historical premeasurement prediction')
                    choice=choices.get(identity,{})
                    item={**pred,'source':'CW' if identity in pending else 'LLM',
                        'true_V1_ml':float(true[0]),'true_V2_ml':float(true[1]),
                        'error_V1_ml':pred['pred_V1_ml']-float(true[0]),
                        'error_V2_ml':pred['pred_V2_ml']-float(true[1]),
                        'reason':choice.get('reason','pending geometric coverage'),
                        'hypothesis_id':choice.get('hypothesis_id')}
                    records.append(item);acquired[identity]=item
                history.append({'seed':public['seed'],'method':self.profile.trajectory_method,'round':r,
                    'batch_freeze_sha256':frozen_envelope['sha256'],'records':records,
                    **{key:plan[key] for key in ('hypotheses','feedback_interpretation','batch_rationale')}})
            expected_l+=list(ids);expected_u=[i for i in expected_u if i not in set(ids)]
        if expected_l!=public['labeled_ids'] or expected_u!=public['candidate_ids']:
            raise ValueError('agent current L/U differ from the ordered frozen/revealed trajectory')
        return history,acquired

    def _model(self, public):
        receipt = self._read(f'fit_{public["round"]:04d}.json')['payload']
        fit = receipt['fit']
        if (fit['checkpoint_hash']!=public['checkpoint_hash'] or fit['seed']!=public['seed']
                or receipt['labeled_ids_hash']!=content_hash(public['labeled_ids'])
                or fit['labeled_ids_hash']!=content_hash(public['labeled_ids'])):
            raise ValueError('agent current model does not match the run-owned fit receipt')
        return self.backend._model(fit),receipt['scope']

    def _predict_current(self, model, candidate_ids):
        from ...predictor.inference import predict_outputs
        from ..representations.latent import predictor_state_hash
        before=predictor_state_hash(model)
        atom,angle=self.backend._graphs(candidate_ids)
        values=predict_outputs(model,atom,angle,list(range(len(candidate_ids))))
        if predictor_state_hash(model)!=before:
            raise ValueError('agent prediction changed the frozen predictor')
        return values

    def _feature_banks(self, model, public):
        from ..representations import Authorization
        from ..representations.gradient import extract_gradient
        from ..representations.latent import predictor_state_hash
        from ...infra.contracts import BaselineProvenance
        authorization=Authorization(tuple(public['labeled_ids']),tuple(public['candidate_ids']))
        rows=[self.backend.features[i] for i in authorization.train_ids+authorization.candidate_ids]
        provenance=BaselineProvenance(**{**self.backend.provenance.to_dict(),
                                        'model_checkpoint_hash':public['checkpoint_hash']})
        kwargs=dict(expected_state_hash=predictor_state_hash(model),authorization=authorization,rows=rows,
                    provenance=provenance,sketch_seed=public['seed']+4_000_037)
        cw=extract_gradient(model,**kwargs,kind='center_width',
                            l0_targets=self.backend.l0_targets,l0_ids=self.backend.l0_ids)
        cw_matrix=np.vstack((cw.train,cw.candidates))
        raw=None
        if self.profile.delivery=='screened_128':
            scales=tuple(self.backend.preprocessing['target_scales'][t] for t in ('V1','V2'))
            gradient=extract_gradient(model,**kwargs,kind='gradient',scales=scales)
            raw=np.vstack((gradient.train,gradient.candidates))
        return cw_matrix,raw

    def _screen_packet(self, public, predictions, cw, raw):
        from rdkit import Chem,DataStructs
        from rdkit.Chem import Descriptors,rdFingerprintGenerator
        import pandas as pd
        n=len(public['labeled_ids']);current=public['labeled_ids']+public['candidate_ids']
        proposal=shortlist(cw,raw,n,public['seed'],public['round'])
        positions=proposal['pending_positions']+proposal['candidate_positions']
        percentile=pd.Series(np.sqrt(_squared_distances(cw[n:],cw[:n]).min(axis=1))).rank(method='average',pct=True).to_numpy()
        smiles=[str(self._features.loc[i]['canonical_smiles']) for i in current]
        mols={s:Chem.MolFromSmiles(s) for s in dict.fromkeys(smiles)}
        if any(m is None for m in mols.values()):raise ValueError('invalid SMILES')
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        fps={s:generator.GetFingerprint(m) for s,m in mols.items()}
        labeled_smiles=smiles[:n];labeled_fps=[fps[s] for s in dict.fromkeys(labeled_smiles)]
        cards=[]
        for position in positions:
            identity=current[position];card=self._card(identity);s=card['smiles'];mol=mols[s];pred=predictions[position-n]
            card.update(descriptors={'MW':round(Descriptors.MolWt(mol),2),'LogP':round(Descriptors.MolLogP(mol),2),
                'TPSA':round(Descriptors.TPSA(mol),2),'HBD':Descriptors.NumHDonors(mol),'HBA':Descriptors.NumHAcceptors(mol)},
                pred_V1_V2=[round(float(pred[1]),3),round(float(pred[4]),3)],
                cw_distance_percentile=round(float(percentile[position-n]),4),
                max_Tanimoto_to_Lt=round(max(DataStructs.BulkTanimotoSimilarity(fps[s],labeled_fps)),4),
                same_molecule_rows_in_Lt=labeled_smiles.count(s))
            cards.append(card)
        packet={'prompt':prompt(self.profile),'labeled_count':n,'pending_CW16':cards[:16],'candidates':cards[16:],
                'units':{'pred_V1_V2':'mL','MW':'g/mol','TPSA':'angstrom^2'},
                'observed_outcome_values_in_packet':0,'validation_or_test_records_in_packet':0}
        # Historical screen response binds the exact pretty-printed packet file.
        import hashlib,json
        packet['packet_sha256']=hashlib.sha256((json.dumps(packet,indent=2)+'\n').encode()).hexdigest()
        validate_packet(self.profile,packet)
        return {'packet':packet,'pending_ids':[current[i] for i in proposal['pending_positions']],
                'fallback_ids':[current[i] for i in proposal['fallback_positions']],
                'binding':{'seed':public['seed'],'round':public['round'],'packet_sha256':packet['packet_sha256']}}

    def __call__(self, public):
        if set(public)!={'seed','round','labeled_ids','candidate_ids','observed_labels','checkpoint_hash'}:
            raise ValueError('agent builder accepts only restricted public current-round context')
        if self.contract_hash!=self._initial_contract or content_hash(self.dataset.feature_frame.to_dict('records'))!=self._data_hash:
            raise ValueError('agent feature/profile/backend contract changed')
        if type(public['round']) is not int or public['round']<0 or public['seed']!=self.dataset.seed:
            raise ValueError('agent outer seed or round mismatch')
        labeled=sample_ids(public['labeled_ids'],name='agent measured L');candidates=sample_ids(public['candidate_ids'],name='agent current U')
        if (len(labeled)!=333+32*public['round'] or set(public['observed_labels'])!=set(labeled)
                or set(labeled)&set(candidates) or set(labeled)|set(candidates)!=set(self.initial_ids+self.original_candidates)):
            raise ValueError('agent context escaped the authorized Row L/U split')
        for identity in labeled:
            values=np.asarray(public['observed_labels'][identity],dtype=float)
            if values.shape!=(2,) or not np.isfinite(values).all():raise ValueError('finite two-endpoint observed values required')
        model,scope=self._model(public)
        history,acquired=self._history(public,scope)
        predictions=np.asarray(self._predict_current(model,candidates))
        if predictions.shape!=(len(candidates),6) or not np.isfinite(predictions).all():
            raise ValueError('current candidate predictions must have six finite outputs in supplied ID order')
        cw,raw=self._feature_banks(model,public);n=len(labeled)
        if np.asarray(cw).shape!=(len(labeled)+len(candidates),512) or not np.isfinite(cw).all():
            raise ValueError('current CW bank identity/dimension mismatch')
        if self.profile.delivery=='screened_128':
            supplied=self._screen_packet(public,predictions,cw,raw)
        else:
            import pandas as pd
            percentile=pd.Series(np.sqrt(_squared_distances(cw[n:],cw[:n]).min(axis=1))).rank(pct=True).to_numpy()
            pending_positions=(lcmd_tp_select(cw[n:],cw[:n],16).selected_pool_positions.tolist()
                               if self.profile.pending_count else [])
            pending=[candidates[p] for p in pending_positions]
            cards=[self._card(identity,pred,distance) for identity,pred,distance in zip(candidates,predictions,percentile)]
            lookup={row['id']:row for row in cards};observed=[]
            for identity in labeled:
                values=public['observed_labels'][identity];card=self._card(identity)
                card.update(V1_ml=float(values[0]),V2_ml=float(values[1]),
                    record_kind='acquired_with_premeasurement_prediction' if identity in acquired else 'initial_L333')
                if identity in acquired:
                    prior=acquired[identity]
                    card.update({f'premeasurement_{kind}_{target}_ml':prior[f'{source}_{target}_ml']
                        for kind,source in [('prediction','pred'),('error','error')] for target in ('V1','V2')})
                observed.append(card)
            catalog={'candidates':[row for row in cards if row['id'] not in set(pending)],
                     'observed':observed,'pending':[lookup[i] for i in pending],'salt':'overridden-by-frozen-profile'}
            packet=build_full_packet(self.profile,catalog,seed=public['seed'],round_index=public['round'],
                history=history,candidate_ids=candidates,observed_ids=labeled)
            supplied={'packet':packet,'pending_ids':pending,'fallback_ids':[]}
        supplied['premeasurement_predictions']={identity:[float(pred[1]),float(pred[4])] for identity,pred in zip(candidates,predictions)}
        return supplied
