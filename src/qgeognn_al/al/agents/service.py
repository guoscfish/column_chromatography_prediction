"""Profile-aware selection with complete packets, bounded correction and receipts."""
from dataclasses import asdict
from pathlib import Path
import json
import math
from ...infra.contracts import content_hash
from .profiles import get_profile
from .prompts import prompt
from .replay_packets import build_messages, correction_messages
from .validation import validate_response, packet_candidate_ids, audit_no_cw, validate_screen_response
from .catalog import decode_table
from .transport import DurableTransport
from ..workflow.state import read_envelope, write_envelope


class AgentSelector:
    """Inject a protocol packet builder; the provider never receives workflow state.

    packet_supplier(public_context) must return packet, ordered pending_ids and
    screen fallback_ids. It receives only current L/U, already revealed L values,
    checkpoint identity and seed/round; no validation/test data or secret handle.
    A supplied packet is checked against that independent current authorization.
    """
    def __init__(self, profile, *, packet_supplier, packet_contract_hash, transport,
                 provider_config, directory, forbidden_output_roots=()):
        self.profile = get_profile(profile)
        self.packet_supplier = packet_supplier
        self.packet_contract_hash = packet_contract_hash
        self.transport,self.config = transport,provider_config
        self.root,self.forbidden = Path(directory),tuple(forbidden_output_roots)

    @property
    def contract_hash(self):
        return content_hash({'profile':asdict(self.profile),'prompt_sha256':content_hash(prompt(self.profile)),
            'packet_contract':self.packet_contract_hash,'provider_config':asdict(self.config),
            'transport_kind':self.transport.kind,'request_recovery':'started_only_fail_closed_v1'})

    def select(self, *, snapshot, state, budget, spec):
        if budget != self.profile.batch_size:
            raise ValueError('named historical agent profile requires B32')
        public = {'seed':spec.seed,'round':state['round'],'labeled_ids':list(snapshot.train_ids),
            'candidate_ids':list(snapshot.candidate_ids),'observed_labels':dict(state['labels']),
            'checkpoint_hash':snapshot.predictor.model_checkpoint_hash}
        supplied = self.packet_supplier(public)
        if set(supplied)-{'packet','pending_ids','fallback_ids','binding','premeasurement_predictions'}:
            raise ValueError('packet supplier returned undeclared capabilities')
        packet = supplied['packet']
        visible = packet_candidate_ids(self.profile,packet)
        pending = tuple(supplied.get('pending_ids',()))
        if self.profile.delivery == 'screened_128':
            pending = tuple(row['id'] for row in packet['pending_CW16'])
            if not set(visible+pending) <= set(snapshot.candidate_ids):
                raise ValueError('screen escaped the current authorized U')
        else:
            if set(visible+pending) != set(snapshot.candidate_ids):
                raise ValueError('full-pool profile must expose every authorized U row')
            observed = decode_table(packet,'observed')
            if set(row['id'] for row in observed) != set(snapshot.train_ids):
                raise ValueError('observed table escaped currently revealed L')
            if any([row['V1_ml'],row['V2_ml']] != list(state['labels'][row['id']]) for row in observed):
                raise ValueError('observed values differ from authorized measured feedback')
        if self.profile.delivery=='screened_128':
            if supplied.get('binding') != {'seed':spec.seed,'round':state['round'],'packet_sha256':packet['packet_sha256']}:
                raise ValueError('screen requires independent seed/round/packet binding')
            if packet['labeled_count'] != len(snapshot.train_ids):
                raise ValueError('screen budget differs from current L')
        elif packet.get('seed') != spec.seed or packet.get('round') != state['round']:
            raise ValueError('packet belongs to another seed or round')
        predictions=supplied.get('premeasurement_predictions',{})
        if predictions and (not set(predictions)<=set(snapshot.candidate_ids) or any(len(v)!=2 or not all(math.isfinite(float(x)) for x in v) for v in predictions.values())):
            raise ValueError('invalid premeasurement prediction authorization or values')
        if self.transport.paid and set(predictions)!=set(snapshot.candidate_ids):
            raise ValueError('online profile requires raw current-U predictions before dispatch')
        messages = build_messages(self.profile,packet)
        count = self.profile.selection_attempts
        if count is None:
            raise NotImplementedError('UNKNOWN historical correction quota; explicit source evidence required')
        durable = DurableTransport(self.transport,self.root/f'round_{state["round"]:04d}',
                                   forbidden_output_roots=self.forbidden,maximum_calls=count)
        provenances = []
        for turn in range(count):
            if not self.profile.cw_visible:
                audit_no_cw(messages)
            answer,provenance = durable.call(messages,self.config,turn=turn)
            provenances.append(provenance)
            if self.transport.paid:
                served=provenance.get('served_model')
                if not isinstance(served,str) or not served:
                    raise ValueError('provider provenance lacks served model')
                identity={'served_model':served,'provider_config_sha256':self.config.sha256}
                path=self.root/'provider_identity.json'
                if path.is_symlink():raise ValueError('served model identity symlink forbidden')
                if path.exists():
                    if {k:v for k,v in read_envelope(path)['payload'].items() if k!='schema_version'}!=identity:
                        raise ValueError('served model changed after first accepted provider response')
                else:write_envelope(path,identity,exclusive=True)
            try:
                chosen = validate_response(self.profile,answer,packet,
                    fallback_ids=supplied.get('fallback_ids',()),pending_ids=pending)
                break
            except ValueError as error:
                if turn+1==count:raise
                messages += correction_messages(self.profile,answer,error)
        if len(set(chosen)) != budget or not set(chosen) <= set(snapshot.candidate_ids):
            raise ValueError('profile selection escaped current authorization')
        if self.profile.delivery=='screened_128':
            accepted=validate_screen_response(answer,list(visible),list(supplied.get('fallback_ids',())),packet['packet_sha256'])
        else:accepted=json.loads(answer)
        predictions=supplied.get('premeasurement_predictions',{})
        if self.transport.paid and not set(chosen)<=set(predictions):
            raise ValueError('online scientific feedback requires raw premeasurement predictions')
        prediction_record=[{'candidate_id':i,'pred_V1_ml':predictions[i][0],'pred_V2_ml':predictions[i][1]} for i in chosen if i in predictions]
        return {'selected_ids':list(chosen),'acquisition':self.profile.name,'seed':spec.acquisition_seed,
            'budget':budget,'feature_hash':snapshot.sha256,
            'trace':[{'sample_id':i,'selection_order':n} for n,i in enumerate(chosen)],
            'audit':{'profile':self.profile.name,'packet_sha256':packet['packet_sha256'],
                     'provider_invoked':self.transport.paid,'response_provenance':provenances,
                     'accepted_raw_response':answer,'messages_sha256':content_hash(messages),
                     'agent_record':{'profile':self.profile.name,'pending_ids':list(pending),
                                     'accepted_response':accepted,'premeasurement_predictions':prediction_record},
                     'trajectory_equivalence':'NOT_VERIFIED'}}
