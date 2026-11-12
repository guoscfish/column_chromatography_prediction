"""Configuration validation, authorized Row construction and read-only reporting."""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
from ...infra.contracts import content_hash, BaselineProvenance
from ...infra.execution import ExecutionGrant, snapshot_resource, runtime_path, protected_worktrees, scientific_source_identity
from .state import RunSpec, RoundWorkflow, LabelBroker, read_envelope, write_envelope, label_values


def configuration_plan(config):
    if config.get('schema_version')==2 and config.get('execution_mode')=='offline_synthetic':
        return {'status':'VALID_CONFIG','source_kind':'synthetic','profile':config['run']['acquisition'],
                'protocol_hash':content_hash(config),'paid_requests':0,'new_training':False}
    if config.get('schema_version')!=3 or config.get('execution_mode') not in ('qgeognn_row','frozen_replay'):
        raise ValueError('expected explicit schema3 qgeognn_row/frozen_replay config or schema2 synthetic')
    from ..numerical_profiles import NUMERICAL_PROFILES
    from ..agents.profiles import profile_names
    profile=config['profile']
    if profile not in NUMERICAL_PROFILES and profile not in profile_names():
        raise ValueError('unknown scientific profile identity')
    seed,target=config['outer_seed'],config['target_budget']
    if type(seed) is not int or seed<0 or type(target) is not int or not 333<target<=3330 or (target-333)%32:
        raise ValueError('Row profiles require explicit outer seed and target L333+32*k')
    if config['execution_mode']=='qgeognn_row':
        if profile in profile_names():
            from ..agents.transport import ProviderConfig
            ProviderConfig(**config.get('provider',{}))
        required={'source','partition','graph_cache','l0_labels','validation_labels','reveal_directory'}
        if set(config.get('resources',{}))!=required:
            raise ValueError('Row run needs five independent snapshot resources and authorized reveal directory')
        for name in required-{'reveal_directory'}:
            resource=config['resources'][name]
            if set(resource)!={'path','sha256','root'} or len(resource['sha256'])!=64 or any(c not in '0123456789abcdef' for c in resource['sha256']):
                raise ValueError('resource requires path, SHA256 and independent root: '+name)
    return {'status':'VALID_CONFIG','source_kind':config['execution_mode'],'profile':profile,
            'protocol_hash':content_hash(config),'resources_opened':False,'paid_requests':0,'new_training':False,
            'real_llm_packet_adapter':'IMPLEMENTED_UNQUALIFIED' if profile in profile_names() else 'not_applicable',
            'online_science':'ONLINE_SCIENCE_NOT_MEASURED'}


def _row_label_values(values,ids):
    import numpy as np
    checked=label_values(values,ids)
    converted={i:np.asarray(checked[i],dtype=np.float32).astype(float).tolist() for i in ids}
    return label_values(converted,ids)


def _label_snapshot(value,ids):
    resource=snapshot_resource(value)
    return _row_label_values(json.loads(resource.verify().read_text()),ids)


def build_row_workflow(config,output,authorization):
    plan=configuration_plan(config)
    grant=ExecutionGrant.read(authorization,config=config,output=output)
    grant.require('training','reveal')
    from ..numerical_profiles import numerical_profile,NUMERICAL_PROFILES
    is_agent=config['profile'] not in NUMERICAL_PROFILES
    if is_agent:grant.require('provider')
    source_identity=scientific_source_identity()
    from ...predictor.row import RowDataset
    from ...predictor.row_fit import sketch_seed
    from ..representations import RepresentationConfig
    from .row_backend import QGeoGNNRowBackend
    from .selectors import NumericalSelector
    resources=config['resources'];root=runtime_path(output);forbidden=protected_worktrees()
    code=subprocess.check_output(['git','--no-optional-locks','-C',str(Path(__file__).resolve().parents[4]),'rev-parse','HEAD'],text=True).strip()
    manifest={'schema_version':2,'protocol_hash':plan['protocol_hash'],'code_commit':code,'source_tree_sha256':source_identity['scientific_tree_sha256'],'output':str(root.resolve())}
    path=root/'run_manifest.json'
    if path.is_symlink():raise ValueError('run manifest may not be a symlink')
    if path.exists():
        if read_envelope(path)['payload']!=manifest:raise ValueError('run identity changed before preparation; existing artifacts preserved')
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('existing output has no matching run manifest')
        root.mkdir(parents=True,exist_ok=True);write_envelope(path,manifest,exclusive=True)
    dataset=RowDataset.from_files(**{k:snapshot_resource(resources[k]) for k in ('source','partition','graph_cache')})
    if dataset.seed!=config['outer_seed'] or len(dataset.l0_ids)!=333:
        raise ValueError('prepared partition does not match outer seed or initial L333')
    initial=_label_snapshot(resources['l0_labels'],dataset.l0_ids)
    validation=_label_snapshot(resources['validation_labels'],dataset.validation_ids)
    import tempfile
    scaler=root/'preprocessing.json'
    if scaler.is_symlink() or (scaler.exists() and scaler.stat().st_nlink!=1):
        raise ValueError('existing preprocessing must be an independent regular file')
    with tempfile.TemporaryDirectory(prefix='prepare-',dir=root) as temporary:
        candidate=Path(temporary)/'preprocessing.json'
        prepared=dataset.prepare([initial[i] for i in dataset.l0_ids],candidate,forbidden_output_roots=forbidden)
        data=candidate.read_bytes()
        if scaler.exists():
            if scaler.read_bytes()!=data:raise ValueError('preprocessing drift; existing scaler preserved')
        else:
            with scaler.open('xb') as stream:stream.write(data)

    profile=None if is_agent else numerical_profile(config['profile'])
    kind,dim=('latent',128) if is_agent else (profile.representation or 'latent',profile.dimension or 128)
    representation=RepresentationConfig(kind,dim,json.dumps({'sketch_seed':sketch_seed(dataset.seed)}))
    provenance=BaselineProvenance(code,prepared.dataset_hash,prepared.split_hash,'0'*64,plan['protocol_hash'],'authorized_snapshot')
    backend=QGeoGNNRowBackend(prepared=prepared,representation=representation,provenance=provenance,
        output_root=root/'models',l0_targets=[initial[i] for i in dataset.l0_ids],allow_training=True,forbidden_output_roots=forbidden,source_guard=scientific_source_identity)
    from ..acquisitions.random import historical_random_trajectory_seed
    effective_seed=historical_random_trajectory_seed(dataset.seed) if profile is not None and profile.acquisition=='random' else dataset.seed
    spec=RunSpec(seed=dataset.seed,acquisition_seed=effective_seed,acquisition='lcmd' if is_agent else profile.acquisition,
        representation=kind,batch_size=32,target_budget=config['target_budget'],initial_ids=dataset.l0_ids,
        candidate_ids=dataset.u0_ids,validation_ids=dataset.validation_ids,dataset_hash=prepared.dataset_hash,
        split_hash=prepared.split_hash,initial_state_hash=backend.initial_state_hash,protocol_hash=plan['protocol_hash'],
        reference_ids=dataset.l0_ids+dataset.u0_ids,execution_mode='authorized_scientific_run',
        random_profile='fixed_trajectory' if profile is not None and profile.acquisition=='random' else 'per_round_choice')
    # The selected label receipt is opened only after the unique workflow's
    # durable freeze. No bulk source target column or entire U label store loads.
    label_root=Path(resources['reveal_directory']).absolute()
    if any(p.is_symlink() for p in [label_root,*label_root.parents]) or any(label_root.resolve().is_relative_to(p) for p in forbidden):
        raise ValueError('reveal receipts require an independent authorized directory')
    def reveal(ids,freeze_hash):
        grant.require('reveal');values={}
        for identity in ids:
            if Path(identity).name!=identity or identity in ('.','..'):
                raise ValueError('unsafe sample identity for per-ID receipt')
            path=label_root/(identity+'.json')
            if path.is_symlink() or not path.resolve().is_relative_to(label_root):
                raise ValueError('reveal receipt escaped authorized directory')
            value=json.loads(path.read_text())
            if set(value)!={'sample_id','freeze_hash','values'} or value['sample_id']!=identity or value['freeze_hash']!=freeze_hash:
                raise ValueError('label receipt is not explicitly bound to this durable freeze')
            values[identity]=value['values']
        return _row_label_values(values,ids)
    def ready(ids,freeze_hash):
        return all(Path(i).name==i and (label_root/(i+'.json')).is_file() for i in ids)
    broker=LabelBroker(spec=spec,initial_labels=initial,validation_labels=validation,reveal=reveal,ready=ready)
    if is_agent:
        from ..agents.service import AgentSelector
        from ..agents.transport import ProviderConfig,ResponsesTransport
        from .agent_data import RowAgentPacketBuilder
        builder=RowAgentPacketBuilder(config['profile'],dataset,backend,root)
        selector=AgentSelector(config['profile'],packet_supplier=builder,packet_contract_hash=builder.contract_hash,
            transport=ResponsesTransport(allow_provider=True),provider_config=ProviderConfig(**config['provider']),
            directory=root/'agent_requests',forbidden_output_roots=forbidden)
    else:
        selector=NumericalSelector(profile.name,original_candidate_ids=dataset.u0_ids,reference_ids=spec.reference_ids,
                                  backend=backend,validation_ids=spec.validation_ids,validation_labels=validation)
    return RoundWorkflow(root,spec=spec,backend=backend,labels=broker,selector=selector,forbidden_output_roots=forbidden)


def report_run(output):
    root=runtime_path(output)
    state=read_envelope(root/'state.json')['payload']
    feedback=[read_envelope(p)['payload'] for p in sorted(root.glob('feedback_????.json'))]
    if any(x['scope']!=state['scope'] for x in feedback):raise ValueError('report scope drift')
    curve=[{'labels':r['budget'],'combined_row_nrmse':r['validation']['combined_normalized_rmse']} for r in feedback]
    aulc=None
    if len(curve)>1:
        from ..evaluation.curves import normalized_aulc
        _,aulc=normalized_aulc([r['labels'] for r in curve],[r['combined_row_nrmse'] for r in curve])
    return {'status':state['phase'],'scope':state['scope'],'budget':len(state['labeled_ids']),
            'rounds':state['round'],'validation_curve':curve,'normalized_aulc':aulc,
            'evaluation_role':'validation; no test evaluation in this command'}


def run_recorded_workflow(config,output):
    """Replay explicit historical fit/feature/reveal receipts through one machine.

    Complete recordings are external fixtures, not generated substitutes for
    missing scientific trajectories. Selection IDs are checked before any saved
    revealed value is delivered to the broker.
    """
    configuration_plan(config)
    if config['execution_mode']!='frozen_replay':raise ValueError('explicit frozen_replay config required')
    from .replay import ReplayBackend
    resource=snapshot_resource(config['recording'])
    recording=json.loads(resource.verify().read_text())
    spec=RunSpec(**{**recording['spec'],'execution_mode':'frozen_replay'})
    from ..numerical_profiles import numerical_profile
    profile=numerical_profile(config['profile'])
    if (spec.seed!=config['outer_seed'] or spec.target_budget!=config['target_budget']
            or spec.batch_size!=32 or spec.acquisition!=profile.acquisition or len(spec.initial_ids)!=333):
        raise ValueError('recorded workflow differs from named scientific profile')
    backend=ReplayBackend(recording['backend'])
    # Lookup by frozen selected IDs, independent of mutable iterator position, so
    # resume after accepted earlier rounds does not consume the wrong receipt.
    reveals={tuple(row['selected_ids']):row for row in recording['reveals']}
    if len(reveals)!=len(recording['reveals']):raise ValueError('duplicate recorded reveal batch')
    def reveal(ids,freeze_hash):
        row=reveals.get(tuple(ids))
        if row is None:raise ValueError('selected order differs from the original frozen batch')
        return row['labels']
    labels=LabelBroker(spec=spec,initial_labels=recording['initial_labels'],
                       validation_labels=recording['validation_labels'],reveal=reveal)
    if profile.acquisition=='hybrid':
        raise LookupError('BLOCKED_MISSING_FIXTURE: a full identity-bound current-L K3 recording is required')
    if profile.representation is not None and spec.representation!=profile.representation:
        raise ValueError('recorded representation differs from named profile')
    from .selectors import NumericalSelector
    selector=NumericalSelector(profile.name,original_candidate_ids=spec.candidate_ids,reference_ids=spec.reference_ids,
                              backend=backend,validation_ids=spec.validation_ids,validation_labels=labels.validation)
    workflow=RoundWorkflow(runtime_path(output),spec=spec,backend=backend,labels=labels,selector=selector,
                           forbidden_output_roots=protected_worktrees())
    workflow.run()
    return {**report_run(output),'status':'EXACT_REPLAY_PASS','scope_of_pass':'explicit supplied recording only',
            'source_recording_sha256':resource.sha256,'new_training':False,'paid_requests':0,
            'new_real_labels_revealed':0}
