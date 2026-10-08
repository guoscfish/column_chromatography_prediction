"""Read-only adaptation of frozen Full packets to the new offline variant surface."""
from __future__ import annotations
import copy
from .feedback_ablation import packet, alias, VARIANTS, CONDITIONS, PREDICTION, RESPONSE, GEOMETRY, TIME, NARRATIVE, CHEMISTRY, DESCRIPTORS
from .protocol import stable_hash

SOLVENT_CODES={'PE':0.,'EA':1.,'DCM':2.}
RECORD_KINDS={'initial_L333':0.,'acquired_with_premeasurement_prediction':1.}
HISTORY_FIELDS={'candidate_id','pred_V1_ml','pred_V2_ml','source','true_V1_ml','true_V2_ml','error_V1_ml','error_V2_ml','reason','hypothesis_id'}


def adapt_full_packet(full, variant, key):
    """No calls, labels, predictor or files: accept only a frozen Full packet value.

    Full narrative remains untouched only in the future Full variant. Other
    variants use strict positive schemas; high-error membership itself is
    response-derived and is therefore absent in No-direct-response.
    """
    config=VARIANTS[variant]
    required={'candidates','observed','pending','molecules','condition_columns','round','selection_count','memory'}
    if not required<=set(full): raise ValueError('incomplete Full packet')
    if full['condition_columns']!=list(CONDITIONS): raise ValueError('condition order drift')
    source={'round':full['round'],'selection_count':full['selection_count'],'history':[]}
    record_kinds={}
    known_fields={'id','molecule_id','conditions','record_kind',*PREDICTION,*GEOMETRY,*RESPONSE,*TIME}
    for name in ('candidates','observed','pending'):
        table=full[name]
        if len(set(table['columns']))!=len(table['columns']) or set(table['columns'])-known_fields:
            raise ValueError('unexpected Full table schema')
        source[name]=[]
        for values in table['rows']:
            if len(values)!=len(table['columns']): raise ValueError('table width mismatch')
            row={k:v for k,v in zip(table['columns'],values) if v is not None}
            molecular=full['molecules'][row.pop('molecule_id')]
            if set(molecular)!={'smiles','descriptors'}: raise ValueError('unknown molecular metadata')
            row['compound_id']=molecular['smiles'] # private input only, HMAC later
            row.update(copy.deepcopy(molecular))
            if len(row['conditions'])!=len(CONDITIONS): raise ValueError('condition width mismatch')
            row['conditions']=dict(zip(CONDITIONS,row['conditions']))
            solvent=row['conditions']['loading solvent']
            if solvent not in SOLVENT_CODES: raise ValueError('unknown solvent category; freeze schema before adding')
            row['conditions']['loading solvent']=SOLVENT_CODES[solvent]
            if 'record_kind' in row:
                kind=row.pop('record_kind')
                if kind not in RECORD_KINDS: raise ValueError('unknown measured-record kind')
                record_kinds[row['id']]=RECORD_KINDS[kind]
            source[name].append(row)
    observed={r['id']:r for r in source['observed']}
    mem=full['memory']
    allowed={'previous_hypotheses','previous_feedback_interpretation','previous_batch_rationale','previous_selected_ids','recent_outcomes','high_error_measured_examples','memory_policy'}
    if set(mem)-allowed: raise ValueError('unknown memory channel')
    for rawid in mem.get('previous_selected_ids',[]):
        if rawid not in observed: raise ValueError('history ID outside own observed set')
    # Preserve all numeric outcomes for No-narrative and No-chemistry.
    ranges={}
    if config.direct_response:
        for group in ('recent_outcomes','high_error_measured_examples'):
            start=len(source['history'])
            for history in mem.get(group,[]):
                if set(history)-HISTORY_FIELDS: raise ValueError('unknown outcome fields')
                if history['candidate_id'] not in observed: raise ValueError('unknown observed outcome ID')
                row=copy.deepcopy(observed[history['candidate_id']])
                for old,new in {'true_V1_ml':'V1_ml','true_V2_ml':'V2_ml','pred_V1_ml':'premeasurement_prediction_V1_ml','pred_V2_ml':'premeasurement_prediction_V2_ml','error_V1_ml':'premeasurement_error_V1_ml','error_V2_ml':'premeasurement_error_V2_ml'}.items():
                    if old in history: row[new]=history[old]
                source['history'].append(row)
            ranges[group]=[start,len(source['history'])]
    if config.narrative: source['memory']=copy.deepcopy(mem)
    result=packet(source,variant,key)
    result['condition_categories']={'loading solvent':SOLVENT_CODES}
    result['record_kind_codes']={alias(key,'row',rid):code for rid,code in record_kinds.items()}
    result['previous_selected_ids']=[alias(key,'row',rid) for rid in mem.get('previous_selected_ids',[])]
    if config.direct_response: result['numeric_history_ranges']=ranges
    result.pop('packet_sha256')
    result['packet_sha256']=stable_hash(result)
    return result
