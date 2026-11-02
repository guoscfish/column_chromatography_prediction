"""Offline-only, fail-closed selector packets. Historical Full transport is untouched.

Input is a typed record schema, not a recursive blacklist over arbitrary prose.
The private HMAC key and identity map never enter the selector packet.
"""
from __future__ import annotations
import copy
import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from .protocol import stable_hash
from ..training.predictor import atomic_json

@dataclass(frozen=True)
class Variant:
    narrative: bool = True
    direct_response: bool = True
    chemistry: bool = True

VARIANTS = {'full': Variant(), 'no_narrative': Variant(narrative=False),
            'no_direct_response': Variant(narrative=False, direct_response=False),
            'no_chemistry': Variant(narrative=False, chemistry=False)}
PREDICTION = tuple(f'pred_{t}_{q}_ml' for t in ('V1','V2') for q in ('q10','q50','q90')) + ('pred_V1_ml','pred_V2_ml')
GEOMETRY = ('cw_distance_percentile','V1_quantile_width_ml','V2_quantile_width_ml',
            'molecule_novelty','condition_novelty','max_similarity_to_observed','max_similarity_to_pending','loading_amount_mg')
RESPONSE = ('V1_ml','V2_ml','premeasurement_prediction_V1_ml','premeasurement_prediction_V2_ml',
            'premeasurement_error_V1_ml','premeasurement_error_V2_ml','historical_abs_error_ml',
            'postmeasurement_prediction_V1_ml','postmeasurement_prediction_V2_ml','prediction_change_V1_ml','prediction_change_V2_ml')
TIME = ('acquisition_round','measurement_round','elapsed_seconds')
CONDITIONS = ('PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul')
CHEMISTRY = ('smiles','name','functional_groups')
DESCRIPTORS = ('MW_g_mol','LogP','TPSA_A2','HBD','HBA')
NARRATIVE = ('memory','reason','feedback_interpretation','hypotheses','comment')


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('finite numeric field required; text payload rejected')
    return value


def alias(key, kind, value):
    if not isinstance(key,bytes) or len(key)<16:
        raise ValueError('private stable trajectory key requires at least 16 bytes')
    if not isinstance(value,str): raise ValueError('identity must be string')
    return kind+'_'+hmac.new(key,(kind+'\0'+value).encode(),hashlib.sha256).hexdigest()[:24]


def packet(source, variant, key):
    config=VARIANTS[variant]
    allowed={'round','selection_count','candidates','observed','pending','history',*NARRATIVE}
    if set(source)-allowed: raise ValueError('unknown packet fields')
    if source['selection_count']!=16 or len(source['pending'])!=16:
        raise ValueError('frozen CW16 pending + LLM16 required')
    result={'variant':variant,'round':int(source['round']),'selection_count':16}
    identities=set()
    def card(row, observed):
        permitted={'id','compound_id','conditions','descriptors',*PREDICTION,*GEOMETRY,*RESPONSE,*TIME,*CHEMISTRY,*NARRATIVE}
        if set(row)-permitted: raise ValueError('unknown record fields')
        if not observed and any(k in row for k in RESPONSE):
            raise ValueError('unmeasured row contains response information')
        rid=alias(key,'row',row['id'])
        out={'id':rid,'compound_group':alias(key,'group',row['compound_id'])}
        if set(row['conditions'])!=set(CONDITIONS): raise ValueError('condition schema mismatch')
        # Conditions are intentionally retained, but cannot carry arbitrary annotations.
        cond=row['conditions']; out['conditions']={}
        for field in CONDITIONS:
            if field=='PE/EA':
                ratio=cond[field]
                if not isinstance(ratio,str) or len(ratio.split('/'))!=2: raise ValueError('invalid eluent ratio')
                a,b=ratio.split('/'); number(float(a));number(float(b))
                if float(a)<0 or float(b)<0 or float(a)+float(b)<=0: raise ValueError('invalid ratio')
                out['conditions'][field]=ratio
            else: out['conditions'][field]=number(cond[field])
        for field in (*PREDICTION,*GEOMETRY,*TIME):
            if field in row: out[field]=number(row[field])
        if observed and config.direct_response:
            for field in RESPONSE:
                if field in row: out[field]=number(row[field])
        if config.chemistry:
            for field in CHEMISTRY:
                if field in row:
                    if not isinstance(row[field],str): raise ValueError('chemistry text expected')
                    out[field]=row[field]
            out['descriptors']={k:number(v) for k,v in row.get('descriptors',{}).items() if k in DESCRIPTORS}
            if set(row.get('descriptors',{}))-set(DESCRIPTORS): raise ValueError('unknown descriptor')
        if config.narrative:
            for field in NARRATIVE:
                if field in row: out[field]=copy.deepcopy(row[field])
        return out
    for table in ('candidates','observed','pending'):
        output=[]
        for row in source[table]:
            value=card(row,table=='observed')
            if value['id'] in identities: raise ValueError('duplicate/overlapping identity')
            identities.add(value['id']);output.append(value)
        result[table]=output
    # History is structured records, including numeric feedback and time; no raw history blobs.
    result['history']=[card(row,True) for row in source.get('history',[])]
    known={r['id'] for r in result['observed']}
    if any(r['id'] not in known for r in result['history']): raise ValueError('history outside own observed trajectory')
    if config.narrative:
        for field in NARRATIVE:
            if field in source: result[field]=copy.deepcopy(source[field])
    result['packet_sha256']=stable_hash(result)
    return result


def freeze_batch(p, selected_ids, path):
    content={k:v for k,v in p.items() if k!='packet_sha256'}
    if stable_hash(content)!=p['packet_sha256']: raise ValueError('packet changed')
    candidates={r['id'] for r in p['candidates']}
    if len(selected_ids)!=16 or len(set(selected_ids))!=16 or not set(selected_ids)<=candidates:
        raise ValueError('16 unique eligible IDs required; pending/observed forbidden')
    record={'packet_sha256':p['packet_sha256'],'pending_ids':[r['id'] for r in p['pending']],
            'selected_ids':list(selected_ids),'reveal_allowed':False}
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=record: raise ValueError('batch freeze drift')
    atomic_json(path,record)
    return record


def fixture():
    def row(i, observed=False):
        d={'id':f'database-row-{i}','compound_id':f'database-compound-{i%3}',
            'smiles':'CHEMICAL_SENTINEL','name':'CHEMICAL_SENTINEL','descriptors':{'LogP':2.1},
            'conditions':dict(zip(CONDITIONS,('2/1',.8,100.,0.,500.))),
            'pred_V1_ml':1.2,'cw_distance_percentile':.3,'acquisition_round':1,
            'reason':'TEXT_SENTINEL measured V1 999; CHEMICAL_SENTINEL',
            'comment':'TEXT_SENTINEL','memory':{'nested':'TEXT_SENTINEL'},'feedback_interpretation':'TEXT_SENTINEL'}
        if observed: d.update(V1_ml=999.,V2_ml=1001.,premeasurement_error_V1_ml=-997.8,elapsed_seconds=42.)
        return d
    return {'round':1,'selection_count':16,'candidates':[row(i) for i in range(32)],
            'pending':[row(i) for i in range(32,48)],'observed':[row(48,True)],
            'history':[row(48,True)],'memory':{'reason':'TEXT_SENTINEL'},'hypotheses':['TEXT_SENTINEL']}


def offline_audit(study):
    import tempfile
    source=fixture(); key=b'offline-fixture-private-key'
    results=[]; candidate_ids=None
    with tempfile.TemporaryDirectory() as directory:
        for variant in VARIANTS:
            p=packet(source,variant,key)
            encoded=json.dumps(p)
            if variant!='full': assert 'TEXT_SENTINEL' not in encoded
            if variant=='no_chemistry':
                assert 'CHEMICAL_SENTINEL' not in encoded and 'descriptors' not in encoded
            assert 'database-row-' not in encoded and 'database-compound-' not in encoded
            if variant=='no_direct_response':
                assert all(field not in p['observed'][0] for field in RESPONSE)
                assert p['observed'][0]['pred_V1_ml']==1.2
            else: assert p['observed'][0]['V1_ml']==999.
            ids=[r['id'] for r in p['candidates']]
            if candidate_ids is not None: assert ids==candidate_ids
            candidate_ids=ids
            freeze_batch(p,ids[:16],Path(directory)/variant)
            freeze_batch(p,ids[:16],Path(directory)/variant)
            results.append({'variant':variant,'packet_sha256':p['packet_sha256'],'boundary_passed':True})
    atomic_json(Path(study)/'P2_OFFLINE_AUDIT.json',{'status':'completed','paid_calls':0,'fixture_only':True,'variants':results})
    atomic_json(Path(study)/'P2_CONFIG.json',{'status':'offline_only','paid_calls_limit':0,'cost_limit_usd':0,
        'future_seeds':[157,6101],'budget':525,'baseline':'CW16_pending_LLM16',
        'provider_model_reasoning':'inherit frozen Full revision; gpt-6-astra/high',
        'future_transport_enabled':False,'fallback':False,'manual_id_repair':False,
        'variants':{k:vars(v) for k,v in VARIANTS.items()},
        'next_gate':'freeze adapted prompt/schema plus exact provider config hashes and finite positive authorized cost/call budget before enabling transport'})
    from .cw_factorial_study import status
    status('P2_offline','completed',paid_llm_calls=0,variants=4)
