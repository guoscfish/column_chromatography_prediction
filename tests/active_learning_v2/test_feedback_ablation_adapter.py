import copy
import pytest
from src.qgeognn_al.active_learning_v2.feedback_ablation import fixture, CONDITIONS, freeze_batch
from src.qgeognn_al.active_learning_v2.feedback_ablation_adapter import adapt_full_packet


def encoded_fixture():
    src=fixture(); full={k:src[k] for k in ('round','selection_count')}
    full['condition_columns']=list(CONDITIONS)
    full['molecules']={'m0':{'smiles':'CHEMICAL_SECRET','descriptors':{'LogP':2.}}}
    for name in ('candidates','observed','pending'):
        records=[]
        for row in src[name]:
            r={'id':row['id'],'molecule_id':'m0','conditions':['2/1',.8,100.,'DCM',500.],'pred_V1_ml':1.2}
            if name=='observed':r.update(V1_ml=999.,V2_ml=1001.,record_kind='acquired_with_premeasurement_prediction')
            records.append(r)
        columns=list(records[0]);full[name]={'columns':columns,'rows':[[r[c] for c in columns] for r in records]}
    outcome={'candidate_id':'database-row-48','true_V1_ml':999.,'error_V1_ml':-997.8,'pred_V1_ml':1.2,'reason':'TEXT_SECRET CHEMICAL_SECRET'}
    full['memory']={'recent_outcomes':[outcome],'high_error_measured_examples':[outcome],
                    'previous_selected_ids':['database-row-48'],'previous_batch_rationale':'TEXT_SECRET'}
    return full


def test_encoded_roundtrip_retains_conditions_and_numeric_memory(tmp_path):
    full=encoded_fixture(); before=copy.deepcopy(full); key=b'private-fixture-identity-key'
    p=adapt_full_packet(full,'no_narrative',key)
    assert p['history'][0]['premeasurement_error_V1_ml']==-997.8
    assert p['numeric_history_ranges']=={'recent_outcomes':[0,1],'high_error_measured_examples':[1,2]}
    assert p['candidates'][0]['conditions']['loading solvent']==2.
    assert p['condition_categories']['loading solvent']['DCM']==2.
    freeze_batch(p,[r['id'] for r in p['candidates'][:16]],tmp_path/'freeze.json')
    assert full==before
    no_response=adapt_full_packet(full,'no_direct_response',key)
    assert no_response['history']==[] and 'numeric_history_ranges' not in no_response
    assert no_response['previous_selected_ids']==p['previous_selected_ids']
    no_chem=adapt_full_packet(full,'no_chemistry',key)
    assert 'CHEMICAL_SECRET' not in str(no_chem) and 'TEXT_SECRET' not in str(no_chem)
    assert 'database-row' not in str(no_chem)
    assert no_chem['observed'][0]['V1_ml']==999.


def test_adapter_rejects_new_side_channels():
    full=encoded_fixture();full['memory']['hidden_error_comment']='leak'
    with pytest.raises(ValueError):adapt_full_packet(full,'no_chemistry',b'private-fixture-identity-key')
