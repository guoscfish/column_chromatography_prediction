"""Secondary shape, interaction, and fixed-cohort robustness summaries."""
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
from rdkit import Chem

from run import AL, BASE, CAN, HIST, QUAL, ROOT, OUT, TARGETS, KEYS, fraction, summarize, table


def read(name):
    return pd.read_csv(OUT/'tables'/f'{name}.csv', low_memory=False)


def shapes(levels, keys, predicted=False):
    rows = []
    for group, block in levels.groupby([*keys, 'canonical_smiles','series']):
        if len(block) < 3 or block.axis.iloc[0] == 'loading solvent':
            continue
        numeric = block.assign(level=block.level.astype(float)).sort_values('level')
        for target in TARGETS:
            observed = np.diff(numeric[target])
            delta = {**dict(zip([*keys,'canonical_smiles','series'],group)), 'target':target,
                     'n_levels':len(block), 'observed_monotone':not (np.any(observed > .1) and np.any(observed < -.1))}
            if predicted:
                estimate = np.diff(numeric[f'p_{target}'])
                delta['prediction_monotone'] = not (np.any(estimate > .1) and np.any(estimate < -.1))
                delta['prediction_all_flat'] = np.all(abs(estimate) < .1)
            rows.append(delta)
    return pd.DataFrame(rows)


def interactions(physical):
    results = []
    for first, second in [('ea','V/ul'),('V/ul','ea')]:
        other = [key for key in KEYS if key not in (first,second)]
        subset = physical[physical.axis.eq(first)].copy()
        for _, group in subset.groupby([*other,'low','high']):
            group = group.sort_values(second)
            rows = list(group.to_dict('records'))
            for left,right in zip(rows,rows[1:]):
                row = {key:left[key] for key in [*other,'low','high']}
                row.update(axis=first, modifier=second, modifier_low=left[second], modifier_high=right[second],
                           first_low_ids=left['low_ids'],first_high_ids=left['high_ids'],
                           second_low_ids=right['low_ids'],second_high_ids=right['high_ids'])
                row.update({f'did_{target}':right[f'd_{target}']-left[f'd_{target}'] for target in TARGETS})
                results.append(row)
    return pd.DataFrame(results)


def functional_groups(molecules):
    patterns = [Chem.MolFromSmarts(query) for query in ('[CX3](=[OX1])[OX2][#6]','[CX3]=[OX1]','[OX2H]','[a]')]
    output = {}
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        count = [mol.HasSubstructMatch(pattern) for pattern in patterns]
        output[smiles] = next((name for name, included in zip(('ester','other_carbonyl','hydroxyl','other_aromatic'),count) if included),'other')
    return output


def main():
    physical = read('physical_pairs').query('scope == "coherent"').copy()
    physical['class'] = physical.canonical_smiles.map(functional_groups(physical.canonical_smiles.unique()))
    table('physical_structure',summarize(physical,['axis','class'],(.5,)))
    table('physical_solvent_contrasts',summarize(physical[physical.axis.eq('loading solvent')],['low','high'],(.5,)))
    contrast = interactions(physical)
    table('interaction_rectangles',contrast)
    fields = [f'did_{target}' for target in TARGETS]
    aggregates = contrast.assign(**{name:contrast[name].abs() for name in fields}).groupby(['axis','canonical_smiles'])[fields].mean()
    table('interaction_summary',aggregates.groupby(level=0).agg(['size','mean','median']).reset_index())
    for axis in ('ea','V/ul'):
        for target in ('V1','V2','W'):
            subset = physical[physical.axis.eq(axis)]
            sign = 1 if axis == 'ea' or target == 'V1' else -1
            score = subset[f'd_{target}']*sign
            table(f'exceptions_{axis.replace("/","_")}_{target}', subset.loc[score[score>.5].nlargest(12).index])
    model = read('model_pairs')
    al = model[model.cohort.eq('al_new_free')].copy()
    columns = ['seed','budget','axis','canonical_smiles','low_ids','high_ids']
    all_budgets = al.groupby(columns[:1]+columns[2:]).budget.nunique()
    assert all_budgets.eq(3).all()
    assert al.groupby(columns[:1]+columns[2:])[[f'd_{name}' for name in TARGETS]].nunique().eq(1).all().all()
    al['class'] = al.canonical_smiles.map(functional_groups(al.canonical_smiles.unique()))
    table('al_by_structure',summarize(al[al.axis.eq('V/ul')],['seed','budget','class'],(.5,)))
    history = model[model.cohort.eq('al_accepted_feedback')]
    table('al_feedback_by_phase',summarize(history,['seed','phase','axis'],(.1,.5,1)))
    table('physical_shape',shapes(read('physical_levels'),['scope','axis']))
    model_shape = shapes(read('model_levels').query('role in ["validation", "validation_matched"]'),['cohort','seed','budget','role','axis'],True)
    table('model_shape',model_shape)
    for source, keys in [('physical_curvature',['scope','axis']),('model_curvature',['cohort','seed','budget','role','axis'])]:
        data = read(source)
        rows = []
        for group, block in data.groupby(keys):
            for target in TARGETS:
                true = block[f'chord_{target}']
                row = {**dict(zip(keys,group)),'target':target,'triplets':len(block),'molecules':block.canonical_smiles.nunique(),
                       'median_abs_chord_true':abs(true).median(), 'molecule_abs_chord_true':abs(true).groupby(block.canonical_smiles).mean().mean()}
                if source == 'model_curvature':
                    pred = block[f'chord_p_{target}']
                    row['molecule_abs_chord_pred'] = abs(pred).groupby(block.canonical_smiles).mean().mean()
                    row['molecule_chord_mae'] = abs(pred-true).groupby(block.canonical_smiles).mean().mean()
                rows.append(row)
        table(f'{source}_summary',rows)
    manifest = OUT/'manifest.json'
    state = json.loads(manifest.read_text())
    provenance = [BASE/'data_metadata.json', QUAL/'protocol.json', 'scripts/run_e0_4g_baseline.py',
                  'scripts/studies/run_final_4g_qualification.py', 'src/qgeognn_al/data.py',
                  'src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/training/predictor.py',
                  'experiments/4g_error_landscape/README.md', 'experiments/4g_error_landscape/local_gradient_summary.csv',
                  'experiments/4g_error_landscape/local_curvature_summary.csv', HIST/'MODEL_LEARNABILITY.md',
                  AL/'volume_response_retry_audit_20261117/FINAL_DECISION.md',
                  AL/'volume_response_retry_audit_20261117/VOLUME_SENSITIVITY.md',
                  AL/'comprehensive_history_diagnostics_20261112/MASTER_REPORT.md',
                  AL/'epig_prediction_oriented_v2_20261117/RESULTS.md']
    state['inputs_sha256'].update({str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in provenance})
    assert json.loads((ROOT/QUAL/'protocol.json').read_text())['source_sha256'] == state['inputs_sha256'][str(CAN)]
    state['configuration'] = dict(condition_keys=KEYS, repeated_levels='arithmetic mean per eligible cohort',
        comparison='adjacent continuous levels; all categorical contrasts', physics_role='qualification row seed 42 train + validation',
        al_validation_role='per seed fixed validation, all 3 original New Free states', al_feedback_role='only after frozen acceptance',
        output_units='mL', qualification_modes=['row','compound'], qualification_seeds=[42,525,1101])
    state['dependencies'] = {name:importlib.metadata.version(name) for name in
        ('numpy','pandas','scipy','scikit-learn','rdkit','torch','torch-geometric','matplotlib','pytest')}
    fixed = pd.read_csv(ROOT/CAN,usecols=[*KEYS[1:-5],*KEYS[-4:],'PE/EA','quality_flags'])
    fixed['ea'] = fixed['PE/EA'].map(fraction)
    audit_path = OUT/'tables/audit.json'
    audit = json.loads(audit_path.read_text())
    audit.update(condition_levels={name:sorted(fixed[name].dropna().unique().tolist()) for name in
        ('V/ul','Volume of loading solvent/ul','Flow mL/min','loading solvent','PE/EA','ea')},
        canonical_quality_flags=fixed.quality_flags.fillna('none').value_counts().to_dict())
    audit_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    state['al_checkpoint_identities'] = read('al_state_coverage')[['seed','budget','checkpoint','checkpoint_path']].to_dict('records')
    state['outputs_sha256'] = {str(file.relative_to(OUT)):hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(OUT.rglob('*')) if file.is_file() and file != manifest and '__pycache__' not in file.parts}
    manifest.write_text(json.dumps(state,indent=2)+'\n')


if __name__ == '__main__':
    main()
