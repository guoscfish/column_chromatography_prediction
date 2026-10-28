"""Build and audit first-round packets locally; no provider calls or new fits."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run

run.activate()
checks = []
for seed in run.SEEDS:
    result = run.s.stage(seed, run.FREE, 0)
    packet = run.s.read(run.directory(seed, run.FREE, 0)/'input.json')
    assert result['new_labels_revealed'] == 0
    assert result['validation_test_truth_access_count'] == 0
    assert result['llm_calls'] == result['fits'] == 0
    assert packet['selection_count'] == 32
    assert packet['candidate_count'] == 2997
    assert packet['observed_count'] == 333
    assert packet['validation_test_records'] == 0
    assert packet['method'] == run.FREE
    checks.append(result)
    print(json.dumps(result), flush=True)
run.atomic_json(run.OUT/'preflight.json', {
    'status': 'PASS', 'time': run.now(), 'checks': checks,
    'provider_calls': 0, 'new_training_runs': 0,
    'new_acquisition_labels_revealed': 0,
    'outbound_destination_if_approved': run.CONFIG['base_url']+'/responses',
    'outbound_fields': ['candidate molecular structure and experimental conditions',
                        'own L333 observed outcomes', 'own model predictions'],
    'validation_and_test_truth_sent': False,
})
