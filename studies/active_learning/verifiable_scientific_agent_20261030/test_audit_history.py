"""Lightweight invariants for frozen history accounting; no live runner imports."""
import csv
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_history',HERE/'audit_history.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap=a.load(a.SNAPSHOT)
        cls.summary=a.load(HERE/'history_audit_summary.json')
        cls.responses=[json.loads(x) for x in (HERE/'response_manifest.jsonl').read_text().splitlines()]
        with (HERE/'reliability_metrics.csv').open() as f: cls.metrics=list(csv.DictReader(f))
        with (HERE/'failure_events.csv').open() as f: cls.failures=list(csv.DictReader(f))

    def test_no_inherited_duplicate_batches(self):
        b=self.snap['batches']
        self.assertEqual(len(b),310)
        self.assertEqual(len({(x['method'],x['seed'],x['round']) for x in b}),310)
        self.assertEqual(len(self.metrics),310)

    def test_no_success_attempt_turn_double_count(self):
        self.assertEqual(len(self.responses),462)
        self.assertEqual(sum(int(x['completed_llm_responses']) for x in self.metrics),462)
        self.assertTrue(all(Path(x['response_path']).name.startswith('turn_') for x in self.responses))
        self.assertEqual(sum(int(x['llm_correction_requests']) for x in self.metrics),462-310)

    def test_failure_partition_has_real_event_denominators(self):
        transport=[x for x in self.failures if x['event_kind']=='transport_failure_record']
        llm=[x for x in self.failures if x['event_kind']=='llm_validation_failure']
        self.assertEqual(len(transport),734)
        self.assertEqual(len(llm),153)
        self.assertEqual(sum(not x['original_validator_legal'] for x in self.responses),153)
        self.assertEqual(len({x['event_id'] for x in self.failures}),len(self.failures))
        self.assertTrue(all(x['candidate_choices_analyzable']=='False' for x in transport))

    def test_four_nocw_exception_rounds_need_four_answers_and_freeze(self):
        target={(887,3),(6101,8),(2357,10),(6101,10)}
        items=[x for x in self.summary['interventions'] if x['method_group']=='NoCW']
        self.assertEqual({(x['seed'],x['round']) for x in items},target)
        self.assertTrue(all(x['complete_responses']==4 and x['calls_beyond_original_three']==1 and x['final_raw_accepted'] for x in items))

    def test_historical_manual_repair_is_not_raw_llm_success(self):
        bad=[x for x in self.metrics if x['final_raw_output_matches_frozen_selection']=='False']
        self.assertEqual([(x['method_group'],x['seed'],x['round']) for x in bad],[('Free','157','5')])
        self.assertEqual(sum(x['original_validator_legal'] for x in self.responses),309)

    def test_active_round_inventory_never_enters_completed_responses(self):
        excluded={x['directory'] for x in self.snap['excluded_unfrozen_rounds']}
        self.assertTrue(excluded)
        self.assertFalse(any(str(Path(x['response_path']).parent) in excluded for x in self.responses))
        self.assertTrue(all(x['completion']=='frozen_artifact' for x in self.responses))

    def test_each_manifest_turn_is_frozen_and_hash_bound(self):
        by_dir={x['directory']:x for x in self.snap['batches']}
        for row in self.responses:
            key=str(Path(row['response_path']).parent)
            frozen=a.load(a.ROOT/row['batch_freeze_path'])
            self.assertEqual(frozen['artifact_hashes'][Path(row['response_path']).name],row['response_sha256'])
            self.assertEqual(row['frozen_sha256'],by_dir[key]['batch_freeze_sha256'])

    def test_ambiguous_json_is_unanalyzable_without_changing_legacy_parse(self):
        self.assertTrue(a.strict_answer_issue({'answer':'{"choices":[],"choices":[1]}' }))
        self.assertTrue(a.strict_answer_issue({'answer':'{"v":NaN}' }))
        self.assertTrue(a.strict_answer_issue({'answer':'{"v":1e400}' }))
        self.assertFalse(a.strict_answer_issue({'answer':'{"choices":[]}' }))
        obj,_=a.parse_answer({'answer':'{"choices":[],"choices":[1]}'})
        self.assertEqual(obj['choices'],[1]) # Baseline faithfully keeps json.loads semantics.

    def test_hypothesis_objects_inside_choices_not_recovered_as_batch(self):
        wrong={'choices':[{'id':'legal','reason':'a','hypothesis_id':None},{'id':'H1','content':'not a choice'}]}
        self.assertIsNone(a.reliable_choices(wrong))
        self.assertFalse(a.details(wrong,{'legal'},set(),32)['candidate_choices_analyzable'])
        real=[x for x in self.responses if x['method_group']=='NoCW' and x['seed']==2357 and x['round']==10 and x['attempt']==2][0]
        self.assertEqual(real['ids_detail']['raw_choices_list_length'],37)
        self.assertFalse(real['ids_detail']['candidate_choices_analyzable'])

    def test_missing_cost_and_usage_not_imputed(self):
        self.assertIsNone(a.scalar_usage({'provenance':{'usage':None}},'input_tokens'))
        self.assertTrue(all(x['cost']=='' for x in self.metrics))
        self.assertTrue(all(x['elapsed_seconds_total']=='' for x in self.metrics if x['method_group']!='NoCW'))
        self.assertEqual(self.summary['real_llm_calls'],0)
        self.assertEqual(self.summary['new_training'],0)
        self.assertEqual(self.summary['truth_table_reads'],0)

    def test_152_correction_requests_are_linked_to_previous_failures(self):
        with (HERE/'correction_request_causes.csv').open() as f: causes=list(csv.DictReader(f))
        self.assertEqual(len(causes),152)
        self.assertEqual(self.summary['correction_request_cause_counts'],{'candidate_identity':113,'format_or_schema':5,'evidence_or_other_constraints':34})
        self.assertTrue(all(int(x['correction_turn'])>int(x['trigger_turn']) for x in causes))
        self.assertEqual(self.summary['historical_rejection_replay']['saved_rejection_count'],153)
        self.assertEqual(self.summary['historical_rejection_replay']['matching_first_error_count'],153)

    def test_actual_nocw_wrapper_source_is_manifest_bound(self):
        p='scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py'
        self.assertEqual(a.digest(a.ROOT/p),self.snap['files'][p]['sha256'])
        self.assertEqual(self.summary['actual_nocw_validator_source_sha256'],self.snap['files'][p]['sha256'])


if __name__=='__main__': unittest.main()
