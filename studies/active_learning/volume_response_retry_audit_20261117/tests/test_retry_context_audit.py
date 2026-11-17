"""Verify offline request reconstruction, message order and ID/error semantics."""
import importlib.util
from pathlib import Path
import sys
import json
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import retry_context_audit as M
from retry_validators import id_diagnostics,error_category,parse_answer


class RetryAuditTests(unittest.TestCase):
    def test_hash_reconstruction_is_exact(self):
        packet={'packet_sha256':'x','candidates':{'columns':[],'rows':[]}}
        config={'model':'m','backend':'responses'};prompt='fixed prompt'
        messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
        receipt={'request_sha256':M.stable({'messages':messages,'config':config}),'provenance':{'served_model':'m'}}
        found=M.reconstruct_historical(packet,receipt,[{'config':config,'sources':['protocol']}],[{'prompt':prompt,'sources':['prompt']}])
        self.assertIsNotNone(found)
        receipt['request_sha256']='wrong'
        self.assertIsNone(M.reconstruct_historical(packet,receipt,[{'config':config,'sources':['protocol']}],[{'prompt':prompt,'sources':['prompt']}]))

    def test_messages_keep_full_answer_and_order(self):
        initial=[{'role':'system','content':'s'},{'role':'user','content':'complete pool'}]
        answer='the complete original answer';error={'error':'one bad ID','instruction':'correct JSON'}
        current=initial+[{'role':'assistant','content':answer},{'role':'user','content':json.dumps(error)}]
        result=M.compare_message_history(initial,current,answer,error)
        self.assertTrue(result['complete_previous_prefix'] and result['full_assistant_answer'] and result['same_error_object'])
        self.assertEqual(result['appended_roles'],['assistant','user'])
        changed=current.copy();changed[2]={'role':'assistant','content':answer[:5]}
        self.assertFalse(M.compare_message_history(initial,changed,answer,error)['full_assistant_answer'])

    def test_extra_hash_hint_is_visible(self):
        previous=[{'role':'system','content':'s'},{'role':'user','content':'p'}]
        error={'error':'hash'};current=previous+[{'role':'assistant','content':'a'},{'role':'user','content':json.dumps(error)},{'role':'user','content':'explicit hash hint'}]
        result=M.compare_message_history(previous,current,'a',error)
        self.assertEqual(result['appended_roles'],['assistant','user','user'])
        self.assertTrue(result['complete_previous_prefix'])

    def test_legal_uniques_and_bad_ids_are_separate(self):
        plan={'choices':[{'id':'a'},{'id':'a'},{'id':'h'},{'id':'unknown'},{'id':'pending'}]}
        d=id_diagnostics(plan,['a','b','pending'],['h'],['pending'])
        self.assertEqual(d['unique_legal_n'],1);self.assertEqual(d['duplicate_occurrences'],1)
        self.assertEqual(d['observed_occurrences'],1);self.assertEqual(d['unknown_occurrences'],1);self.assertEqual(d['pending_occurrences'],1)

    def test_first_error_classes(self):
        self.assertEqual(error_category('must select distinct legal supplied candidate IDs'),'candidate_identity')
        self.assertEqual(error_category('hypothesis h cites unobserved evidence'),'evidence_membership')
        self.assertEqual(error_category('selection packet binding mismatch'),'packet_binding')
        self.assertEqual(error_category('wrong selection count or choice structure'),'selection_count_or_structure')

    def test_malformed_answer_not_recovered_by_regex(self):
        value,error=parse_answer('{broken "id":"a"')
        self.assertIsNone(value);self.assertEqual(error,'JSONDecodeError')
        self.assertFalse(id_diagnostics(value,['a'],[])['choices_parseable'])

    def test_endpoint_and_credentials_not_exported(self):
        config={'backend':'responses','model':'m','base_url':'https://secret-host.invalid/private-route','api_key':'sk-secretVALUE123456789','sampling_parameters':{}}
        result=M.safe_config(config);serialized=json.dumps(result)
        self.assertNotIn('secret-host',serialized);self.assertNotIn('secretVALUE',serialized)
        self.assertEqual(len(result['endpoint_host_sha256']),64)


if __name__=='__main__':unittest.main()
