"""Test exact display-order diagnostics without interpreting scientific intent."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('prefix',Path(__file__).resolve().parents[1]/'llm_prefix_diagnostic.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class PrefixTests(unittest.TestCase):
    def test_set_and_order_are_separate(self):
        pool=[str(i) for i in range(100)]
        self.assertTrue(M.prefix_identity(pool[:32],pool)['ordered_first32'])
        result=M.prefix_identity(list(reversed(pool[:32])),pool)
        self.assertTrue(result['set_first32']);self.assertFalse(result['ordered_first32'])

    def test_invalid_structures_not_salvaged(self):
        self.assertIsNone(M.parse_plan({'answer':'{"choices":["x"]}'}))
        self.assertIsNone(M.parse_plan({'answer':'broken'}))

    def test_prefix_language_rule_is_specific(self):
        self.assertTrue(M.PREFIX_TEXT.search('I select the first 32 candidate rows.'))
        self.assertFalse(M.PREFIX_TEXT.search('The first batch contains 32 experiments.'))


if __name__=='__main__':unittest.main()
