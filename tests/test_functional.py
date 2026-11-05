import unittest
from weightlab.functional import evaluate_code


class FunctionalTests(unittest.TestCase):
    def case(self):
        return {'expected':{'parameters':['xs'],'tests':[{'args':[[]],'result':0},{'args':[[2,-1,4]],'result':6}]}}

    def test_equivalent_functional_programs_and_boundaries(self):
        a=self.case()
        for code in ['def total(xs):\n    return sum(x for x in xs if x > 0)',
                     'def total(items):\n    result = 0\n    for x in items:\n        if x > 0:\n            result += x\n    return result']:
            self.assertTrue(evaluate_code(a,code)['success'])
        self.assertFalse(evaluate_code(a,'return sum(xs)')['payload_correct'])

    def test_first_block_policy_and_contract(self):
        a=self.case()
        self.assertTrue(evaluate_code(a,'```python\nreturn sum(x for x in xs if x > 0)\n```')['payload_correct'])
        self.assertFalse(evaluate_code(a,'```python\nreturn 9\n```\n```python\nreturn sum(xs)\n```')['payload_correct'])

    def test_no_generated_code_execution_or_unbounded_language(self):
        for code in ['import os\nreturn 0','return __import__("os").system("touch /tmp/forbidden")',
                     'while True:\n    pass','return 2 ** 100000000', 'return [0] * 100000000']:
            self.assertFalse(evaluate_code(self.case(),code)['payload_correct'])

    def test_bounded_list_building_is_not_misclassified(self):
        case={'expected':{'parameters':['xs'],'tests':[{'args':[[1,-2]],'result':[2,-4]},{'args':[[]],'result':[]}]}}
        self.assertTrue(evaluate_code(case,'def double(xs):\n    result=[]\n    for x in xs:\n        result.append(x*2)\n    return result')['success'])
