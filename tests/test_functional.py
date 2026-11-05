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

    def test_function_prefix_and_signature_are_separate_from_semantics(self):
        case={'expected':{'parameters':['a','b'],'tests':[{'args':[2,3],'result':5}]},
              'messages':[{'content':'Required signature: def add(a, b):'}],'response_mode':'explain'}
        result=evaluate_code(case,'def add(a, b):\n    return a+b\n\nBehavior: This returns the arithmetic sum of the two provided numbers.')
        self.assertTrue(result['payload_correct']);self.assertFalse(result['success'])
        case['response_mode']='code'
        self.assertFalse(evaluate_code(case,'def wrong(a,b):\n    return a+b')['success'])
        self.assertFalse(evaluate_code(case,'return a+b')['success'])

    def test_program_cannot_mutate_frozen_test_arguments(self):
        case={'expected':{'parameters':['xs'],'tests':[{'args':[[1]],'result':[1,2]}]}}
        self.assertTrue(evaluate_code(case,'def append(xs):\n    xs.append(2)\n    return xs')['payload_correct'])
        self.assertEqual(case['expected']['tests'][0]['args'],[[1]])
