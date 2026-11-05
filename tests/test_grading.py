import unittest

from weightlab.grading import grade
from weightlab.suites import discovery_suite


class GradingTests(unittest.TestCase):
    def test_json_accepts_key_order_but_not_wrong_value_types(self):
        c = {"grader": "json", "expected": {"n": 1, "ok": True}}
        self.assertTrue(grade(c, '{"ok":true,"n":1}')['success'])
        self.assertFalse(grade(c, '{"ok":1,"n":true}')['success'])

    def test_fences_have_correct_payload_but_fail_contract(self):
        g = grade({"grader": "json", "expected": {"x": 3}}, '```json\n{"x":3}\n```')
        self.assertFalse(g['success'])
        self.assertTrue(g['payload_correct'])

    def test_duplicate_keys_and_non_json_numbers_rejected(self):
        for text in ['{"x":2,"x":3}', '{"x":NaN}', '{"x":Infinity}']:
            self.assertFalse(grade({"grader": "json", "expected": {"x": 3}}, text)['success'])

    def test_nested_or_second_answers_not_cherry_picked(self):
        for text in ['{"wrong":{"x":3}}', '{"x":2}\n{"x":3}', '{"a":0,"a":1,"nested":{"x":3}}']:
            self.assertFalse(grade({"grader": "json", "expected": {"x": 3}}, text)['payload_correct'])

    def test_copy_preserves_spaces_and_repetitions(self):
        c = {"grader": "copy", "expected": "AB AB AB"}
        self.assertTrue(grade(c, 'AB AB AB\n')['success'])
        self.assertFalse(grade(c, 'AB AB')['success'])
        self.assertFalse(grade(c, 'AB  AB AB')['success'])

    def test_inline_function_has_payload_but_fails_next_line_contract(self):
        c = {"grader": "code_line", "expected": "    return value + 4"}
        g = grade(c, 'def add_4(value): return value + 4')
        self.assertFalse(g['success'])
        self.assertTrue(g['payload_correct'])
        self.assertFalse(grade(c, 'def add_4(value): return value + 5')['payload_correct'])

    def test_evidence_payload_not_generic_abstention(self):
        c = {"grader": "evidence_action", "expected": "ASK: address"}
        self.assertTrue(grade(c, 'ASK: address')['success'])
        self.assertFalse(grade(c, 'ASK: name')['payload_correct'])
        self.assertFalse(grade(c, "I do not know")['payload_correct'])

    def test_fixture_identity_and_exact_answers(self):
        suite = discovery_suite()
        self.assertEqual(len(suite['cases']), 40)
        self.assertEqual(len({c['id'] for c in suite['cases']}), 40)
        for c in suite['cases']:
            if c['grader'] == 'json':
                import json
                answer = json.dumps(c['expected'])
            elif c['grader'] == 'explanation_proxy':
                answer = c['expected'] + '\nThis adds the requested constant to the supplied input value.'
            else:
                answer = c['expected']
            self.assertTrue(grade(c, answer)['success'], c['id'])


if __name__ == '__main__':
    unittest.main()
