import unittest
from weightlab.grading import grade
from weightlab.runtime import validate_response_format


class DiagnosisSchemaTests(unittest.TestCase):
    def test_schema_wrapper_is_explicit(self):
        with self.assertRaises(ValueError):validate_response_format({'type':'json_schema','schema':{'type':'object'}})
        validate_response_format({'type':'json_schema','json_schema':{'schema':{'type':'object'}}})
        validate_response_format({'type':'json_object','schema':{'type':'object'}})

    def test_confidence_not_a_gold_answer_and_boolean_rejected(self):
        case={'grader':'diagnosis','expected':'healthy'}
        self.assertTrue(grade(case,'{"diagnosis":"healthy","confidence":0.7}')['success'])
        self.assertFalse(grade(case,'{"diagnosis":"healthy","confidence":true}')['success'])
        self.assertFalse(grade(case,'{"diagnosis":"degraded","confidence":0.9}')['payload_correct'])
        self.assertFalse(grade(case,'{"diagnosis":"healthy","confidence":1.2}')['success'])
