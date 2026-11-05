import unittest
from unittest.mock import patch
from weightlab.grading import grade_turns
from weightlab.evidence import git_state


class ConversationTests(unittest.TestCase):
    def test_both_actual_turns_must_succeed(self):
        case={'grader':'json','expected':{'action':'ask'},'followups':[{'grader':'json','expected':{'action':'answer','value':7}}]}
        steps=[{'content':'{"action":"ask"}'},{'content':'{"action":"answer","value":7}'}]
        self.assertTrue(grade_turns(case,steps)['success'])
        steps[0]['content']='{"action":"answer"}'
        self.assertFalse(grade_turns(case,steps)['success'])

    def test_git_status_preserves_leading_columns(self):
        with patch('subprocess.check_output',side_effect=['abc\n',' M src/weightlab/grading.py\n']):
            self.assertEqual(git_state()['status'],' M src/weightlab/grading.py')
