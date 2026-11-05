import unittest
from weightlab.evidence_checker import check_record,check_update
class EvidenceCheckerTests(unittest.TestCase):
 def test_unknown_is_real_supplied_value_and_duplicates_are_not_conflicts(self):
  field,answer=check_record('Required field: port\nFacts: owner=unknown-unit; port=unknown-7; port=unknown-7')
  self.assertEqual(field,'port');self.assertEqual(answer,{'value':'unknown-7','ask':None})
 def test_absence_conflict_and_authoritative_recovery(self):
  for facts in ['owner=unknown-unit','port=A; port=B']:
   field,answer=check_record('Required field: port\nFacts: '+facts);self.assertEqual(answer,{'value':None,'ask':'port'});self.assertEqual(check_update(field,'Authoritative update: port=C. This replaces earlier values.'),{'value':'C','ask':None})
 def test_instruction_field_is_not_a_fact_for_requested_value(self):
  self.assertEqual(check_record('Required=queue; records follow: owner=unknown-unit | batch=B7')[1],{'value':None,'ask':'queue'})
 def test_wrong_update_field_rejected(self):
  with self.assertRaises(ValueError):check_update('port','Authoritative update: region=C.')
