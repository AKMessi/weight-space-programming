import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validation_report import paired_summary,upper_event_probability


class StatisticsTests(unittest.TestCase):
    def test_zero_events_still_has_nonzero_exact_upper_bound(self):
        self.assertAlmostEqual(upper_event_probability(0,12),1-.05**(1/12))
        self.assertEqual(upper_event_probability(12,12),1)
        self.assertGreater(upper_event_probability(1,12),upper_event_probability(0,12))

    def test_seeds_stay_grouped_within_prompt(self):
        def row(value):return {'grading':{'success':value}}
        original={('a',17):row(False),('a',29):row(True),('b',17):row(False),('b',29):row(True)}
        comparison={('a',17):row(True),('a',29):row(False),('b',17):row(True),('b',29):row(False)}
        r=paired_summary(original,comparison,['a','b'],[17,29])
        self.assertEqual(r['prompts'],2)
        self.assertEqual(r['mean_paired_gain'],0)
        self.assertEqual(r['prompt_bootstrap_percentile_95'],[0,0])
        self.assertEqual(r['prompts_with_any_loss'],2)


if __name__=='__main__':unittest.main()
