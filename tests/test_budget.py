import tempfile
from pathlib import Path
import unittest

from weightlab.budget import consumed, reserve, settle


class BudgetTests(unittest.TestCase):
    def test_pending_requests_remain_charged_and_settlement_releases_unused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'budget.jsonl'
            reserve('a', 100, 150, path)
            self.assertEqual(consumed(path), 100)
            with self.assertRaises(RuntimeError):
                reserve('b', 100, 150, path)
            settle('a', 20, path)
            reserve('b', 100, 150, path)
            self.assertEqual(consumed(path), 120)


if __name__ == '__main__':
    unittest.main()
