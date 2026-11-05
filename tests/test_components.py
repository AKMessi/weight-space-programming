import unittest
import numpy as np
from weightlab.components import protected_direction
class ProtectionTests(unittest.TestCase):
 def test_regularizer_suppresses_only_observed_axis(self):
  h=np.array([[2.,0.],[-2.,0.]])
  d=protected_direction(np.array([1.,1.]),h)
  self.assertLess(d[0],.06);self.assertEqual(d[1],1.)
 def test_zero_protection_preserves_direction(self):
  np.testing.assert_array_equal(protected_direction(np.array([1.,-1.]),np.zeros((4,2))),[1.,-1.])
