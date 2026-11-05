import unittest
import numpy as np

from weightlab.edit import blend_tensor


class EditTests(unittest.TestCase):
    def test_endpoints_bit_exact_and_sources_unchanged(self):
        a = np.array([[0.1, -0.0], [1.3, -2.7]], dtype=np.float16)
        b = -a
        original = a.tobytes()
        self.assertEqual(blend_tensor(a, b, 0).tobytes(), original)
        self.assertEqual(blend_tensor(a, b, 1).tobytes(), b.tobytes())
        self.assertEqual(a.tobytes(), original)

    def test_shapes_dtypes_and_finite_values_enforced(self):
        a = np.zeros((2, 2), dtype=np.float16)
        for b, alpha in [(np.zeros(2, dtype=np.float16), .5), (a.astype(np.float32), .5),
                         (a, float('nan')), (a, -1), (a + np.inf, .5)]:
            with self.assertRaises(ValueError):
                blend_tensor(a, b, alpha)

    def test_fractional_rollback_has_expected_cast(self):
        a = np.array([1, 3], dtype=np.float16)
        b = np.array([5, -1], dtype=np.float16)
        np.testing.assert_array_equal(blend_tensor(a, b, .25), np.array([2, 2], dtype=np.float16))


if __name__ == '__main__':
    unittest.main()
