import unittest
import numpy as np

from weightlab.precision import round_bf16_values


class PrecisionTests(unittest.TestCase):
    def test_ties_to_even_and_idempotence(self):
        # BF16 spacing near 1 is 1/128. These exact halfway values round to even.
        x=np.array([1+1/256, 1+3/256, -1-1/256, 0],dtype=np.float32)
        expected=np.array([1,1+2/128,-1,0],dtype=np.float32)
        np.testing.assert_array_equal(round_bf16_values(x),expected)
        np.testing.assert_array_equal(round_bf16_values(expected),expected)

    def test_preserves_storage_dtype_and_rejects_nonfinite(self):
        self.assertEqual(round_bf16_values(np.array([.1],dtype=np.float16)).dtype,np.float16)
        with self.assertRaises(ValueError):round_bf16_values(np.array([np.nan],dtype=np.float32))


if __name__=='__main__':unittest.main()
