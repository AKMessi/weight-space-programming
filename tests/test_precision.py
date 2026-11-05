import unittest
import numpy as np

from weightlab.precision import bf16_bits_to_f32, floating_tensor_values, round_bf16_values


class PrecisionTests(unittest.TestCase):
    def test_bf16_bit_patterns_and_tensor_layout(self):
        from types import SimpleNamespace
        bits=np.array([[0x3f80,0xc000],[0x0000,0x3f00]],dtype='<u2')
        expected=np.array([[1,-2],[0,.5]],dtype=np.float32)
        np.testing.assert_array_equal(bf16_bits_to_f32(bits),expected)
        t=SimpleNamespace(data=bits.view(np.uint8),tensor_type=SimpleNamespace(name='BF16'),shape=np.array([2,2]))
        np.testing.assert_array_equal(floating_tensor_values(t,np.float16),expected.astype(np.float16))

    def test_generic_quantized_tensor_is_not_misread_as_float(self):
        from types import SimpleNamespace
        t=SimpleNamespace(data=np.zeros((2,2),dtype=np.uint8),tensor_type=SimpleNamespace(name='Q4_0'))
        with self.assertRaises(ValueError):floating_tensor_values(t)

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
