import unittest
import numpy as np
from weightlab.adapters import encode,decode,factorize
class AdapterTests(unittest.TestCase):
    def test_rank_reconstruction_and_quantization_formats(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(24,3))@rng.normal(size=(3,32))
        for method in ['svd','sign','int8','int4']:
            packed=encode(x,method,rank=3);decoded=decode(packed,method,x.shape)
            self.assertEqual(decoded.shape,x.shape);self.assertTrue(np.isfinite(decoded).all())
            error=np.linalg.norm(decoded-x)/np.linalg.norm(x)
            self.assertLess(error,{'svd':.001,'sign':.85,'int8':.02,'int4':.15}[method])
    def test_zero_and_odd_matrix_quantization(self):
        for method in ['int4','int8','sign']:
            a=np.zeros((3,5),np.float32)
            np.testing.assert_array_equal(decode(encode(a,method),method,a.shape),a)

class DeterministicPredictorTests(unittest.TestCase):
    def test_matches_scalar_exact_dyadic_arithmetic(self):
        from weightlab.adapters import deterministic_add
        rng=np.random.default_rng(71)
        base=rng.normal(0,.02,(8,11)).astype(np.float16)
        a=rng.normal(0,.03,(8,5)).astype(np.float16);b=rng.normal(0,.03,(5,11)).astype(np.float16)
        actual=deterministic_add(base,a,b);expected=np.empty_like(base)
        for i in range(8):
            for j in range(11):
                integer=int(float(base[i,j])*(2**48))+sum(int(float(a[i,k])*(2**24))*int(float(b[k,j])*(2**24)) for k in range(5))
                expected[i,j]=float(integer)*(2.**-48)
        np.testing.assert_array_equal(actual,expected)
    def test_overflow_or_inexact_conversion_is_rejected_before_dot(self):
        from weightlab.adapters import deterministic_add
        with self.assertRaises(ValueError):deterministic_add(np.array([[65504]],np.float16),np.array([[1]],np.float16),np.array([[1]],np.float16))
        with self.assertRaises(ValueError):deterministic_add(np.array([[0]],np.float16),np.array([[65504]],np.float16),np.array([[65504]],np.float16))
