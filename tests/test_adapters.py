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
