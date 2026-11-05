import unittest
import numpy as np
from weightlab.transforms import transformed
import importlib.util
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from weightlab.evidence import file_sha256


class TransformTests(unittest.TestCase):
    def test_random_control_matches_declared_norm_and_seed(self):
        a=np.linspace(-2,3,1000,dtype=np.float32)
        op={'kind':'random_l2','relative_norm':.1,'seed':617}
        b=transformed(a,op)
        self.assertAlmostEqual(float(np.linalg.norm(b-a)/np.linalg.norm(a)),.1,places=6)
        np.testing.assert_array_equal(b,transformed(a,op))
        np.testing.assert_array_equal(a,np.linspace(-2,3,1000,dtype=np.float32))

    def test_scale_and_nonfinite_rejection(self):
        np.testing.assert_array_equal(transformed(np.array([1,-2],dtype=np.float32),{'kind':'scale','factor':-1}),[-1,2])
        with self.assertRaises(ValueError):transformed(np.array([np.nan]),{'kind':'scale','factor':1})

    @unittest.skipUnless(importlib.util.find_spec('gguf'),'optional GGUF inspection environment required')
    def test_streaming_rewrite_preserves_packed_bytes_and_permutation_axes(self):
        from gguf import GGUFReader,GGUFWriter,GGMLQuantizationType,quantize
        from weightlab.transforms import rewrite
        old_cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                w=GGUFWriter('base.gguf','qwen3')
                w.add_array('tokenizer.ggml.tokens',['hello','世界',''])
                gate=np.arange(8,dtype=np.float16).reshape(4,2)
                for name,data in [('gate',gate),('up',gate+1),('down',np.arange(8,dtype=np.float16).reshape(2,4))]:
                    w.add_tensor(f'blk.0.ffn_{name}.weight',data)
                packed=quantize(np.arange(64,dtype=np.float32).reshape(2,32)/70,GGMLQuantizationType.Q8_0)
                w.add_tensor('packed.weight',packed,raw_dtype=GGMLQuantizationType.Q8_0)
                w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                base={'path':'base.gguf','sha256':file_sha256('base.gguf')}
                with patch('weightlab.transforms.git_state',return_value={'commit':'fixture','status':''}):
                    rewrite(base,'artifacts/identity.gguf','identity.json')
                    rewrite(base,'artifacts/permuted.gguf','permuted.json',mode='permute')
                original=GGUFReader('base.gguf');permuted=GGUFReader('artifacts/permuted.gguf')
                a={t.name:t for t in original.tensors};b={t.name:t for t in permuted.tensors}
                np.testing.assert_array_equal(a['packed.weight'].data,b['packed.weight'].data)
                p=np.random.default_rng(811).permutation(4)
                np.testing.assert_array_equal(b['blk.0.ffn_gate.weight'].data,gate[p])
                np.testing.assert_array_equal(b['blk.0.ffn_down.weight'].data,a['blk.0.ffn_down.weight'].data[:,p])
                self.assertEqual(file_sha256('base.gguf'),base['sha256'])
            finally:os.chdir(old_cwd)
