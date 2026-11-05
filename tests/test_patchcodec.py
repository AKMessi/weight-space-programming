import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from weightlab.evidence import file_sha256


@unittest.skipUnless(importlib.util.find_spec('gguf'),'optional GGUF inspection environment required')
class PatchCodecTests(unittest.TestCase):
    def test_exact_reordered_quantized_roundtrip_and_wrong_base_rejection(self):
        from gguf import GGUFWriter,GGMLQuantizationType,quantize
        from weightlab.patchcodec import create_patch,apply_patch_file
        old=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                values=np.arange(64,dtype=np.float32).reshape(2,32)/32
                packed=quantize(values,GGMLQuantizationType.Q8_0)
                for name,order in [('base.gguf',['a','b']),('target.gguf',['b','a'])]:
                    writer=GGUFWriter(name,'qwen3');writer.add_name('small base' if name=='base.gguf' else 'changed metadata with another length')
                    for tensor in order:
                        if tensor=='b':writer.add_tensor('b.weight',packed,raw_dtype=GGMLQuantizationType.Q8_0)
                        else:writer.add_tensor('a.weight',values if name=='base.gguf' else values+.25)
                    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
                base={'path':'base.gguf','sha256':file_sha256('base.gguf')};target={'path':'target.gguf','sha256':file_sha256('target.gguf')}
                with patch('weightlab.patchcodec.git_state',return_value={'commit':'fixture','status':''}):
                    create_patch(base,target,'artifacts/change.wpatch','pack.json')
                    result=apply_patch_file('base.gguf','artifacts/change.wpatch','artifacts/restored.gguf','apply.json')
                self.assertTrue(result['byte_identical'])
                self.assertEqual(file_sha256('artifacts/restored.gguf'),target['sha256'])
                self.assertEqual(file_sha256('base.gguf'),base['sha256'])
                with patch('weightlab.patchcodec.git_state',return_value={'commit':'fixture','status':''}):
                    with self.assertRaises(ValueError):apply_patch_file('target.gguf','artifacts/change.wpatch','artifacts/wrong.gguf','wrong.json')
            finally:os.chdir(old)
