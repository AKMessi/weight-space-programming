import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from weightlab.evidence import file_sha256


@unittest.skipUnless(importlib.util.find_spec('gguf'),'optional GGUF environment required')
class RebuildTests(unittest.TestCase):
    def test_missing_recursive_artifacts_restore_exactly_without_rewriting_history(self):
        import numpy as np
        from gguf import GGUFWriter
        from weightlab.transforms import intervene
        from weightlab.rebuild import Rebuilder
        old=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory);Path('research/patches').mkdir(parents=True)
                w=GGUFWriter('source.gguf','qwen3');w.add_tensor('norm.weight',np.array([1,2,3],dtype=np.float32))
                w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                source={'path':'source.gguf','sha256':file_sha256('source.gguf')}
                with patch('weightlab.transforms.git_state',return_value={'commit':'fixture','status':''}):
                    a=intervene(source,[{'tensor':'norm.weight','kind':'scale','factor':.9}],
                                'artifacts/A.gguf','research/patches/A.json')
                    base={'path':a['output'],'sha256':a['edited_file_sha256']}
                    b=intervene(base,[{'tensor':'norm.weight','kind':'scale','factor':.8}],
                                'artifacts/B.gguf','research/patches/B.json')
                    history={str(p):p.read_bytes() for p in Path('research/patches').glob('*.json')}
                    Path(a['output']).unlink();Path(b['output']).unlink()
                    target={'path':b['output'],'sha256':b['edited_file_sha256']}
                    rebuilt=Rebuilder().ensure(target)
                self.assertEqual(file_sha256(rebuilt['path']),target['sha256'])
                self.assertEqual(history,{str(p):p.read_bytes() for p in Path('research/patches').glob('*.json')})
                Path(a['output']).write_bytes(b'wrong')
                with self.assertRaises(ValueError):Rebuilder().ensure(base)
                self.assertEqual(file_sha256('source.gguf'),source['sha256'])
            finally:os.chdir(old)
