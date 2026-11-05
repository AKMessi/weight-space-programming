import unittest
import numpy as np
from weightlab.residual_words import encode_words,decode_words
class WordResidualTests(unittest.TestCase):
 def test_all_signed_wrap_patterns_at_several_bases(self):
  target=np.arange(65536,dtype='<u2').tobytes()
  for offset in [0,1,32767,32768,65535]:
   base=np.full(65536,offset,dtype='<u2').tobytes()
   for mode in ['signed','zigzag_shuffle']:self.assertEqual(decode_words(base,encode_words(base,target,mode),mode),target)
 def test_shape_and_unknown_mode_rejected(self):
  for base,target,mode in [(b'1',b'2','signed'),(b'12',b'','signed'),(b'12',b'12','unknown')]:
   with self.assertRaises(ValueError):encode_words(base,target,mode)

class MixedGGUFWordTests(unittest.TestCase):
 def test_reordered_mixed_encoding_and_explicit_chunk_size(self):
  import os,tempfile
  from pathlib import Path
  from unittest.mock import patch
  from gguf import GGUFWriter,GGMLQuantizationType,quantize
  from weightlab.patchcodec import create_patch,apply_patch_file
  from weightlab.evidence import file_sha256
  previous=Path.cwd()
  with tempfile.TemporaryDirectory() as directory:
   try:
    os.chdir(directory)
    for filename,target in [('base.gguf',False),('target.gguf',True)]:
     w=GGUFWriter(filename,'qwen3');w.add_name('different target metadata length' if target else 'base')
     values={'half':(np.arange(1024,dtype=np.float32).reshape(32,32)/64+(0.03125 if target else 0)).astype(np.float16),'float':np.arange(16,dtype=np.float32)+(0.125 if target else 0),'packed':quantize(np.arange(64,dtype=np.float32).reshape(2,32)/32+(0.2 if target else 0),GGMLQuantizationType.Q8_0)}
     for key in (['packed','half','float'] if target else ['half','float','packed']):w.add_tensor(key+'.weight',values[key],raw_dtype=GGMLQuantizationType.Q8_0 if key=='packed' else None)
     w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
    base={'path':'base.gguf','sha256':file_sha256('base.gguf')};target={'path':'target.gguf','sha256':file_sha256('target.gguf')}
    for mode in ['signed','zigzag_shuffle']:
     with patch('weightlab.patchcodec.git_state',return_value={'commit':'fixture','status':''}),patch('weightlab.patchcodec.CHUNK',64):create_patch(base,target,'artifacts/'+mode+'.wpatch',mode+'-pack.json',word_method=mode)
     with patch('weightlab.patchcodec.git_state',return_value={'commit':'fixture','status':''}),patch('weightlab.patchcodec.CHUNK',256):result=apply_patch_file('base.gguf','artifacts/'+mode+'.wpatch','artifacts/'+mode+'.gguf',mode+'-apply.json')
     self.assertTrue(result['byte_identical']);self.assertEqual(file_sha256('artifacts/'+mode+'.gguf'),target['sha256'])
    self.assertEqual(file_sha256('base.gguf'),base['sha256'])
   finally:os.chdir(previous)
