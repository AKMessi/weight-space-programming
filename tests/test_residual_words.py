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
