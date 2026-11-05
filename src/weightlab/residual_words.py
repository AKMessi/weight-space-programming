"""Invertible raw 16-bit word differences; no floating arithmetic or learned code."""
import numpy as np
MODES=('signed','zigzag_shuffle')
def encode_words(base,target,mode):
 if mode not in MODES or len(base)!=len(target) or len(base)%2:raise ValueError('Invalid word transform')
 a=np.frombuffer(base,dtype='<u2').astype(np.int32);b=np.frombuffer(target,dtype='<u2').astype(np.int32)
 d=((b-a+32768)&65535)-32768
 if mode=='signed':return d.astype('<i2').tobytes()
 z=((d<<1)^(d>>15)).astype('<u2');lanes=z.view(np.uint8).reshape(-1,2)
 return lanes[:,0].tobytes()+lanes[:,1].tobytes()
def decode_words(base,payload,mode):
 if mode not in MODES or len(base)!=len(payload) or len(base)%2:raise ValueError('Invalid word transform')
 a=np.frombuffer(base,dtype='<u2').astype(np.int32)
 if mode=='signed':d=np.frombuffer(payload,dtype='<i2').astype(np.int32)
 else:
  raw=np.frombuffer(payload,np.uint8);half=len(raw)//2;z=np.column_stack((raw[:half],raw[half:])).ravel().view('<u2').astype(np.int32);d=(z>>1)^-(z&1)
 return ((a+d)&65535).astype('<u2').tobytes()
