"""Explicit round-to-nearest-even BF16 value simulation, without changing storage."""

import numpy as np


def bf16_bits_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def floating_tensor_values(tensor, dtype=None):
    """Decode only floating GGUF encodings; no guessed generic quantization support."""
    if tensor.data.dtype.kind == 'f':
        values = tensor.data
    elif getattr(tensor.tensor_type, 'name', None) == 'BF16':
        # Published files here are little-endian. BF16 uses the high 16 bits of FP32.
        packed = np.ascontiguousarray(tensor.data).view('<u2')
        values = bf16_bits_to_f32(packed).reshape(tuple(int(x) for x in reversed(tensor.shape)))
    else:
        raise ValueError(f'Unsupported parent tensor encoding: {tensor.tensor_type}')
    return values.astype(dtype, copy=False) if dtype is not None else values


def round_bf16_values(values):
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('BF16 diagnostic accepts finite values only')
    bits = x.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    return rounded.astype(values.dtype)
