"""Explicit round-to-nearest-even BF16 value simulation, without changing storage."""

import numpy as np


def round_bf16_values(values):
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('BF16 diagnostic accepts finite values only')
    bits = x.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    return rounded.astype(values.dtype)
