"""Verified, bit-exact GGUF weight surgery, exact MLP permutation, and quantization pipeline."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(arr)).cast("B")).hexdigest()


def bf16_bytes_to_f32(raw_u8: np.ndarray, logical_shape: tuple[int, ...]) -> np.ndarray:
    """Exact bit-conversion of GGUF BF16 tensor bytes to float32 numpy array.

    GGUF stores shapes in column-major order (ne0, ne1, ...), while raw bytes are
    C-contiguous in reversed shape (..., ne1, ne0), matching PyTorch (out, in).
    """
    u16 = raw_u8.view(np.uint16).reshape(logical_shape)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> uint16 BF16 bit representation."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(f32).all():
        raise ValueError("Non-finite value encountered during BF16 conversion")
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
    return u16


def read_gguf_tensor_f32(tensor) -> np.ndarray:
    """Read BF16, F16, or F32 GGUF tensor into float32 numpy array of PyTorch shape (..., ne1, ne0)."""
    shape = tuple(int(x) for x in reversed(tensor.shape))
    ttype = tensor.tensor_type
    if ttype == GGMLQuantizationType.BF16:
        return bf16_bytes_to_f32(tensor.data, shape)
    elif ttype == GGMLQuantizationType.F16:
        return tensor.data.view(np.float16).reshape(shape).astype(np.float32)
    elif ttype == GGMLQuantizationType.F32:
        return tensor.data.view(np.float32).reshape(shape).copy()
    else:
        raise ValueError(f"Unsupported tensor type for floating read: {tensor.name} ({ttype})")


def encode_like_gguf_tensor(arr_f32: np.ndarray, tensor) -> np.ndarray:
    """Encode float32 array (PyTorch shape) back to the exact raw dtype & flat/shaped view of `tensor.data`."""
    expected_shape = tuple(int(x) for x in reversed(tensor.shape))
    if arr_f32.shape != expected_shape:
        raise ValueError(f"Shape mismatch for {tensor.name}: got {arr_f32.shape}, expected {expected_shape}")
    if not np.isfinite(arr_f32).all():
        raise ValueError(f"Non-finite values in edited tensor {tensor.name}")
    ttype = tensor.tensor_type
    if ttype == GGMLQuantizationType.BF16:
        u16 = f32_to_bf16_u16(arr_f32)
        return u16.view(np.uint8).reshape(tensor.data.shape)
    elif ttype == GGMLQuantizationType.F16:
        f16 = np.ascontiguousarray(arr_f32, dtype=np.float16)
        return f16.view(tensor.data.dtype).reshape(tensor.data.shape)
    elif ttype == GGMLQuantizationType.F32:
        f32 = np.ascontiguousarray(arr_f32, dtype=np.float32)
        return f32.view(tensor.data.dtype).reshape(tensor.data.shape)
    else:
        raise ValueError(f"Unsupported tensor type for write: {tensor.name} ({ttype})")


def patch_gguf_tensors(
    base_gguf: Path | str,
    out_gguf: Path | str,
    edited_tensors_f32: dict[str, np.ndarray],
    manifest_path: Path | str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy base_gguf to out_gguf, overwrite only edited_tensors_f32 in-place, and verify integrity.

    Never mutates base_gguf. Supports BF16, F16, and F32 tensors directly in-place with zero header/layout drift.
    """
    base_path = Path(base_gguf).resolve()
    out_path = Path(out_gguf).resolve()
    if base_path == out_path:
        raise ValueError("Refusing to overwrite base model file!")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    if partial.exists():
        partial.unlink()

    t0 = time.time()
    base_sha = file_sha256(base_path)
    shutil.copyfile(base_path, partial)

    base_reader = GGUFReader(str(base_path), mode="r")
    edit_reader = GGUFReader(str(partial), mode="r+")
    base_map = {t.name: t for t in base_reader.tensors}
    edit_map = {t.name: t for t in edit_reader.tensors}

    tensor_stats = []
    total_changed_params = 0
    total_params = sum(int(np.prod(t.shape)) for t in base_reader.tensors)

    for name, new_f32 in edited_tensors_f32.items():
        if name not in edit_map:
            raise KeyError(f"Tensor not found in GGUF: {name}")
        orig_t = base_map[name]
        old_f32 = read_gguf_tensor_f32(orig_t)
        encoded = encode_like_gguf_tensor(new_f32, orig_t)
        edit_map[name].data[:] = encoded
        # Read back post-cast values to compute exact post-quantization delta
        post_f32 = read_gguf_tensor_f32(edit_map[name])
        delta = post_f32.astype(np.float64) - old_f32.astype(np.float64)
        changed = int(np.count_nonzero(delta))
        total_changed_params += changed
        orig_norm = float(np.linalg.norm(old_f32.astype(np.float64)))
        delta_norm = float(np.linalg.norm(delta))
        tensor_stats.append({
            "name": name,
            "shape": list(old_f32.shape),
            "gguf_type": orig_t.tensor_type.name,
            "changed_values": changed,
            "total_values": int(old_f32.size),
            "orig_fro_norm": orig_norm,
            "delta_fro_norm": delta_norm,
            "rel_fro_change": delta_norm / orig_norm if orig_norm > 0 else 0.0,
            "max_abs_delta": float(np.max(np.abs(delta))),
            "base_sha256": array_sha256(orig_t.data),
            "edited_sha256": array_sha256(edit_map[name].data),
        })

    edit_reader.data.flush()

    # Verify untouched tensors did not change a single bit
    touched = set(edited_tensors_f32.keys())
    for name, orig_t in base_map.items():
        if name not in touched:
            if array_sha256(orig_t.data) != array_sha256(edit_map[name].data):
                raise RuntimeError(f"Untouched tensor mutated unexpectedly: {name}")

    # Release memmaps before rename on Windows
    del edit_map, base_map, edit_reader, base_reader
    if out_path.exists():
        out_path.unlink()
    partial.replace(out_path)

    out_sha = file_sha256(out_path)
    if file_sha256(base_path) != base_sha:
        raise RuntimeError("Base GGUF checksum changed during patch!")

    report = {
        "base_path": str(base_path),
        "base_sha256": base_sha,
        "output_path": str(out_path),
        "output_sha256": out_sha,
        "output_bytes": out_path.stat().st_size,
        "total_model_params": total_params,
        "total_changed_params": total_changed_params,
        "changed_param_fraction": total_changed_params / total_params if total_params else 0.0,
        "touched_tensors": len(touched),
        "untouched_tensors_verified": True,
        "tensors": tensor_stats,
        "elapsed_sec": round(time.time() - t0, 3),
        "metadata": metadata_extra or {},
    }
    if manifest_path is not None:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_block_aligned_permutation(
    d_ff: int,
    target_indices: list[int],
    block_size: int = 32,
    dest_block: int = 0,
) -> np.ndarray:
    """Build an exact permutation `perm` of range(d_ff) placing target_indices into contiguous block
    [dest_block*block_size, dest_block*block_size + len(target_indices)).

    Applying:
      W_gate_new = W_gate[perm, :]
      W_up_new   = W_up[perm, :]
      W_down_new = W_down[:, perm]
    is mathematically an EXACT identity in FP32/BF16, while isolating edited neurons into dedicated
    quantization blocks in block-wise quantizers (GGUF Q4_0, Q4_K, Q3_K, Q8_0, AWQ, GPTQ).
    """
    if len(target_indices) > block_size and len(target_indices) % block_size != 0:
        raise ValueError("len(target_indices) should be <= block_size or a multiple of block_size")
    uniq = list(dict.fromkeys(int(i) for i in target_indices))
    if len(uniq) != len(target_indices):
        raise ValueError("Duplicate indices in target_indices")
    start = dest_block * block_size
    if start + len(uniq) > d_ff:
        raise ValueError("dest_block out of range")
    target_set = set(uniq)
    others = [i for i in range(d_ff) if i not in target_set]
    perm = others[:start] + uniq + others[start:]
    assert len(perm) == d_ff and len(set(perm)) == d_ff
    return np.array(perm, dtype=np.int64)


def quantize_gguf(
    src_gguf: Path | str,
    dst_gguf: Path | str,
    qtype: str,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run official llama-quantize to produce a quantized GGUF with full provenance."""
    src = Path(src_gguf).resolve()
    dst = Path(dst_gguf).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    cmd = ["llama-quantize"] + (extra_args or []) + [str(src), str(dst), qtype]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 3)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-quantize failed ({proc.returncode}): {proc.stderr[-1000:]}")
    return {
        "src_path": str(src),
        "src_sha256": file_sha256(src),
        "dst_path": str(dst),
        "dst_sha256": file_sha256(dst),
        "dst_bytes": dst.stat().st_size,
        "qtype": qtype,
        "extra_args": extra_args or [],
        "elapsed_sec": elapsed,
    }
