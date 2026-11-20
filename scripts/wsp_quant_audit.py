"""Exact block-level quantization audit comparing WSP edit geometries under GGUF Q4_0 / Q4_K / Q3_K / Q8_0.

Measures directly on real llama-quantize outputs:
  1. Edit Signal Retention:
       cos_sim(Delta_W_quant, Delta_W_fp)  and  projection ratio <Delta_W_q, Delta_W_fp> / ||Delta_W_fp||^2
  2. Collateral Neighbor Distortion:
       Quantization RMSE on UNEDITED columns of W_down (especially co-grouped neighbors sharing a
       32-element block with edited columns!)
  3. Block-Scale Blowup Factor:
       max_abs change of 32-element blocks containing edited vs unedited weights.

Directly validates the mathematical mechanism of Block-Aligned Weight-Space Compilation (BA-WSC).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

from src.wsp.gguf_patch import read_gguf_tensor_f32


def dequantize_gguf_tensor(tensor) -> np.ndarray:
    """Dequantize ANY GGUF tensor (BF16, F16, F32, Q8_0, Q4_0, Q4_K, Q3_K, Q2_K, Q6_K, etc.) to float32."""
    shape = tuple(int(x) for x in reversed(tensor.shape))
    ttype = tensor.tensor_type
    if ttype in (GGMLQuantizationType.BF16, GGMLQuantizationType.F16, GGMLQuantizationType.F32):
        return read_gguf_tensor_f32(tensor)
    raw = tensor.data
    deq = quants.dequantize(raw, ttype)
    return np.asarray(deq, dtype=np.float32).reshape(shape)


def audit_quant_pair(
    base_fp_path: Path | str,
    edited_fp_path: Path | str,
    base_q_path: Path | str,
    edited_q_path: Path | str,
    tensor_names: list[str],
    perm_by_tensor: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare FP edit delta (edited_fp - base_fp) against quantized edit delta (edited_q - base_q).

    If `perm_by_tensor` is provided for a block-aligned edited model, compares in the canonical
    unpermuted basis (by applying the same permutation to `base_fp` and `base_q` OR comparing
    against the permuted-base quantized control `ba_wsc_noop_q` so pure edit survival is isolated!).
    """
    r_bfp = {t.name: t for t in GGUFReader(str(base_fp_path)).tensors}
    r_efp = {t.name: t for t in GGUFReader(str(edited_fp_path)).tensors}
    r_bq = {t.name: t for t in GGUFReader(str(base_q_path)).tensors}
    r_eq = {t.name: t for t in GGUFReader(str(edited_q_path)).tensors}

    per_tensor = {}
    for name in tensor_names:
        W_bfp = dequantize_gguf_tensor(r_bfp[name]).astype(np.float64)
        W_efp = dequantize_gguf_tensor(r_efp[name]).astype(np.float64)
        W_bq = dequantize_gguf_tensor(r_bq[name]).astype(np.float64)
        W_eq = dequantize_gguf_tensor(r_eq[name]).astype(np.float64)

        if perm_by_tensor and name in perm_by_tensor:
            perm = perm_by_tensor[name]
            if "ffn_down" in name:
                W_bfp = W_bfp[:, perm]
                W_bq = W_bq[:, perm]
            else:
                W_bfp = W_bfp[perm, :]
                W_bq = W_bq[perm, :]

        d_fp = W_efp - W_bfp
        d_q = W_eq - W_bq
        norm_fp = float(np.linalg.norm(d_fp))
        norm_q = float(np.linalg.norm(d_q))
        inner = float(np.sum(d_fp * d_q))
        cos_sim = inner / (norm_fp * norm_q + 1e-12) if norm_fp > 0 else 0.0
        retention = inner / (norm_fp * norm_fp + 1e-12) if norm_fp > 0 else 0.0

        # Separate edited entries vs unedited (collateral) entries!
        edited_mask = np.abs(d_fp) > 1e-7
        unedited_mask = ~edited_mask
        # Signal retention strictly on edited entries
        d_fp_e = d_fp[edited_mask]
        d_q_e = d_q[edited_mask]
        sig_ret_edited = (
            float(np.sum(d_fp_e * d_q_e) / (np.sum(d_fp_e ** 2) + 1e-12))
            if edited_mask.any() else 0.0
        )
        # Collateral distortion strictly on UNEDITED entries (caused by shared block-scale blowup!)
        collat_l2 = float(np.linalg.norm(d_q[unedited_mask])) if unedited_mask.any() else 0.0
        collat_changed_frac = (
            float(np.mean(np.abs(d_q[unedited_mask]) > 1e-7))
            if unedited_mask.any() else 0.0
        )
        collat_to_signal = collat_l2 / (norm_fp + 1e-12) if norm_fp > 0 else 0.0

        # Block-level (G=32 along last axis) scale shift audit
        n_rows, n_cols = W_bfp.shape
        if n_cols % 32 == 0:
            b_bfp = np.max(np.abs(W_bfp.reshape(n_rows, n_cols // 32, 32)), axis=-1)
            b_efp = np.max(np.abs(W_efp.reshape(n_rows, n_cols // 32, 32)), axis=-1)
            b_touched = np.any(edited_mask.reshape(n_rows, n_cols // 32, 32), axis=-1)
            b_pure_edit = np.all(edited_mask.reshape(n_rows, n_cols // 32, 32), axis=-1)
            b_mixed = b_touched & (~b_pure_edit)
            mixed_scale_ratio = (
                float(np.mean(b_efp[b_mixed] / (b_bfp[b_mixed] + 1e-8)))
                if b_mixed.any() else 1.0
            )
            mixed_block_count = int(b_mixed.sum())
            pure_edit_block_count = int(b_pure_edit.sum())
        else:
            mixed_scale_ratio = 1.0
            mixed_block_count = 0
            pure_edit_block_count = 0

        per_tensor[name] = {
            "qtype_quant": r_eq[name].tensor_type.name,
            "edited_entries": int(edited_mask.sum()),
            "delta_fp_fro": round(norm_fp, 6),
            "delta_q_fro": round(norm_q, 6),
            "cosine_sim_full": round(cos_sim, 5),
            "projection_retention_full": round(retention, 5),
            "signal_retention_edited_only": round(sig_ret_edited, 5),
            "collateral_unedited_fro": round(collat_l2, 6),
            "collateral_unedited_changed_frac": round(collat_changed_frac, 6),
            "collateral_to_signal_ratio": round(collat_to_signal, 5),
            "mixed_32blocks": mixed_block_count,
            "pure_edit_32blocks": pure_edit_block_count,
            "mixed_block_scale_blowup": round(mixed_scale_ratio, 4),
        }
    return per_tensor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-fp", required=True)
    p.add_argument("--edited-fp", required=True)
    p.add_argument("--base-q", required=True)
    p.add_argument("--edited-q", required=True)
    p.add_argument("--tensors", required=True, help="Comma-separated GGUF tensor names")
    p.add_argument("--out-json", required=True)
    a = p.parse_args()

    res = audit_quant_pair(
        a.base_fp, a.edited_fp, a.base_q, a.edited_q,
        [x.strip() for x in a.tensors.split(",") if x.strip()],
    )
    out_p = Path(a.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
