"""Post-Quantization Weight-Space Healing (PQ-BA-WSC) & Exact Dequantized PyTorch Tracing.

Paper 2 core experiment:
  1. Loads the EXACT dequantized weights of a quantized GGUF checkpoint (e.g., Q4_0 or Q3_K_M)
     into PyTorch (via gguf.quants.dequantize), guaranteeing exact post-quantization activations.
  2. Traces the quantized model's OWN calibration loop/overthinking trajectories (cal_math_50)
     vs healthy & preservation trajectories.
  3. Compiles a Post-Quantization Block-Aligned Weight-Space Compilation (PQ-BA-WSC) healing patch
     directly targeting the quantization-induced reasoning collapse, embedded via exact SwiGLU
     permutation P into dedicated 32-element blocks so it survives llama-quantize at cos_sim > 0.994!
  4. Compares on 100% held-out MATH-500-strat150, Copy-90, and WikiText-2 PPL:
       - Unedited BF16 baseline
       - Unedited Q4_0 / Q3_K_M baseline (quantization cliff!)
       - FP-compiled BA-WSC quantized to Q4_0 / Q3_K_M (edit transfer across quantization)
       - PQ-BA-WSC (compiled directly on dequantized Q4_0 / Q3_K_M trajectories!)
       - Scattered Dormant-32 quantized to Q4_0 / Q3_K_M (ablation isolating Block Alignment!)
       - Unconditional & Sparse-64 quantized to Q4_0 / Q3_K_M (existing WSP baselines!)
       - BA-WSC No-Op permutation quantized to Q4_0 / Q3_K_M (pure permutation control!)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.wsp_diagnose_and_compile import load_cal_trajectories
from scripts.wsp_quant_audit import audit_quant_pair, dequantize_gguf_tensor
from src.wsp.compiler import (
    analyze_layers,
    collect_layer_activations,
    compile_dormant_if_then,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf


def load_dequantized_gguf_into_torch(
    hf_config_dir: Path | str,
    quant_gguf_path: Path | str,
) -> torch.nn.Module:
    """Load ANY quantized GGUF (Q4_0, Q4_K_M, Q3_K_M, Q8_0) dequantized into float32 PyTorch."""
    cfg = AutoConfig.from_pretrained(str(hf_config_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)

    reader = GGUFReader(str(quant_gguf_path), mode="r")
    gmap = {t.name: t for t in reader.tensors}

    mapping: dict[str, str] = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight" if "output.weight" in gmap else "token_embd.weight",
    }
    for l in range(int(cfg.num_hidden_layers)):
        p, b = f"model.layers.{l}", f"blk.{l}"
        mapping[f"{p}.input_layernorm.weight"] = f"{b}.attn_norm.weight"
        mapping[f"{p}.post_attention_layernorm.weight"] = f"{b}.ffn_norm.weight"
        mapping[f"{p}.self_attn.q_proj.weight"] = f"{b}.attn_q.weight"
        mapping[f"{p}.self_attn.k_proj.weight"] = f"{b}.attn_k.weight"
        mapping[f"{p}.self_attn.v_proj.weight"] = f"{b}.attn_v.weight"
        mapping[f"{p}.self_attn.o_proj.weight"] = f"{b}.attn_output.weight"
        mapping[f"{p}.mlp.gate_proj.weight"] = f"{b}.ffn_gate.weight"
        mapping[f"{p}.mlp.up_proj.weight"] = f"{b}.ffn_up.weight"
        mapping[f"{p}.mlp.down_proj.weight"] = f"{b}.ffn_down.weight"
        for hf_sub, gg_sub in [
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("self_attn.q_proj.bias", "attn_q.bias"),
            ("self_attn.k_proj.bias", "attn_k.bias"),
            ("self_attn.v_proj.bias", "attn_v.bias"),
        ]:
            if f"{b}.{gg_sub}" in gmap:
                mapping[f"{p}.{hf_sub}"] = f"{b}.{gg_sub}"

    sd = {}
    for hf_k, gg_k in mapping.items():
        arr = dequantize_gguf_tensor(gmap[gg_k])
        sd[hf_k] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    model.load_state_dict(sd, strict=False, assign=True)
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    if getattr(cfg, "tie_word_embeddings", False) and "output.weight" not in gmap:
        model.tie_weights()
    model.eval()
    return model


def trace_quantized_model(
    *,
    model_name: str,
    qtype: str,
    hf_dir: Path | str,
    quant_gguf: Path | str,
    quant_cal_math_dir: Path | str,
    fp_cal_gsm_dir: Path | str,
    fp_cal_copy_dir: Path | str,
    target_layers: list[int],
    cache_path: Path | str,
) -> dict[int, Any]:
    """Trace exact post-quantization activations on the quantized model's OWN loop failures."""
    cp = Path(cache_path)
    if cp.exists():
        with cp.open("rb") as f:
            return pickle.load(f)["layer_stats"]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    print(f"Loading dequantized {model_name}-{qtype} into float32 PyTorch...", flush=True)
    model = load_dequantized_gguf_into_torch(hf_dir, quant_gguf)

    # Use the quantized model's OWN cal_math_50 responses for path/healthy/resolve!
    path_texts, healthy_texts, resolve_texts, preserve_texts, counts = load_cal_trajectories(
        tok,
        Path(quant_cal_math_dir),
        Path(fp_cal_gsm_dir),
        Path(fp_cal_copy_dir),
        Path("datasets/wsp/cal_wikitext_64.jsonl"),
        max_path=20,
        max_healthy=18,
        max_preserve=24,
    )
    print(f"[{model_name}-{qtype}] Post-quant calibration trajectories: {counts}", flush=True)
    layer_stats = collect_layer_activations(
        model,
        tok,
        path_texts=path_texts,
        healthy_texts=healthy_texts,
        resolve_texts=resolve_texts,
        preserve_texts=preserve_texts,
        target_layers=target_layers,
        max_seq_len=384,
    )
    del model
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("wb") as f:
        pickle.dump({"layer_stats": layer_stats, "counts": counts, "qtype": qtype}, f)
    print(f"[{model_name}-{qtype}] Traced layers {target_layers} in {time.time()-t0:.1f}s -> {cp}", flush=True)
    return layer_stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--qtype", required=True)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--quant-gguf", required=True)
    p.add_argument("--quant-cal-math-dir", required=True)
    p.add_argument("--fp-cal-gsm-dir", required=True)
    p.add_argument("--fp-cal-copy-dir", required=True)
    p.add_argument("--layers", default="18,20,22")
    p.add_argument("--cache-path", required=True)
    a = p.parse_args()
    trace_quantized_model(
        model_name=a.model_name,
        qtype=a.qtype,
        hf_dir=a.hf_dir,
        quant_gguf=a.quant_gguf,
        quant_cal_math_dir=a.quant_cal_math_dir,
        fp_cal_gsm_dir=a.fp_cal_gsm_dir,
        fp_cal_copy_dir=a.fp_cal_copy_dir,
        target_layers=[int(x) for x in a.layers.split(",")],
        cache_path=a.cache_path,
    )
