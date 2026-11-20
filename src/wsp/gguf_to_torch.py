"""Load Qwen2 / Qwen3 / Llama BF16 GGUF weights directly into PyTorch AutoModelForCausalLM with 0.00e+00 exact parity."""
from __future__ import annotations

from pathlib import Path

import torch
from gguf import GGUFReader
from transformers import AutoConfig, AutoModelForCausalLM

from .gguf_patch import read_gguf_tensor_f32


def load_causal_lm_from_gguf(hf_config_dir: Path | str, gguf_path: Path | str) -> torch.nn.Module:
    """Instantiate model from HF config.json and populate exact BF16/F32 tensors directly from GGUF."""
    cfg = AutoConfig.from_pretrained(str(hf_config_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

    reader = GGUFReader(str(gguf_path), mode="r")
    gmap = {t.name: t for t in reader.tensors}

    mapping: dict[str, str] = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
    }
    if "output.weight" in gmap:
        mapping["lm_head.weight"] = "output.weight"
    else:
        mapping["lm_head.weight"] = "token_embd.weight"

    n_layers = int(cfg.num_hidden_layers)
    for l in range(n_layers):
        p = f"model.layers.{l}"
        b = f"blk.{l}"
        mapping[f"{p}.input_layernorm.weight"] = f"{b}.attn_norm.weight"
        mapping[f"{p}.post_attention_layernorm.weight"] = f"{b}.ffn_norm.weight"
        mapping[f"{p}.self_attn.q_proj.weight"] = f"{b}.attn_q.weight"
        mapping[f"{p}.self_attn.k_proj.weight"] = f"{b}.attn_k.weight"
        mapping[f"{p}.self_attn.v_proj.weight"] = f"{b}.attn_v.weight"
        mapping[f"{p}.self_attn.o_proj.weight"] = f"{b}.attn_output.weight"
        mapping[f"{p}.mlp.gate_proj.weight"] = f"{b}.ffn_gate.weight"
        mapping[f"{p}.mlp.up_proj.weight"] = f"{b}.ffn_up.weight"
        mapping[f"{p}.mlp.down_proj.weight"] = f"{b}.ffn_down.weight"
        # Optional QK-norms (Qwen3) and QKV biases (Qwen2 / DeepSeek-R1-Distill)
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
    for hf_key, gg_key in mapping.items():
        arr = read_gguf_tensor_f32(gmap[gg_key])
        sd[hf_key] = torch.from_numpy(arr).to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    # Ignore non-persistent buffers like rotary inv_freq if any
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing:
        raise RuntimeError(f"Missing weights loading GGUF into PyTorch: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected weights loading GGUF into PyTorch: {unexpected}")
    # Materialize rotary_emb (which lives on meta device if not in state_dict)
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rot_cls = type(model.model.rotary_emb)
        model.model.rotary_emb = rot_cls(config=cfg)
    # Tie lm_head if tied_word_embeddings
    if getattr(cfg, "tie_word_embeddings", False) and "output.weight" not in gmap:
        model.tie_weights()
    model.eval()
    return model
