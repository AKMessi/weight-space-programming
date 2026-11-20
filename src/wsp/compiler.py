"""Zero-gradient Weight-Space Programming (WSP) activation tracer, mechanistic analyzer, and compiler.

Implements five controlled weight-space compilation operators on pretrained SwiGLU LLMs:
  1. uncond_subspace   : Unconditional rank-r direction excision in W_down (Takehi / Steer2Edit style)
  2. sparse_cols       : Top-K loop-contributing column excision in W_down (Takehi et al. exact style)
  3. oblique_nullspace : Linear conditional null-space-gated oblique projection in post-SwiGLU space
  4. dormant_if_then   : Nonlinear thresholded IF-THEN control-flow compiled into dormant SwiGLU neurons
                         using a closed-form Virtual Constant-1 Bias Register b*
  5. ba_wsc            : Block-Aligned Gauge-Balanced Weight-Space Compilation (exact MLP permutation P
                         isolating edited dormant neurons into dedicated 32-element GGUF quant blocks
                         + exact SwiGLU gauge scaling eta for 4-bit / 3-bit survival)

Zero gradients. Zero backpropagation. Zero extra parameters or runtime hooks.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gguf_patch import build_block_aligned_permutation, read_gguf_tensor_f32


def get_transformer_layers(model: torch.nn.Module):
    """Locate decoder layers for Qwen3, Qwen2 (DeepSeek-R1-Distill), Qwen3.5, Llama, etc."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    raise AttributeError(f"Could not locate decoder layers on model class {type(model)}")


def _auroc_binary(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact AUROC via rank statistics (no sklearn dependency overhead)."""
    n_p, n_n = len(pos_scores), len(neg_scores)
    if n_p == 0 or n_n == 0:
        return 0.5
    both = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_p, dtype=np.int8), np.zeros(n_n, dtype=np.int8)])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1, dtype=np.float64)
    # Handle ties with average rank
    sorted_v = both[order]
    i = 0
    while i < len(sorted_v):
        j = i + 1
        while j < len(sorted_v) and sorted_v[j] == sorted_v[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    sum_pos_ranks = ranks[:n_p].sum()
    return float((sum_pos_ranks - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


@dataclass
class LayerStats:
    layer: int
    d_model: int
    d_ff: int
    # Sampled activations (float32)
    X_path: np.ndarray       # [N_p, d_model] post-ffn-norm input to MLP
    X_healthy: np.ndarray    # [N_h, d_model] healthy reasoning post-ffn-norm
    X_preserve: np.ndarray   # [N_pr, d_model] healthy + copy + wikitext post-ffn-norm
    Z_path: np.ndarray       # [N_p, d_ff] post-SwiGLU hidden input to W_down
    Z_healthy: np.ndarray    # [N_h, d_ff]
    Z_preserve: np.ndarray   # [N_pr, d_ff]
    Y_path_mean: np.ndarray  # [d_model] mean MLP output on path tokens
    Y_healthy_mean: np.ndarray  # [d_model] mean MLP output on healthy tokens
    H_path_mean: np.ndarray  # [d_model] mean post-layer residual on path tokens
    H_healthy_mean: np.ndarray  # [d_model] mean post-layer residual on healthy tokens
    H_resolve_mean: np.ndarray  # [d_model] mean post-layer residual near </think> resolution tokens
    neuron_rms_preserve: np.ndarray  # [d_ff] RMS of z_j over all preserve tokens
    neuron_max_preserve: np.ndarray  # [d_ff] max |z_j| over all preserve tokens


def collect_layer_activations(
    model: torch.nn.Module,
    tokenizer,
    *,
    path_texts: list[tuple[str, int]],       # (full_text, onset_char_idx)
    healthy_texts: list[tuple[str, int]],    # (full_text, think_start_char_idx)
    resolve_texts: list[tuple[str, int]],    # (full_text, close_think_char_idx)
    preserve_texts: list[str],               # copy + wikitext preservation texts
    target_layers: list[int] | None = None,
    max_seq_len: int = 1024,
    max_samples_per_seq: int = 32,
) -> dict[int, LayerStats]:
    """Single-pass, zero-gradient PyTorch hook collector for exact MLP/residual diagnostics."""
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    selected = list(range(n_layers)) if target_layers is None else list(target_layers)

    # Storage buffers
    buf: dict[int, dict[str, list[np.ndarray]]] = {
        l: {
            "X_path": [], "X_healthy": [], "X_preserve": [],
            "Z_path": [], "Z_healthy": [], "Z_preserve": [],
            "Y_path": [], "Y_healthy": [],
            "H_path": [], "H_healthy": [], "H_resolve": [],
        }
        for l in selected
    }
    # Running neuron RMS/max across ALL tokens in preserve/healthy sequences (no subsampling!)
    d_ff = layers[selected[0]].mlp.down_proj.in_features
    d_model = layers[selected[0]].mlp.down_proj.out_features
    neuron_sq_sum = {l: np.zeros(d_ff, dtype=np.float64) for l in selected}
    neuron_max_abs = {l: np.zeros(d_ff, dtype=np.float64) for l in selected}
    neuron_tok_count = {l: 0 for l in selected}

    # Register lightweight forward hooks
    curr_cache: dict[int, dict[str, torch.Tensor]] = {}
    handles = []

    def make_mlp_pre_hook(l_idx: int):
        def hook(mod, inp):
            curr_cache.setdefault(l_idx, {})["x"] = inp[0].detach()[0].float()
        return hook

    def make_down_hook(l_idx: int):
        def hook(mod, inp, out):
            curr_cache.setdefault(l_idx, {})["z"] = inp[0].detach()[0].float()
            curr_cache[l_idx]["y"] = out.detach()[0].float()
        return hook

    def make_layer_out_hook(l_idx: int):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            curr_cache.setdefault(l_idx, {})["h"] = h.detach()[0].float()
        return hook

    for l_idx in selected:
        blk = layers[l_idx]
        handles.append(blk.mlp.register_forward_pre_hook(make_mlp_pre_hook(l_idx)))
        handles.append(blk.mlp.down_proj.register_forward_hook(make_down_hook(l_idx)))
        handles.append(blk.register_forward_hook(make_layer_out_hook(l_idx)))

    def run_seq(text: str, start_char: int = 0, end_char: int | None = None, window_tail: int | None = None):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 4:
            return None, None
        # Find token indices overlapping [start_char, end_char)
        idxs = []
        for ti, (c0, c1) in enumerate(offsets):
            if ti < 2:
                continue
            if c1 >= start_char and (end_char is None or c0 <= end_char):
                idxs.append(ti)
        if window_tail is not None and idxs:
            idxs = idxs[-window_tail:]
        if not idxs:
            idxs = list(range(max(2, seq_len - 16), seq_len))
        if len(idxs) > max_samples_per_seq:
            step = np.linspace(0, len(idxs) - 1, max_samples_per_seq, dtype=int)
            idxs = [idxs[i] for i in step]
        curr_cache.clear()
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        return idxs, seq_len

    try:
        # 1. Pathological (loop / cutoff tail) sequences
        for txt, onset_c in path_texts:
            idxs, _ = run_seq(txt, start_char=onset_c)
            if not idxs:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long)
            for l_idx in selected:
                c = curr_cache[l_idx]
                buf[l_idx]["X_path"].append(c["x"][idx_t].numpy())
                buf[l_idx]["Z_path"].append(c["z"][idx_t].numpy())
                buf[l_idx]["Y_path"].append(c["y"][idx_t].numpy())
                buf[l_idx]["H_path"].append(c["h"][idx_t].numpy())

        # 2. Healthy reasoning sequences
        for txt, start_c in healthy_texts:
            idxs, seq_len = run_seq(txt, start_char=start_c)
            if not idxs:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long)
            for l_idx in selected:
                c = curr_cache[l_idx]
                z_all = c["z"][2:].numpy().astype(np.float64)
                neuron_sq_sum[l_idx] += np.sum(z_all * z_all, axis=0)
                neuron_max_abs[l_idx] = np.maximum(neuron_max_abs[l_idx], np.max(np.abs(z_all), axis=0))
                neuron_tok_count[l_idx] += z_all.shape[0]
                x_s = c["x"][idx_t].numpy()
                z_s = c["z"][idx_t].numpy()
                buf[l_idx]["X_healthy"].append(x_s)
                buf[l_idx]["Z_healthy"].append(z_s)
                buf[l_idx]["X_preserve"].append(x_s)
                buf[l_idx]["Z_preserve"].append(z_s)
                buf[l_idx]["Y_healthy"].append(c["y"][idx_t].numpy())
                buf[l_idx]["H_healthy"].append(c["h"][idx_t].numpy())

        # 3. Resolution / transition-to-</think> tokens (where model cleanly concludes!)
        for txt, close_c in resolve_texts:
            # Capture the last 12 tokens right up to and including </think>
            idxs, _ = run_seq(txt, start_char=max(0, close_c - 120), end_char=close_c + 16, window_tail=12)
            if not idxs:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long)
            for l_idx in selected:
                c = curr_cache[l_idx]
                buf[l_idx]["H_resolve"].append(c["h"][idx_t].numpy())

        # 4. General preservation (exact copy + WikiText-2) sequences
        for txt in preserve_texts:
            idxs, _ = run_seq(txt, start_char=0)
            if not idxs:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long)
            for l_idx in selected:
                c = curr_cache[l_idx]
                z_all = c["z"][2:].numpy().astype(np.float64)
                neuron_sq_sum[l_idx] += np.sum(z_all * z_all, axis=0)
                neuron_max_abs[l_idx] = np.maximum(neuron_max_abs[l_idx], np.max(np.abs(z_all), axis=0))
                neuron_tok_count[l_idx] += z_all.shape[0]
                buf[l_idx]["X_preserve"].append(c["x"][idx_t].numpy())
                buf[l_idx]["Z_preserve"].append(c["z"][idx_t].numpy())
    finally:
        for h in handles:
            h.remove()
        curr_cache.clear()

    stats: dict[int, LayerStats] = {}
    for l_idx in selected:
        b = buf[l_idx]
        X_p = np.concatenate(b["X_path"], axis=0)
        X_h = np.concatenate(b["X_healthy"], axis=0)
        X_pr = np.concatenate(b["X_preserve"], axis=0)
        Z_p = np.concatenate(b["Z_path"], axis=0)
        Z_h = np.concatenate(b["Z_healthy"], axis=0)
        Z_pr = np.concatenate(b["Z_preserve"], axis=0)
        Y_pm = np.concatenate(b["Y_path"], axis=0).mean(axis=0)
        Y_hm = np.concatenate(b["Y_healthy"], axis=0).mean(axis=0)
        H_pm = np.concatenate(b["H_path"], axis=0).mean(axis=0)
        H_hm = np.concatenate(b["H_healthy"], axis=0).mean(axis=0)
        H_rm = np.concatenate(b["H_resolve"], axis=0).mean(axis=0) if b["H_resolve"] else H_hm
        rms = np.sqrt(neuron_sq_sum[l_idx] / max(1, neuron_tok_count[l_idx])).astype(np.float32)
        stats[l_idx] = LayerStats(
            layer=l_idx,
            d_model=d_model,
            d_ff=d_ff,
            X_path=X_p,
            X_healthy=X_h,
            X_preserve=X_pr,
            Z_path=Z_p,
            Z_healthy=Z_h,
            Z_preserve=Z_pr,
            Y_path_mean=Y_pm.astype(np.float32),
            Y_healthy_mean=Y_hm.astype(np.float32),
            H_path_mean=H_pm.astype(np.float32),
            H_healthy_mean=H_hm.astype(np.float32),
            H_resolve_mean=H_rm.astype(np.float32),
            neuron_rms_preserve=rms,
            neuron_max_preserve=neuron_max_abs[l_idx].astype(np.float32),
        )
    return stats


def solve_virtual_bias_register(X_preserve: np.ndarray, ridge: float = 1e-2) -> tuple[np.ndarray, dict[str, float]]:
    """Solve b* = argmin_b E[(b^T x - 1)^2] + ridge * ||b||^2 on post-RMSNorm activations.

    Returns (b_star, diagnostics) where b_star^T x ≈ 1 for all tokens x, creating an exact
    Virtual Constant-1 Bias Register in bias-free SwiGLU layers!
    """
    X = X_preserve.astype(np.float64)
    n, d = X.shape
    gram = (X.T @ X) / n
    rhs = X.mean(axis=0)
    b = np.linalg.solve(gram + ridge * np.eye(d), rhs)
    vals = X @ b
    mean_v = float(vals.mean())
    std_v = float(vals.std())
    # Rescale so mean is exactly 1.0
    if abs(mean_v) > 1e-8:
        b = b / mean_v
        vals = vals / mean_v
        std_v = float(vals.std())
    diag = {
        "b_norm": float(np.linalg.norm(b)),
        "mean": float(vals.mean()),
        "std": std_v,
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
    }
    return b.astype(np.float32), diag


def solve_ridge_detector(
    A_pos: np.ndarray,
    A_neg: np.ndarray,
    ridge_scale: float = 0.1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fisher/Ridge-whitened linear discriminant w = (Cov_neg + lam*I)^-1 (mu_pos - mu_neg)."""
    P = A_pos.astype(np.float64)
    N = A_neg.astype(np.float64)
    mu_p = P.mean(axis=0)
    mu_n = N.mean(axis=0)
    diff = mu_p - mu_n
    Nc = N - mu_n
    cov = (Nc.T @ Nc) / max(1, len(N) - 1)
    tr_scale = float(np.trace(cov) / cov.shape[0])
    lam = max(1e-6, ridge_scale * tr_scale)
    w = np.linalg.solve(cov + lam * np.eye(cov.shape[0]), diff)
    w_norm = float(np.linalg.norm(w))
    if w_norm > 0:
        w = w / w_norm
    s_pos = P @ w
    s_neg = N @ w
    auroc = _auroc_binary(s_pos, s_neg)
    fisher = float((s_pos.mean() - s_neg.mean()) / (math.sqrt(0.5 * (s_pos.var() + s_neg.var())) + 1e-8))
    diag = {
        "auroc": round(auroc, 4),
        "fisher_d": round(fisher, 3),
        "neg_mean": float(s_neg.mean()),
        "neg_std": float(s_neg.std()),
        "neg_p95": float(np.percentile(s_neg, 95)),
        "neg_p99": float(np.percentile(s_neg, 99)),
        "pos_mean": float(s_pos.mean()),
        "pos_p10": float(np.percentile(s_pos, 10)),
        "pos_median": float(np.median(s_pos)),
    }
    return w.astype(np.float32), diag


def analyze_layers(
    layer_stats: dict[int, LayerStats],
    gguf_tensors: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce full mechanistic diagnostics across all traced layers (zero test data used)."""
    rows = []
    for l_idx in sorted(layer_stats):
        st = layer_stats[l_idx]
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"])
        col_norms = np.linalg.norm(W_down.astype(np.float64), axis=0)
        contrib_rms = st.neuron_rms_preserve.astype(np.float64) * col_norms

        # Residual loop direction u_loop = unit(H_path - H_healthy)
        d_h = (st.H_path_mean - st.H_healthy_mean).astype(np.float64)
        d_h_norm = float(np.linalg.norm(d_h))
        u_loop = d_h / (d_h_norm + 1e-12)

        # MLP contribution along u_loop
        d_y = (st.Y_path_mean - st.Y_healthy_mean).astype(np.float64)
        mlp_loop_align = float(np.dot(d_y, u_loop))
        mlp_delta_norm = float(np.linalg.norm(d_y))

        # Detector separability in X (pre-MLP) and Z (post-SwiGLU)
        _, diag_x = solve_ridge_detector(st.X_path, st.X_preserve, ridge_scale=0.1)
        # Fast diagonal-regularized Fisher in Z (since d_ff can be 3072..8960)
        z_mu_p = st.Z_path.mean(axis=0).astype(np.float64)
        z_mu_n = st.Z_preserve.mean(axis=0).astype(np.float64)
        z_var_n = st.Z_preserve.var(axis=0).astype(np.float64)
        w_z_diag = (z_mu_p - z_mu_n) / (z_var_n + 0.1 * float(z_var_n.mean()) + 1e-8)
        w_z_diag /= np.linalg.norm(w_z_diag) + 1e-12
        auroc_z = _auroc_binary(st.Z_path @ w_z_diag, st.Z_preserve @ w_z_diag)

        # Virtual bias register quality
        _, vb_diag = solve_virtual_bias_register(st.X_preserve, ridge=1e-2)

        # Dormancy quantiles
        sorted_c = np.sort(contrib_rms)
        med_c = float(np.median(contrib_rms))
        dormant_ratio_32 = float(sorted_c[31] / (med_c + 1e-12))

        rows.append({
            "layer": l_idx,
            "res_diff_norm": round(d_h_norm, 4),
            "mlp_diff_norm": round(mlp_delta_norm, 4),
            "mlp_loop_proj": round(mlp_loop_align, 4),
            "auroc_x": diag_x["auroc"],
            "fisher_x": diag_x["fisher_d"],
            "auroc_z_diag": round(auroc_z, 4),
            "vb_std": round(vb_diag["std"], 4),
            "vb_norm": round(vb_diag["b_norm"], 3),
            "neuron_contrib_median": round(med_c, 5),
            "neuron_contrib_32nd": round(float(sorted_c[31]), 5),
            "dormant_32_to_median": round(dormant_ratio_32, 4),
        })
    return rows


def get_action_vector(st: LayerStats, action_mode: str = "resolve") -> np.ndarray:
    """Compute unit corrective residual direction v_action."""
    if action_mode == "anti_loop":
        v = (st.H_healthy_mean - st.H_path_mean).astype(np.float64)
    elif action_mode == "resolve":
        v = (st.H_resolve_mean - st.H_path_mean).astype(np.float64)
    elif action_mode == "anti_mlp_loop":
        v = (st.Y_healthy_mean - st.Y_path_mean).astype(np.float64)
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")
    v_norm = float(np.linalg.norm(v))
    return (v / (v_norm + 1e-12)).astype(np.float32)


def compile_uncond_subspace(
    gguf_tensors: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    layers: list[int],
    alpha: float,
    rank: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Baseline 1: Unconditional rank-r direction excision in W_down: W <- (I - alpha * U U^T) W."""
    edited: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"method": "uncond_subspace", "layers": layers, "alpha": alpha, "rank": rank}
    for l_idx in layers:
        st = layer_stats[l_idx]
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"]).astype(np.float64)
        u = (st.H_path_mean - st.H_healthy_mean).astype(np.float64)
        u /= np.linalg.norm(u) + 1e-12
        if rank == 1:
            U = u[:, None]
        else:
            # Top-r directions spanning path - healthy mean + top PCs of (Y_path - Y_healthy)
            u2 = (st.Y_path_mean - st.Y_healthy_mean).astype(np.float64)
            u2 = u2 - np.dot(u2, u) * u
            u2 /= np.linalg.norm(u2) + 1e-12
            U = np.column_stack([u, u2])[:, :rank]
        W_new = W_down - alpha * (U @ (U.T @ W_down))
        edited[f"blk.{l_idx}.ffn_down.weight"] = W_new.astype(np.float32)
    return edited, info


def compile_sparse_cols(
    gguf_tensors: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    layers: list[int],
    alpha: float,
    top_k: int = 64,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Baseline 2: Takehi et al. style sparse column excision on top-K loop-writing columns of W_down."""
    edited: dict[str, np.ndarray] = {}
    selected_cols: dict[int, list[int]] = {}
    for l_idx in layers:
        st = layer_stats[l_idx]
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"]).astype(np.float64)
        u = (st.H_path_mean - st.H_healthy_mean).astype(np.float64)
        u /= np.linalg.norm(u) + 1e-12
        # Per-neuron excess contribution along loop direction: (u^T W_down[:, j]) * (E_path[z_j] - E_healthy[z_j])
        dz = (st.Z_path.mean(axis=0) - st.Z_healthy.mean(axis=0)).astype(np.float64)
        col_proj = u @ W_down  # [d_ff]
        scores = col_proj * dz
        cols = np.argsort(scores)[::-1][:top_k]
        selected_cols[l_idx] = [int(c) for c in cols]
        W_new = W_down.copy()
        # Subtract alpha * u * (u^T W[:, cols]) only from selected columns
        W_new[:, cols] -= alpha * np.outer(u, col_proj[cols])
        edited[f"blk.{l_idx}.ffn_down.weight"] = W_new.astype(np.float32)
    return edited, {
        "method": "sparse_cols",
        "layers": layers,
        "alpha": alpha,
        "top_k": top_k,
        "selected_cols": selected_cols,
    }


def compile_oblique_nullspace(
    gguf_tensors: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    layers: list[int],
    alpha: float,
    action_mode: str = "resolve",
    ridge_scale: float = 0.2,
    null_rank: int = 64,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Method A: Null-Space-Gated Oblique Rank-1 Surgery in W_down.

    Writes Delta W_down = alpha * v_action * w_det^T where w_det lives in the
    orthogonal complement of the top-`null_rank` principal subspace of Z_preserve
    (so Delta W_down @ z_preserve ≈ 0, while Delta W_down @ z_path ≈ alpha * v_action).
    """
    edited: dict[str, np.ndarray] = {}
    per_layer_diag: dict[int, dict[str, float]] = {}
    for l_idx in layers:
        st = layer_stats[l_idx]
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"]).astype(np.float64)
        v_act = get_action_vector(st, action_mode).astype(np.float64)

        Z_pr = st.Z_preserve.astype(np.float64)
        Z_pa = st.Z_path.astype(np.float64)
        # Dual-SVD (fast even when d_ff=8960, since N_pr ~ 1500)
        # Compute top-r right singular vectors of Z_pr (uncentered, so also orthogonal to mean!)
        _, S_pr, Vt_pr = np.linalg.svd(Z_pr, full_matrices=False)
        r = min(null_rank, Vt_pr.shape[0] - 1)
        V_keep = Vt_pr[:r]  # [r, d_ff]

        # Contrastive hidden direction projected strictly into null space of V_keep
        d_z = Z_pa.mean(axis=0) - Z_pr.mean(axis=0)
        # Variance-weighted in residual complement for stability
        z_var = Z_pr.var(axis=0)
        d_z_w = d_z / (z_var + ridge_scale * float(z_var.mean()) + 1e-8)
        w_det = d_z_w - V_keep.T @ (V_keep @ d_z_w)
        # Re-orthogonalize once for FP64 precision
        w_det = w_det - V_keep.T @ (V_keep @ w_det)

        s_pa = Z_pa @ w_det
        s_pr = Z_pr @ w_det
        gap = float(s_pa.mean() - s_pr.mean())
        if abs(gap) < 1e-12:
            raise RuntimeError(f"Zero null-space gap at layer {l_idx}")
        # Normalize so E_path[w_det^T z] - E_preserve[w_det^T z] == 1.0
        w_det = w_det / gap
        s_pa = Z_pa @ w_det
        s_pr = Z_pr @ w_det

        # Scale alpha relative to typical layer residual step magnitude (RMS of Y_healthy)
        # Wait: alpha is directly the residual-stream injection magnitude (in L2 units) on pathological tokens!
        delta_W = alpha * np.outer(v_act, w_det)
        W_new = W_down + delta_W
        edited[f"blk.{l_idx}.ffn_down.weight"] = W_new.astype(np.float32)

        per_layer_diag[l_idx] = {
            "auroc": round(_auroc_binary(s_pa, s_pr), 4),
            "preserve_rms": round(float(np.sqrt(np.mean(s_pr ** 2))), 5),
            "preserve_mean": round(float(np.mean(s_pr)), 5),
            "path_mean": round(float(np.mean(s_pa)), 5),
            "delta_fro": round(float(np.linalg.norm(delta_W)), 5),
            "delta_max_abs": round(float(np.max(np.abs(delta_W))), 6),
        }

    return edited, {
        "method": "oblique_nullspace",
        "layers": layers,
        "alpha": alpha,
        "action_mode": action_mode,
        "null_rank": null_rank,
        "per_layer": per_layer_diag,
    }


def compile_dormant_if_then(
    gguf_tensors: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    layers: list[int],
    alpha: float,
    *,
    k_neurons: int = 32,
    gamma: float = 4.0,
    tau_quantile: float = 0.96,
    action_mode: str = "resolve",
    block_align: bool = False,
    gauge_balance: bool = True,
    block_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Method B & C: Nonlinear Thresholded IF-THEN Control-Flow Compilation into Dormant SwiGLU Neurons.

    If `block_align=True` (BA-WSC), applies exact MLP permutation symmetry P to gather the
    `k_neurons` edited dormant slots into contiguous quantization block [0 : k_neurons]
    in W_gate, W_up, and W_down, isolating them from all normal weights under 4-bit/3-bit GGUF quantization!
    """
    edited: dict[str, np.ndarray] = {}
    per_layer_diag: dict[int, dict[str, Any]] = {}

    for l_idx in layers:
        st = layer_stats[l_idx]
        W_gate = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_gate.weight"]).astype(np.float64)
        W_up = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_up.weight"]).astype(np.float64)
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"]).astype(np.float64)

        # 1. Rank neurons by exact healthy/preserve residual contribution energy
        col_norms = np.linalg.norm(W_down, axis=0)
        contrib_rms = st.neuron_rms_preserve.astype(np.float64) * col_norms
        dormant_idx = [int(i) for i in np.argsort(contrib_rms)[:k_neurons]]
        evicted_energy_max = float(contrib_rms[dormant_idx[-1]])
        median_energy = float(np.median(contrib_rms))

        # 2. Solve Virtual Constant-1 Bias Register b* and Ridge Trigger u_trig in post-RMSNorm space
        b_star, vb_diag = solve_virtual_bias_register(st.X_preserve, ridge=1e-2)
        b_star_64 = b_star.astype(np.float64)
        u_trig, det_diag = solve_ridge_detector(st.X_path, st.X_preserve, ridge_scale=0.1)
        u_trig_64 = u_trig.astype(np.float64)

        # 3. Choose threshold tau from calibration preserve quantile (zero test leakage!)
        s_pr = st.X_preserve.astype(np.float64) @ u_trig_64
        s_pa = st.X_path.astype(np.float64) @ u_trig_64
        tau = float(np.quantile(s_pr, tau_quantile))
        # Ensure tau sits strictly below median pathological score so circuit actually fires on loops
        if tau >= float(np.median(s_pa)):
            tau = 0.5 * (float(np.quantile(s_pr, 0.90)) + float(np.median(s_pa)))

        # Compute exact expected SiLU activation on path vs preserve with virtual bias
        w_gate_vec = gamma * (u_trig_64 - tau * b_star_64)
        pre_pr = st.X_preserve.astype(np.float64) @ w_gate_vec
        pre_pa = st.X_path.astype(np.float64) @ w_gate_vec
        silu_pr = pre_pr / (1.0 + np.exp(-np.clip(pre_pr, -40, 40)))
        silu_pa = pre_pa / (1.0 + np.exp(-np.clip(pre_pa, -40, 40)))
        b_pr = st.X_preserve.astype(np.float64) @ b_star_64
        b_pa = st.X_path.astype(np.float64) @ b_star_64
        # Raw unscaled single-neuron output coefficient: silu(w_gate^T x) * (b*^T x)
        act_pr = silu_pr * b_pr
        act_pa = silu_pa * b_pa
        mean_act_pa = float(np.mean(act_pa))
        if mean_act_pa <= 1e-4:
            # Soft fallback if tau was too steep
            tau = float(np.quantile(s_pr, 0.85))
            w_gate_vec = gamma * (u_trig_64 - tau * b_star_64)
            pre_pr = st.X_preserve.astype(np.float64) @ w_gate_vec
            pre_pa = st.X_path.astype(np.float64) @ w_gate_vec
            silu_pr = pre_pr / (1.0 + np.exp(-np.clip(pre_pr, -40, 40)))
            silu_pa = pre_pa / (1.0 + np.exp(-np.clip(pre_pa, -40, 40)))
            act_pr = silu_pr * b_pr
            act_pa = silu_pa * b_pa
            mean_act_pa = float(np.mean(act_pa))

        # 4. Target total residual jump on pathological tokens = alpha * v_act
        # Each of the k_neurons contributes (1 / k_neurons) of the total jump!
        v_act = get_action_vector(st, action_mode).astype(np.float64)
        per_neuron_scale = alpha / (k_neurons * max(mean_act_pa, 1e-4))

        # 5. Exact SwiGLU Gauge Scaling eta between W_up and W_down:
        #    (eta * b*) and (per_neuron_scale / eta) * v_act
        #    Choose eta so max(|W_up_row|) and max(|W_down_col|) are balanced in 4-bit dynamic range!
        w_up_base = b_star_64
        w_down_base = per_neuron_scale * v_act
        if gauge_balance:
            m_up = float(np.max(np.abs(w_up_base))) + 1e-12
            m_dn = float(np.max(np.abs(w_down_base))) + 1e-12
            eta = math.sqrt(m_dn / m_up)
        else:
            eta = 1.0
        w_up_vec = eta * w_up_base
        w_down_vec = w_down_base / eta

        # Write compiled IF-THEN circuit into the k_neurons dormant slots
        W_gate_new = W_gate.copy()
        W_up_new = W_up.copy()
        W_down_new = W_down.copy()
        for j in dormant_idx:
            W_gate_new[j, :] = w_gate_vec
            W_up_new[j, :] = w_up_vec
            W_down_new[:, j] = w_down_vec

        # 6. Optional Block-Aligned Exact MLP Permutation (BA-WSC!)
        perm_applied = False
        if block_align:
            perm = build_block_aligned_permutation(st.d_ff, dormant_idx, block_size=block_size, dest_block=0)
            W_gate_new = W_gate_new[perm, :]
            W_up_new = W_up_new[perm, :]
            W_down_new = W_down_new[:, perm]
            perm_applied = True

        edited[f"blk.{l_idx}.ffn_gate.weight"] = W_gate_new.astype(np.float32)
        edited[f"blk.{l_idx}.ffn_up.weight"] = W_up_new.astype(np.float32)
        edited[f"blk.{l_idx}.ffn_down.weight"] = W_down_new.astype(np.float32)

        # Compute exact post-compilation firing contrast on calibration tokens
        inj_pr_rms = float(np.sqrt(np.mean((k_neurons * per_neuron_scale * act_pr) ** 2)))
        inj_pr_p95 = float(np.percentile(np.abs(k_neurons * per_neuron_scale * act_pr), 95))
        inj_pa_mean = float(np.mean(k_neurons * per_neuron_scale * act_pa))
        per_layer_diag[l_idx] = {
            "dormant_idx_first8": dormant_idx[:8],
            "k_neurons": k_neurons,
            "evicted_max_rms": round(evicted_energy_max, 6),
            "median_neuron_rms": round(median_energy, 6),
            "detector_auroc": det_diag["auroc"],
            "vb_std": round(vb_diag["std"], 4),
            "tau": round(tau, 4),
            "gamma": gamma,
            "eta_gauge": round(eta, 4),
            "w_gate_max_abs": round(float(np.max(np.abs(w_gate_vec))), 4),
            "w_up_max_abs": round(float(np.max(np.abs(w_up_vec))), 4),
            "w_down_max_abs": round(float(np.max(np.abs(w_down_vec))), 4),
            "inj_preserve_rms": round(inj_pr_rms, 5),
            "inj_preserve_p95": round(inj_pr_p95, 5),
            "inj_path_mean": round(inj_pa_mean, 4),
            "selectivity_ratio": round(inj_pa_mean / (inj_pr_rms + 1e-8), 2),
            "block_aligned": perm_applied,
        }

    return edited, {
        "method": "ba_wsc" if block_align else "dormant_if_then",
        "layers": layers,
        "alpha": alpha,
        "k_neurons": k_neurons,
        "gamma": gamma,
        "tau_quantile": tau_quantile,
        "action_mode": action_mode,
        "block_align": block_align,
        "gauge_balance": gauge_balance,
        "per_layer": per_layer_diag,
    }


def compile_ba_wsc_noop(
    gguf_tensors: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    layers: list[int],
    k_neurons: int = 32,
    block_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pure permutation control: applies exact same BA-WSC permutation with ZERO weight edit!"""
    edited: dict[str, np.ndarray] = {}
    for l_idx in layers:
        st = layer_stats[l_idx]
        W_gate = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_gate.weight"])
        W_up = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_up.weight"])
        W_down = read_gguf_tensor_f32(gguf_tensors[f"blk.{l_idx}.ffn_down.weight"])
        col_norms = np.linalg.norm(W_down.astype(np.float64), axis=0)
        contrib_rms = st.neuron_rms_preserve.astype(np.float64) * col_norms
        dormant_idx = [int(i) for i in np.argsort(contrib_rms)[:k_neurons]]
        perm = build_block_aligned_permutation(st.d_ff, dormant_idx, block_size=block_size, dest_block=0)
        edited[f"blk.{l_idx}.ffn_gate.weight"] = W_gate[perm, :]
        edited[f"blk.{l_idx}.ffn_up.weight"] = W_up[perm, :]
        edited[f"blk.{l_idx}.ffn_down.weight"] = W_down[:, perm]
    return edited, {"method": "ba_wsc_noop", "layers": layers, "k_neurons": k_neurons}


def compile_random_matched(
    gguf_tensors: dict[str, Any],
    layers: list[int],
    reference_edited: dict[str, np.ndarray],
    seed: int = 999,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Negative control: random Gaussian perturbation matching exact per-tensor Frobenius delta norm."""
    rng = np.random.default_rng(seed)
    edited: dict[str, np.ndarray] = {}
    for name, ref_w in reference_edited.items():
        orig_w = read_gguf_tensor_f32(gguf_tensors[name]).astype(np.float64)
        delta_norm = float(np.linalg.norm(ref_w.astype(np.float64) - orig_w))
        noise = rng.standard_normal(orig_w.shape)
        noise *= delta_norm / (float(np.linalg.norm(noise)) + 1e-12)
        edited[name] = (orig_w + noise).astype(np.float32)
    return edited, {"method": "random_matched", "layers": layers, "seed": seed}
