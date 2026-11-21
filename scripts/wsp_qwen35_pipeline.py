"""Model 2 Pipeline: Qwen3.5-0.8B (Hybrid 18 Gated-DeltaNet + 6 Full-Attention, the Sparks of WSP model!).

Runs:
  Stage 0: Baseline characterization across BF16, Q8_0, Q4_K_M, Q4_0, Q3_K_M on cal_math_50
           under BOTH greedy (temp=0.0) and official stochastic sampling (temp=0.6, top_p=0.95)
           + WikiText-2 PPL + cal_copy_36!
  Stage 1: Exact float32 PyTorch mechanistic layer profiling across ALL 24 layers, explicitly
           contrasting Gated-DeltaNet linear-recurrent layers vs Full-Attention layers (3,7,11,15,19,23)!
  Stage 2: Empirical Theorem 1 block-level quantization audit on Qwen3.5-0.8B (Q4_K_M, Q4_0, Q3_K_M)!
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.wsp_diagnose_and_compile import load_cal_trajectories
from scripts.wsp_quant_audit import audit_quant_pair
from src.wsp.compiler import (
    analyze_layers,
    collect_layer_activations,
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_sparse_cols,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.llama_eval import measure_perplexity, run_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["baseline", "profile", "theorem1", "all"], default="all")
    p.add_argument("--max-tokens", type=int, default=2048)
    a = p.parse_args()

    model_name = "qwen35_0.8b"
    base_gguf = Path("local/models/Qwen3.5-0.8B-BF16.gguf")
    hf_dir = Path("local/hf_models/Qwen3.5-0.8B")
    base_out = Path("runs/wsp_baselines") / model_name
    art_base = Path("artifacts/wsp") / model_name / "baselines"
    rep_dir = Path("research/wsp") / model_name
    cache_dir = Path("local/cache") / model_name
    for d in (base_out, art_base, rep_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    if a.stage in ("baseline", "all"):
        print("\n================ [Qwen3.5-0.8B] BASELINE CHARACTERIZATION ================", flush=True)
        res_path = base_out / "baseline_summary.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {"model_name": model_name, "precisions": {}}

        if "FP" not in res["precisions"]:
            ppl_fp = measure_perplexity(base_gguf, chunks=32, ctx_size=512)
            print(f"  PPL(FP): {ppl_fp['ppl']:.4f} +/- {ppl_fp['ppl_se']:.4f}", flush=True)
            cal_math_greedy = run_benchmark(
                model_path=base_gguf, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
                run_dir=base_out / "fp_cal_math50", max_tokens=a.max_tokens, temperature=0.0,
                parallel=4, ctx_per_slot=2560, tag="qwen35_fp_cal_math50_greedy",
            )
            cal_math_stoch = run_benchmark(
                model_path=base_gguf, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
                run_dir=base_out / "fp_cal_math50_stoch", max_tokens=a.max_tokens,
                temperature=0.6, top_p=0.95, top_k=20, seed=42,
                parallel=4, ctx_per_slot=2560, tag="qwen35_fp_cal_math50_stoch",
            )
            cal_gsm = run_benchmark(
                model_path=base_gguf, dataset_path="datasets/wsp/cal_gsm8k_48.jsonl", benchmark="gsm8k",
                run_dir=base_out / "fp_cal_gsm48", max_tokens=a.max_tokens,
                temperature=0.6, top_p=0.95, top_k=20, seed=42,
                parallel=4, ctx_per_slot=2560, tag="qwen35_fp_cal_gsm48_stoch",
            )
            cal_copy = run_benchmark(
                model_path=base_gguf, dataset_path="datasets/wsp/cal_copy_36.jsonl", benchmark="copy",
                run_dir=base_out / "fp_cal_copy36", max_tokens=512, temperature=0.0,
                enable_thinking=False, parallel=4, ctx_per_slot=1024, tag="qwen35_fp_cal_copy36",
            )
            res["precisions"]["FP"] = {
                "ppl": ppl_fp,
                "cal_math50_greedy": cal_math_greedy,
                "cal_math50_stoch": cal_math_stoch,
                "cal_gsm48_stoch": cal_gsm,
                "cal_copy36": cal_copy,
            }
            res_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

        for qt in ["Q8_0", "Q4_K_M", "Q4_0", "Q3_K_M"]:
            if qt in res["precisions"]:
                continue
            qp = art_base / f"{model_name}-{qt}.gguf"
            qm = quantize_gguf(base_gguf, qp, qt) if not qp.exists() else {"dst_path": str(qp), "dst_bytes": qp.stat().st_size, "qtype": qt}
            ppl_q = measure_perplexity(qp, chunks=32, ctx_size=512)
            print(f"  PPL({qt}): {ppl_q['ppl']:.4f} +/- {ppl_q['ppl_se']:.4f} | {qm['dst_bytes']/1e6:.1f} MB", flush=True)
            ev_q = run_benchmark(
                model_path=qp, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
                run_dir=base_out / f"{qt}_cal_math50_stoch", max_tokens=a.max_tokens,
                temperature=0.6, top_p=0.95, top_k=20, seed=42,
                parallel=4, ctx_per_slot=2560, tag=f"qwen35_{qt}_cal_math50_stoch",
            )
            res["precisions"][qt] = {"quant_meta": qm, "ppl": ppl_q, "cal_math50_stoch": ev_q}
            res_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    if a.stage in ("profile", "all"):
        print("\n================ [Qwen3.5-0.8B] HYBRID DELTANET vs ATTENTION PROFILING ================", flush=True)
        stats_pkl = cache_dir / "layer_stats.pkl"
        reader = GGUFReader(str(base_gguf), mode="r")
        gmap = {t.name: t for t in reader.tensors}
        if stats_pkl.exists():
            with stats_pkl.open("rb") as f:
                pl = pickle.load(f)
            layer_stats, counts = pl["layer_stats"], pl["counts"]
        else:
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(str(hf_dir))
            model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.float32).eval()
            # Use stochastic cal_math50 + cal_gsm48 trajectories (unconfounded by greedy!)
            path_t, healthy_t, resolve_t, preserve_t, counts = load_cal_trajectories(
                tok,
                base_out / "fp_cal_math50_stoch",
                base_out / "fp_cal_gsm48",
                base_out / "fp_cal_copy36",
                Path("datasets/wsp/cal_wikitext_64.jsonl"),
                max_path=18, max_healthy=18, max_preserve=24,
            )
            print("Qwen3.5-0.8B cal trajectories:", counts, flush=True)
            layer_stats = collect_layer_activations(
                model, tok, path_texts=path_t, healthy_texts=healthy_t,
                resolve_texts=resolve_t, preserve_texts=preserve_t, max_seq_len=384,
            )
            del model
            with stats_pkl.open("wb") as f:
                pickle.dump({"layer_stats": layer_stats, "counts": counts}, f)
            print(f"Collected 24 hybrid layers in {time.time()-t0:.1f}s", flush=True)

        prof = analyze_layers(layer_stats, gmap)
        for row in prof:
            # Annotate exact layer type: Full Attention at 3,7,11,15,19,23 vs Gated-DeltaNet!
            row["layer_type"] = "full_attn" if (row["layer"] + 1) % 4 == 0 else "gated_deltanet"
        (rep_dir / "layer_profile.json").write_text(
            json.dumps({"model_name": model_name, "counts": counts, "layers": prof}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{'L':>3s} {'type':14s} {'resDiff':>8s} {'mlpProj':>8s} {'aucX':>6s} {'vbStd':>7s} {'d32/med':>8s}")
        for r in prof:
            print(f"{r['layer']:3d} {r['layer_type']:14s} {r['res_diff_norm']:8.3f} {r['mlp_loop_proj']:8.4f} {r['auroc_x']:6.3f} {r['vb_std']:7.4f} {r['dormant_32_to_median']:8.4f}")

    if a.stage in ("theorem1", "all"):
        print("\n================ [Qwen3.5-0.8B] THEOREM 1 VERIFICATION (HYBRID L15 & L19) ================", flush=True)
        with (cache_dir / "layer_stats.pkl").open("rb") as f:
            stats = pickle.load(f)["layer_stats"]
        gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}
        layers = [15, 19]  # Exact Full-Attention layers from Sparks of WSP!
        tmp = Path("artifacts/wsp") / model_name / "theorem1_audit"
        tmp.mkdir(parents=True, exist_ok=True)
        ed_noop, _ = compile_ba_wsc_noop(gmap, stats, layers, k_neurons=32)
        noop_fp = tmp / "noop.gguf"
        patch_gguf_tensors(base_gguf, noop_fp, ed_noop)
        noop_q = {}
        for qt in ["Q4_0", "Q4_K_M", "Q3_K_M"]:
            noop_q[qt] = tmp / f"noop-{qt}.gguf"
            quantize_gguf(noop_fp, noop_q[qt], qt)

        cases = [
            ("uncond_a0.6", compile_uncond_subspace(gmap, stats, layers, alpha=0.6), False),
            ("sparse64_a1.0", compile_sparse_cols(gmap, stats, layers, alpha=1.0, top_k=64), False),
            ("oblique_a0.25", compile_oblique_nullspace(gmap, stats, layers, alpha=0.25, action_mode="resolve"), False),
            ("dormant32_scattered", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, block_align=False), False),
            ("ba_wsc_32_aligned", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, block_align=True), True),
        ]
        all_out = {}
        for name, (ed, cinfo), is_ba in cases:
            fp_p = tmp / f"{name}.gguf"
            patch_gguf_tensors(base_gguf, fp_p, ed)
            all_out[name] = {"compile": cinfo, "quants": {}}
            for qt in ["Q4_0", "Q4_K_M", "Q3_K_M"]:
                bq = art_base / f"{model_name}-{qt}.gguf"
                qp = tmp / f"{name}-{qt}.gguf"
                quantize_gguf(fp_p, qp, qt)
                ref_fp = noop_fp if is_ba else base_gguf
                ref_q = noop_q[qt] if is_ba else bq
                res_a = audit_quant_pair(ref_fp, fp_p, ref_q, qp, ["blk.15.ffn_down.weight", "blk.19.ffn_down.weight"])
                all_out[name]["quants"][qt] = res_a
                r19 = res_a["blk.19.ffn_down.weight"]
                print(f"{name:22s} | {qt:6s} | L19 cos={r19['cosine_sim_full']:.5f} collat/sig={r19['collateral_to_signal_ratio']:.5f} mix={r19['mixed_32blocks']} pure={r19['pure_edit_32blocks']}", flush=True)
                qp.unlink()
            fp_p.unlink()
        for p_f in list(tmp.glob("*.gguf")):
            p_f.unlink()
        (rep_dir / "theorem1_quant_audit.json").write_text(json.dumps(all_out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
