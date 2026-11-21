"""Paper 2 Quantization Survival & Post-Quantization Healing Matrix on Qwen3-0.6B (Q4_K_M, Q4_0, Q3_K_M).

Evaluates on 100% untouched held-out MATH-500-strat150 + Copy-90 + 32-chunk WikiText-2 PPL + exact
Theorem 1 block-level tensor audit:
  1. Unedited quantized baselines: Q4_K_M, Q4_0, Q3_K_M (measures held-out quantization cliff)
  2. FP-compiled edits quantized directly:
       - oblique_res_a0.25 (diffuse rank-1 null-space oblique)
       - bawsc32_res_a0.25 (Theorem 1 Block-Aligned dormant SwiGLU)
       - dormant32_res_a0.25 (exact scattered ablation without Block Alignment)
       - uncond_a0.8 (existing unconditional WSP baseline)
  3. Post-Quantization Weight-Space Healing (PQ-Oblique & PQ-BA-WSC):
       Compiled directly on the exact dequantized Q4_0 / Q3_K_M activations and post-quantization
       calibration loop trajectories, testing whether zero-gradient WSP heals quantization collapse!
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from gguf import GGUFReader

from scripts.wsp_post_quant_heal import trace_quantized_model
from scripts.wsp_quant_audit import audit_quant_pair
from src.wsp.compiler import (
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.llama_eval import mcnemar_exact, measure_perplexity, run_benchmark


def eval_quant_arm(
    *,
    tag: str,
    q_path: Path,
    run_root: Path,
    ds_math: Path,
    ids: list[str],
    base_std: list[bool],
    base_eos: list[bool],
    base_cls: list[bool],
    base_ppl: float,
    audit: dict[str, Any] | None = None,
    q_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ppl = measure_perplexity(q_path, chunks=32, ctx_size=512)
    cp = run_benchmark(
        model_path=q_path,
        dataset_path="datasets/wsp/test_copy_90.jsonl",
        benchmark="copy",
        run_dir=run_root / f"{tag}_copy90",
        max_tokens=512,
        temperature=0.0,
        enable_thinking=False,
        parallel=4,
        ctx_per_slot=1024,
        tag=f"{tag}_copy90",
    )
    ev = run_benchmark(
        model_path=q_path,
        dataset_path=ds_math,
        benchmark="math",
        run_dir=run_root / f"{tag}_math150",
        max_tokens=2048,
        temperature=0.0,
        parallel=4,
        ctx_per_slot=2560,
        tag=f"{tag}_math150",
    )
    a_map = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_root / f"{tag}_math150" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std_f = [bool(a_map[i]["grade"]["correct"]) for i in ids]
    eos_f = [bool(a_map[i]["grade"]["correct"]) and a_map[i]["finish_reason"] == "stop" for i in ids]
    cls_f = [bool(a_map[i]["grade"]["correct"]) and a_map[i]["grade"]["has_think_close"] for i in ids]
    mc_std = mcnemar_exact(base_std, std_f)
    mc_eos = mcnemar_exact(base_eos, eos_f)
    mc_cls = mcnemar_exact(base_cls, cls_f)
    print(
        f"QUANT {tag:28s}: PPL={ppl['ppl']:.4f} ({ppl['ppl']-base_ppl:+.4f}) | "
        f"STD={sum(std_f):2d}/150 (p={mc_std['mcnemar_p_two_sided']:.4f}) | "
        f"CLOSED={sum(cls_f):2d}/150 (p={mc_cls['mcnemar_p_two_sided']:.4f}) | "
        f"EOS={sum(eos_f):2d}/150 (p={mc_eos['mcnemar_p_two_sided']:.4f}) | "
        f"lim={ev['hit_limit_count']} lp={ev['loop_count']} | COPY={cp['correct']}/90",
        flush=True,
    )
    return {
        "ppl": ppl,
        "test_math500_strat150": ev,
        "test_copy_90": cp,
        "clean_eos_correct": sum(eos_f),
        "closed_think_correct": sum(cls_f),
        "mcnemar_vs_same_quant_base": {"std": mc_std, "clean_eos": mc_eos, "closed_think": mc_cls},
        "block_audit": audit,
        "quant_meta": q_meta,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qtypes", default="Q4_0,Q3_K_M,Q4_K_M")
    p.add_argument("--with-pq-heal", action="store_true")
    a = p.parse_args()

    base_fp = Path("local/models/Qwen3-0.6B-BF16.gguf")
    gmap = {t.name: t for t in GGUFReader(str(base_fp)).tensors}
    with open("local/cache/qwen3_0.6b/layer_stats.pkl", "rb") as f:
        fp_stats = pickle.load(f)["layer_stats"]

    layers = [18, 20, 22]
    ds_math = Path("datasets/wsp/test_math500_strat150.jsonl")
    ids = [json.loads(l)["id"] for l in ds_math.read_text(encoding="utf-8").splitlines() if l.strip()]

    run_root = Path("runs/wsp_study/qwen3_0.6b/quant")
    art_q = Path("artifacts/wsp/qwen3_0.6b/quant")
    art_fp = Path("artifacts/wsp/qwen3_0.6b/frozen")
    run_root.mkdir(parents=True, exist_ok=True)
    art_q.mkdir(parents=True, exist_ok=True)
    art_fp.mkdir(parents=True, exist_ok=True)

    out_path = Path("research/wsp/qwen3_0.6b/quant_matrix_results.json")
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    # Prepare FP frozen checkpoints for all transfer arms
    fp_specs = {
        "ba_wsc_noop": compile_ba_wsc_noop(gmap, fp_stats, layers, k_neurons=32),
        "oblique_res_a0.25": compile_oblique_nullspace(gmap, fp_stats, layers, alpha=0.25, action_mode="resolve", null_rank=64),
        "bawsc32_res_a0.25": compile_dormant_if_then(gmap, fp_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True),
        "dormant32_res_a0.25": compile_dormant_if_then(gmap, fp_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=False),
        "uncond_a0.8": compile_uncond_subspace(gmap, fp_stats, layers, alpha=0.8, rank=1),
    }
    fp_paths: dict[str, Path] = {}
    for nm, (ed, cinfo) in fp_specs.items():
        fp_p = art_fp / f"{nm}.gguf"
        if not fp_p.exists():
            patch_gguf_tensors(base_fp, fp_p, ed, metadata_extra=cinfo)
        fp_paths[nm] = fp_p

    for qt in [x.strip() for x in a.qtypes.split(",") if x.strip()]:
        print(f"\n================ QWEN3-0.6B QUANTIZATION MATRIX: {qt} ================", flush=True)
        base_q = Path(f"artifacts/wsp/qwen3_0.6b/baselines/qwen3_0.6b-{qt}.gguf")
        if not base_q.exists():
            quantize_gguf(base_fp, base_q, qt)

        # 1. Unedited quantized baseline
        b_tag = f"baseline_{qt}"
        if b_tag not in results:
            ppl_b = measure_perplexity(base_q, chunks=32, ctx_size=512)
            cp_b = run_benchmark(
                model_path=base_q, dataset_path="datasets/wsp/test_copy_90.jsonl", benchmark="copy",
                run_dir=run_root / f"{b_tag}_copy90", max_tokens=512, temperature=0.0,
                enable_thinking=False, parallel=4, ctx_per_slot=1024, tag=f"{b_tag}_copy90",
            )
            ev_b = run_benchmark(
                model_path=base_q, dataset_path=ds_math, benchmark="math",
                run_dir=run_root / f"{b_tag}_math150", max_tokens=2048, temperature=0.0,
                parallel=4, ctx_per_slot=2560, tag=f"{b_tag}_math150",
            )
            bm = {json.loads(l)["id"]: json.loads(l) for l in (run_root / f"{b_tag}_math150" / "responses.jsonl").read_text().splitlines() if l.strip()}
            b_eos_n = sum(bool(bm[i]["grade"]["correct"]) and bm[i]["finish_reason"] == "stop" for i in ids)
            b_cls_n = sum(bool(bm[i]["grade"]["correct"]) and bm[i]["grade"]["has_think_close"] for i in ids)
            print(f"QUANT {b_tag:28s}: PPL={ppl_b['ppl']:.4f} | STD={ev_b['correct']:2d}/150 | CLOSED={b_cls_n:2d}/150 | EOS={b_eos_n:2d}/150 | lim={ev_b['hit_limit_count']} lp={ev_b['loop_count']} | COPY={cp_b['correct']}/90", flush=True)
            results[b_tag] = {
                "ppl": ppl_b, "test_math500_strat150": ev_b, "test_copy_90": cp_b,
                "clean_eos_correct": b_eos_n, "closed_think_correct": b_cls_n,
            }
            out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        bm = {json.loads(l)["id"]: json.loads(l) for l in (run_root / f"{b_tag}_math150" / "responses.jsonl").read_text().splitlines() if l.strip()}
        base_std = [bool(bm[i]["grade"]["correct"]) for i in ids]
        base_eos = [bool(bm[i]["grade"]["correct"]) and bm[i]["finish_reason"] == "stop" for i in ids]
        base_cls = [bool(bm[i]["grade"]["correct"]) and bm[i]["grade"]["has_think_close"] for i in ids]
        base_ppl = results[b_tag]["ppl"]["ppl"]

        # Build noop permuted quant file for canonical Theorem 1 audit of BA-WSC
        noop_q = art_q / f"ba_wsc_noop-{qt}.gguf"
        if not noop_q.exists():
            quantize_gguf(fp_paths["ba_wsc_noop"], noop_q, qt)

        # 2. Transfer arms (FP-compiled -> quantized)
        transfer_arms = ["oblique_res_a0.25", "bawsc32_res_a0.25", "dormant32_res_a0.25", "uncond_a0.8"]
        for nm in transfer_arms:
            tag = f"{nm}_{qt}"
            if tag in results:
                continue
            q_p = art_q / f"{nm}-{qt}.gguf"
            q_meta = quantize_gguf(fp_paths[nm], q_p, qt)
            ref_fp = fp_paths["ba_wsc_noop"] if nm.startswith("bawsc") else base_fp
            ref_q = noop_q if nm.startswith("bawsc") else base_q
            audit = audit_quant_pair(ref_fp, fp_paths[nm], ref_q, q_p, ["blk.20.ffn_down.weight"])
            results[tag] = eval_quant_arm(
                tag=tag, q_path=q_p, run_root=run_root, ds_math=ds_math, ids=ids,
                base_std=base_std, base_eos=base_eos, base_cls=base_cls, base_ppl=base_ppl,
                audit=audit, q_meta=q_meta,
            )
            out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            if nm in ("dormant32_res_a0.25", "uncond_a0.8") and q_p.exists():
                q_p.unlink()

        # 3. Post-Quantization Weight-Space Healing (PQ-Oblique & PQ-BA-WSC) on Q4_0 and Q3_K_M!
        if a.with_pq_heal and qt in ("Q4_0", "Q3_K_M"):
            pq_cache = Path(f"local/cache/qwen3_0.6b/pq_{qt}_layer_stats.pkl")
            pq_stats = trace_quantized_model(
                model_name="qwen3_0.6b",
                qtype=qt,
                hf_dir="local/hf_models/Qwen3-0.6B",
                quant_gguf=base_q,
                quant_cal_math_dir=f"runs/wsp_baselines/qwen3_0.6b/{qt}_cal_math50",
                fp_cal_gsm_dir="runs/wsp_baselines/qwen3_0.6b/fp_cal_gsm48",
                fp_cal_copy_dir="runs/wsp_baselines/qwen3_0.6b/fp_cal_copy36",
                target_layers=layers,
                cache_path=pq_cache,
            )
            pq_specs = [
                (f"pq_oblique_a0.25_{qt}", compile_oblique_nullspace(gmap, pq_stats, layers, alpha=0.25, action_mode="resolve", null_rank=64), False),
                (f"pq_bawsc32_a0.35_{qt}", compile_dormant_if_then(gmap, pq_stats, layers, alpha=0.35, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True), True),
            ]
            for tag, (ed, cinfo), is_ba in pq_specs:
                if tag in results:
                    continue
                fp_tmp = art_q / f"{tag}_fp.gguf"
                q_p = art_q / f"{tag}.gguf"
                patch_gguf_tensors(base_fp, fp_tmp, ed, metadata_extra=cinfo)
                q_meta = quantize_gguf(fp_tmp, q_p, qt)
                ref_fp = fp_paths["ba_wsc_noop"] if is_ba else base_fp
                ref_q = noop_q if is_ba else base_q
                audit = audit_quant_pair(ref_fp, fp_tmp, ref_q, q_p, ["blk.20.ffn_down.weight"])
                results[tag] = eval_quant_arm(
                    tag=tag, q_path=q_p, run_root=run_root, ds_math=ds_math, ids=ids,
                    base_std=base_std, base_eos=base_eos, base_cls=base_cls, base_ppl=base_ppl,
                    audit=audit, q_meta=q_meta,
                )
                results[tag]["compile_info"] = cinfo
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                if fp_tmp.exists():
                    fp_tmp.unlink()


if __name__ == "__main__":
    main()
