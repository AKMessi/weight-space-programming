"""Paper 2 Quantization Survival & Post-Quantization Healing Matrix on Qwen3-0.6B (Q4_K_M, Q4_0, Q3_K_M).

Evaluates on 100% untouched held-out MATH-500-strat150 + Copy-90 + 32-chunk WikiText-2 PPL + exact
Theorem 1 block-level tensor audit, under EITHER greedy (temp=0.0) OR official stochastic sampling
(temp=0.6, top_p=0.95, top_k=20) with explicit negative controls:
  1. Unedited quantized baselines: Q4_K_M, Q4_0, Q3_K_M (measures held-out quantization cliff)
  2. Negative controls quantized:
       - ba_wsc_noop (exact SwiGLU permutation no-op quantized -> isolates permutation effects)
       - random_matched_oblique (matched Frobenius Gaussian noise quantized -> controls perturbation bias!)
  3. FP-compiled edits quantized directly:
       - oblique_res_a0.25 (diffuse rank-1 null-space oblique)
       - bawsc32_res_a0.25 (Theorem 1 Block-Aligned dormant SwiGLU)
       - dormant32_res_a0.25 (exact scattered ablation without Block Alignment)
       - uncond_a0.8 (existing unconditional WSP baseline)
  4. Post-Quantization Weight-Space Healing (PQ-Oblique & PQ-BA-WSC):
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
    compile_random_matched,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.llama_eval import mcnemar_exact, measure_perplexity, run_benchmark


def _extract_flags(run_dir: Path, ids: list[str]) -> tuple[list[bool], list[bool], list[bool]]:
    a_map = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std_f = [bool(a_map[i]["grade"]["correct"]) for i in ids]
    eos_f = [bool(a_map[i]["grade"]["correct"]) and a_map[i]["finish_reason"] == "stop" for i in ids]
    cls_f = [bool(a_map[i]["grade"]["correct"]) and a_map[i]["grade"]["has_think_close"] for i in ids]
    return std_f, eos_f, cls_f


def eval_quant_arm(
    *,
    tag: str,
    q_path: Path,
    run_root: Path,
    ds_math: Path,
    ids: list[str],
    base_std: list[bool] | None,
    base_eos: list[bool] | None,
    base_cls: list[bool] | None,
    base_ppl: float | None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
    run_ppl_copy: bool = True,
    audit: dict[str, Any] | None = None,
    q_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ppl = measure_perplexity(q_path, chunks=32, ctx_size=512) if run_ppl_copy else None
    cp = (
        run_benchmark(
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
        if run_ppl_copy
        else None
    )
    ev = run_benchmark(
        model_path=q_path,
        dataset_path=ds_math,
        benchmark="math",
        run_dir=run_root / f"{tag}_math150",
        max_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        parallel=4,
        ctx_per_slot=2560,
        tag=f"{tag}_math150",
    )
    std_f, eos_f, cls_f = _extract_flags(run_root / f"{tag}_math150", ids)
    mc_block = None
    if base_std is not None and base_eos is not None and base_cls is not None:
        mc_std = mcnemar_exact(base_std, std_f)
        mc_eos = mcnemar_exact(base_eos, eos_f)
        mc_cls = mcnemar_exact(base_cls, cls_f)
        mc_block = {"std": mc_std, "clean_eos": mc_eos, "closed_think": mc_cls}
        ppl_str = f"PPL={ppl['ppl']:.4f} ({ppl['ppl']-base_ppl:+.4f}) | " if (ppl and base_ppl) else ""
        cp_str = f" | COPY={cp['correct']}/90" if cp else ""
        print(
            f"QUANT {tag:34s}: {ppl_str}"
            f"STD={sum(std_f):2d}/150 (p={mc_std['mcnemar_p_two_sided']:.4f}) | "
            f"CLOSED={sum(cls_f):2d}/150 (p={mc_cls['mcnemar_p_two_sided']:.4f}) | "
            f"EOS={sum(eos_f):2d}/150 (p={mc_eos['mcnemar_p_two_sided']:.4f}) | "
            f"lim={ev['hit_limit_count']} lp={ev['loop_count']}{cp_str}",
            flush=True,
        )
    else:
        ppl_str = f"PPL={ppl['ppl']:.4f} | " if ppl else ""
        cp_str = f" | COPY={cp['correct']}/90" if cp else ""
        print(
            f"QUANT {tag:34s}: {ppl_str}"
            f"STD={sum(std_f):2d}/150 | CLOSED={sum(cls_f):2d}/150 | EOS={sum(eos_f):2d}/150 | "
            f"lim={ev['hit_limit_count']} lp={ev['loop_count']}{cp_str}",
            flush=True,
        )
    return {
        "ppl": ppl,
        "test_math500_strat150": ev,
        "test_copy_90": cp,
        "clean_eos_correct": sum(eos_f),
        "closed_think_correct": sum(cls_f),
        "mcnemar_vs_same_quant_base": mc_block,
        "block_audit": audit,
        "quant_meta": q_meta,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qtypes", default="Q4_0,Q3_K_M,Q4_K_M")
    p.add_argument("--with-pq-heal", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seeds", default="42")
    p.add_argument("--arms", default="ba_wsc_noop,random_matched_oblique,oblique_res_a0.25,bawsc32_res_a0.25,dormant32_res_a0.25,uncond_a0.8")
    p.add_argument("--out-json", default="")
    a = p.parse_args()

    is_stoch = a.temperature > 0.0
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    req_arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    base_fp = Path("local/models/Qwen3-0.6B-BF16.gguf")
    gmap = {t.name: t for t in GGUFReader(str(base_fp)).tensors}
    with open("local/cache/qwen3_0.6b/layer_stats.pkl", "rb") as f:
        fp_stats = pickle.load(f)["layer_stats"]

    layers = [18, 20, 22]
    ds_math = Path("datasets/wsp/test_math500_strat150.jsonl")
    ids = [json.loads(l)["id"] for l in ds_math.read_text(encoding="utf-8").splitlines() if l.strip()]

    mode_sub = "quant_stoch" if is_stoch else "quant"
    run_root = Path(f"runs/wsp_study/qwen3_0.6b/{mode_sub}")
    art_q = Path("artifacts/wsp/qwen3_0.6b/quant")
    art_fp = Path("artifacts/wsp/qwen3_0.6b/frozen")
    run_root.mkdir(parents=True, exist_ok=True)
    art_q.mkdir(parents=True, exist_ok=True)
    art_fp.mkdir(parents=True, exist_ok=True)

    default_out = (
        "research/wsp/qwen3_0.6b/quant_matrix_stoch_results.json"
        if is_stoch
        else "research/wsp/qwen3_0.6b/quant_matrix_results.json"
    )
    out_path = Path(a.out_json or default_out)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    ed_oblique, cinfo_oblique = compile_oblique_nullspace(gmap, fp_stats, layers, alpha=0.25, action_mode="resolve", null_rank=64)
    fp_specs = {
        "ba_wsc_noop": compile_ba_wsc_noop(gmap, fp_stats, layers, k_neurons=32),
        "random_matched_oblique": compile_random_matched(gmap, layers, ed_oblique, seed=999),
        "oblique_res_a0.25": (ed_oblique, cinfo_oblique),
        "bawsc32_res_a0.25": compile_dormant_if_then(gmap, fp_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True),
        "dormant32_res_a0.25": compile_dormant_if_then(gmap, fp_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=False),
        "uncond_a0.8": compile_uncond_subspace(gmap, fp_stats, layers, alpha=0.8, rank=1),
    }

    def ensure_fp(nm: str) -> Path:
        fp_p = art_fp / f"{nm}.gguf"
        if not fp_p.exists():
            ed, cinfo = fp_specs[nm]
            patch_gguf_tensors(base_fp, fp_p, ed, metadata_extra=cinfo)
        return fp_p

    for qt in [x.strip() for x in a.qtypes.split(",") if x.strip()]:
        print(f"\n================ QWEN3-0.6B QUANT MATRIX ({mode_sub}): {qt} ================", flush=True)
        base_q = Path(f"artifacts/wsp/qwen3_0.6b/baselines/qwen3_0.6b-{qt}.gguf")
        if not base_q.exists():
            quantize_gguf(base_fp, base_q, qt)

        noop_fp = ensure_fp("ba_wsc_noop")
        noop_q = art_q / f"ba_wsc_noop-{qt}.gguf"
        if not noop_q.exists():
            quantize_gguf(noop_fp, noop_q, qt)

        for sd in seeds:
            suf = f"_s{sd}" if is_stoch else ""
            # 1. Unedited quantized baseline
            b_tag = f"baseline_{qt}{suf}"
            if b_tag not in results:
                results[b_tag] = eval_quant_arm(
                    tag=b_tag, q_path=base_q, run_root=run_root, ds_math=ds_math, ids=ids,
                    base_std=None, base_eos=None, base_cls=None, base_ppl=None,
                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=sd,
                    run_ppl_copy=(sd == seeds[0]),
                )
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

            base_std, base_eos, base_cls = _extract_flags(run_root / f"{b_tag}_math150", ids)
            base_ppl = results[f"baseline_{qt}" + (f"_s{seeds[0]}" if is_stoch else "")]["ppl"]["ppl"]

            # 2. Transfer arms (FP-compiled -> quantized)
            for nm in req_arms:
                tag = f"{nm}_{qt}{suf}"
                if tag in results:
                    continue
                fp_p = ensure_fp(nm)
                q_p = noop_q if nm == "ba_wsc_noop" else (art_q / f"{nm}-{qt}.gguf")
                q_meta = quantize_gguf(fp_p, q_p, qt) if not q_p.exists() else {"dst_path": str(q_p), "dst_bytes": q_p.stat().st_size, "qtype": qt}
                ref_fp = noop_fp if nm.startswith("bawsc") else base_fp
                ref_q = noop_q if nm.startswith("bawsc") else base_q
                audit = (
                    audit_quant_pair(ref_fp, fp_p, ref_q, q_p, ["blk.20.ffn_down.weight"])
                    if nm != "ba_wsc_noop"
                    else {"note": "exact SwiGLU permutation no-op control"}
                )
                results[tag] = eval_quant_arm(
                    tag=tag, q_path=q_p, run_root=run_root, ds_math=ds_math, ids=ids,
                    base_std=base_std, base_eos=base_eos, base_cls=base_cls, base_ppl=base_ppl,
                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=sd,
                    run_ppl_copy=(sd == seeds[0]), audit=audit, q_meta=q_meta,
                )
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                if nm == "random_matched_oblique" and fp_p.exists():
                    fp_p.unlink()

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
                    (f"pq_bawsc32_a0.25_{qt}", compile_dormant_if_then(gmap, pq_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True), True),
                    (f"pq_bawsc32_a0.35_{qt}", compile_dormant_if_then(gmap, pq_stats, layers, alpha=0.35, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True), True),
                ]
                for base_name, (ed, cinfo), is_ba in pq_specs:
                    tag = f"{base_name}{suf}"
                    if tag in results:
                        continue
                    fp_tmp = art_q / f"{base_name}_fp.gguf"
                    q_p = art_q / f"{base_name}.gguf"
                    if not q_p.exists():
                        patch_gguf_tensors(base_fp, fp_tmp, ed, metadata_extra=cinfo)
                        q_meta = quantize_gguf(fp_tmp, q_p, qt)
                        ref_fp = noop_fp if is_ba else base_fp
                        ref_q = noop_q if is_ba else base_q
                        audit = audit_quant_pair(ref_fp, fp_tmp, ref_q, q_p, ["blk.20.ffn_down.weight"])
                        if fp_tmp.exists():
                            fp_tmp.unlink()
                    else:
                        q_meta = {"dst_path": str(q_p), "dst_bytes": q_p.stat().st_size, "qtype": qt}
                        audit = None
                    results[tag] = eval_quant_arm(
                        tag=tag, q_path=q_p, run_root=run_root, ds_math=ds_math, ids=ids,
                        base_std=base_std, base_eos=base_eos, base_cls=base_cls, base_ppl=base_ppl,
                        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=sd,
                        run_ppl_copy=(sd == seeds[0]), audit=audit, q_meta=q_meta,
                    )
                    results[tag]["compile_info"] = cinfo
                    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
