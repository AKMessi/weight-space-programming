"""Complete zero-gradient WSP study runner:
  Stage 1 (Dev Selection - strictly MATH-train / GSM8K-train):
    Sweeps pre-registered hyperparameter grid for each WSP operator + non-weight baselines
    (DRY sampler, presence penalty, concise prompt) on dev_math_60 + WikiText-2 PPL.
    Freezes exact winning config per operator BEFORE touching ANY held-out test set.

  Stage 2 (Frozen Confirmatory Evaluation - 100% Held-Out Test Sets):
    Evaluates Baseline + Non-weight Controls + All Frozen WSP Operators + Negative Controls
    (ba_wsc_noop, random_matched) on:
      - MATH-500 (strat150 for all methods, full 500 for primary comparison)
      - GSM8K-250
      - ARC-Challenge-250 (OOD science reasoning)
      - Copy/KV/Cycle-Induction-90 (Dual-Function Entanglement probe)
      - WikiText-2 test perplexity

  Stage 3 (Paper 2 — Quantization Survival & Block-Aligned Compilation Matrix):
    Quantizes frozen FP operators to Q8_0, Q4_K_M, Q4_0, Q3_K_M via official llama-quantize,
    runs exact tensor/block-level audit (signal retention vs neighbor collateral distortion),
    and evaluates on RX 6400 Vulkan across MATH-500-strat150, Copy-90, and WikiText-2 PPL.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader

from src.wsp.compiler import (
    LayerStats,
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_random_matched,
    compile_sparse_cols,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import (
    build_block_aligned_permutation,
    patch_gguf_tensors,
    quantize_gguf,
    read_gguf_tensor_f32,
)
from src.wsp.llama_eval import mcnemar_exact, measure_perplexity, run_benchmark
from scripts.wsp_quant_audit import audit_quant_pair


CONCISE_SYS_PROMPT = (
    "Solve the problem accurately and concisely. Do not repeat steps or second-guess "
    "verified calculations. Once you obtain the answer, immediately finish your reasoning "
    "and put the final answer in \\boxed{}."
)


def compile_by_spec(
    spec: dict[str, Any],
    gguf_map: dict[str, Any],
    layer_stats: dict[int, LayerStats],
    ref_edited: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    m = spec["method"]
    layers = spec["layers"]
    if m == "uncond_subspace":
        return compile_uncond_subspace(gguf_map, layer_stats, layers, alpha=spec["alpha"], rank=spec.get("rank", 1))
    elif m == "sparse_cols":
        return compile_sparse_cols(gguf_map, layer_stats, layers, alpha=spec["alpha"], top_k=spec.get("top_k", 64))
    elif m == "oblique_nullspace":
        return compile_oblique_nullspace(
            gguf_map, layer_stats, layers,
            alpha=spec["alpha"],
            action_mode=spec.get("action_mode", "resolve"),
            ridge_scale=spec.get("ridge_scale", 0.2),
            null_rank=spec.get("null_rank", 64),
        )
    elif m in ("dormant_if_then", "ba_wsc"):
        return compile_dormant_if_then(
            gguf_map, layer_stats, layers,
            alpha=spec["alpha"],
            k_neurons=spec.get("k_neurons", 32),
            gamma=spec.get("gamma", 4.0),
            tau_quantile=spec.get("tau_quantile", 0.96),
            action_mode=spec.get("action_mode", "resolve"),
            block_align=(m == "ba_wsc"),
            gauge_balance=spec.get("gauge_balance", True),
        )
    elif m == "ba_wsc_noop":
        return compile_ba_wsc_noop(gguf_map, layer_stats, layers, k_neurons=spec.get("k_neurons", 32))
    elif m == "random_matched":
        assert ref_edited is not None
        return compile_random_matched(gguf_map, layers, ref_edited, seed=spec.get("seed", 999))
    else:
        raise ValueError(f"Unknown method: {m}")


def load_correct_flags(run_dir: Path, dataset_path: Path) -> list[bool]:
    ids = [json.loads(l)["id"] for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {}
    for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            by_id[r["id"]] = bool(r["grade"].get("correct"))
    return [by_id.get(i, False) for i in ids]


def eval_candidate_dev(
    *,
    name: str,
    model_path: Path,
    run_root: Path,
    max_tokens: int,
    parallel: int,
    ctx_per_slot: int,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    sampling_extra: dict[str, Any] | None = None,
    skip_ppl: bool = False,
) -> dict[str, Any]:
    out = {}
    if not skip_ppl:
        out["ppl"] = measure_perplexity(model_path, chunks=24, ctx_size=512)
    out["dev_math60"] = run_benchmark(
        model_path=model_path,
        dataset_path="datasets/wsp/dev_math_60.jsonl",
        benchmark="math",
        run_dir=run_root / f"{name}_dev_math60",
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        sampling_extra=sampling_extra,
        parallel=parallel,
        ctx_per_slot=ctx_per_slot,
        tag=f"{name}_dev_math60",
    )
    out["dev_copy36"] = run_benchmark(
        model_path=model_path,
        dataset_path="datasets/wsp/cal_copy_36.jsonl",
        benchmark="copy",
        run_dir=run_root / f"{name}_dev_copy36",
        max_tokens=512,
        temperature=0.0,
        enable_thinking=False,
        parallel=parallel,
        ctx_per_slot=1024,
        tag=f"{name}_dev_copy36",
    )
    return out


def eval_heldout_suite(
    *,
    name: str,
    model_path: Path,
    run_root: Path,
    max_tokens: int,
    parallel: int,
    ctx_per_slot: int,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    sampling_extra: dict[str, Any] | None = None,
    full_math500: bool = False,
    skip_ppl: bool = False,
    run_arc_gsm: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "model_path": str(model_path)}
    if not skip_ppl:
        out["ppl"] = measure_perplexity(model_path, chunks=32, ctx_size=512)
    math_ds = "datasets/wsp/test_math500.jsonl" if full_math500 else "datasets/wsp/test_math500_strat150.jsonl"
    math_key = "test_math500" if full_math500 else "test_math500_strat150"
    out[math_key] = run_benchmark(
        model_path=model_path,
        dataset_path=math_ds,
        benchmark="math",
        run_dir=run_root / f"{name}_{math_key}",
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        sampling_extra=sampling_extra,
        parallel=parallel,
        ctx_per_slot=ctx_per_slot,
        tag=f"{name}_{math_key}",
    )
    if run_arc_gsm:
        out["test_gsm8k_250"] = run_benchmark(
            model_path=model_path,
            dataset_path="datasets/wsp/test_gsm8k_250.jsonl",
            benchmark="gsm8k",
            run_dir=run_root / f"{name}_test_gsm8k_250",
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            sampling_extra=sampling_extra,
            parallel=parallel,
            ctx_per_slot=ctx_per_slot,
            tag=f"{name}_test_gsm8k_250",
        )
        out["test_arc_250"] = run_benchmark(
            model_path=model_path,
            dataset_path="datasets/wsp/test_arc_challenge_250.jsonl",
            benchmark="arc",
            run_dir=run_root / f"{name}_test_arc_250",
            max_tokens=min(max_tokens, 1536),
            temperature=temperature,
            system_prompt=system_prompt,
            sampling_extra=sampling_extra,
            parallel=parallel,
            ctx_per_slot=min(ctx_per_slot, 2048),
            tag=f"{name}_test_arc_250",
        )
    out["test_copy_90"] = run_benchmark(
        model_path=model_path,
        dataset_path="datasets/wsp/test_copy_90.jsonl",
        benchmark="copy",
        run_dir=run_root / f"{name}_test_copy_90",
        max_tokens=512,
        temperature=0.0,
        enable_thinking=False,
        parallel=parallel,
        ctx_per_slot=1024,
        tag=f"{name}_test_copy_90",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON experiment specification")
    p.add_argument("--stage", choices=["dev", "test", "quant", "all"], default="dev")
    a = p.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    base_gguf = Path(cfg["base_gguf"])
    stats_pkl = Path(cfg["layer_stats_pkl"])
    run_root = Path("runs/wsp_study") / model_name
    art_root = Path("artifacts/wsp") / model_name
    rep_root = Path("research/wsp") / model_name
    run_root.mkdir(parents=True, exist_ok=True)
    art_root.mkdir(parents=True, exist_ok=True)
    rep_root.mkdir(parents=True, exist_ok=True)

    with stats_pkl.open("rb") as f:
        layer_stats: dict[int, LayerStats] = pickle.load(f)["layer_stats"]
    reader = GGUFReader(str(base_gguf), mode="r")
    gguf_map = {t.name: t for t in reader.tensors}

    max_toks = cfg.get("max_tokens", 2048)
    parallel = cfg.get("parallel", 4)
    ctx_slot = cfg.get("ctx_per_slot", 2560)
    temp = cfg.get("temperature", 0.0)

    if a.stage in ("dev", "all"):
        dev_results: dict[str, Any] = {}
        print(f"\n================ [{model_name}] STAGE 1: DEV SELECTION ================", flush=True)
        # Baseline + non-weight controls on dev
        dev_results["baseline_fp"] = eval_candidate_dev(
            name="baseline_fp", model_path=base_gguf, run_root=run_root / "dev",
            max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
        )
        dev_results["ctrl_concise_prompt"] = eval_candidate_dev(
            name="ctrl_concise_prompt", model_path=base_gguf, run_root=run_root / "dev",
            max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
            system_prompt=CONCISE_SYS_PROMPT, skip_ppl=True,
        )
        dev_results["ctrl_dry_sampler"] = eval_candidate_dev(
            name="ctrl_dry_sampler", model_path=base_gguf, run_root=run_root / "dev",
            max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
            sampling_extra={"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2},
            skip_ppl=True,
        )
        dev_results["ctrl_presence_pen"] = eval_candidate_dev(
            name="ctrl_presence_pen", model_path=base_gguf, run_root=run_root / "dev",
            max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
            sampling_extra={"presence_penalty": 1.5},
            skip_ppl=True,
        )

        # Sweep pre-registered WSP candidate grid
        for cand in cfg["dev_candidates"]:
            cname = cand["name"]
            print(f"\n--- Compiling dev candidate: {cname} ({cand}) ---", flush=True)
            edited, cinfo = compile_by_spec(cand, gguf_map, layer_stats)
            out_gguf = art_root / "dev" / f"{cname}.gguf"
            man_path = rep_root / "manifests" / f"dev_{cname}.json"
            patch_rep = patch_gguf_tensors(
                base_gguf, out_gguf, edited, manifest_path=man_path, metadata_extra=cinfo
            )
            res = eval_candidate_dev(
                name=cname, model_path=out_gguf, run_root=run_root / "dev",
                max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
            )
            res["compile_info"] = cinfo
            res["patch_summary"] = {
                "changed_params": patch_rep["total_changed_params"],
                "changed_frac": patch_rep["changed_param_fraction"],
                "output_sha256": patch_rep["output_sha256"],
            }
            dev_results[cname] = res
            # Delete temporary non-winning dev gguf immediately to keep disk lean (deterministic rebuild!)
            if out_gguf.exists():
                out_gguf.unlink()
            # Persist incremental dev results after every single candidate!
            (rep_root / "dev_results.json").write_text(json.dumps(dev_results, indent=2) + "\n", encoding="utf-8")

        print(f"\nDev selection complete -> {rep_root / 'dev_results.json'}", flush=True)

    if a.stage in ("test", "all") and "frozen_test_specs" in cfg:
        print(f"\n================ [{model_name}] STAGE 2: FROZEN HELD-OUT TEST ================", flush=True)
        test_results: dict[str, Any] = {}
        # Baseline + non-weight controls
        test_results["baseline_fp"] = eval_heldout_suite(
            name="baseline_fp", model_path=base_gguf, run_root=run_root / "test",
            max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
            full_math500=cfg.get("full_math500_primary", False),
        )
        for ctrl_name, sp, se in [
            ("ctrl_dry_sampler", None, {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2}),
            ("ctrl_presence_pen", None, {"presence_penalty": 1.5}),
            ("ctrl_concise_prompt", CONCISE_SYS_PROMPT, None),
        ]:
            test_results[ctrl_name] = eval_heldout_suite(
                name=ctrl_name, model_path=base_gguf, run_root=run_root / "test",
                max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
                system_prompt=sp, sampling_extra=se, skip_ppl=True,
                full_math500=False,
            )

        compiled_cache: dict[str, dict[str, np.ndarray]] = {}
        for spec in cfg["frozen_test_specs"]:
            sname = spec["name"]
            print(f"\n--- Building & Testing Frozen Checkpoint: {sname} ---", flush=True)
            ref_ed = compiled_cache.get(spec.get("ref_name", ""))
            edited, cinfo = compile_by_spec(spec, gguf_map, layer_stats, ref_edited=ref_ed)
            compiled_cache[sname] = edited
            out_gguf = art_root / "frozen" / f"{sname}.gguf"
            man_path = rep_root / "manifests" / f"frozen_{sname}.json"
            patch_rep = patch_gguf_tensors(
                base_gguf, out_gguf, edited, manifest_path=man_path, metadata_extra=cinfo
            )
            res = eval_heldout_suite(
                name=sname, model_path=out_gguf, run_root=run_root / "test",
                max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
                full_math500=spec.get("full_math500", False),
                run_arc_gsm=spec.get("run_arc_gsm", True),
            )
            res["compile_info"] = cinfo
            res["patch_summary"] = {
                "changed_params": patch_rep["total_changed_params"],
                "changed_frac": patch_rep["changed_param_fraction"],
                "output_sha256": patch_rep["output_sha256"],
            }
            test_results[sname] = res
            (rep_root / "test_results.json").write_text(json.dumps(test_results, indent=2) + "\n", encoding="utf-8")

    if a.stage in ("quant", "all") and "quant_matrix" in cfg:
        print(f"\n================ [{model_name}] STAGE 3: QUANTIZATION SURVIVAL MATRIX ================", flush=True)
        q_results: dict[str, Any] = {}
        qtypes = cfg["quant_matrix"]["qtypes"]
        arms = cfg["quant_matrix"]["arms"]

        # Ensure unedited base quant files exist
        base_q_paths: dict[str, Path] = {}
        for qt in qtypes:
            bp = art_root / "baselines" / f"{model_name}-{qt}.gguf"
            if not bp.exists():
                quantize_gguf(base_gguf, bp, qt)
            base_q_paths[qt] = bp
            # Evaluate unedited quantized baseline on held-out strat150 + copy90 + PPL
            q_results[f"baseline_{qt}"] = eval_heldout_suite(
                name=f"baseline_{qt}", model_path=bp, run_root=run_root / "quant",
                max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
                full_math500=False, run_arc_gsm=False,
            )

        # Ensure BA-WSC noop permuted quant baselines exist if needed
        for arm in arms:
            aname = arm["name"]
            fp_path = art_root / "frozen" / f"{aname}.gguf"
            if not fp_path.exists():
                edited, cinfo = compile_by_spec(arm, gguf_map, layer_stats)
                patch_gguf_tensors(base_gguf, fp_path, edited, metadata_extra=cinfo)
            touched_tensors = []
            for l_idx in arm["layers"]:
                if arm["method"] in ("dormant_if_then", "ba_wsc", "ba_wsc_noop"):
                    touched_tensors.extend([
                        f"blk.{l_idx}.ffn_gate.weight",
                        f"blk.{l_idx}.ffn_up.weight",
                        f"blk.{l_idx}.ffn_down.weight",
                    ])
                else:
                    touched_tensors.append(f"blk.{l_idx}.ffn_down.weight")

            # Build permutation map for BA-WSC audit comparison
            perm_map = None
            if arm["method"] == "ba_wsc":
                perm_map = {}
                for l_idx in arm["layers"]:
                    st = layer_stats[l_idx]
                    W_dn = read_gguf_tensor_f32(gguf_map[f"blk.{l_idx}.ffn_down.weight"]).astype(np.float64)
                    c_rms = st.neuron_rms_preserve.astype(np.float64) * np.linalg.norm(W_dn, axis=0)
                    d_idx = [int(i) for i in np.argsort(c_rms)[: arm.get("k_neurons", 32)]]
                    perm = build_block_aligned_permutation(st.d_ff, d_idx, block_size=32, dest_block=0)
                    perm_map[f"blk.{l_idx}.ffn_gate.weight"] = perm
                    perm_map[f"blk.{l_idx}.ffn_up.weight"] = perm
                    perm_map[f"blk.{l_idx}.ffn_down.weight"] = perm

            for qt in qtypes:
                q_path = art_root / "quant" / f"{aname}-{qt}.gguf"
                q_meta = quantize_gguf(fp_path, q_path, qt)
                # For BA-WSC, compare quantized delta against quantized permuted-base (`ba_wsc_noop`) if available!
                audit_base_q = base_q_paths[qt]
                audit_base_fp = base_gguf
                audit_perm = perm_map
                if arm["method"] == "ba_wsc":
                    noop_q = art_root / "quant" / f"ba_wsc_noop-{qt}.gguf"
                    noop_fp = art_root / "frozen" / "ba_wsc_noop.gguf"
                    if noop_q.exists() and noop_fp.exists():
                        audit_base_q = noop_q
                        audit_base_fp = noop_fp
                        audit_perm = None  # already in identical permuted basis!

                audit = audit_quant_pair(
                    audit_base_fp, fp_path, audit_base_q, q_path, touched_tensors, perm_by_tensor=audit_perm
                )
                ev = eval_heldout_suite(
                    name=f"{aname}_{qt}", model_path=q_path, run_root=run_root / "quant",
                    max_tokens=max_toks, parallel=parallel, ctx_per_slot=ctx_slot, temperature=temp,
                    full_math500=False, run_arc_gsm=False,
                )
                ev["quant_meta"] = q_meta
                ev["block_audit"] = audit
                q_results[f"{aname}_{qt}"] = ev
                (rep_root / "quant_results.json").write_text(json.dumps(q_results, indent=2) + "\n", encoding="utf-8")
                # Clean up non-final intermediate quant binary if requested to save disk
                if arm.get("cleanup_quant", False) and q_path.exists():
                    q_path.unlink()


if __name__ == "__main__":
    main()
