"""Model 3 Pipeline: DeepSeek-R1-Distill-Qwen-1.5B (Pure Qwen2 1.5B, d=1536, d_ff=8960, RL-distilled).

Uses zero-duplicate BF16 GGUF -> float32 PyTorch loading (src/wsp/gguf_to_torch.py) for exact activation
tracing, and 100%-VRAM-resident Q8_0 / Q4_K_M / Q4_0 / Q3_K_M on RX 6400 4 GB Vulkan for fast evaluation!

Stages:
  baseline : BF16 PPL (ngl=20 for 4 GB VRAM safety) + Q8_0/Q4_K_M/Q4_0/Q3_K_M PPL & cal_math_50 (stochastic
             temp=0.6, top_p=0.95 + greedy on Q8_0) + cal_gsm8k_48 + cal_copy_36.
  profile  : Exact float32 PyTorch mechanistic profiling across all 28 layers on calibration trajectories.
  theorem1 : Empirical Theorem 1 block-level quantization audit (Q4_K_M, Q4_0, Q3_K_M) on d_ff=8960 SwiGLU.
  study    : Strictly held-out MATH-500-strat150 + Copy-90 + WikiText-2 PPL under official multi-seed
             stochastic sampling (temp=0.6, top_p=0.95) at Q8_0, Q4_0, and Q3_K_M comparing baseline,
             random_matched, uncond_a0.8, oblique_res_a0.25, and bawsc32_res_a0.25.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from gguf import GGUFReader
from transformers import AutoTokenizer

from scripts.wsp_diagnose_and_compile import load_cal_trajectories
from scripts.wsp_quant_audit import audit_quant_pair
from src.wsp.compiler import (
    analyze_layers,
    collect_layer_activations,
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_random_matched,
    compile_sparse_cols,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.gguf_to_torch import load_causal_lm_from_gguf
from src.wsp.llama_eval import mcnemar_exact, measure_perplexity, run_benchmark


def _extract_flags(run_dir: Path, ids: list[str]) -> tuple[list[bool], list[bool], list[bool]]:
    mp = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std = [bool(mp[i]["grade"]["correct"]) for i in ids]
    eos = [
        bool(mp[i]["grade"]["correct"])
        and mp[i]["finish_reason"] == "stop"
        and bool(mp[i]["grade"]["has_think_close"])
        for i in ids
    ]
    cls = [bool(mp[i]["grade"]["correct"]) and bool(mp[i]["grade"]["has_think_close"]) for i in ids]
    return std, eos, cls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["baseline", "profile", "theorem1", "study", "all"], default="all")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--study-layers", default="18,20,22")
    p.add_argument("--study-seeds", default="17,42")
    a = p.parse_args()

    model_name = "deepseek_r1_1.5b"
    base_gguf = Path("local/models/DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf")
    hf_dir = Path("local/hf_models/DeepSeek-R1-Distill-Qwen-1.5B")
    base_out = Path("runs/wsp_baselines") / model_name
    study_out = Path("runs/wsp_study") / model_name
    art_base = Path("artifacts/wsp") / model_name / "baselines"
    art_study = Path("artifacts/wsp") / model_name / "study"
    rep_dir = Path("research/wsp") / model_name
    cache_dir = Path("local/cache") / model_name
    for d in (base_out, study_out, art_base, art_study, rep_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Note: DeepSeek-R1-Distill-Qwen-1.5B Q8_0 is ~1.89 GB -> fits 100% in 4 GB RX 6400 with parallel=4, ctx=2560!
    # Q4_K_M / Q4_0 / Q3_K_M are 0.9-1.1 GB -> blazing fast at parallel=4!
    if a.stage in ("baseline", "all"):
        print("\n================ [DeepSeek-R1-1.5B] BASELINE CHARACTERIZATION ================", flush=True)
        res_path = base_out / "baseline_summary.json"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"model_name": model_name, "precisions": {}}

        if "FP" not in res["precisions"]:
            # Hybrid ngl=18 fits comfortably in 4 GB VRAM for 3.55 GB BF16 PPL check
            ppl_fp = measure_perplexity(base_gguf, chunks=32, ctx_size=512, ngl=18)
            print(f"  PPL(FP, ngl=18): {ppl_fp['ppl']:.4f} +/- {ppl_fp['ppl_se']:.4f}", flush=True)
            res["precisions"]["FP"] = {"ppl": ppl_fp}
            res_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

        for qt in ["Q8_0", "Q4_K_M", "Q4_0", "Q3_K_M"]:
            if qt in res["precisions"]:
                continue
            qp = art_base / f"{model_name}-{qt}.gguf"
            qm = quantize_gguf(base_gguf, qp, qt) if not qp.exists() else {"dst_path": str(qp), "dst_bytes": qp.stat().st_size, "qtype": qt}
            ppl_q = measure_perplexity(qp, chunks=32, ctx_size=512, ngl=99)
            print(f"  PPL({qt}): {ppl_q['ppl']:.4f} +/- {ppl_q['ppl_se']:.4f} | {qm['dst_bytes']/1e6:.1f} MB", flush=True)
            par = 2 if qt == "Q8_0" else 4  # Leave ample KV headroom in 4 GB VRAM for 1.89 GB Q8_0!
            ev_stoch = run_benchmark(
                model_path=qp, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
                run_dir=base_out / f"{qt}_cal_math50_stoch", max_tokens=a.max_tokens,
                temperature=0.6, top_p=0.95, top_k=20, seed=42,
                parallel=par, ctx_per_slot=2560, tag=f"ds15b_{qt}_cal_math50_stoch",
            )
            rec = {"quant_meta": qm, "ppl": ppl_q, "cal_math50_stoch": ev_stoch}
            if qt == "Q8_0":
                rec["cal_math50_greedy"] = run_benchmark(
                    model_path=qp, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
                    run_dir=base_out / "Q8_0_cal_math50_greedy", max_tokens=a.max_tokens,
                    temperature=0.0, parallel=par, ctx_per_slot=2560, tag="ds15b_Q8_0_cal_math50_greedy",
                )
                rec["cal_gsm48_stoch"] = run_benchmark(
                    model_path=qp, dataset_path="datasets/wsp/cal_gsm8k_48.jsonl", benchmark="gsm8k",
                    run_dir=base_out / "Q8_0_cal_gsm48", max_tokens=a.max_tokens,
                    temperature=0.6, top_p=0.95, top_k=20, seed=42,
                    parallel=par, ctx_per_slot=2560, tag="ds15b_Q8_0_cal_gsm48_stoch",
                )
                rec["cal_copy36"] = run_benchmark(
                    model_path=qp, dataset_path="datasets/wsp/cal_copy_36.jsonl", benchmark="copy",
                    run_dir=base_out / "Q8_0_cal_copy36", max_tokens=1024,
                    temperature=0.0, enable_thinking=False,
                    parallel=par, ctx_per_slot=1536, tag="ds15b_Q8_0_cal_copy36",
                )
            res["precisions"][qt] = rec
            res_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    if a.stage in ("profile", "all"):
        print("\n================ [DeepSeek-R1-1.5B] 28-LAYER MECHANISTIC PROFILING ================", flush=True)
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
            model = load_causal_lm_from_gguf(hf_dir, base_gguf, dtype=torch.float32)
            path_t, healthy_t, resolve_t, preserve_t, counts = load_cal_trajectories(
                tok,
                base_out / "Q8_0_cal_math50_stoch",
                base_out / "Q8_0_cal_gsm48",
                base_out / "Q8_0_cal_copy36",
                Path("datasets/wsp/cal_wikitext_64.jsonl"),
                max_path=16, max_healthy=16, max_preserve=20,
            )
            print("DeepSeek-R1-1.5B cal trajectories:", counts, flush=True)
            layer_stats = collect_layer_activations(
                model, tok, path_texts=path_t, healthy_texts=healthy_t,
                resolve_texts=resolve_t, preserve_texts=preserve_t, max_seq_len=384,
            )
            del model
            with stats_pkl.open("wb") as f:
                pickle.dump({"layer_stats": layer_stats, "counts": counts}, f)
            print(f"Collected 28 layers in {time.time()-t0:.1f}s", flush=True)

        prof = analyze_layers(layer_stats, gmap)
        (rep_dir / "layer_profile.json").write_text(
            json.dumps({"model_name": model_name, "counts": counts, "layers": prof}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{'L':>3s} {'resDiff':>8s} {'mlpProj':>8s} {'aucX':>6s} {'vbStd':>7s} {'d32/med':>8s}")
        for r in prof:
            print(f"{r['layer']:3d} {r['res_diff_norm']:8.3f} {r['mlp_loop_proj']:8.4f} {r['auroc_x']:6.3f} {r['vb_std']:7.4f} {r['dormant_32_to_median']:8.4f}")

    if a.stage in ("theorem1", "all"):
        print("\n================ [DeepSeek-R1-1.5B] THEOREM 1 VERIFICATION (d_ff=8960) ================", flush=True)
        with (cache_dir / "layer_stats.pkl").open("rb") as f:
            stats = pickle.load(f)["layer_stats"]
        gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}
        layers = [20]
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
                res_a = audit_quant_pair(ref_fp, fp_p, ref_q, qp, ["blk.20.ffn_down.weight"])
                all_out[name]["quants"][qt] = res_a
                r20 = res_a["blk.20.ffn_down.weight"]
                print(f"{name:22s} | {qt:6s} | L20 cos={r20['cosine_sim_full']:.5f} collat/sig={r20['collateral_to_signal_ratio']:.5f} mix={r20['mixed_32blocks']} pure={r20['pure_edit_32blocks']}", flush=True)
                qp.unlink()
            fp_p.unlink()
        for p_f in list(tmp.glob("*.gguf")):
            p_f.unlink()
        (rep_dir / "theorem1_quant_audit.json").write_text(json.dumps(all_out, indent=2) + "\n", encoding="utf-8")

    if a.stage in ("study", "all"):
        print("\n================ [DeepSeek-R1-1.5B] HELD-OUT STOCHASTIC & QUANTIZED STUDY ================", flush=True)
        with (cache_dir / "layer_stats.pkl").open("rb") as f:
            stats = pickle.load(f)["layer_stats"]
        gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}
        layers = [int(x) for x in a.study_layers.split(",") if x.strip()]
        seeds = [int(x) for x in a.study_seeds.split(",") if x.strip()]
        ds_math = Path("datasets/wsp/test_math500_strat150.jsonl")
        ids = [json.loads(l)["id"] for l in ds_math.read_text(encoding="utf-8").splitlines() if l.strip()]

        out_path = rep_dir / "test_stochastic_and_quant.json"
        out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

        ed_oblique, ci_oblique = compile_oblique_nullspace(gmap, stats, layers, alpha=0.25, action_mode="resolve", null_rank=64)
        arm_specs = [
            ("baseline", None),
            ("random_matched", compile_random_matched(gmap, layers, ed_oblique, seed=999)),
            ("uncond_a0.8", compile_uncond_subspace(gmap, stats, layers, alpha=0.8, rank=1)),
            ("oblique_res_a0.25", (ed_oblique, ci_oblique)),
            ("bawsc32_res_a0.25", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True)),
        ]

        for qt in ["Q8_0", "Q4_0", "Q3_K_M"]:
            par = 2 if qt == "Q8_0" else 4
            q_out = out.setdefault(qt, {})
            base_flags = {}
            base_ppl = None
            for name, comp in arm_specs:
                if comp is None:
                    q_p = art_base / f"{model_name}-{qt}.gguf"
                    cinfo = None
                else:
                    q_p = art_study / f"{name}-{qt}.gguf"
                    ed, cinfo = comp
                    if not q_p.exists():
                        fp_tmp = art_study / f"{name}_fp_tmp.gguf"
                        patch_gguf_tensors(base_gguf, fp_tmp, ed, metadata_extra=cinfo)
                        quantize_gguf(fp_tmp, q_p, qt)
                        fp_tmp.unlink()

                arm_rec = q_out.setdefault(name, {"seeds": {}, "compile_info": cinfo})
                if "ppl" not in arm_rec:
                    arm_rec["ppl"] = measure_perplexity(q_p, chunks=32, ctx_size=512, ngl=99)
                if "copy_90" not in arm_rec:
                    arm_rec["copy_90"] = run_benchmark(
                        model_path=q_p, dataset_path="datasets/wsp/test_copy_90.jsonl", benchmark="copy",
                        run_dir=study_out / f"{qt}_{name}_copy90", max_tokens=1024, temperature=0.0,
                        enable_thinking=False, parallel=par, ctx_per_slot=1536, tag=f"ds15b_{qt}_{name}_copy90",
                    )
                if name == "baseline":
                    base_ppl = arm_rec["ppl"]["ppl"]

                for sd in seeds:
                    skey = f"s{sd}"
                    rdir = study_out / f"{qt}_{name}_math150_{skey}"
                    if skey not in arm_rec["seeds"]:
                        ev = run_benchmark(
                            model_path=q_p, dataset_path=ds_math, benchmark="math",
                            run_dir=rdir, max_tokens=a.max_tokens,
                            temperature=0.6, top_p=0.95, top_k=20, seed=sd,
                            parallel=par, ctx_per_slot=2560, tag=f"ds15b_{qt}_{name}_m150_{skey}",
                        )
                        std_f, eos_f, cls_f = _extract_flags(rdir, ids)
                        srec = {
                            "summary": ev,
                            "clean_eos_correct": sum(eos_f),
                            "closed_think_correct": sum(cls_f),
                        }
                        if name != "baseline":
                            b_std, b_eos, b_cls = base_flags[sd]
                            srec["mcnemar_std"] = mcnemar_exact(b_std, std_f)
                            srec["mcnemar_clean_eos"] = mcnemar_exact(b_eos, eos_f)
                            srec["mcnemar_closed_think"] = mcnemar_exact(b_cls, cls_f)
                        arm_rec["seeds"][skey] = srec
                        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
                    std_f, eos_f, cls_f = _extract_flags(rdir, ids)
                    if name == "baseline":
                        base_flags[sd] = (std_f, eos_f, cls_f)
                    ev = arm_rec["seeds"][skey]["summary"]
                    dppl = arm_rec["ppl"]["ppl"] - base_ppl if base_ppl else 0.0
                    print(
                        f"[DS1.5B {qt} {skey}] {name:20s}: PPL={arm_rec['ppl']['ppl']:.3f} ({dppl:+.3f}) | "
                        f"STD={ev['correct']:2d}/150 | EOS={arm_rec['seeds'][skey]['clean_eos_correct']:2d}/150 | "
                        f"lim={ev['hit_limit_count']} lp={ev['loop_count']} | COPY={arm_rec['copy_90']['correct']}/90",
                        flush=True,
                    )
                if name != "baseline" and q_p.exists():
                    q_p.unlink()


if __name__ == "__main__":
    main()
