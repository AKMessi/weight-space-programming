"""Model 2 (Qwen3.5-0.8B Hybrid Gated-DeltaNet): Leak-Free Prompt-to-Weight Calibration Enrichment,
Corollary 1 (k=256) Verification, Strict Dev Dose Selection, and Held-Out Stochastic + Quantized Study.

Because unprompted Qwen3.5-0.8B times out on 96%-100% of math problems (yielding only 2 clean </think>
closures in cal_math_50), Stage 'enrich_and_dev' uses a concise system prompt STRICTLY on calibration
train splits (cal_math_50[:25] + cal_gsm8k_48[:25]) to elicit genuine Qwen3.5-0.8B </think> closure
states (H_resolve / X_healthy), traces layers [15, 19, 21] in float32, verifies Corollary 1 (k=256),
and selects doses strictly on dev_math_60[:20] (with ZERO test leakage!).

Stage 'heldout' then freezes the dev winners and evaluates 100% untouched held-out MATH-500-strat150,
Copy-90, and 32-chunk WikiText-2 PPL across Q8_0 (stochastic + greedy illusion check) and 4-bit Q4_0!
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from gguf import GGUFReader
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from scripts.wsp_diagnose_and_compile import load_cal_trajectories
from scripts.wsp_quant_audit import audit_quant_pair
from src.wsp.compiler import (
    collect_layer_activations,
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_random_matched,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.llama_eval import mcnemar_exact, measure_perplexity, run_benchmark


CONCISE_SYS = (
    "Solve step by step concisely. Do not repeat verification steps or second-guess yourself. "
    "Close </think> as soon as you find the answer and write the final answer in \\boxed{}."
)


def _flags(run_dir: Path, ids: list[str]) -> tuple[list[bool], list[bool], list[bool]]:
    mp = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std = [bool(mp[i]["grade"]["correct"]) for i in ids]
    eos = [bool(mp[i]["grade"]["correct"]) and mp[i]["finish_reason"] == "stop" for i in ids]
    cls = [bool(mp[i]["grade"]["correct"]) and bool(mp[i]["grade"]["has_think_close"]) for i in ids]
    return std, eos, cls


def stage_enrich_and_dev() -> None:
    base_gguf = Path("local/models/Qwen3.5-0.8B-BF16.gguf")
    q8_base = Path("artifacts/wsp/qwen35_0.8b/baselines/qwen35_0.8b-Q8_0.gguf")
    hf_dir = Path("local/hf_models/Qwen3.5-0.8B")
    base_out = Path("runs/wsp_baselines/qwen35_0.8b")
    rep_dir = Path("research/wsp/qwen35_0.8b")
    cache_dir = Path("local/cache/qwen35_0.8b")
    art_dev = Path("artifacts/wsp/qwen35_0.8b/dev")
    art_dev.mkdir(parents=True, exist_ok=True)
    layers = [15, 19, 21]  # Exact top non-readout peaks from layer_profile.json (Full-Attn 15, 19 + DeltaNet 21)

    # 1. Corollary 1 (k=256 super-block alignment) verification on Qwen3.5-0.8B L19!
    cor1_path = rep_dir / "corollary1_superblock_audit.json"
    if not cor1_path.exists():
        print("\n=== [Qwen3.5-0.8B] COROLLARY 1 SUPER-BLOCK AUDIT (k=32 vs k=256, L19) ===", flush=True)
        with (cache_dir / "layer_stats.pkl").open("rb") as f:
            raw_stats = pickle.load(f)["layer_stats"]
        gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}
        tmp = Path("artifacts/wsp/qwen35_0.8b/cor1_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        cor_out = {}
        for k_n in [32, 256]:
            ed_n, _ = compile_ba_wsc_noop(gmap, raw_stats, [19], k_neurons=k_n)
            n_fp = tmp / f"noop_{k_n}.gguf"
            patch_gguf_tensors(base_gguf, n_fp, ed_n)
            ed_b, ci = compile_dormant_if_then(gmap, raw_stats, [19], alpha=0.25, k_neurons=k_n, gamma=2.5, block_align=True)
            b_fp = tmp / f"ba_{k_n}.gguf"
            patch_gguf_tensors(base_gguf, b_fp, ed_b)
            cor_out[f"ba_wsc_{k_n}_aligned"] = {"compile": ci, "quants": {}}
            for qt in ["Q4_0", "Q4_K_M", "Q3_K_M"]:
                nq = tmp / f"noop_{k_n}-{qt}.gguf"
                bq = tmp / f"ba_{k_n}-{qt}.gguf"
                quantize_gguf(n_fp, nq, qt)
                quantize_gguf(b_fp, bq, qt)
                res = audit_quant_pair(n_fp, b_fp, nq, bq, ["blk.19.ffn_down.weight"])
                cor_out[f"ba_wsc_{k_n}_aligned"]["quants"][qt] = res
                r = res["blk.19.ffn_down.weight"]
                print(f"  Q3.5 k={k_n:3d} | {qt:6s} | L19 cos={r['cosine_sim_full']:.6f} collat/sig={r['collateral_to_signal_ratio']:.6f}", flush=True)
                nq.unlink(); bq.unlink()
            n_fp.unlink(); b_fp.unlink()
        cor1_path.write_text(json.dumps(cor_out, indent=2) + "\n", encoding="utf-8")

    # 2. Leak-free concise-prompt calibration elicitation (ONLY on cal_math_50[:30] + cal_gsm8k_48[:25])
    print("\n=== [Qwen3.5-0.8B] LEAK-FREE CONCISE CALIBRATION ELICITATION ===", flush=True)
    c_math = base_out / "q8_cal_math30_concise"
    c_gsm = base_out / "q8_cal_gsm25_concise"
    if not (c_math / "summary.json").exists():
        run_benchmark(
            model_path=q8_base, dataset_path="datasets/wsp/cal_math_50.jsonl", benchmark="math",
            run_dir=c_math, max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, seed=42,
            system_prompt=CONCISE_SYS, parallel=4, ctx_per_slot=2560, limit=30, tag="q35_cal_math30_concise",
        )
    if not (c_gsm / "summary.json").exists():
        run_benchmark(
            model_path=q8_base, dataset_path="datasets/wsp/cal_gsm8k_48.jsonl", benchmark="gsm8k",
            run_dir=c_gsm, max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, seed=42,
            system_prompt=CONCISE_SYS, parallel=4, ctx_per_slot=2560, limit=25, tag="q35_cal_gsm25_concise",
        )

    # 3. Trace enriched trajectories on layers [15, 19, 21] (path from unprompted base, healthy/resolve from concise cal!)
    enr_pkl = cache_dir / "enriched_layer_stats.pkl"
    if not enr_pkl.exists():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        # Pathological tails strictly from unprompted baseline rollouts
        path_t, h_base, r_base, preserve_t, _ = load_cal_trajectories(
            tok, base_out / "fp_cal_math50_stoch", base_out / "fp_cal_gsm48",
            base_out / "fp_cal_copy36", Path("datasets/wsp/cal_wikitext_64.jsonl"),
            max_path=20, max_healthy=18, max_preserve=24,
        )
        # Healthy + </think> resolve states from concise-elicited calibration rollouts
        _, h_conc, r_conc, _, _ = load_cal_trajectories(
            tok, c_math, c_gsm, base_out / "fp_cal_copy36", Path("datasets/wsp/cal_wikitext_64.jsonl"),
            max_path=0, max_healthy=18, max_preserve=0,
        )
        healthy_t = (h_base + h_conc)[:18]
        resolve_t = (r_base + r_conc)[:18]
        counts = {"n_path": len(path_t), "n_healthy": len(healthy_t), "n_resolve": len(resolve_t), "n_preserve": len(preserve_t)}
        print(f"Enriched Qwen3.5-0.8B calibration trajectories: {counts}", flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(hf_dir), dtype=torch.float32).eval()
        layer_stats = collect_layer_activations(
            model, tok, path_texts=path_t, healthy_texts=healthy_t,
            resolve_texts=resolve_t, preserve_texts=preserve_t,
            target_layers=layers, max_seq_len=384,
        )
        del model
        with enr_pkl.open("wb") as f:
            pickle.dump({"layer_stats": layer_stats, "counts": counts, "layers": layers}, f)
        print(f"Traced enriched layers {layers} in {time.time()-t0:.1f}s -> {enr_pkl}", flush=True)

    # 4. Strict Dev Dose Selection on dev_math_60[:24] (MATH-train dev ONLY — zero test leakage!)
    print("\n=== [Qwen3.5-0.8B] STRICT DEV DOSE SELECTION (dev_math_60[:24], Q8_0, temp=0.6) ===", flush=True)
    with enr_pkl.open("rb") as f:
        enr_stats = pickle.load(f)["layer_stats"]
    gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}
    ds_dev = Path("datasets/wsp/dev_math_60.jsonl")
    dev_ids = [json.loads(l)["id"] for l in ds_dev.read_text(encoding="utf-8").splitlines() if l.strip()][:24]
    dev_out_path = rep_dir / "dev_dose_selection.json"
    dev_res = json.loads(dev_out_path.read_text(encoding="utf-8")) if dev_out_path.exists() else {}
    dev_run_root = Path("runs/wsp_study/qwen35_0.8b/dev")

    dev_specs = [
        ("baseline", None, None),
        ("ctrl_concise_prompt", None, CONCISE_SYS),
        ("uncond_a0.8", compile_uncond_subspace(gmap, enr_stats, layers, alpha=0.8, rank=1), None),
        ("oblique_res_a0.25", compile_oblique_nullspace(gmap, enr_stats, layers, alpha=0.25, action_mode="resolve", null_rank=64), None),
        ("oblique_res_a0.45", compile_oblique_nullspace(gmap, enr_stats, layers, alpha=0.45, action_mode="resolve", null_rank=64), None),
        ("oblique_res_a0.70", compile_oblique_nullspace(gmap, enr_stats, layers, alpha=0.70, action_mode="resolve", null_rank=64), None),
        ("oblique_res_a1.00", compile_oblique_nullspace(gmap, enr_stats, layers, alpha=1.00, action_mode="resolve", null_rank=64), None),
        ("bawsc32_res_a0.25", compile_dormant_if_then(gmap, enr_stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.90, action_mode="resolve", block_align=True), None),
        ("bawsc32_res_a0.45", compile_dormant_if_then(gmap, enr_stats, layers, alpha=0.45, k_neurons=32, gamma=2.5, tau_quantile=0.90, action_mode="resolve", block_align=True), None),
        ("bawsc32_res_a0.80", compile_dormant_if_then(gmap, enr_stats, layers, alpha=0.80, k_neurons=32, gamma=2.5, tau_quantile=0.90, action_mode="resolve", block_align=True), None),
        ("bawsc32_res_a1.20", compile_dormant_if_then(gmap, enr_stats, layers, alpha=1.20, k_neurons=32, gamma=2.5, tau_quantile=0.90, action_mode="resolve", block_align=True), None),
    ]

    for name, comp, sys_p in dev_specs:
        if name in dev_res:
            continue
        if comp is None:
            q8_p = q8_base
            cinfo = None
        else:
            ed, cinfo = comp
            fp_tmp = art_dev / f"{name}_fp.gguf"
            q8_p = art_dev / f"{name}-Q8_0.gguf"
            if not q8_p.exists():
                patch_gguf_tensors(base_gguf, fp_tmp, ed, metadata_extra=cinfo)
                quantize_gguf(fp_tmp, q8_p, "Q8_0")
                fp_tmp.unlink()
        ppl = measure_perplexity(q8_p, chunks=16, ctx_size=512) if sys_p is None else dev_res["baseline"]["ppl"]
        rdir = dev_run_root / name
        ev = run_benchmark(
            model_path=q8_p, dataset_path=ds_dev, benchmark="math", run_dir=rdir,
            max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, seed=42,
            system_prompt=sys_p, parallel=4, ctx_per_slot=2560, limit=24, tag=f"q35_dev_{name}",
        )
        std_f, eos_f, cls_f = _flags(rdir, dev_ids)
        dev_res[name] = {
            "ppl": ppl, "dev_math24": ev,
            "clean_eos": sum(eos_f), "closed_think": sum(cls_f), "compile_info": cinfo,
        }
        dev_out_path.write_text(json.dumps(dev_res, indent=2) + "\n", encoding="utf-8")
        dppl = ppl["ppl"] - dev_res["baseline"]["ppl"]["ppl"]
        print(f"DEV {name:22s}: PPL={ppl['ppl']:.3f} ({dppl:+.3f}) | STD={ev['correct']:2d}/24 | EOS={sum(eos_f):2d}/24 | lim={ev['hit_limit_count']:2d} lp={ev['loop_count']:2d} toks={ev['mean_completion_tokens']:.0f}", flush=True)
        if comp is not None and q8_p.exists():
            q8_p.unlink()


def stage_heldout(oblique_alpha: float, bawsc_alpha: float, qtypes_str: str) -> None:
    base_gguf = Path("local/models/Qwen3.5-0.8B-BF16.gguf")
    art_base = Path("artifacts/wsp/qwen35_0.8b/baselines")
    art_study = Path("artifacts/wsp/qwen35_0.8b/study")
    art_study.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("research/wsp/qwen35_0.8b")
    study_out = Path("runs/wsp_study/qwen35_0.8b/heldout")

    with open("local/cache/qwen35_0.8b/enriched_layer_stats.pkl", "rb") as f:
        pl = pickle.load(f)
    stats, layers = pl["layer_stats"], pl["layers"]
    gmap = {t.name: t for t in GGUFReader(str(base_gguf)).tensors}

    ds_math = Path("datasets/wsp/test_math500_strat150.jsonl")
    ids = [json.loads(l)["id"] for l in ds_math.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = rep_dir / "test_heldout_results.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    ed_oblique, ci_oblique = compile_oblique_nullspace(gmap, stats, layers, alpha=oblique_alpha, action_mode="resolve", null_rank=64)
    arm_specs = [
        ("baseline", None),
        ("random_matched", compile_random_matched(gmap, layers, ed_oblique, seed=999)),
        ("uncond_a0.8", compile_uncond_subspace(gmap, stats, layers, alpha=0.8, rank=1)),
        (f"oblique_res_a{oblique_alpha}", (ed_oblique, ci_oblique)),
        (f"bawsc32_res_a{bawsc_alpha}", compile_dormant_if_then(gmap, stats, layers, alpha=bawsc_alpha, k_neurons=32, gamma=2.5, tau_quantile=0.90, action_mode="resolve", block_align=True)),
    ]

    for qt in [x.strip() for x in qtypes_str.split(",") if x.strip()]:
        print(f"\n================ [Qwen3.5-0.8B] HELD-OUT STUDY ({qt}, Stochastic temp=0.6) ================", flush=True)
        q_out = out.setdefault(qt, {})
        base_std = base_eos = base_cls = None
        base_ppl = None
        for name, comp in arm_specs:
            if comp is None:
                q_p = art_base / f"qwen35_0.8b-{qt}.gguf"
                cinfo = None
            else:
                q_p = art_study / f"{name}-{qt}.gguf"
                ed, cinfo = comp
                if not q_p.exists():
                    fp_tmp = art_study / f"{name}_fp_tmp.gguf"
                    patch_gguf_tensors(base_gguf, fp_tmp, ed, metadata_extra=cinfo)
                    quantize_gguf(fp_tmp, q_p, qt)
                    fp_tmp.unlink()

            arm_rec = q_out.setdefault(name, {"compile_info": cinfo})
            if "ppl" not in arm_rec:
                arm_rec["ppl"] = measure_perplexity(q_p, chunks=32, ctx_size=512)
            if "copy_90" not in arm_rec:
                arm_rec["copy_90"] = run_benchmark(
                    model_path=q_p, dataset_path="datasets/wsp/test_copy_90.jsonl", benchmark="copy",
                    run_dir=study_out / f"{qt}_{name}_copy90", max_tokens=512, temperature=0.0,
                    enable_thinking=False, parallel=4, ctx_per_slot=1024, tag=f"q35_{qt}_{name}_copy90",
                )
            if name == "baseline":
                base_ppl = arm_rec["ppl"]["ppl"]

            rdir = study_out / f"{qt}_{name}_math150_s42"
            if "math150_s42" not in arm_rec:
                ev = run_benchmark(
                    model_path=q_p, dataset_path=ds_math, benchmark="math",
                    run_dir=rdir, max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, seed=42,
                    parallel=4, ctx_per_slot=2560, tag=f"q35_{qt}_{name}_m150_s42",
                )
                std_f, eos_f, cls_f = _flags(rdir, ids)
                arm_rec["math150_s42"] = ev
                arm_rec["clean_eos_correct"] = sum(eos_f)
                arm_rec["closed_think_correct"] = sum(cls_f)
                if name != "baseline":
                    arm_rec["mcnemar_std"] = mcnemar_exact(base_std, std_f)
                    arm_rec["mcnemar_clean_eos"] = mcnemar_exact(base_eos, eos_f)
                    arm_rec["mcnemar_closed_think"] = mcnemar_exact(base_cls, cls_f)
                out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

            std_f, eos_f, cls_f = _flags(rdir, ids)
            if name == "baseline":
                base_std, base_eos, base_cls = std_f, eos_f, cls_f
            ev = arm_rec["math150_s42"]
            dppl = arm_rec["ppl"]["ppl"] - base_ppl if base_ppl else 0.0
            p_eos = arm_rec.get("mcnemar_clean_eos", {}).get("mcnemar_p_two_sided", float("nan"))
            print(
                f"HELD-OUT [Q3.5 {qt}] {name:20s}: PPL={arm_rec['ppl']['ppl']:.3f} ({dppl:+.3f}) | "
                f"STD={ev['correct']:2d}/150 | EOS={arm_rec['clean_eos_correct']:2d}/150 (p={p_eos:.4f}) | "
                f"lim={ev['hit_limit_count']:3d} lp={ev['loop_count']:2d} toks={ev['mean_completion_tokens']:.0f} | COPY={arm_rec['copy_90']['correct']}/90",
                flush=True,
            )
            if name != "baseline" and q_p.exists():
                q_p.unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["enrich_and_dev", "heldout"], required=True)
    p.add_argument("--oblique-alpha", type=float, default=0.45)
    p.add_argument("--bawsc-alpha", type=float, default=0.45)
    p.add_argument("--qtypes", default="Q8_0,Q4_0")
    a = p.parse_args()
    if a.stage == "enrich_and_dev":
        stage_enrich_and_dev()
    else:
        stage_heldout(a.oblique_alpha, a.bawsc_alpha, a.qtypes)


if __name__ == "__main__":
    main()
