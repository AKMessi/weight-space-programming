"""Decisive test of the Greedy-Perturbation Illusion vs. True Causal WSP under multi-seed stochastic sampling.

Tests on Qwen3-0.6B (using Q8_0 for 2x RX 6400 speed with identical PPL 19.28 vs BF16 19.31):
  Part A (greedy_rand_seeds):
    - Multi-seed random Gaussian noise (seeds 999, 1000, 1001) at greedy temp=0.0 vs Q8_0 baseline,
      proving whether asymmetric argmax escape causes systematic positive bias for random edits at temp=0.0.
  Part B (stochastic_q8):
    - Multi-seed official stochastic sampling (temp=0.6, top_p=0.95, top_k=20; seeds 17, 42, 101)
      looped seed-outer / arm-inner so every seed yields a complete 5-arm paired comparison:
      baseline vs random_s999 vs oblique_res_a0.25 vs bawsc32_res_a0.25 vs uncond_a0.8 on held-out MATH-500-strat150.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from gguf import GGUFReader

from src.wsp.compiler import (
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_random_matched,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors, quantize_gguf
from src.wsp.llama_eval import mcnemar_exact, run_benchmark


def _flags(run_dir: Path, ids: list[str]) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
    mp = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std = [bool(mp[i]["grade"]["correct"]) for i in ids]
    eos = [bool(mp[i]["grade"]["correct"]) and mp[i]["finish_reason"] == "stop" for i in ids]
    cls = [bool(mp[i]["grade"]["correct"]) and mp[i]["grade"]["has_think_close"] for i in ids]
    lp = [bool(mp[i]["grade"]["loop_detected"]) for i in ids]
    return std, eos, cls, lp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["greedy_rand_seeds", "stochastic_q8", "both"], default="both")
    p.add_argument("--seeds", default="17,42,101")
    p.add_argument("--max-tokens", type=int, default=2048)
    a = p.parse_args()

    base_fp = Path("local/models/Qwen3-0.6B-BF16.gguf")
    gmap = {t.name: t for t in GGUFReader(str(base_fp)).tensors}
    with open("local/cache/qwen3_0.6b/layer_stats.pkl", "rb") as f:
        stats = pickle.load(f)["layer_stats"]
    layers = [18, 20, 22]

    ds_math = Path("datasets/wsp/test_math500_strat150.jsonl")
    ids = [json.loads(l)["id"] for l in ds_math.read_text(encoding="utf-8").splitlines() if l.strip()]

    art_dir = Path("artifacts/wsp/qwen3_0.6b/illusion_test")
    art_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path("runs/wsp_study/qwen3_0.6b/illusion_test")
    out_path = Path("research/wsp/qwen3_0.6b/greedy_vs_stochastic_results.json")
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    ed_oblique, _ = compile_oblique_nullspace(gmap, stats, layers, alpha=0.25, action_mode="resolve", null_rank=64)

    def get_q8(name: str, compiled) -> Path:
        q8_p = art_dir / f"{name}-Q8_0.gguf"
        if q8_p.exists():
            return q8_p
        if compiled is None:
            return Path("artifacts/wsp/qwen3_0.6b/baselines/qwen3_0.6b-Q8_0.gguf")
        ed, cinfo = compiled
        fp_tmp = art_dir / f"{name}_tmp.gguf"
        patch_gguf_tensors(base_fp, fp_tmp, ed, metadata_extra=cinfo)
        quantize_gguf(fp_tmp, q8_p, "Q8_0")
        fp_tmp.unlink()
        return q8_p

    if a.mode in ("greedy_rand_seeds", "both"):
        print("\n=== PART A: MULTI-SEED RANDOM GAUSSIAN NOISE AT GREEDY temp=0.0 (Q8_0) ===", flush=True)
        q8_base = get_q8("baseline", None)
        if "greedy_q8_baseline" not in results:
            ev = run_benchmark(
                model_path=q8_base, dataset_path=ds_math, benchmark="math",
                run_dir=run_root / "greedy_q8_baseline", max_tokens=a.max_tokens,
                temperature=0.0, parallel=4, ctx_per_slot=2560, tag="greedy_q8_baseline",
            )
            std_f, eos_f, cls_f, _ = _flags(run_root / "greedy_q8_baseline", ids)
            ev["clean_eos_correct"] = sum(eos_f)
            ev["closed_think_correct"] = sum(cls_f)
            results["greedy_q8_baseline"] = ev
            out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        b_std, b_eos, _, _ = _flags(run_root / "greedy_q8_baseline", ids)
        print(f"  greedy_q8_baseline            : STD={sum(b_std):2d}/150 | EOS={sum(b_eos):2d}/150 | lim={results['greedy_q8_baseline']['hit_limit_count']} lp={results['greedy_q8_baseline']['loop_count']}", flush=True)

        for rseed in [999, 1000, 1001]:
            key = f"greedy_q8_rand_s{rseed}"
            if key not in results:
                q8_r = get_q8(f"random_s{rseed}", compile_random_matched(gmap, layers, ed_oblique, seed=rseed))
                ev = run_benchmark(
                    model_path=q8_r, dataset_path=ds_math, benchmark="math",
                    run_dir=run_root / key, max_tokens=a.max_tokens,
                    temperature=0.0, parallel=4, ctx_per_slot=2560, tag=key,
                )
                std_f, eos_f, cls_f, _ = _flags(run_root / key, ids)
                ev["clean_eos_correct"] = sum(eos_f)
                ev["closed_think_correct"] = sum(cls_f)
                ev["mcnemar_std_vs_base"] = mcnemar_exact(b_std, std_f)
                ev["mcnemar_eos_vs_base"] = mcnemar_exact(b_eos, eos_f)
                results[key] = ev
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                if rseed != 999 and q8_r.exists():
                    q8_r.unlink()
            ev = results[key]
            print(f"  {key:30s}: STD={ev['correct']:2d}/150 | EOS={ev.get('clean_eos_correct', ev['correct']):2d}/150 (p={ev.get('mcnemar_eos_vs_base', {}).get('mcnemar_p_two_sided', 1.0):.4f}) | lim={ev['hit_limit_count']} lp={ev['loop_count']}", flush=True)

    if a.mode in ("stochastic_q8", "both"):
        print("\n=== PART B: MULTI-SEED OFFICIAL STOCHASTIC SAMPLING (temp=0.6, top_p=0.95, top_k=20) ===", flush=True)
        arm_specs = [
            ("baseline", None),
            ("random_s999", compile_random_matched(gmap, layers, ed_oblique, seed=999)),
            ("oblique_res_a0.25", (ed_oblique, {"method": "oblique_nullspace"})),
            ("bawsc32_res_a0.25", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True)),
            ("uncond_a0.8", compile_uncond_subspace(gmap, stats, layers, alpha=0.8, rank=1)),
        ]
        # Pre-build Q8_0 files once
        q8_paths = {name: get_q8(name, comp) for name, comp in arm_specs}
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

        # Seed-outer, arm-inner: complete 5-arm comparison after EVERY single seed!
        for sd in seeds:
            print(f"\n--- Stochastic Seed {sd} (temp=0.6, top_p=0.95, top_k=20) ---", flush=True)
            seed_flags = {}
            for name, _ in arm_specs:
                key = f"stoch_q8_{name}_s{sd}"
                rdir = run_root / key
                if key not in results:
                    ev = run_benchmark(
                        model_path=q8_paths[name], dataset_path=ds_math, benchmark="math",
                        run_dir=rdir, max_tokens=a.max_tokens,
                        temperature=0.6, top_p=0.95, top_k=20, seed=sd,
                        parallel=4, ctx_per_slot=2560, tag=key,
                    )
                    std_f, eos_f, cls_f, lp_f = _flags(rdir, ids)
                    ev["clean_eos_correct"] = sum(eos_f)
                    ev["closed_think_correct"] = sum(cls_f)
                    if "baseline" in seed_flags:
                        ev["mcnemar_std_vs_base"] = mcnemar_exact(seed_flags["baseline"][0], std_f)
                        ev["mcnemar_eos_vs_base"] = mcnemar_exact(seed_flags["baseline"][1], eos_f)
                        ev["mcnemar_lp_vs_base"] = mcnemar_exact(seed_flags["baseline"][3], lp_f)
                    if "random_s999" in seed_flags and name != "random_s999":
                        ev["mcnemar_eos_vs_rand"] = mcnemar_exact(seed_flags["random_s999"][1], eos_f)
                        ev["mcnemar_lp_vs_rand"] = mcnemar_exact(seed_flags["random_s999"][3], lp_f)
                    results[key] = ev
                    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                seed_flags[name] = _flags(rdir, ids)
                ev = results[key]
                eos_n = ev.get("clean_eos_correct", sum(seed_flags[name][1]))
                p_b = ev.get("mcnemar_eos_vs_base", {}).get("mcnemar_p_two_sided", float("nan"))
                print(
                    f"  {key:30s}: STD={ev['correct']:2d}/150 | EOS={eos_n:2d}/150 (p_vsB={p_b:.4f}) | "
                    f"lim={ev['hit_limit_count']} lp={ev['loop_count']} close={100*ev['think_close_rate']:.1f}% toks={ev['mean_completion_tokens']:.0f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
