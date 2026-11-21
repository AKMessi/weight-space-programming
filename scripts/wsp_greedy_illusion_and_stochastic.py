"""Decisive test of the Greedy-Perturbation Illusion vs. True Causal WSP under multi-seed stochastic sampling.

Tests on Qwen3-0.6B (using Q8_0 for 2x RX 6400 speed with identical PPL 19.28 vs BF16 19.31, plus BF16 parity check):
  1. Multi-seed random Gaussian noise (seeds 999, 1000, 1001) at greedy temp=0.0 -> proves whether
     asymmetric argmax escape causes systematic positive bias for random perturbations at temp=0.0.
  2. Multi-seed official stochastic sampling (temp=0.6, top_p=0.95, top_k=20; seeds 17, 42, 101)
     comparing baseline vs random_matched vs uncond_a0.8 vs oblique_res_a0.25 vs bawsc32_res_a0.25
     on held-out MATH-500-strat150!
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

    # Ensure Q8_0 versions exist for fast 2x RX 6400 multi-seed stochastic sweeps (PPL identical to BF16: 19.28 vs 19.31!)
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
            results["greedy_q8_baseline"] = run_benchmark(
                model_path=q8_base, dataset_path=ds_math, benchmark="math",
                run_dir=run_root / "greedy_q8_baseline", max_tokens=a.max_tokens,
                temperature=0.0, parallel=4, ctx_per_slot=2560, tag="greedy_q8_baseline",
            )
            out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        for rseed in [999, 1000, 1001]:
            key = f"greedy_q8_rand_s{rseed}"
            if key in results:
                continue
            q8_r = get_q8(f"rand_s{rseed}", compile_random_matched(gmap, layers, ed_oblique, seed=rseed))
            results[key] = run_benchmark(
                model_path=q8_r, dataset_path=ds_math, benchmark="math",
                run_dir=run_root / key, max_tokens=a.max_tokens,
                temperature=0.0, parallel=4, ctx_per_slot=2560, tag=key,
            )
            out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            if rseed != 999 and q8_r.exists():
                q8_r.unlink()

    if a.mode in ("stochastic_q8", "both"):
        print("\n=== PART B: MULTI-SEED OFFICIAL STOCHASTIC SAMPLING (temp=0.6, top_p=0.95, top_k=20) ===", flush=True)
        arm_specs = [
            ("baseline", None),
            ("random_s999", compile_random_matched(gmap, layers, ed_oblique, seed=999)),
            ("oblique_res_a0.25", (ed_oblique, {"method": "oblique_nullspace"})),
            ("bawsc32_res_a0.25", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True)),
            ("uncond_a0.8", compile_uncond_subspace(gmap, stats, layers, alpha=0.8, rank=1)),
        ]
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        for name, comp in arm_specs:
            q8_p = get_q8(name, comp)
            for sd in seeds:
                key = f"stoch_q8_{name}_s{sd}"
                if key in results:
                    continue
                ev = run_benchmark(
                    model_path=q8_p, dataset_path=ds_math, benchmark="math",
                    run_dir=run_root / key, max_tokens=a.max_tokens,
                    temperature=0.6, top_p=0.95, top_k=20, seed=sd,
                    parallel=4, ctx_per_slot=2560, tag=key,
                )
                results[key] = ev
                print(f"  {key:30s}: acc={ev['correct']:2d}/150 ({100*ev['accuracy']:.1f}%) | lim={ev['hit_limit_count']} lp={ev['loop_count']} close={100*ev['think_close_rate']:.1f}% toks={ev['mean_completion_tokens']:.0f}", flush=True)
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
