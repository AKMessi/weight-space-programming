"""Held-out OOD Transfer & Preservation Evaluation (GSM8K-100 & ARC-Challenge-100) on Qwen3-0.6B (Q8_0).

Evaluates under official stochastic sampling (temp=0.6, top_p=0.95, top_k=20) with strict Clean-EOS
auditing and paired exact McNemar tests against BOTH unedited baseline AND matched random noise.
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


def _flags(run_dir: Path, ids: list[str]) -> tuple[list[bool], list[bool]]:
    mp = {
        json.loads(l)["id"]: json.loads(l)
        for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    std = [bool(mp[i]["grade"]["correct"]) for i in ids]
    eos = [bool(mp[i]["grade"]["correct"]) and mp[i]["finish_reason"] == "stop" for i in ids]
    return std, eos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    base_fp = Path("local/models/Qwen3-0.6B-BF16.gguf")
    gmap = {t.name: t for t in GGUFReader(str(base_fp)).tensors}
    with open("local/cache/qwen3_0.6b/layer_stats.pkl", "rb") as f:
        stats = pickle.load(f)["layer_stats"]
    layers = [18, 20, 22]

    art_dir = Path("artifacts/wsp/qwen3_0.6b/illusion_test")
    art_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path("runs/wsp_study/qwen3_0.6b/ood")
    out_path = Path("research/wsp/qwen3_0.6b/ood_gsm8k_arc_results.json")
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

    arm_specs = [
        ("baseline", None),
        ("random_s999", compile_random_matched(gmap, layers, ed_oblique, seed=999)),
        ("oblique_res_a0.25", (ed_oblique, {"method": "oblique_nullspace"})),
        ("bawsc32_res_a0.25", compile_dormant_if_then(gmap, stats, layers, alpha=0.25, k_neurons=32, gamma=2.5, tau_quantile=0.95, action_mode="resolve", block_align=True)),
        ("uncond_a0.8", compile_uncond_subspace(gmap, stats, layers, alpha=0.8, rank=1)),
    ]

    benchmarks = [
        ("gsm8k", Path("datasets/wsp/test_gsm8k_250.jsonl"), 2048, 2560, True),
        ("arc", Path("datasets/wsp/test_arc_challenge_250.jsonl"), 1024, 1536, True),
    ]

    for bname, ds_path, max_tok, ctx_slot, think in benchmarks:
        ids = [json.loads(l)["id"] for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()][: a.limit]
        b_res = results.setdefault(bname, {})
        base_std = base_eos = None

        for name, comp in arm_specs:
            q8_p = get_q8(name, comp)
            rdir = run_root / f"{bname}_{name}_s{a.seed}"
            if name not in b_res:
                ev = run_benchmark(
                    model_path=q8_p,
                    dataset_path=ds_path,
                    benchmark=bname,
                    run_dir=rdir,
                    max_tokens=max_tok,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    seed=a.seed,
                    enable_thinking=think,
                    parallel=4,
                    ctx_per_slot=ctx_slot,
                    limit=a.limit,
                    tag=f"{bname}_{name}_s{a.seed}",
                )
                std_f, eos_f = _flags(rdir, ids)
                entry = {"summary": ev, "clean_eos_correct": sum(eos_f)}
                if base_std is not None:
                    entry["mcnemar_std"] = mcnemar_exact(base_std, std_f)
                    entry["mcnemar_clean_eos"] = mcnemar_exact(base_eos, eos_f)
                b_res[name] = entry
                out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            std_f, eos_f = _flags(rdir, ids)
            if name == "baseline":
                base_std, base_eos = std_f, eos_f
            ev = b_res[name]["summary"]
            p_str = ""
            if "mcnemar_clean_eos" in b_res[name]:
                p_str = f" (p_eos={b_res[name]['mcnemar_clean_eos']['mcnemar_p_two_sided']:.4f})"
            print(
                f"OOD {bname:5s} | {name:20s}: STD={ev['correct']:3d}/{a.limit} | "
                f"EOS={b_res[name]['clean_eos_correct']:3d}/{a.limit}{p_str} | "
                f"lim={ev['hit_limit_count']} lp={ev['loop_count']} toks={ev['mean_completion_tokens']:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
