"""Baseline characterization of target OSS models across precisions on calibration splits & PPL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.wsp.gguf_patch import quantize_gguf
from src.wsp.llama_eval import measure_perplexity, run_benchmark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--base-gguf", required=True)
    p.add_argument("--out-root", default="runs/wsp_baselines")
    p.add_argument("--art-root", default="artifacts/wsp")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--ctx-per-slot", type=int, default=2560)
    p.add_argument("--quants", default="Q8_0,Q4_K_M,Q4_0,Q3_K_M")
    p.add_argument("--run-quant-cal", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args()

    out_root = Path(a.out_root) / a.model_name
    art_root = Path(a.art_root) / a.model_name / "baselines"
    out_root.mkdir(parents=True, exist_ok=True)
    art_root.mkdir(parents=True, exist_ok=True)

    results = {"model_name": a.model_name, "base_gguf": a.base_gguf, "precisions": {}}

    # 1. Floating-point calibration runs
    print(f"\n=== [{a.model_name}] BF16/F16 Calibration & PPL ===", flush=True)
    ppl_fp = measure_perplexity(a.base_gguf, chunks=32, ctx_size=512)
    print(f"  PPL(FP): {ppl_fp['ppl']:.4f} +/- {ppl_fp['ppl_se']:.4f} ({ppl_fp['elapsed_sec']}s)", flush=True)

    cal_math = run_benchmark(
        model_path=a.base_gguf,
        dataset_path="datasets/wsp/cal_math_50.jsonl",
        benchmark="math",
        run_dir=out_root / "fp_cal_math50",
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        parallel=a.parallel,
        ctx_per_slot=a.ctx_per_slot,
        tag=f"{a.model_name}_fp_cal_math50",
    )
    cal_gsm = run_benchmark(
        model_path=a.base_gguf,
        dataset_path="datasets/wsp/cal_gsm8k_48.jsonl",
        benchmark="gsm8k",
        run_dir=out_root / "fp_cal_gsm48",
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        parallel=a.parallel,
        ctx_per_slot=a.ctx_per_slot,
        tag=f"{a.model_name}_fp_cal_gsm48",
    )
    cal_copy = run_benchmark(
        model_path=a.base_gguf,
        dataset_path="datasets/wsp/cal_copy_36.jsonl",
        benchmark="copy",
        run_dir=out_root / "fp_cal_copy36",
        max_tokens=512,
        temperature=0.0,
        enable_thinking=False,
        parallel=a.parallel,
        ctx_per_slot=1024,
        tag=f"{a.model_name}_fp_cal_copy36",
    )
    results["precisions"]["FP"] = {
        "ppl": ppl_fp,
        "cal_math50": cal_math,
        "cal_gsm48": cal_gsm,
        "cal_copy36": cal_copy,
    }

    # 2. Quantize & measure PPL (+ optional cal_math50 on quantized baselines!)
    for qt in [x.strip() for x in a.quants.split(",") if x.strip()]:
        q_path = art_root / f"{a.model_name}-{qt}.gguf"
        if not q_path.exists():
            print(f"\n  Quantizing {a.model_name} -> {qt}...", flush=True)
            q_meta = quantize_gguf(a.base_gguf, q_path, qt)
        else:
            q_meta = {"dst_path": str(q_path), "dst_bytes": q_path.stat().st_size, "qtype": qt}
        ppl_q = measure_perplexity(q_path, chunks=32, ctx_size=512)
        print(f"  PPL({qt}): {ppl_q['ppl']:.4f} +/- {ppl_q['ppl_se']:.4f} | {q_meta['dst_bytes']/1e6:.1f} MB", flush=True)
        entry = {"quant_meta": q_meta, "ppl": ppl_q}
        if a.run_quant_cal:
            entry["cal_math50"] = run_benchmark(
                model_path=q_path,
                dataset_path="datasets/wsp/cal_math_50.jsonl",
                benchmark="math",
                run_dir=out_root / f"{qt}_cal_math50",
                max_tokens=a.max_tokens,
                temperature=a.temperature,
                parallel=a.parallel,
                ctx_per_slot=a.ctx_per_slot,
                tag=f"{a.model_name}_{qt}_cal_math50",
            )
        results["precisions"][qt] = entry

    rep_path = out_root / "baseline_summary.json"
    rep_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nSaved baseline report to", rep_path, flush=True)


if __name__ == "__main__":
    main()
