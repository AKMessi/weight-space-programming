"""Run mechanistic layer profiling on calibration rollouts and compile candidate WSP checkpoints."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.wsp.compiler import (
    LayerStats,
    analyze_layers,
    collect_layer_activations,
    compile_ba_wsc_noop,
    compile_dormant_if_then,
    compile_oblique_nullspace,
    compile_random_matched,
    compile_sparse_cols,
    compile_uncond_subspace,
)
from src.wsp.gguf_patch import patch_gguf_tensors
from src.wsp.gguf_to_torch import load_causal_lm_from_gguf
from src.wsp.llama_eval import format_messages


def load_cal_trajectories(
    tokenizer,
    cal_math_dir: Path,
    cal_gsm_dir: Path,
    cal_copy_dir: Path,
    cal_wikitext_path: Path,
    *,
    max_path: int = 28,
    max_healthy: int = 28,
    max_preserve: int = 40,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], list[str], dict[str, int]]:
    """Build strictly-calibration-only trajectory sets for PyTorch tracing."""
    # Map dataset items by id so we reconstruct the EXACT chat-templated prompt + generated output
    math_items = {
        json.loads(l)["id"]: json.loads(l)
        for l in Path("datasets/wsp/cal_math_50.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    gsm_items = {
        json.loads(l)["id"]: json.loads(l)
        for l in Path("datasets/wsp/cal_gsm8k_48.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    copy_items = {
        json.loads(l)["id"]: json.loads(l)
        for l in Path("datasets/wsp/cal_copy_36.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    def render_prompt(item: dict[str, Any], bench: str, enable_thinking: bool = True) -> str:
        msgs = format_messages(item, bench)
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    path_texts: list[tuple[str, int]] = []
    healthy_texts: list[tuple[str, int]] = []
    resolve_texts: list[tuple[str, int]] = []

    for run_dir, item_map, bench in [
        (cal_math_dir, math_items, "math"),
        (cal_gsm_dir, gsm_items, "gsm8k"),
    ]:
        resp_file = run_dir / "responses.jsonl"
        if not resp_file.exists():
            continue
        for line in resp_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            it = item_map.get(rec["id"])
            if not it or rec.get("error"):
                continue
            raw = rec["raw_response"]
            g = rec["grade"]
            prompt_str = render_prompt(it, bench, enable_thinking=True)
            full = prompt_str + raw
            p_len = len(prompt_str)

            # Pathological: loop detected OR hit token limit without closing think
            if (g.get("loop_detected") or g.get("hit_limit")) and not g.get("correct"):
                onset_w = g.get("loop_onset_word", -1)
                if onset_w is not None and onset_w > 0:
                    # Map word index approximately to char offset inside raw
                    words = raw.split()
                    prefix = " ".join(words[: min(onset_w, len(words))])
                    onset_char = p_len + len(prefix)
                else:
                    # Final 45% stalled tail of generation
                    onset_char = p_len + int(len(raw) * 0.55)
                # Keep the window near onset so max_seq_len=1024 captures the actual pathology!
                # Construct prompt + truncated slice centered around pathology if raw is huge
                if len(full) > 3600:
                    # Keep prompt + first 600 chars of think + 2400 chars starting near onset
                    raw_onset = max(0, onset_char - p_len)
                    tail_slice = raw[max(0, raw_onset - 200) : raw_onset + 2400]
                    full_trim = prompt_str + raw[:400] + "\n...\n" + tail_slice
                    onset_trim = len(prompt_str) + 405 + min(200, raw_onset)
                    path_texts.append((full_trim, onset_trim))
                else:
                    path_texts.append((full, onset_char))

            # Healthy: correct, non-looping, finished cleanly
            elif g.get("correct") and not g.get("loop_detected") and not g.get("hit_limit"):
                healthy_texts.append((full, p_len))
                if g.get("has_think_close") and "</think>" in raw:
                    close_char = p_len + raw.find("</think>")
                    resolve_texts.append((full, close_char))

    # Preservation texts: exact-copy prompt+gold AND WikiText-2 validation passages
    preserve_texts: list[str] = []
    for cid, it in list(copy_items.items())[: max_preserve // 2]:
        p_str = render_prompt(it, "copy", enable_thinking=False)
        preserve_texts.append(p_str + it["answer"])
    wt_rows = [
        json.loads(l) for l in cal_wikitext_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    for wr in wt_rows[: max_preserve // 2]:
        preserve_texts.append(wr["text"])

    path_texts = path_texts[:max_path]
    healthy_texts = healthy_texts[:max_healthy]
    resolve_texts = resolve_texts[:max_healthy]
    counts = {
        "n_path": len(path_texts),
        "n_healthy": len(healthy_texts),
        "n_resolve": len(resolve_texts),
        "n_preserve": len(preserve_texts),
    }
    return path_texts, healthy_texts, resolve_texts, preserve_texts, counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--base-gguf", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--cache-dir", default="local/cache")
    p.add_argument("--report-dir", default="research/wsp")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--force-retrace", action="store_true")
    a = p.parse_args()

    cache_dir = Path(a.cache_dir) / a.model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_pkl = cache_dir / "layer_stats.pkl"
    rep_dir = Path(a.report_dir) / a.model_name
    rep_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(a.baseline_dir)
    reader = GGUFReader(a.base_gguf, mode="r")
    gguf_map = {t.name: t for t in reader.tensors}

    if stats_pkl.exists() and not a.force_retrace:
        print(f"Loading cached layer_stats from {stats_pkl}...", flush=True)
        with stats_pkl.open("rb") as f:
            payload = pickle.load(f)
        layer_stats: dict[int, LayerStats] = payload["layer_stats"]
        counts = payload["counts"]
    else:
        t0 = time.time()
        print(f"Loading PyTorch model & tokenizer from {a.hf_dir}...", flush=True)
        tok = AutoTokenizer.from_pretrained(a.hf_dir)
        has_st = any(Path(a.hf_dir).glob("*.safetensors"))
        if has_st:
            model = AutoModelForCausalLM.from_pretrained(a.hf_dir, dtype=torch.bfloat16)
        else:
            print(f"No safetensors in {a.hf_dir}; loading exact BF16 weights from {a.base_gguf}...", flush=True)
            model = load_causal_lm_from_gguf(a.hf_dir, a.base_gguf)
        model.eval()

        path_texts, healthy_texts, resolve_texts, preserve_texts, counts = load_cal_trajectories(
            tok,
            base_dir / "fp_cal_math50",
            base_dir / "fp_cal_gsm48",
            base_dir / "fp_cal_copy36",
            Path("datasets/wsp/cal_wikitext_64.jsonl"),
        )
        print("Calibration trajectory counts:", counts, flush=True)
        if counts["n_path"] < 3 or counts["n_healthy"] < 3:
            raise RuntimeError(f"Insufficient pathological or healthy trajectories: {counts}")

        print("Collecting exact layer-by-layer activations (zero gradients)...", flush=True)
        layer_stats = collect_layer_activations(
            model,
            tok,
            path_texts=path_texts,
            healthy_texts=healthy_texts,
            resolve_texts=resolve_texts,
            preserve_texts=preserve_texts,
            max_seq_len=a.max_seq_len,
        )
        print(f"Collected {len(layer_stats)} layers in {time.time() - t0:.1f}s.", flush=True)
        with stats_pkl.open("wb") as f:
            pickle.dump({"layer_stats": layer_stats, "counts": counts}, f)
        del model

    print("Running mechanistic layer localization & dormancy audit...", flush=True)
    profile = analyze_layers(layer_stats, gguf_map)
    prof_path = rep_dir / "layer_profile.json"
    prof_path.write_text(
        json.dumps({"model_name": a.model_name, "counts": counts, "layers": profile}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved layer profile to {prof_path}")
    print(f"{'L':>3s} {'resDiff':>8s} {'mlpProj':>8s} {'aucX':>6s} {'fishX':>6s} {'aucZ':>6s} {'vbStd':>7s} {'d32/med':>8s}")
    for row in profile:
        print(
            f"{row['layer']:3d} {row['res_diff_norm']:8.3f} {row['mlp_loop_proj']:8.4f} "
            f"{row['auroc_x']:6.3f} {row['fisher_x']:6.2f} {row['auroc_z_diag']:6.3f} "
            f"{row['vb_std']:7.4f} {row['dormant_32_to_median']:8.4f}"
        )


if __name__ == "__main__":
    main()
