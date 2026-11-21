"""Automated publication-grade figure and statistical table generator verified from raw JSON logs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.wsp.llama_eval import mcnemar_exact, wilson_ci


def load_run_flags(run_dir: Path, dataset_path: Path) -> tuple[list[bool], dict[str, Any]]:
    ids = [json.loads(l)["id"] for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {}
    for l in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            by_id[r["id"]] = bool(r["grade"].get("correct"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return [by_id.get(i, False) for i in ids], summary


def plot_theorem1_quant_survival(audit_json: Path, out_png: Path) -> None:
    """Plot exact ΔW Cosine Similarity and Collateral-to-Signal Ratio across Q4_K_M / Q4_0 / Q3_K_M."""
    d = json.loads(audit_json.read_text(encoding="utf-8"))
    methods = [
        ("uncond_a0.6", "Uncond Rank-1\n(Steer2Edit)"),
        ("sparse64_a1.0", "Sparse-64 Cols\n(Takehi et al.)"),
        ("oblique_a0.35", "Oblique Null-Space\n(Ours A)"),
        ("dormant32_scattered", "Dormant-32 Scattered\n(No Align)"),
        ("ba_wsc_32_aligned", "BA-WSC-32 Aligned\n(Ours C)"),
    ]
    qtypes = ["Q4_K_M", "Q4_0", "Q3_K_M"]
    colors = ["#2563eb", "#d97706", "#dc2626"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    x = np.arange(len(methods))
    w = 0.25

    for qi, qt in enumerate(qtypes):
        cos_vals = []
        col_vals = []
        for mk, _ in methods:
            t_dict = d[mk]["quants"][qt]
            t_rec = next(iter(t_dict.values()))
            cos_vals.append(t_rec["cosine_sim_full"])
            col_vals.append(t_rec["collateral_to_signal_ratio"])
        ax1.bar(x + (qi - 1) * w, cos_vals, width=w, label=qt, color=colors[qi], edgecolor="black", linewidth=0.5)
        ax2.bar(x + (qi - 1) * w, col_vals, width=w, label=qt, color=colors[qi], edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in methods], fontsize=8.5)
    ax1.set_ylim(0.0, 1.08)
    ax1.set_ylabel("Full $\\Delta W$ Cosine Similarity (Higher = Better)")
    ax1.set_title("(a) Edit Signal Retention Under Quantization")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(frameon=True, fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in methods], fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_ylim(5e-4, 5.0)
    ax2.set_ylabel("Collateral / Signal Norm Ratio (Log, Lower = Better)")
    ax2.set_title("(b) Unedited Neighbor Block-Scale Distortion")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(frameon=True, fontsize=8.5)

    fig.suptitle("Theorem 1 Empirical Verification on Real llama-quantize GGUFs (Qwen3-0.6B L20 ffn_down)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", out_png)


def plot_quant_cliff_baseline(baseline_json: Path, out_png: Path) -> None:
    """Plot Quantization-Induced Reasoning Collapse & Repetition Loop Explosion on Qwen3-0.6B."""
    d = json.loads(baseline_json.read_text(encoding="utf-8"))
    order = ["FP", "Q8_0", "Q4_K_M", "Q4_0", "Q3_K_M"]
    labels = ["BF16", "Q8_0", "Q4_K_M", "Q4_0", "Q3_K_M"]
    accs = [100 * d["precisions"][k]["cal_math50"]["accuracy"] for k in order]
    loops = [100 * d["precisions"][k]["cal_math50"]["loop_rate"] for k in order]
    lims = [100 * d["precisions"][k]["cal_math50"]["hit_limit_rate"] for k in order]
    ppls = [d["precisions"][k]["ppl"]["ppl"] for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=300)
    x = np.arange(len(order))
    ax1.plot(x, accs, marker="o", linewidth=2.2, color="#16a34a", label="MATH Accuracy (%)")
    ax1.plot(x, loops, marker="s", linewidth=2.2, color="#dc2626", label="Repetition Loop Rate (%)")
    ax1.plot(x, lims, marker="^", linewidth=1.8, linestyle="--", color="#d97706", label="Hit 2048-Token Limit (%)")
    for i in range(len(order)):
        ax1.annotate(f"{accs[i]:.0f}%", (x[i], accs[i]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#16a34a")
        ax1.annotate(f"{loops[i]:.0f}%", (x[i], loops[i]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color="#dc2626")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 90)
    ax1.set_ylabel("Percentage of Problems (%)")
    ax1.set_title("(a) Accuracy Collapse vs. Loop Explosion")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=8.5)

    ax2.bar(x, ppls, color=["#334155", "#2563eb", "#0ea5e9", "#d97706", "#dc2626"], edgecolor="black", linewidth=0.6)
    for i, v in enumerate(ppls):
        ax2.text(x[i], v + 0.4, f"{v:.2f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(ppls) + 4)
    ax2.set_ylabel("WikiText-2 Perplexity")
    ax2.set_title("(b) WikiText-2 Perplexity Across Quantization Formats")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Quantization-Induced Reasoning Collapse in Qwen3-0.6B (cal_math_50, 2048 max tokens)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", out_png)


if __name__ == "__main__":
    plot_theorem1_quant_survival(
        Path("research/wsp/qwen3_0.6b/theorem1_quant_audit.json"),
        Path("research/wsp/figures/fig_theorem1_qwen3_0.6b.png"),
    )
    plot_quant_cliff_baseline(
        Path("runs/wsp_baselines/qwen3_0.6b/baseline_summary.json"),
        Path("research/wsp/figures/fig_quant_cliff_qwen3_0.6b.png"),
    )
