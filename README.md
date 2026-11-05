# Weight-space research on this machine

Five bounded experimental pilots have now been executed across **Qwen3-0.6B, LFM3-1.7B and Fable 230M**. The response-repair candidates did not pass their development/confirmation gates. No generally useful weight repair or publication-ready editing method is established.

Read the [plain-language findings](research/FINDINGS.md), [five experiment reports](research/portfolio-v3/README.md), [current completion status](research/PROGRESS.md), and [reproduction instructions](docs/REPRODUCING.md). The [original five-study design](FIVE_EXPERIMENTS.md) retains the stronger research targets and publication criteria.

The [next-experiment plan](NEXT_EXPERIMENTS.md) prioritizes a verified behavioral repair and compact Fable adapter recovery, followed by the speed anomaly and gated deployment/reliability studies. No new experiments were started when writing that plan.

The practical results include:

- An exact GGUF patch utility: the original 461.9 MB Fable checkpoint can be reconstructed from its **matching cached base plus a 20.3 MB patch**. A within-source rounding control needs 44.7 MB, versus 167.7 MB before rounding. This uses established XOR/deflate; patch sizes exclude the base.
- A device-specific timing surprise: Fable F16 decodes at approximately **195 tokens/s**, versus 13 for Q8 and 22 for Q4 in the matched short-context follow-up. Larger files can be faster on this runtime/GPU; the mechanism is unresolved.
- A blinded diagnosis benchmark with explicit fault screening, healthy controls, permutation checks, raw editor replies, and simple baselines. The local 0.6B editor's weight-summary diagnosis matched the majority baseline and flagged every healthy presentation as degraded.
- Reversible GGUF edits, precision/provenance audits, controlled quantization, actual two-turn conversations, bounded functional Python checks, versioned grading corrections, and immutable request/response evidence.

Run on a Ryzen 3 3200G, 16 GB RAM and RX 6400 4 GB using the existing Vulkan llama.cpp build. Model binaries stay local and ignored. Ordinary LFM3 floating/Q8/Q4 comparisons use one source lineage and 18 GPU layers; QAD is a separate trained lineage. The Fable parent is an additional control, not a fourth target.

Compact raw evidence is under [runs/](runs/), artifact recipes under [research/patches/](research/patches/), and every decision/correction is retained in the [journal](research/JOURNAL.md). See [exact patch delivery](research/EXACT_PATCH_DELIVERY.md) for the usable utility. Earlier findings remain in [the first-milestone record](research/FINDINGS_FIRST_MILESTONE.md).
