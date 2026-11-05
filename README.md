# Weight-space research on this machine

The expanded research portfolio is in [FIVE_EXPERIMENTS.md](FIVE_EXPERIMENTS.md): five distinct studies, explained in plain language, with model roles, comparisons, publication criteria, and a staged local budget. It is the latest experiment-design document; the earlier plan below retains useful hardware and artifact details.

Final research targets: **Qwen3-0.6B**, **LiquidAI LFM3-1.7B**, and **AKMESSI/lfm2.5-230m-fable-5**. The Fable parent is a matched control. SmolLM2 is no longer in the plan.

The goal is to repair a demonstrated behavioral failure with small, reversible weight edits while preserving useful behavior and verifying the actual quantized deployment. Qwen is the first mechanistic reference; Fable tests whether coding specialization can be preserved while repairing response behavior; LFM3 tests practical deployment and architecture-specific replication.

Read [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the earlier detailed strategy. [configs/pilot.toml](configs/pilot.toml) records its single-study envelope; it has not been converted into an execution configuration for the expanded five-study portfolio. Arm counts and datasets will be frozen after baseline characterization.

Evidence:

- [Inspected hardware](research/hardware.json)
- [Pinned model and artifact metadata](research/model_catalog.json)
- [User-reported LFM3 throughput](research/user_observations.json)

Important artifact distinction: the Fable release is GGUF-only, including F16; LFM3 QAD-Q4_0 is a separately trained quantized checkpoint, not ordinary PTQ of the public BF16 model. The plan handles those lineages separately.

**Status: implementation authorized and underway.** The local inference/evidence pipeline and grader tests are implemented. See [running instructions](docs/RUNNING.md), [progress](research/PROGRESS.md), and the append-only [journal](research/JOURNAL.md). Model improvements are not yet established.
