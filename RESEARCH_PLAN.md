# Final research strategy: targeted weight repairs on three local models

Version 2, 2026-11-05. This replaces the previous model selection. Strategy is finalized; dataset IDs, local artifact hashes, and measured runtime settings are frozen after baseline characterization. Implementation has not started and will wait for the user's instruction.

## Decision and scope

Use exactly three research targets:

1. **Qwen/Qwen3-0.6B**: first implementation and main mechanistic reference. Investigate unwanted repetition or failure to finish, provided it occurs under sensible intended-use settings.
2. **AKMESSI/lfm2.5-230m-fable-5**: main personalized application. Investigate whether we can recover strict response-format/tool-argument behavior while preserving useful code continuation.
3. **LiquidAI/LFM3-1.7B**: main deployment and hybrid-architecture target, including the user's working QAD-Q4_0 artifact. Test the same repair procedure on a demonstrated failure of its own, not an assumed Qwen failure.

SmolLM2 and Qwen3.5 are removed from the experiment schedule. The unmodified **LiquidAI/LFM2.5-230M parent** is a necessary matched control for Fable, not an additional research target. No models are trained or fine-tuned in this study; existing trained checkpoints are experimental subjects.

The overarching question is: **can a small, reversible, gradient-free weight edit repair a specific behavioral failure, preserve valuable existing behavior, and remain useful in the actual local deployment?**

The strongest personalized result would separate Fable's coding gains from an unwanted response habit. The broader scientific result would show which elements of the editing procedure work across a conventional transformer and hybrid architectures, under honest compute and precision controls. Improvements on different task suites are separate results; they cannot be averaged into a cross-model capability claim.

## Evidence driving the revision

The user's observed approximately **50 decode tokens/second on LFM3-1.7B with QAD 4-bit** makes that model a practical evaluation target. This is a user measurement, not a benchmark reproduced in this project. Preserve the working launch configuration and artifact before changing anything. Record the exact file, hash, runtime commit, backend, threads, GPU offload, context, cache types, prompt length, and generated length. A separately inspected CPU-only llama.cpp build does not establish which runtime produced the user's result. [Recorded observation](research/user_observations.json).

Fable's public release provides F16, Q8_0, and Q4_K_M GGUF artifacts, with no merged Transformers weights or adapter listed. Its card identifies LiquidAI/LFM2.5-230M as the parent and reports local code-continuation gains alongside some task/format regressions. These motivate fresh controlled tests; the published evaluations are prior observations, not our held-out evidence. [Fable model card](https://huggingface.co/AKMESSI/lfm2.5-230m-fable-5).

The official LFM3 repository distinguishes its **QAD-Q4_0 checkpoint, trained to recover quantization accuracy**, from ordinary converted quantizations. Consequently, re-quantizing the public BF16 model cannot be assumed to recreate the QAD checkpoint. [Official GGUF documentation](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).

The original weight-space report motivates investigating localized behavioral repair with an LLM editor and feedback, without gradient descent. It does not establish that arbitrary abilities can be added by editing a few neurons. [Motivating report](https://rikiyat.github.io/blog/weight-space-programming/).

## Verified model and architecture details

| Target | Architecture relevant to editing | Public artifacts checked | Research role |
|---|---|---|---|
| Qwen3-0.6B | 28 conventional attention/MLP layers; width 1024; MLP width 3072 | BF16 safetensors, 1.503 GB serialized | First adapter, causal localization, loop/completion study |
| Fable 230M | Parent has 14 hybrid convolution/attention layers; width 1024; MLP width 2560 | F16 GGUF 0.462 GB; Q8 0.247 GB; Q4_K_M 0.153 GB | Repair specialization side effects while preserving coding behavior |
| LFM3-1.7B | 24 layers: convolution, sliding attention, global attention; width 2048; MLP width 7680; sandwich normalization | BF16 safetensors 3.458 GB; QAD-Q4_0 GGUF 1.050 GB | Practical deployment and procedure replication on hybrid architecture |

Sizes are decimal GB and exclude inference overhead. Exact revisions and configurations are in [research/model_catalog.json](research/model_catalog.json). Official references: [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), [LFM2.5 parent](https://huggingface.co/LiquidAI/LFM2.5-230M), [LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B).

Architecture adapters must be explicit. Qwen's MLP output projection is `down_proj`; LFM3's inspected implementation uses `feed_forward.w2`, followed by `post_ffn_norm` before residual addition. A projection edit does not have the same effect through that normalization as in Qwen. Initially edit feed-forward outputs only; leave convolution state, attention, embeddings, output heads, and normalization weights unchanged. Validate hooks before and after normalization. [LFM3 implementation](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py).

No neuron index or residual direction is assumed portable between checkpoints. Replication transfers the procedure and measures whether it works; it does not blindly copy Qwen's directions into LFM.

## Hardware and runtime policy

Inspected system: Ryzen 3 3200G, four CPU cores, AVX2, approximately 16 GB RAM, RX 6400 with approximately 4 GB VRAM, Ubuntu 26.04, system Python 3.14.4, and ample free workspace storage. Vulkan detects the RX 6400. The specifically inspected llama.cpp build had GPU backends disabled. [Hardware snapshot](research/hardware.json).

Reuse the user's proven LFM3 inference setup first. Create an isolated project environment/build only as necessary; preserve the existing installation. Reference tensor inspection uses CPU PyTorch/NumPy with one model at a time, short windows, inference mode, and streamed statistics. Benchmark threads and dtypes rather than assuming BF16/FP16 is faster on this CPU. Reserve at least roughly 3 GiB system memory and avoid sustained swapping.

Qwen FP32 weights occupy roughly 2.4 GB when tied storage is deduplicated; LFM3 FP32 weights alone occupy roughly 6.9 GB. A naive load or duplicate FP32 model can exceed available RAM. LFM3 therefore needs memory-mapped/streamed inspection and bounded CPU reference runs, or selected GGUF tensor inspection. Its 50 tok/s quantized generation rate is not a promise about full-precision hooked inference.

Use batch one and an initial 2,048-token context. Characterize 4,096 only where necessary. Do not begin with 32K-token reasoning, large attention caches, exhaustive neuron sweeps, or multiple resident models. Builds and caches preferably live on native Linux storage; project artifacts remain in this workspace.

## Two primary studies and one replication

### A. Qwen: unwanted looping versus useful repetition

Hypothesis: a small set of feed-forward contributions may promote unrequested repetition; modifying them may restore useful completion while preserving deliberate copying.

Use recommended Qwen sampling separately for thinking and non-thinking modes. Thinking loops, ordinary repetition, and administrative token cutoffs are separate labels. Greedy thinking is a stress condition, not the sole baseline. [Qwen sampling guidance](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices).

Characterize arithmetic/reasoning, short continuations, exact copying, requested repeats, and structured responses. Compare prompt-only changes and validation-selected sampling controls first. Start the loop study only if the intended-use development condition has a meaningful reproducible failure, provisionally at least 10% across sufficiently many distinct prompts. If loops are absent, do not manufacture a bad setting; report this and prioritize study B.

Measure correct-and-complete answers, loop incidence, truncation, and preservation of useful repetition. A patch that just increases early EOS fails.

### B. Fable: preserve specialization while repairing response behavior

Hypothesis: part of the fine-tune's response-style change can be reduced without erasing useful coding behavior. This is a hypothesis about this checkpoint, not a judgment from parameter count.

Compare parent and fine-tune on fresh matched tasks. Separate four things: correct semantic payload, strict output contract, correct function selection/arguments, and code continuation. For example, a correct value wrapped in an unsolicited trace is a format failure; a wrong value is a semantic failure. Record both scores. Do not claim that removing explanations restores arithmetic reasoning.

The target is excessive trace/preamble output, schema violations, or argument filling only if the baseline confirms a systematic error with underlying competence. Preserve held-out code continuation, helpful explanation when explicitly requested, and ordinary response ability. Use new coding fixtures and schema families; include parse checks and exact-value graders. Broader repository benchmarks can be an extension with explicitly reserved IDs.

The first personalized edit comparison uses layerwise or selected-matrix rollback toward the matched parent and low-rank/column-selective edits. The rollback baseline is data-dependent weight arithmetic without training; it may change many weights and must not be described as sparse. Compare it with deterministic localization and an LLM-proposed targeted patch at equal evaluation budgets.

For the parent/fine-tune delta, first align architecture, tokenizer, tensor orientation, tied weights, and dtype. Compare both at the same precision. Do not attribute BF16-versus-F16 conversion differences to fine-tuning. A fine-tune delta is not itself an identified behavior circuit.

### C. LFM3: practical replication

Establish the user's existing QAD deployment baseline, then screen structured-output/tool-argument reliability and the common preservation tasks. No real external tools execute: requests and tool schemas are inert evaluation fixtures.

If the same failure occurs with baseline competence, replicate the selected procedure with LFM3-specific localization. Otherwise, use a separate named failure and report a separate case study. If neither failure is present, LFM3 serves as a robustness/control model; it is not edited merely to fill a results table.

The primary practical result is measured in the same runtime and QAD artifact lineage the user actually uses. A public BF16/PTQ experiment may provide mechanism evidence, but is not automatically evidence about QAD.

## Artifact handling and the QAD distinction

**Qwen:** inspect the pinned safetensors checkpoint; edit in FP32; evaluate after saving/casting; convert original and edited checkpoints using the same converter and ordinary quantizer. Compare floating runtime parity before attributing effects to quantization.

**Fable:** preserve all released GGUF files unchanged. The F16 GGUF is the starting weight artifact. Implement verified tensor reading/mapping and an identity round trip first. Direct GGUF edits can support weight arithmetic and deployment evaluation. Activation-based analysis requires either the matching merged checkpoint/adapter with its tokenizer or a validated mapping of the F16 GGUF into a reference model. Do not assume GGUF is directly loadable with `AutoModelForCausalLM`, and do not assume exact reverse conversion is already available. If reference parity is unresolved, run direct GGUF/weight-delta studies and label mechanistic conclusions accordingly. Access to an original training checkpoint would simplify the path, but is not required to finalize this strategy.

**LFM3 standard lineage:** separately test BF16-derived floating and ordinary PTQ checkpoints if memory and runtime permit. These give controlled editing/quantization comparisons.

**LFM3 QAD lineage:** pin the original QAD GGUF hash and compare every patch with that exact baseline. If a matching floating QAD checkpoint is available, use it. Otherwise, dequantization yields the effective quantized weights, not the missing pre-quantization trained checkpoint. Read/dequantize only needed tensors; preserve untouched quantized blocks where the writer permits. Candidate output formats include re-quantized touched tensors or mixed precision, with explicit runtime support and size/latency checks. Include a no-op rewrite/dequantize/repack control before behavioral edits. If a reliable writer or identity control is unavailable, keep QAD as an unmodified deployment reference and report that limitation rather than substituting BF16 results.

No-op controls must expose numerical drift, metadata errors, changed stop tokens, or unintended changes in untouched tensors. Report logical changed entries, physical artifact changes, update norms, and final file size separately. QAD training is upstream provenance; our edits introduce no new distillation or gradient training.

## Implementation sequence after the user's go-ahead

### Milestone 1 — Baseline report before weight editing

Inventory local artifacts/runtime settings, establish exact hashes and revisions, reuse the known working LFM3 setup, and smoke-test Qwen first. Add Fable GGUF and parent controls. Create evaluation graders and immutable run manifests before edit operators.

Per research target, screen 64 development prompts at two seeds and up to 256 output tokens; allow eight longer development cases at 1,024 tokens. This survey is exploratory, never the final test. Measure prefill/decode throughput, peak RAM/VRAM, and behavior under appropriate per-model templates and sampling.

Validate no-op save/load and, where needed, Torch/GGUF parity on fixed contexts. Confirm real failure prevalence and baseline task competence. Select one primary failure per model before intervention searches.

**Deliverable:** baseline table, workload estimate, precise target definitions, and which studies pass the feasibility gates. Finding no useful failure is a valid gate result. This checkpoint prevents spending the experiment budget on an unsupported premise.

### Milestone 2 — Causal diagnosis and edit operators

For passing studies, collect up to 48 matched productive/failure contexts, controlling task and approximate token position. Use short teacher-forced branches and streamed activation summaries where a verified reference exists. Include useful-copy/code cases and corresponding task-specific preservation data.

Test reversible activation interventions before permanent edits where supported. Implement attenuation, sparse directional projection, and Fable parent rollback as distinct methods. Begin with 16 or 32 channels at one or two feed-forward sites and small validation doses. For an output matrix W with shape [residual width, MLP width], selected columns J, unit direction u:

`W_new[:, J] = W_original[:, J] - alpha * u[:, None] * (u @ W_original[:, J])[None, :]`

This operation is a local matrix edit, not a general theorem about the resulting network. LFM normalization requires separate causal checks. A variant projects a failure direction away from a small basis associated with preserved behavior; it is a hypothesis, not an assumed solution. If the directions are inseparable, report that result.

### Milestone 3 — Controlled editor comparison

Use a fixed heuristic and an LLM editor with the same development evidence, feedback allocation, and maximum eight candidates per target. Log every hypothesis, requested tensor, edit, rejection, and outcome. Give the editor selected weights and diagnostic tools; withhold the test suite. No gradient descent, learned optimizer, hidden training loop, or finite-difference optimization is part of the primary study.

Controls: original checkpoint, best prompt/decoding-only configuration, deterministic edit, LLM edit, and support/norm-matched random edits. Include Fable parent rollback in development comparisons. Select a random-control draw using a declared independent seed rather than the most favorable test outcome. Fixed heuristics are essential: a successful edit alone does not show the LLM added value.

### Milestone 4 — Freeze, test, and deploy

Split by task/template family and code repository/fixture lineage as appropriate. Use distinct development and validation pools; reserve at least 48 validation prompts per target. Keep a 120-prompt pilot test per target: 40 failure-sensitive, 40 task-specific preservation, and 40 general controls. Qwen's loop-sensitive generation cap is 512; Fable/LFM3 initial caps are 256. These caps are locked after development characterization and increased only in a separately budgeted experiment.

Freeze patch bytes, tokenizer/template, sampling, grader, split IDs, seeds, and runtime before test evaluation. Test all cases, reporting baseline failures rescued and baseline successes broken. First pilot uses one seed. A larger independent follow-up uses more prompts and three seeds if the pilot warrants it; additional seeds are not independent new questions.

The eight-arm planning envelope per target is: original float, best decoding-only float, deterministic edited float, LLM edited float, random edited float, original quantized, selected edited quantized, and decoding-only quantized. For an inaccessible floating QAD lineage, replace unavailable arms with correctly labeled effective-weight/no-op controls or omit them; do not invent a floating QAD reference. Some arms are unnecessary if an earlier gate fails.

Deployment acceptance requires better useful task completion, preserved control behavior, artifact integrity, and measured latency/memory. Initially target at least 30% relative target-failure reduction and no more than a three percentage point preservation regression. These are decision targets; 40-item categories cannot establish a tight preservation margin by themselves. Report paired uncertainty and call a result inconclusive when the interval is too wide. Use prompt-clustered intervals with seeds grouped, plus exact upper bounds for rare regressions. Do not interpret an insignificant regression as proof of no harm.

## Budget and scheduling

The revised three-target investigation has a **1,800,000 generated-token ceiling in total**, not per model, and a separate initial ceiling of 1,000,000 teacher-forced scored tokens. No paid compute or unattended external API calls are assumed. The interactive LLM editor can operate in this session when implemented; any separate API experiment needs its own explicit usage budget.

| Allocation calculation | Worst-case generated tokens |
|---|---:|
| Survey: 64 prompts × 2 seeds × 256 × 3 models | 98,304 |
| Longer development: 8 × 1,024 × 3 | 24,576 |
| Candidate screens: 8 candidates × 2 editors × 16 prompts × 2 seeds × 256 × 3 | 393,216 |
| Eight-arm pilot: 120 × 8 × (512 + 256 + 256) | 983,040 |
| Smoke tests: 16 × 64 × 3 | 3,072 |
| Reserved for identity checks, retries, and additional controls | 297,792 |
| **Total cap** | **1,800,000** |

Within the survey, reuse the shared preservation prompts across models where meaningful; use task-specific prompts for different failure targets. Count teacher-forced and generated tokens separately. Count sequences and feedback calls too. Do not silently exceed the cap by adding precision variants, longer contexts, or repeated tests.

At the user's observed 50 tok/s, 1.8M tokens would take 10 hours of pure decode if every arm had that speed. That is only an optimistic reference: CPU reference models, hook collection, prefill, exports, and the other models can be slower. Measure each arm, then predict total time. Release the experiment budget in stages: approximately 200K for baseline characterization first; spend the rest on studies that pass. Reserve 25–40 GB for environments, originals, matched-parent artifacts, exports, and logs; keep candidate deltas rather than full copies.

Expected order is approximately one to two work sessions for baselines/compatibility, two to four for diagnosis and edits, and two to four for frozen evaluation and deployment. This is a work sequence, not a guaranteed calendar deadline.

## Value, novelty, and deliverables

The user-facing outcome is a reproducible patch that improves a model we actually use, with measured tradeoffs. The scientific outcome could be a demonstrated separation between a fine-tune's useful specialization and an unwanted habit, a failure to separate those effects, a gap between ordinary PTQ and QAD repair, or an LLM editor that matches/beats/falls short of a fixed algorithm.

Relevant prior work includes [Repetition Neurons](https://aclanthology.org/2025.naacl-short.41/), [translation error repair](https://aclanthology.org/2024.emnlp-main.879/), [ROME](https://rome.baulab.info/), and [Composable Interventions](https://arxiv.org/abs/2407.06483). Sparse editing, weight interpolation, and quantization-aware methods are not novel by themselves. Any new claim must be narrower and supported by these specific controls. No publication or positive result is promised.

Deliverables: model/artifact manifests; evaluation fixtures and graders; a baseline report; validated inspect/trace/patch/restore/export tools; editor transcripts and fixed budgets; compact patches tied to exact hashes if successful; raw results with confidence intervals; and a short report with both successes and regressions. Implement the CLI/data pipeline first. A visual comparison UI follows reliable results.

## Current status

Only planning documents and metadata snapshots have been updated. No environments installed, model weights downloaded, model files modified, harness implemented, or experiments run. Next action, after the user instructs us to start, is **Milestone 1: baseline and artifact compatibility**, beginning with Qwen and the user's existing LFM3 runtime, then the Fable/parent pair.
