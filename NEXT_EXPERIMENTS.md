# Next experiments: obtain one useful, defensible weight-space result

**Planned:** 2026-11-05. **Status:** planning only; no new model inference, editing, training or downloads started for this plan. The completed pilot and its raw evidence remain unchanged.

## Recommendation

Make the next milestone **one useful patch or recovered adapter with a fresh evaluation**, plus a measured explanation of the Fable speed anomaly. Give the most attention to the first two experiments below. Quantized deployment and missing-evidence repair follow only when their prerequisites pass.

Our previous measurements tested fixed normalization changes and parent rollbacks. They did not establish a trustworthy activation-derived repair, did not demonstrate Fable's repository-context advantage independently, and did not test a learned semantic edit against a published component-editing baseline. Those are the gaps to resolve next. More arbitrary attenuation variants would not answer them.

| Priority | Experiment | What success would let us do | Main model |
|---|---|---|---|
| 1 | A conditional response repair that preserves useful answers | Install a small patch that follows strict instructions while retaining requested explanations and coding behavior | Qwen first; Fable and LFM3 after gates |
| 2 | Recover a compact adapter from merged Fable weights | Distribute/rebuild Fable from its matching base and a small approximate adapter, without the original training adapter | Fable and its aligned parent |
| 3 | Explain and reduce the Fable quantization slowdown | Choose a faster deployment format using operator measurements, quality, memory and latency | Fable; Qwen as a backend control |
| 4 | Preserve a real patch during 4-bit deployment | Ship the validated behavior or adapter at a measured storage/speed cost | Whichever model passes 1 or 2; LFM3 QAD separately |
| 5 | Make clarification follow the evidence | Ask for an absent/conflicting field, then answer once it is supplied, without extra checker calls | LFM3 first if competent; Qwen replication |

These are a ranked program, not five promised papers. Experiments 1, 2 and 3 can yield separate useful results. Experiments 4 and 5 are conditional extensions. Composition and another weight-only diagnosis benchmark have lower priority until we have actual useful edits and a suitable editor.

## First resolve the measurement gate

**Question:** are we observing the exact internal state that produced the native model's logits?

The old separate embedding/generation path preserved top-1 in six contexts but failed its centered-logit tolerance. That does not make activation methods unavailable. It means that path was not validated.

There is a concrete alternative: the working binary directory already contains `llama-debug` and `llama-cvector-generator`. The inspected llama.cpp source exposes evaluation callbacks; the upstream debug example saves logits and token IDs and can filter tensor traces. These are starting points, not proof that their semantics match our working build. [Official debug example](https://github.com/ggml-org/llama.cpp/tree/master/examples/debug).

Before editing:

1. Fingerprint the existing debug binaries and colocated libraries. Match their version to the working `llama-server`. The build cache records a source directory that is currently absent; recover the exact source revision into an isolated checkout if a build is necessary. Do not replace the working installation.
2. Start with Qwen floating weights on CPU. Capture the final state and pre-sampling logits **from the same forward execution**, including exact token IDs, positions, normalization boundary, output matrix and any output transformation. Do not infer equivalence merely from matching token counts.
3. Use 24 visible diagnostic contexts spanning strict answers, explanations, code and different prefix lengths. Include teacher-forced continuation positions as well as the last prompt position. These are calibration fixtures, not repair-quality tests.
4. Reconstruct the output projection in FP32 and compare centered logits and token ranks with native outputs. Proposed CPU acceptance: maximum centered error <=0.001 and relative centered error <=0.001 on the declared comparison set. Freeze the exact set and low-norm error policy before collecting results. Preserve the original failed gate; never relabel it as passed.
5. Repeat with Vulkan. Retain the earlier GPU thresholds of maximum error <=0.05 and relative error <=1%, and test an instrumentation-only no-op against uninstrumented generation. If graph tracing changes kernel selection or output, quantify that difference before using the trace to select edits.
6. Confirm a genuine causal intervention: a known change at the traced tensor must have the predicted immediate effect. A deliberately injected fault is only an instrumentation positive control, never a natural repair result.

If CPU passes and GPU tracing does not, use the verified CPU path for bounded diagnostics and validate each exported patch separately under the untouched GPU server. If both paths fail after a fixed diagnostic budget, close this method route and prioritize Experiments 2 and 3; do not spend the budget on undocumented instrumentation changes or claim activation editing impossible.

## 1. A conditional response repair that preserves useful answers

### In simple words

Can the model answer briefly when asked, explain when asked, and keep solving the task after a small permanent weight change?

A model that becomes terse everywhere has not learned the switch. A model that outputs valid JSON with a wrong value has not improved.

### Why this is the main scientific bet

It directly tests an unimplemented part of our first design: choosing an edit using both failures to repair and successes to protect. Published training-free component editing is already available through [Steer2Edit](https://arxiv.org/abs/2602.09870v3). Null-space preservation already appears in [AlphaEdit](https://arxiv.org/abs/2410.02355). Therefore a steering vector, rank-one update or preservation projection alone is not our novelty claim.

The candidate contribution is a better measured tradeoff for **instruction-dependent response switching in small deployed models**, followed by an honest deployment test. Reproducing a known method on Qwen is a baseline result, not sufficient evidence of a new method.

### Competence gate before repair search

Begin with Qwen3-0.6B's existing floating source. Keep its community-export lineage distinct from the official Q8 checkpoint. Verify underlying task competence using clear prompts or a suitable output contract. Require at least 80% content correctness on the selected development task family and a reproducible response-contract failure rate of at least 20%. Do not manufacture a failure by using unsuitable decoding.

Use matched task pairs: plain answer versus requested explanation; next-line continuation versus requested code explanation; plain JSON versus an explicitly requested fenced example. The desired output must differ because of the instruction, not because one group has different subject matter or answers. Include familiar and unfamiliar schema forms.

Keep a protection panel of correct factual extraction, copying, requested explanation and code continuation. For Fable, first verify repository-context continuation against its parent using repository-disjoint data. Do not call a toy function score its specialization. If a model cannot perform the underlying task, record it as an ineligible target for this behavior rather than trying to erase its inability.

### Methods and controls

- Original model, one frozen stronger-prompt policy, and grammar-constrained structured output where applicable.
- The old attenuation baseline and an equal-size random edit.
- A mean-difference activation intervention as a reversible causal baseline.
- A faithful, documented Steer2Edit implementation on supported Qwen components.
- A candidate edit that uses protected examples to constrain damage, compared with an unconstrained version at the same edited-value and feedback budget.

Collect directions at a verified MLP/attention output boundary. Test whether reversible intervention changes the target behavior before exporting a permanent edit. A classifier that separates outputs does not establish that changing the direction repairs them. Architecture adaptations to LFM hybrid blocks require their own trace/intervention checks; Qwen component semantics must not be copied blindly.

Do not start with tied output embeddings as a shortcut. A change there can alter both input representations and the readout. Any such arm would require an explicit untied-head no-op control and a separate interpretation.

### Evaluation and selection

Use new development/calibration cases, then a **fresh 240-case confirmation panel for the first eligible model**, split by instruction family and code/repository lineage. The previous portfolio's confirmation sets are already exposed and cannot serve again as independent confirmation. Keep repository relatives and paired instructions together in splits and uncertainty analysis.

Screen at most twelve predeclared candidate settings on a small development panel. Shortlist at most three before full development scoring. Freeze the winning patch, decoder/prompt controls, extraction rules and output caps before the fresh test. Primary decoding is deterministic; a smaller, separately declared recommended-sampling panel checks stability with matched seeds.

Proposed pilot target: >=30% relative reduction in the specified strict-response failure, no observed material decline on protected skills, and a useful tradeoff against the strongest simple control. Report absolute gains, rescues, new failures and uncertainty. A strong preservation claim additionally needs the one-sided confidence bound to exclude a loss larger than three percentage points; if the sample cannot resolve that, call preservation inconclusive rather than proved. Longer explanations need matched adequate caps across all arms.

**Deliverable:** a reversible edited GGUF, exact delta, clear success/failure report and an instruction-switching evaluation. If prompting/grammar dominates the edit, recommend it and retain that result.

## 2. Recover a compact adapter from merged Fable weights

### In simple words

Can we recover a small package that makes the parent behave like your released Fable model, even without the original LoRA file?

This has the strongest immediate practical case. The model card reports rank-32 LoRA training, while our effective-precision audit found substantial differences concentrated in 18 Q/K/V matrices. That makes low-rank recovery a motivated hypothesis. It does not prove the residual is exactly rank 32 or identify the historical parent. [Fable model card](https://huggingface.co/AKMESSI/lfm2.5-230m-fable-5).

### Experiment

1. Reuse the pinned original Fable, public F16 parent, same-source BF16-rounded parent and separately published aligned BF16-value parent. Keep source choice and within-source rounding as distinct comparisons.
2. Measure singular-value spectra of each substantial matrix difference, one matrix at a time. Compare the apparent rank before and after precision alignment. Add a small known low-rank matrix/rounding control to separate mathematical rank from rounding noise; it is a numerical control, not target-model training.
3. Construct approximate adapters at ranks 4, 8, 16, 32 and 64, then at matched actual byte budgets. The rank-32 cue is declared in advance. Never infer that recovered factors are the exact original LoRA factors: factorization is non-unique and merged rounding can destroy exact low rank.
4. Compare aligned and unaligned SVD, ordinary 4/8-bit delta packing, an uncalibrated 1-bit sign-and-mean-scale baseline, and our exact XOR patch. Preserve tiny residual tensors explicitly where cheap. Count factor metadata, scales and any residual sidecar in the package size.
5. Select on calibration only, then compare the reconstructed model with original Fable and its parent on a fresh repository-disjoint next-line panel, generic extraction/tool contracts, and held-out teacher-forced distributions.

Use a bounded RepoBench-C-style subset with a fixed context-window policy; pin source revisions and retain repository identities. It is a short-context adaptation, not an official full RepoBench score. Exact line match measures text agreement, not general functional equivalence. [RepoBench](https://arxiv.org/abs/2306.03091).

Approximate delta compression already exists in [BitDelta](https://arxiv.org/abs/2402.10193), [DeltaZip](https://arxiv.org/abs/2312.05215) and [Delta-CoMe](https://arxiv.org/abs/2406.08903). Full BitDelta scale distillation uses optimization and is outside the current no-training scope; label its uncalibrated baseline accurately. A relevant training-free Delta-CoMe implementation is a stronger comparator than SVD alone. Do not declare a compression winner over an unimplemented relevant method.

### Success and value

A **<=5 MB package** versus the existing 20.3 MB exact patch is a useful target, not a forecast. The base remains required and excluded from patch size. Report quality as absolute differences from Fable with uncertainty; do not use retention percentages when the Fable-parent advantage is small or unverified. Seek no more than a three-point loss on the predeclared task panel, with an adequate uncertainty bound before claiming preservation, plus distributional fidelity on unseen contexts.

If Fable's coding advantage does not replicate, a compact approximation can still be a useful model-delivery result, but not preservation of a proven superior specialist. Export a native adapter only after its loader/scaling no-op checks pass; otherwise merge into a new F16 GGUF and distinguish delivery size from resident model size.

**Possible research contribution:** how effective precision and parent provenance distort apparent adapter rank and behavior-preserving compression. One Fable case supports a case study/tool, not a universal claim; broader publication needs additional independently sourced fine-tunes with known provenance.

## 3. Explain and reduce the quantized Fable slowdown

### In simple words

Why does the larger F16 model run much faster here, and can we keep its speed while storing fewer bytes?

The observed medians were 194.60 tokens/s for F16, 13.40 for Q8_0 and roughly 22 for Q4. The verbose log reports layer offload, which does not identify slow operators or exclude CPU work within a graph. The mechanism is unresolved.

### Experiment

- Recheck pinned F16/Q8_0/Q4_0/Q4_K_M with matched native settings and CPU versus Vulkan as separate conditions. Never mix their timings into one average.
- Randomize format order across at least three sessions. Use seven post-warmup requests per condition at prompt lengths around 128, 512 and 1,536 tokens, with a fixed actual decode length and disabled prompt reuse. Archive actual token counts and templates.
- Measure time to first token, prefill, decode, total useful-answer latency, process memory and sampled device-wide VRAM. Timing fixtures and quality fixtures stay separate.
- Collect bounded per-operator/backend traces, with profiling overhead measured separately. Inspect whether a hybrid operation, conversion, synchronization or CPU fallback dominates. Treat each as a hypothesis until measured.
- If a bottleneck is localized, test a small predeclared hybrid GGUF family that keeps the implicated tensors F16 and quantizes the rest. Compare at equal storage or VRAM budgets, include no-op rewrites, and rerun quality tests. Do not assume the fastest format has the best useful output.

Use the existing Qwen format panel as a control for whether an effect is Fable-specific; run only a small matched control if the old panel's settings do not match. LFM3 is a later check on the identified mechanism, not a full expensive sweep by default.

**Success target:** explain the measured gap with operator-level evidence and find a deployment with materially lower bytes/memory and close-to-F16 useful latency, or document that the full F16 is the best choice here. A desired target is >=30% smaller with <=10% latency penalty and no material measured quality loss; it may not be achievable.

**Value:** a reproducible backend issue, format-selection tool, or validated hybrid export. Results on one RX 6400/runtime are a systems case study until replicated on another backend/device. No kernel rewrite is assumed before profiling.

## 4. Make an actually useful patch survive 4-bit deployment

**Gate:** Experiment 1 supplies a confirmed repair, or Experiment 2 supplies an adapter with verified useful task fidelity. Label these separately: retaining an existing fine-tune and repairing a behavior are different objectives.

Freeze the source patch before precision selection. Compare unedited and edited outputs under the same ordinary floating/Q8/Q4 lineage. Use edits to real matrix values; our old F32 normalization changes could not test a patch being rounded away.

At matched extra-byte budgets compare ordinary whole-model Q4, Q8, all touched tensors retained at F16, random precision allocation, magnitude/reconstruction-error allocation, and behavioral calibration allocation. Candidate higher-precision pieces can be tensors or supported quantization blocks; unsupported block layouts are not silently simulated as native deployment. Include the option of an FP16 adapter over a quantized base only after native loader parity is verified.

For QAD, retain its released integer codes/scales, use the existing exact no-op control, and evaluate any direct/mixed export against the original QAD baseline. Do not treat ordinary LFM3 Q4 as a reconstruction of QAD. Every modified artifact is a new deployment.

Keep precision calibration separate from a new final test. Report absolute behavioral gains, useful latency, bytes and VRAM. An unedited same-precision baseline is mandatory. Do not report a repair-retention percentage over negligible floating gain. A matched-byte advantage over the strongest simple alternative, repeated on fresh tasks, is the proposed contribution. Compression/edit interactions are already established in [Composable Interventions](https://arxiv.org/abs/2407.06483).

## 5. Make clarification depend on evidence

**Gate:** verified internal access and development evidence that the chosen model can perform the target decision task. Begin with LFM3 because the first pilot showed correct supported answers alongside unsupported guesses; validate this on new development examples before selection. The old 4/12 score is not a claim that this method will work.

Use complete/absent/conflicting versions of the same supplied record, followed by an authoritative update in an actual second turn. Also include a supplied value containing words such as "unknown", distractor fields, reordered facts and paraphrased instructions, so lexical cues do not determine the decision.

Derive an evidence-sensitive direction, then compare reversible steering, a fixed sparse component edit and an edit constrained by supported-answer examples. Match answer coverage rather than rewarding "ask" everywhere. Compare strong few-shot prompting, schema decoding, a separate answerability call with its cost, and a deterministic required-field checker where structured records make one possible.

Hold out complete field/schema and layout families, not random variants. Primary outcome is full two-turn semantic success; also report unsupported non-null values, wrong clarification fields, supported-answer loss and recovery after the update. Proposed target: halve unsupported non-null answers with supported-answer loss <=3 points and improved joint success, with uncertainty sufficiently narrow before calling it a reliable repair.

A grammar that prevents unsupported values by encoding the correct answerability decision is a deterministic checker baseline, not evidence that the model learned it. Probe classification accuracy or reduced willingness to answer is insufficient.

**Value if successful:** a more reliable local extraction/tool assistant with a merged edit and no extra inference call. If the deterministic checker or prompting is clearly better, use it and close this editing hypothesis.

## Hardware and finite budget

Stay with the Ryzen 3 3200G, 16 GB RAM and RX 6400 4 GB. One model resident; CPU diagnostics are short and selective, GPU execution handles repeated quality panels. Avoid a whole-model FP32 duplicate, full activation dumps or full covariance inversions. Capture selected token positions and tensor sites; use diagonal/low-rank covariance approximations as explicitly labeled adaptations where necessary. Process Fable delta matrices sequentially.

Proposed operational caps are 10 GB host resident memory and about 3 GB process GPU allocation, allowing desktop headroom. The existing VRAM sampler is device-wide and cannot enforce a process allocation cap by itself. If measured headroom is inadequate, reduce GPU layers/batch size in a separately recorded condition. Ordinary LFM3 previously used 18 GPU layers; QAD is a separate configuration. Do not assume the reported short-decode speed applies to CPU tracing or long contexts.

| Work | Additional generated-token cap | Additional diagnostic/scored-token cap |
|---|---:|---:|
| Measurement/compatibility gate | 10,000 | 50,000 |
| Conditional response repair | 200,000 | 100,000 |
| Fable adapter recovery | 80,000 | 25,000 |
| Speed diagnosis | 15,000 | 0 |
| Quantized deployment, if eligible | 100,000 | 50,000 |
| Evidence-sensitive clarification, if eligible | 120,000 | 50,000 |
| Declared retries/controls reserve | 50,000 | 25,000 |
| **Total next-program ceiling** | **575,000** | **300,000** |

Combined with the completed record, those ceilings would reach at most 724,296 generated tokens and 300,582 separately scored/diagnostic tokens, below the existing overall ceilings. Ordinary chat prompt usage still stays in each raw response; teacher-forced/activation-only work has its own reservations. A cap is not an instruction to spend it. Unused budgets are not silently reassigned, and diagnostics are bounded even if they are slow.

These are planning caps, not frozen executable arm counts. Before each stage, commit the exact dataset revisions/splits, prompt/token limits, methods, shortlist policy, sampling, CPU/GPU conditions, run counts, resource checks, analysis and stopping rules. Adjust the size of a confirmation before seeing outcomes if the maximum-token arithmetic does not fit. A precision or clarification stage that fails its gate does not consume its allocation by default.

No target training loops, gradient optimization, paid compute, unattended external editor API or subagents are part of this plan. No new model download is needed to start. Any later extra fine-tune used for generalization requires pinned provenance and a separately declared expansion.

## Execution order and decisions

1. **First work package:** freeze a Qwen same-forward tracing gate and a fresh task-competence screen. Separately freeze the Fable-parent repository-context baseline. No large patch search before these checks.
2. **First practical artifact:** inspect Fable delta spectra and compare approximate adapters at fixed byte budgets. This can proceed if the activation gate fails, and uses no target training.
3. **First serious repair:** reversible causal intervention, published component-editing baseline, preservation-aware candidate, then one frozen fresh confirmation. Replicate the procedure on the other architectures only after its sites are verified; do not transfer exact neuron indices.
4. **Bounded systems detour:** profile Fable's format anomaly and test a small hybrid-format family if the trace supports it. Stop after the fixed condition set; avoid an open-ended kernel project.
5. **Expand only with evidence:** deploy a validated patch across precisions; then attempt the harder clarification behavior. Compose repairs only after two independent useful patches actually exist.

All old confirmation data stays exposed development/history. A new patch that wins after examining a test needs another independent test, not a renamed analysis. Preserve every failed export, nuisance-variable correction, raw request and candidate; append interpretations to the journal and push each coherent result. Publication readiness requires relevant implemented baselines, independent task/source coverage, adequate uncertainty and a fresh novelty review. The motivating report itself distinguishes removing a behavioral obstacle from adding absent ability; our competence gate follows that distinction. [Original report](https://rikiyat.github.io/blog/weight-space-programming/).

**The best first scientific bet is the conditional repair. The best first practical bet is a compact, behavior-faithful Fable adapter. A credible next milestone is one of these working on new tasks with all costs and regressions measured.**
