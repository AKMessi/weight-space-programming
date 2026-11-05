# Five experiments for useful weight-space research

**Models:** Qwen3-0.6B, LiquidAI LFM3-1.7B, and AKMESSI/lfm2.5-230m-fable-5.  
**Designed:** 2026-11-05. **Execution update:** implementation and the first discovery/validation milestone have now been completed. The proposals below retain their original research targets; see [current progress](research/PROGRESS.md) and [actual findings](research/FINDINGS.md) for what has and has not been established.

## The goal in plain words

We want to discover how to repair a model in a way that is useful, measurable, and understandable. A good result could be a better version of your own model, a reliable way to deliver weight patches, or a benchmark that exposes what AI weight editing can and cannot do.

These are five different research questions. They share tools and model files, but each has its own main result, competing methods, and test. A successful demonstration alone does not make a paper: the result must also survive serious comparisons, fresh examples, and tests for hidden damage. The proposed contributions below are hypotheses, not claims of established novelty or promised publication.

**My recommendation:** begin with Experiment 1 for the quickest personal value. Experiment 3 has the strongest practical systems angle. Experiment 5 is the most direct test of the exciting claim that an AI can understand weights. Experiments 2 and 4 offer broader reliability and method-development opportunities.

| # | Experiment | The useful outcome | Distinct research contribution to aim for |
|---|---|---|---|
| 1 | Preserve skills while repairing response habits | Your model follows the requested response style while retaining useful skills | A method and benchmark for separating a repair from loss of specialization |
| 2 | Ask when information is missing | A local assistant stops inventing missing values and answers once evidence arrives | A weight repair that responds to actual evidence, rather than becoming reluctant to answer |
| 3 | Make repairs survive 4-bit deployment | A working repair stays useful in your fast local model | A measured way to allocate a small precision/storage budget to preserve behavioral repairs |
| 4 | Combine repairs without breaking them | Several improvements coexist in one checkpoint | A compatibility test and composition method for small behavioral patches |
| 5 | Blindly diagnose a model from its weights | We learn whether an AI really extracts useful evidence from weights | A controlled benchmark separating weight reading, behavioral probing, and superficial clues |

## Why these three models are a good research set

**Qwen3-0.6B** is the first reference model because its conventional transformer structure makes inspection easier. **Fable** is our personalized specialization case: its unmodified LFM2.5-230M parent gives us a meaningful before/after comparison. **LFM3-1.7B** is the practical deployment case, informed by your reported approximately 50 tokens/second with QAD 4-bit.

All five studies include all three as subjects. We adapt task difficulty using development data so that each model can already do the underlying task. A model with no target failure is a useful control; we do not deliberately choose broken decoding settings to create an improvement. We report all three outcomes separately.

Three checkpoints do not prove a result about every model or isolate architecture from model size and training. We transfer the editing procedure between models, not exact neuron numbers or weight directions.

The matched Fable parent is an additional control, not a fourth main research target. The editor is a capable external LLM working through the research harness; these small models are the objects being examined. We are not assuming that the 230M model can diagnose the 1.7B model.

---

## Experiment 1 — Preserve skills while repairing response habits

### The idea

Can we fix a model's habit of answering in the wrong way while keeping the skill that makes it useful?

Consider two requests about the same short program:

- “Return only the next line of code.”
- “Explain which line comes next and why.”

A useful model should switch correctly between them. Permanently making it terse is not enough. For your Fable model, the interesting outcome would be keeping its coding specialization while reducing unwanted traces or extra text when a strict answer is requested.

### Why this matters

Fine-tuning can produce a useful specialist with inconvenient habits. Users currently compensate through repeated prompting, output cleaning, or another training run. A small repair that preserves the specialization would offer a cheaper way to maintain personalized models.

The Fable card reports a shift toward coding-agent behavior and local evaluation differences. That motivates this hypothesis; it does not establish that a separable “bad habit” exists in its weights. [Fable model card](https://huggingface.co/AKMESSI/lfm2.5-230m-fable-5).

### How all three models participate

| Model | Main test | Behavior to protect |
|---|---|---|
| Qwen | Exact answers, JSON, and code-only responses versus explicitly requested explanations | Correct answers, helpful explanation, copying |
| Fable | Code continuation and strict tool/response contracts | Useful coding continuation and requested trace-style explanation |
| LFM3 | Structured output and tool arguments across unfamiliar schemas | Correct values, instruction following, ordinary explanations |

### What we do

1. Build matched examples where the content stays the same but the requested output changes. Use fresh schemas, code fixtures, and wording.
2. Score content correctness separately from response-format correctness. Valid JSON with the wrong value is a failure.
3. Find components that differ between correct and incorrect response behavior. Where verified activation access exists, first test reversible interventions.
4. Compare simple attenuation, a standard directional edit, and an edit selected to preserve correct coding/explanation examples. For Fable, also compare partial rollback toward the parent.
5. Ask the LLM editor for a bounded set of mechanistic proposals. Compare with fixed algorithms using the same feedback budget.
6. Freeze the chosen patch and test new instruction families and code fixtures, including cases that explicitly request detailed explanation.

**Proposed method worth testing:** choose the smallest edit that improves strict-response examples while changing the model's behavior as little as possible on matched examples where its existing skill is wanted. Useful examples actively constrain the patch. This is an experimental selection rule, not proof that we have found a clean “style neuron.”

### Comparisons that could defeat our idea

Use strong prompting, a practical grammar-constrained decoder for structured output, ordinary directional editing, matched random edits, and Fable parent interpolation. Adapt a published component-level editing method where supported, and clearly label architecture adaptations. Count prompt length, latency, and edit-search cost.

If prompting plus a grammar decoder achieves the same semantic accuracy and preservation with less effort, that is the useful recommendation. Grammar-constrained decoding must not be omitted merely because it makes the comparison harder.

### What would make the result compelling

The new edit should improve **correct content in the requested form**, preserve requested explanation and code ability, and work on unseen schemas and wording. Plot the full tradeoff between repair and retained skill, rather than selecting a flattering example or dose.

For Fable, show that the repaired model retains a meaningful part of any freshly verified coding advantage over its parent. For Qwen and LFM3, measure preservation relative to their own original checkpoints. Do not call the same underlying inability a style failure.

**Potential paper claim:** under a fixed editing budget, preservation examples produce a better repair/skill tradeoff than established alternatives in these small deployed models. The method must outperform relevant baselines; testing known editing on new model names alone is a weaker contribution.

**Practical deliverable:** a reversible repair for your Fable model and a benchmark of “same task, different requested response.”

**Useful negative result:** useful coding and unwanted response habits cannot be separated at our edit sizes, or a simple decoder is the better solution.

**Closest work:** task-vector arithmetic already changes model behavior through weight differences, and Steer2Edit already converts steering signals into component-level edits. Our candidate contribution is the measured preservation of specialization and response switching, not inventing weight subtraction. [Task Arithmetic](https://arxiv.org/abs/2212.04089), [Steer2Edit](https://arxiv.org/abs/2602.09870).

---

## Experiment 2 — Ask when information is missing

### The idea

Can a weight edit make a model notice that it lacks a required piece of information, ask for it, and then answer correctly when the information is supplied?

Example: a mock tool requires a person's name and an address. The user provides only the name. The improved model asks for the address instead of inventing one. Once the address is provided, it makes the correct mock tool call.

### Why this matters

An assistant that guesses missing arguments can be unreliable even when its output looks polished. A model that constantly says “I don't know” is also unhelpful. We want a model that changes its decision when the available evidence changes.

This study focuses on information contained in the prompt and conversation. We can know exactly what is missing, without trying to measure all the facts inside the model's memory.

### How all three models participate

- **Qwen:** short document questions and simple calculations with complete or incomplete information.
- **Fable:** coding tasks with missing function signatures, configuration values, or required tool parameters.
- **LFM3:** mock tool use and structured extraction with complete, missing, or conflicting fields.

Each model also sees a shared set of simple context-only tasks. Model-specific tasks are reported separately from that common panel.

### What we do

1. Create families of examples with three versions: enough evidence, one essential fact missing, and conflicting evidence with no stated resolution rule.
2. Randomize names, values, wording, and field position. Match lengths with harmless distractors so that “shorter prompt” does not predict missing information.
3. Make counterfactual pairs: add the missing fact or resolve the conflict, keeping everything else similar. The correct action should switch from clarification to answering.
4. Diagnose the difference between evidence-sensitive decisions and unsupported guesses. Test small edits selected using both incomplete and complete examples.
5. Evaluate an entire two-turn exchange: appropriate clarification, then correct completion once the answer becomes possible.
6. Test new document layouts, schema families, and phrases. Keep lexical markers such as “unknown” out of a diagnostic subset.

**Proposed method worth testing:** measure a direction associated with missing-evidence decisions, remove the portion associated with failing easy but answerable tasks, and test a small component edit based on the remaining signal. If those signals cannot be separated, record that. We do not assume that subtracting vectors creates genuine understanding.

### Comparisons that could defeat our idea

Compare explicit prompting, a separate answerability prompt, simple confidence thresholds, and ordinary unconditional steering/editing. Include a deterministic required-field checker on tasks where it applies. For the common task panel, add a simple retrieval or parsing rule where that can solve the task.

Count the extra calls and tokens needed by a separate checker. Compare error versus coverage: **among the questions the model chooses to answer, how often is it right, and how many answerable questions does it still answer?**

### What would make the result compelling

The model should invent fewer values at a matched rate of answering answerable questions. It should correctly resume answering when the missing fact appears, without being triggered by surface words or simply producing more abstentions.

Score missing-field identification, supported-answer accuracy, unsupported-answer rate, and complete two-turn success. A JSON wrapper or a generic clarification sentence cannot earn a semantic success by itself.

**Potential paper claim:** a small merged weight repair improves evidence-dependent decisions and clarification under held-out layouts, including the transition back to answering, without an extra inference call. A strong benchmark finding about failure to achieve this could also be valuable.

**Practical deliverable:** a more reliable local tool assistant or extraction model, plus a small exact-answer benchmark of missing information.

**Useful negative result:** edits change willingness to answer but do not improve sensitivity to evidence. That is a concrete limitation worth exposing.

**Closest work:** abstention is already an established topic; monitoring and activation interventions already address unanswerable questions. Recent work also shows abstention can follow prompt artifacts. Our test therefore requires counterfactual evidence changes and matched answer coverage. [AbstentionBench](https://arxiv.org/abs/2506.09038), [Abstention intervention](https://arxiv.org/abs/2508.18760), [Prompt-induced abstention](https://aclanthology.org/2026.emnlp-main.549/).

---

## Experiment 3 — Make repairs survive 4-bit deployment

### The idea

Can we deliver a weight repair that still works in the small, fast model file you actually run?

A repair can work in a floating-point model and disappear when its numbers are rounded into a 4-bit format. Keeping the whole model at higher precision would lose part of the reason for using a local model. We want to find how little extra precision is needed to keep the repair useful.

### Why this matters

This is the bridge between a research demo and an improvement someone can install. A good result could become a tool that takes a working behavioral repair and produces an appropriate GGUF deployment while reporting its accuracy, speed, and size.

### How all three models participate

| Model | Controlled comparison |
|---|---|
| Qwen | Same original checkpoint and repair, converted consistently into floating, Q8, and ordinary Q4 artifacts |
| Fable | Original F16 GGUF and repaired F16, with controlled Q8 and Q4 conversions |
| LFM3 | Ordinary BF16-derived quantizations as one lineage; the actual QAD-Q4_0 checkpoint as a separate deployment lineage |

The official QAD artifact was trained for quantization accuracy. Re-quantizing public BF16 weights does not recreate it. A result on ordinary Q4 is not automatically a result on QAD. [Official LFM3 GGUF documentation](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).

### What we do

1. Obtain at least one independently verified useful repair per model from any defensible method. Experiments 1 or 2 can supply it, but this study does not require their proposed method to win.
2. Freeze it before precision selection. Compare each deployed repair with its own same-lineage unedited baseline.
3. Test whether ordinary quantization preserves, weakens, erases, or reverses the behavioral improvement.
4. Compare simple alternatives: stronger edit dose, Q8 everywhere, higher precision for every touched tensor, and a budgeted subset of touched tensors at higher precision.
5. Select that subset on calibration examples using both reconstruction error and the measured effect on the repaired behavior. Treat behavior-based precision allocation as a candidate method, not an established novelty.
6. Freeze the deployment choice and evaluate new prompts, file size, memory, speed, and useful completion.

For QAD, preserve the original artifact and inspect/dequantize only needed tensors. An edited mixed-precision GGUF is a new deployment with a measured cost. Include an otherwise identical no-op rewrite to detect conversion drift. If its writer or parity check fails, stop the QAD modification arm rather than replacing it with BF16 evidence.

### Comparisons that could defeat our idea

Compare ordinary Q4 conversion, Q8, magnitude-only precision selection, random selection, and all-touched-tensors at higher precision. Give methods equal additional-byte budgets. Include the no-op control and matched runtime/template checks.

If simple Q8 or preserving all touched tensors gives the same outcome at negligible practical cost, an elaborate allocation method has little value. If numerical error predicts repair retention just as well as behavior-sensitive selection, prefer the simpler method.

### What would make the result compelling

Show a repeatable size/speed/repair tradeoff: the proposed allocation preserves useful behavioral gains with fewer extra bytes than competing choices. Measure behavior, not only closeness between matrices. The same precision policy should be selected without inspecting the final test.

Do not divide by a nearly zero floating-point gain to claim a huge “retention percentage.” Report absolute gains with uncertainty. QAD gains are measured against QAD's original behavior and are not normalized by a different BF16 checkpoint's gain.

**Potential paper claim:** behavioral calibration improves the cost of shipping sparse model repairs across these deployment formats, with a separately verified QAD case. The broad observation that compression can interfere with editing is already known.

**Practical deliverable:** a GGUF patch/export utility with a report showing which precision is needed, what changed, and how to restore the original artifact.

**Useful negative result:** repairs that look equally strong before conversion have systematically different deployment robustness, or the QAD lineage needs a different repair altogether.

**Closest work:** intervention/compression interactions and updates to quantized models already exist. Our possible contribution is the deployment budget and behavioral objective for small gradient-free repairs, including honest QAD handling. [Composable Interventions](https://arxiv.org/abs/2407.06483), [In-Cell Learning](https://arxiv.org/abs/2608.20873).

---

## Experiment 4 — Combine repairs without breaking them

### The idea

Suppose one patch makes JSON responses reliable and another makes the model ask for missing information. Can both improvements live in one model?

The combined model should return correct JSON when it has enough information, and a valid JSON clarification when a required field is missing. It should also preserve ordinary coding or explanation behavior.

### Why this matters

Users need models with several good behaviors at once. If each repair damages a previous one, maintaining a personalized model becomes difficult. A compatibility test could help decide whether to merge patches, weaken one, or keep them separate.

### How all three models participate

For each model independently, obtain two modest but verified patches that improve different behaviors. Use strict format and evidence-sensitive clarification where both have a real baseline problem. A loop/copying repair can supply a third behavior if genuinely needed. The same two behaviors across models provide the cleanest comparison; otherwise report different case studies explicitly.

These primitive patches can come from established algorithms. This experiment is about their interaction, not whether our earlier new methods succeed. If a model has fewer than two independently useful patches, that model cannot support a successful composition claim.

### What we do

1. Evaluate the original model, patch A alone, patch B alone, and their simple sum on the same mixed task panel.
2. Measure joint task success. Separate improvements on two disjoint task lists are not enough: include prompts requiring both behaviors together.
3. Build a small compatibility table from calibration examples: what does each patch do to its own target, the other's target, and preserved skills?
4. Compare whether that behavioral table predicts conflicts better than weight overlap or similarity between the patch vectors.
5. Test a bounded composition method: choose patch strengths jointly; optionally remove edit components that calibration identifies as damaging the other behavior. Use closed-form projections and a small declared search, without gradient training.
6. Freeze the composition and test unseen templates. A stronger study reserves previously untested patch pairs or sites as well as unseen prompts.

**Important mathematical control:** adding two fixed deltas commutes. Applying the exact same saved A and B in opposite orders should produce the same floating-point weights within declared numerical tolerance. Ordering effects require state-dependent recalculation, rounding, or another operation. We will not present ordinary addition as mysteriously order-sensitive.

### Comparisons that could defeat our idea

Compare naive addition, weaker scalar combinations, sign/overlap-based merging, an adapted TIES-style baseline, and a small direct joint search with the same evaluation budget. Where reference hooks exist, conditional activation steering is a useful runtime comparator; record its inference overhead and any trained components separately.

Include no-op combinations and compatible control pairs. Do not create impossible requirements such as demanding both prose and no prose on the exact same input. Every task must have a logically compatible expected output.

### What would make the result compelling

The combined checkpoint should retain a substantial part of both individual gains and perform better on prompts requiring both behaviors. A compatibility score becomes interesting if it predicts problems on held-out pairs/sites, not just the calibration examples used to build it.

Start with one pair per model. A paper-level extension needs a larger collection of patch pairs, including successful and unsuccessful combinations. Different doses of one patch are not independent discoveries, and repeated prompts are not independent patch pairs.

**Potential paper claim:** a small calibration-based compatibility test improves selection/composition of behavioral weight patches, including useful joint-task performance after deployment. The general idea that model edits can interfere is already established.

**Practical deliverable:** a patch compatibility report and a way to assemble several improvements into one local checkpoint.

**Useful negative result:** patches look independent in weight space but strongly interfere in behavior, or composition works before Q4 conversion and fails afterward.

**Closest work:** model merging and composing steering directions already address related problems. Our candidate contribution must be specific to merged behavioral patches, their prediction/validation, and deployment cost. [TIES-Merging](https://arxiv.org/abs/2306.01708), [Steer2Adapt](https://arxiv.org/abs/2602.07276), [Contextual composition](https://arxiv.org/abs/2610.24906).

---

## Experiment 5 — Blindly diagnose a model from its weights

### The idea

Can an AI identify a model's specific problem from its weights, without seeing its answers or being told which checkpoint it is?

Then make the test harder: rearrange equivalent internal neurons while keeping the model's behavior unchanged. Does the diagnosis remain correct?

### Why this matters

The striking weight-space claim is that an AI can inspect numerical parameters and infer useful behavior. A controlled blind test can separate that ability from recognizing a filename, noticing obvious damage, or learning the answer through repeated behavioral tests.

Even a negative result can improve how the field evaluates these claims. A reliable benchmark would be useful beyond this project.

### How all three models participate

Create anonymized cases from each model. Each includes its architecture information and the tools needed to inspect weights, but no healthy reference checkpoint, public repository name, original tensor hashes, or network search. Test within each architecture so that recognizing “this is Qwen” cannot solve the task.

Cases include healthy controls and controlled faulty variants. All remain language models of the same shape. We must verify what the fault actually does before using it in the benchmark.

### What we do

1. Prepare a hidden library of bounded interventions: for example, attenuate or perturb small feed-forward components. Use several independent sites and seeds.
2. An evaluator checks whether each variant has a clear behavioral phenotype, such as degraded copying, premature completion, or response-contract failure. Retain only cases meeting criteria set before the editor sees them; disclose the screening process.
3. Include healthy cases and perturbations with similar numerical size but no selected behavioral fault. A change's norm alone should not reveal its label.
4. Give the editor a fresh anonymous session per case. In the **weights-only arm**, it can compute statistics and inspect matrices but cannot run any model forward pass, including random-input layer tests.
5. In a separate **behavior-only arm**, provide a fixed small number of output probes and no weights. A later **weights-plus-activations arm** measures what additional internal evidence buys us. Keep input/evaluation budgets explicit for each arm.
6. Ask for a diagnosis, a predicted measurable effect, and an optional bounded repair. Freeze the answer before evaluating on hidden behavior tests.
7. Repeat on function-preserving neuron permutations and compare both diagnosis and repair quality.

### The crucial rearrangement control

In a gated MLP, permute matching rows in the two input projections and matching columns in the output projection. With the correct mapping, the network computes the same function in exact arithmetic. Verify numerical equivalence in our actual floating implementation before blinding.

Run this symmetry control in floating precision first. Re-quantizing after a permutation can change block rounding, so quantized behavior cannot simply be assumed identical. Arbitrary rotations across nonlinear layers are not valid replacements for this control.

### Comparisons that could defeat our idea

Compare a statistics-only diagnosis, a deterministic anomaly detector, the behavior-probe arm, and matched random repairs. All editor runs use the same editor version and predeclared tool budget. Withhold the known original weights; otherwise a simple diff can reveal injected changes without behavioral understanding.

For the pilot, use one healthy case and two independently generated fault cases per model. Two equivalent layouts give 18 case presentations; two evidence arms give 36 editor sessions. This is a feasibility test, not enough evidence for a sweeping claim. A strong benchmark needs many more independently generated cases, sites, healthy controls, and repeated editor runs.

### What would make the result compelling

Measure correct diagnoses, false alarms on healthy cases, localization, confidence quality, and hidden behavioral improvement after repair. The known injected site is the location of our intervention, not automatically the unique natural circuit for that behavior.

A strong positive result would diagnose and repair new faults beyond numerical baselines, remain reliable under equivalent neuron layouts, and generalize to held-out sites. A failure under permutation would show that the editor is sensitive to representation choices even though model behavior did not change.

**Potential paper claim:** a controlled benchmark identifies when weight inspection adds value over behavioral probes and how robust that value is to harmless reparameterization. Results on injected faults do not establish diagnosis of naturally occurring failures; add natural cases as a separate external-validity test.

**Practical deliverable:** an auditable weight-diagnosis benchmark and harness for comparing editors.

**Useful negative result:** weights alone do not reliably identify these behaviors, or a few output probes outperform expensive weight inspection. That meaningfully narrows the original claim.

**Closest work:** the motivating report already studies weight reading; weight-space symmetry is also established. Our contribution would be the blinded language-model diagnosis/repair benchmark and its controls. [Weight-space report](https://rikiyat.github.io/blog/weight-space-programming/), [Permutation Equivariant Neural Functionals](https://arxiv.org/abs/2302.14040).

---

## Shared rules that make these credible research

### Start with abilities and failures we can measure

Use development data to check that a model can solve the underlying task when the troublesome behavior is absent. Do not interpret a 230M model's missing knowledge as a removable habit. Record models with no target failure as controls and state when an experiment's feasibility condition fails.

Use exact graders for values, fields, copying, code fixtures, and complete mock tool arguments. For code, prefer short fixed fixtures with known tests. No generated shell commands or real tool actions are executed as part of scoring. A blinded human audit checks ambiguous outputs and detector errors; an LLM judge is supplementary.

### Hold out the right things

For Experiments 1 and 2, begin with 64 discovery and 64 validation examples per model, and a 60-example unseen pilot. Pilot categories and seeds are frozen before test feedback. The stronger extension uses at least 240 fresh test cases per model and three decoding seeds, with additional data if a tight preservation claim needs it.

Split by instruction/schema family, document layout, and code fixture lineage, not only random rows. If a common panel is reused, record that reuse. Experiments 3 and 4 need their own calibration/test partitions; exposing a patch's earlier test examples to a later allocator makes those examples calibration data for that later experiment. Experiment 5 holds out fault sites and fault instances as well as prompts.

Release all attempted edits and exclusions. The successful patch must be frozen before final evaluation. Report category-level results, rescues, new failures, and uncertainty. Keep a prompt's multiple seeds together in statistical analysis. Include exact upper bounds for rare regressions; zero observed failures in a tiny pilot does not establish zero risk.

### Define success before seeing the result

For a repair pilot, a useful opening target is a 30% relative reduction in the specified failure with better useful completion. A preservation loss of more than three percentage points is a provisional concern, but the confidence interval matters. These are design targets, not promises or universal publication thresholds.

For a strong claim, the new method must beat the relevant simple and published baselines at matched task utility, compute, or storage. A new model name and attractive examples are insufficient. If it only matches a simpler method, the release may still be a useful engineering artifact; the method novelty claim should be reduced.

### Treat editing cost as part of the result

Count every candidate, feedback call, sequence, generated token, scored token, wall-clock time, memory peak, and failed export. Comparing an LLM's long search to a heuristic's single attempt is not a fair editor comparison. No target-model gradient training is planned. A method estimated from examples is still data-dependent, even when it uses no gradients.

Input/output token use by the external editor is separate from local target-model tokens. In particular, Experiment 5's editor sessions have a real time/usage cost. Its first pilot should cap each session's inspections and context; no paid unattended API loop is assumed.

## Keeping this practical on our computer

We have the inspected Ryzen 3 3200G, approximately 16 GB RAM, and RX 6400 with approximately 4 GB VRAM. Reuse your working LFM3 setup. Your reported 50 tok/s is an observation for that quantized configuration, not a speed guarantee for Qwen, Fable, or hooked floating-point analysis.

Keep one model resident, use memory-mapped weights and short diagnostic windows, and start with roughly 2K context. LFM3's FP32 weights alone take about 6.9 GB, so avoid duplicate full-precision copies and unrestricted activation storage. A first implementation gate verifies reference loading and actual memory peaks.

The Fable release's F16 GGUF supports direct artifact inspection, but reference activation analysis needs a matching checkpoint or a verified mapping into its model implementation. LFM3 includes hybrid layers and normalization around feed-forward outputs, so Qwen's intervention semantics cannot simply be assumed. If mapping/parity is unresolved, label the direct artifact experiment honestly and defer the unsupported mechanistic arm.

Cache immutable originals, save deltas/manifests, and verify exact rollback. Quantized files have block layouts and metadata: editing an array and exporting it are separate operations that require a no-op control. These artifact constraints carry over from [the existing hardware and implementation plan](RESEARCH_PLAN.md).

## A staged budget for all five pilots

This expands the previous single-study plan into a five-study portfolio. It does not authorize five copies of the old budget. The existing 1.8-million generated-token ceiling can be provisionally divided as follows; these are caps and must be translated into explicit arm counts before execution.

| Work | Initial local generated-token cap |
|---|---:|
| Shared compatibility and baseline characterization | 200,000 |
| Experiment 1 pilot across all three models | 350,000 |
| Experiment 2 pilot across all three models | 350,000 |
| Experiment 3 pilot across all three models | 300,000 |
| Experiment 4 pilot across all three models | 250,000 |
| Experiment 5 local evaluation pilot | 200,000 |
| Reserved retries and controls | 150,000 |
| **Total pilot ceiling** | **1,800,000** |

Short Experiment 1/2 pilot outputs start at 128 new tokens where the task permits. Six methods × 60 test examples × 128 tokens × three models uses at most 138,240 generated tokens per experiment, leaving room for bounded candidate screens within those caps. Longer tasks need smaller predeclared pilot panels or a separately declared extension; never silently truncate a reasoning task and label it a loop.

The initial teacher-forced diagnostic ceiling remains one million scored tokens, logged separately. Reused local outputs can reduce spending when scientifically valid. Larger confirmation studies and Experiment 5's external-editor input/output usage need separately declared allocations. This budget can screen all five ideas; it is not a claim that five publication-ready studies fit inside it.

At 50 tok/s, 1.8 million tokens would take ten hours of pure decode if every run achieved that speed. Actual elapsed time includes slower reference runs, prompt processing, analysis, exports, and editor sessions. Plan for multiple work sessions and measure before estimating a completion date. No GPU purchase or cloud rental is needed to begin.

## Recommended execution order and stopping decisions

1. **Shared baseline and artifact checks.** Capture the working runtime and establish trustworthy evaluation. No weight editing before these pass.
2. **Experiment 1 first.** It offers the clearest path to a useful personalized Fable improvement and exercises the common editing tools.
3. **Experiment 2 next.** It supplies a separate reliability question and, if successful, another useful patch.
4. **Experiment 3 once any useful patch exists.** Test deployment before investing in a polished demo.
5. **Experiment 4 once two useful patches exist.** Composition is a distinct question, but it needs real primitive improvements. No such pair means the gate has not passed.
6. **Experiment 5 can proceed independently after artifact access is reliable.** Start with the small blind pilot before funding a large editor benchmark.

Do not force every proposal into a positive result. If the first pilot shows that simple prompting wins, preserve that finding and move the research effort to a stronger question. If an experiment depends on a repair that cannot be established, report the failed prerequisite instead of filling the table with meaningless variants.

## What we would ultimately own

- Useful, reversible patches for models we actually run, when the evidence supports them.
- A benchmark of response switching and missing evidence, with exact graders and fresh held-out families.
- A measured GGUF deployment path for edited models, including explicit QAD provenance.
- A compatibility test for combining improvements.
- A blinded benchmark for evaluating claims about reading weights.

Each study can support its own report if it produces a substantial, well-controlled finding. They share infrastructure and may share primitive patches, so they are not five guaranteed independent papers. Keep separate hypotheses, splits, budgets, and primary outcomes; combine closely related results if that makes the scientific story stronger.

**Recommended first bet:** Experiment 1 for a model improvement you can use. **Strongest deployment bet:** Experiment 3. **Most fundamental weight-space question:** Experiment 5.

The initial design preceded implementation. Following authorization, the baseline/artifact checks and first gated experiments were executed; [the progress record](research/PROGRESS.md) tracks remaining work. No publication or successful repair is promised by this design.
