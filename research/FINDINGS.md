# Findings from the five local pilots

All five tracks have relevant executed measurements on Qwen3-0.6B, LFM3-1.7B and Fable 230M. The bounded pilots close with **no accepted general behavioral repair**. The strongest practical results are exact checkpoint delivery, a measured deployment-speed surprise, and controls showing why attractive weight-editing demos need careful validation. These results support a reproducible technical report and tools; they do not establish a new paper-ready editing method.

See the [five detailed reports](portfolio-v3/README.md), [machine-readable analysis](portfolio-v3/results.json), [deployment/symmetry follow-ups](portfolio-v3/supplement/README.md), [journal](JOURNAL.md), and [reproduction guide](../docs/REPRODUCING.md). The [first-milestone findings](FINDINGS_FIRST_MILESTONE.md) retain the earlier 30-run history.

## 1. Small edits changed responses, but did not produce a reliable repair

**In simple words:** making a model answer differently is easy. Showing that it follows instructions better while keeping its useful skills is harder.

For each deployment we tested fixed last-layer FFN-normalization attenuation, attention-normalization attenuation, and a seeded perturbation with matched numerical size. Fable also had parent-rollback controls. Selection used development examples before fresh confirmation; no candidate qualified. A retained negative diagnostic was confirmed on new functions, schemas and copy forms at two seeds. None met the declared repair/preservation gate.

The new code checks actually test simple Python functions on alternate inputs through a bounded interpreter. They distinguish correct content from following the requested format. They do not establish Fable's repository-level coding specialization. The Fable parent control was measured rather than assumed inferior. Code extraction/signature grading was corrected as disclosed post-hoc sensitivity analysis; original grades and selections remain intact.

Grammar-constrained JSON is a useful practical baseline: on the matched confirmation control, all three models produced the correct exact object on 8/8 authored extraction cases. Strong prompting alone gave Qwen and Fable 0/8 strict objects despite 8/8 correct payloads; LFM3 already passed 8/8. This is a small synthetic panel and one fixed prompt policy, not proof that grammar always wins or that better prompting is impossible.

Exact-copy contracts remain hard on the new BEGIN/END forms, even when the payload itself appears correctly. The stronger system prompt explicitly excludes markers, but the models still often include them. An aggregate zero strict score is a floor on this contract, not proof of no copying ability. Previous V-only Fable rollback recovered 4/4 old development examples and 0/24 fresh validation copies; it is not a general repair.

[Study 1](portfolio-v3/experiment-1.md) contains all confirmation arms, paired gains/losses, control results and limitations.

## 2. Valid JSON did not mean the model knew when to ask

**In simple words:** a model can produce a perfectly shaped tool call while inventing a value that was never supplied.

We tested complete, absent and conflicting evidence, then supplied an authoritative update in a real second turn containing the model's actual previous reply. Twelve conversations cover four authored field families; they are correlated synthetic tasks.

LFM3's original achieved 4/12 correct whole conversations. Fable's schema control improved the complete/update formatting, also reaching 4/12 whole conversations, but answered all eight absent/conflicting first turns with unsupported non-null values. LFM3's schema arm did the same. Correct final answers after evidence arrives cannot erase the first-turn guess. Qwen struggled with the requested action protocol and update transition on this panel; this is not a general model ranking.

Neither fixed weight candidate improved the declared joint outcome. A deterministic parser/checker solved 12/12 conversations using only the supplied explicit records and authoritative updates. For a tool with known required fields, checking those fields is a useful deployment baseline. This parser is task-specific and does not solve general ambiguity.

[Study 2](portfolio-v3/experiment-2.md) separates answer coverage, unsupported action, invented value, and first/second-turn correctness.

## 3. We can ship an exact checkpoint change cheaply when the base matches

**In simple words:** instead of sending another whole model, send the exact differences and rebuild the same file locally.

The utility streams named-tensor XOR/deflate patches, handles metadata/layout changes and tensor reordering, preserves originals, rejects a wrong base and verifies the **entire restored target hash**. It uses established delta encoding, not a new compression algorithm.

The same 461.9 MB original Fable GGUF was reconstructed exactly from three bases:

| Matching base | Patch size | What this comparison means |
|---|---:|---|
| Published F16 parent | 167.7 MB | Unaligned baseline |
| That same F16 parent, deterministically BF16-rounded | 44.7 MB | Within-source precision control; 3.75× smaller |
| Separately published aligned BF16-value parent in F16 layout | 20.3 MB | Source-choice result; 8.26× smaller than unaligned |

The base is required and excluded from patch size. This is useful when it is already cached or shared. The extra aligned-source saving cannot be assigned solely to rounding; conversion provenance and double rounding remain possible factors. Tiny diagnostic Qwen and LFM3 QAD normalization changes also restore exactly from approximately 19 KB patches, but those diagnostics have no confirmed general behavioral utility.

This follows the earlier precision finding: two F16 files can encode different effective value precision. After explicit alignment, 105/132 Fable/parent tensors match exactly; substantial residuals lie in 18 Q/K/V projections, with twelve scalar residuals in nine other tensors. The exact historical training parent remains unverified. This is numerical evidence about these artifacts, not a unique behavior circuit.

We also executed same-source floating/Q8/Q4 comparisons for each model and verified a QAD no-op rewrite. The edited normalization vectors stayed F32 under quantization. Therefore those stress tests cannot show a sparse floating repair being rounded away. No useful-repair retention ratio or precision allocator is claimed because the prerequisite useful repair failed.

[Study 3](portfolio-v3/experiment-3.md) and [the codec guide](EXACT_PATCH_DELIVERY.md) provide the data and runnable commands.

## 4. Combining arbitrary changes is not the same as combining improvements

The two fixed changes touch disjoint tensors. AB and BA were byte-identical in all three models and produced identical content/token counts on their six matched order probes. This validates the implementation of this composition, not compatibility of two useful repairs: those independent repairs were never established.

The initial engineering comparison mixed 128- and 192-token output caps. The audit caught this, preserved the runs, declared the correction and repeated every affected AB/BA panel at 192 before final interpretation. Joint-evidence arms already matched at 128. The primary report uses the corrected engineering runs. Aggregate equality can hide per-case changes and floor effects. The matched audit found three Fable payload interactions despite a strict-score floor: one A-only functional success is lost when B is combined, one JSON payload appears only under AB, and one positive interaction merely reflects two edits losing the same original success. A positive interaction is not automatically an improvement. [Per-case values](portfolio-v3/comparison-audit.md) are retained.

[Study 4](portfolio-v3/experiment-4.md) reports the matched comparisons and the failed useful-repair prerequisite.

## 5. The local editor did not demonstrate useful weight diagnosis

**In simple words:** giving a small AI summaries of weights did not make it a reliable mechanic for these models.

We created healthy states and two declared injected fault types for each architecture, then screened their behavior before editor calls. A local Qwen3-0.6B editor received anonymous bounded weight summaries/raw samples or two fixed behavioral outputs. It never received labels, screening scores, filenames, target hashes or a healthy reference. This tests a limited local editor, not the frontier editor or full interactive weight access in the motivating blog.

On twelve retained presentations, weights-only diagnosis scored **7/12**, exactly the majority-label baseline, and flagged all **5/5 healthy presentations** as degraded. Behavior-only diagnosis scored 8/12. A simple output-normalization sign detector scored 8/12 and an exact checker on the same two exposed behavior probes scored 12/12. The small correlated sample supports no general claim that an LLM can or cannot understand weights.

All nine MLP permutation checks passed their FP32 numerical tolerance, but only three full-runtime layouts had identical greedy text on all twelve probes. Coarse task scores were equal in every original/permuted state, yet six layouts failed the stricter predeclared parity gate and were excluded before editor inference. Equal scores are not proof of full functional equivalence; mathematical symmetry is not automatically bitwise generation parity.

The initial editor context was too short for one packet. The rejected request generated zero tokens; after a tokenizer preflight the entire unchanged batch was rerun at 4096 context. The partial attempt and full retry are separate. No long case was dropped to improve the result.

[Study 5](portfolio-v3/experiment-5.md) includes confidence, false alarms, baselines, screening and exclusions.

## The deployment-speed surprise

The matched timing follow-up used the same prompt and 64 actual generated tokens per call, with one warmup and three repetitions per format:

| Fable format | Median decode tokens/s |
|---|---:|
| F16 | 194.60 |
| Q8_0 | 13.40 |
| Q4_0 | 22.07 |
| Q4_K_M | 21.52 |

F16 is approximately 14.5× faster than Q8 on this workload while taking more storage/VRAM. The Q8 and Q4_K_M conversions match their pinned published hashes. The verbose Q8 trace reports all 15/15 layers offloaded; it does not establish which kernel/path causes the slowdown. Three repeats in one session on one device are descriptive, not a universal ranking or an isolated kernel benchmark. There is no quality score on these timing fixtures.

Finally, native raw-final-state readouts retained the next-token top-1 in all six calibration contexts but failed the predeclared centered-logit agreement gate. No directional edit was derived from an unverified readout. This is a boundary/backend compatibility failure, not proof that activation methods are impossible. [The supplement](portfolio-v3/supplement/README.md) preserves every discrepancy.

## What is complete, and what remains a research hypothesis

The bounded local portfolio has implemented tools, executed all five pilot tracks, tested controls, closed failed gates, and produced source-linked reports. No target training, paid compute or external editor API was used. Model originals were preserved; compact evidence is committed while binaries remain ignored.

The original paper-level goals remain hypotheses: a preservation-constrained behavioral editing method, evidence-sensitive directional repair, behavior-guided precision allocation, a library of independently useful composable patches, and frontier interactive weight diagnosis. They have not been silently relabeled as successes. A broader independently sampled task set, stronger methods/editor and proper novelty assessment would be needed before a paper-level claim. The immediate usable outputs are the exact patch/rebuild utilities, the precision audit, the small benchmark and the measured device-specific deployment result.
