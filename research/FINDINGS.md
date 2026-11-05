# First experimental findings

Updated 2026-11-05. **Implementation and the first experimental milestone are complete; the five-study portfolio is ongoing. No generally useful or skill-preserving repair has been established.**

We ran **30 local inference runs, 1,238 requests, and 41,908 generated tokens** on the inspected machine. Every request has raw evidence and settled token accounting. Nineteen tests pass. These are small synthetic experiments, not general model rankings or publication-ready results. The [integrity audit](evidence-integrity-latest.json) checks consistency of the records, not the truth of a scientific conclusion.

## 1. A large apparent weight change can be a precision artifact

**In simple words:** two files can both say “F16,” yet one contains numbers rounded more coarsely than the other. Subtracting the files then mixes actual model changes with rounding differences. Editing that difference may change precision instead of repairing a learned behavior.

Fable's embedding and FFN output values lie on the BF16 value grid despite their F16 storage. The public F16 parent's values mostly do not. A first partial FFN rollback therefore was not a clean removal of a learned FFN update.

We fetched the separately published BF16 parent, decoded its values exactly, and explicitly aligned the remaining floating tensors to the same effective precision. After that alignment:

- **105 of 132 tensors match Fable exactly.**
- **18 Q/K/V attention projection matrices carry substantial differences.**
- The remaining 9 tensors differ at only 12 scalar positions in total.

This sharply narrows where substantial differences between these particular artifacts reside. It does **not** prove the exact historical training revision, identify a unique behavior circuit, or establish that every residual difference was learned. Fable's card does not pin its historical parent revision.

**Value:** a reusable precision/provenance audit before weight subtraction, interpolation, or claims about where fine-tuning changed a model. This is a promising methods note or replication lead; novelty and broader relevance still need independent examples and a literature comparison.

Evidence: [initial audit](fable-parent-tensor-audit-v2.json), [official BF16 comparison](fable-vs-official-bf16-parent.json), [exact decoder and alignment checks](bf16-codec-and-alignment-check.json), [patch manifests](patches/).

## 2. A visually exciting repair failed fresh examples

**In simple words:** changing a few attention matrices made Fable copy the first four examples correctly. That looked promising. New kinds of text exposed that it was not a general copying repair.

We restored selected Fable projections toward the pinned public parent's values. These are rollback baselines, not a new editing algorithm. Restoring V projections across six attention layers recovered 4/4 original literal-copy examples, versus Fable's 0/4. However, it also regressed a simple explanation proxy from 4/4 to 0/4. Restoring all Q/K/V projections recovered more original response contracts but could erase specialization.

Before further selection, we committed a fresh 48-example validation panel, fixed candidates and seeds, and then ran all ten declared arms. The literal-copy primary result was:

| Arm | New exact copies | Interpretation |
|---|---:|---|
| Original Fable | 0/24 | 12 prompts, two seeds |
| V-only restoration | 0/24 | No exact-copy generalization |
| All-QKV restoration | 0/24 | Broad rollback also insufficient |
| Precision-aligned parent | 0/24 | These new forms also challenge this parent |
| Qwen3-0.6B Q8 reference | 12/12 | One seed; different model/settings |
| LFM3-1.7B QAD Q4 reference | 7/12 | One seed; different model/settings |

The seeds are repeated samples of the same prompts, not 24 independent prompt families. Some Fable failures were truncated outputs. A 128-token cap is part of this measured deployment contract; a cutoff does not prove an infinite loop or inability under a larger cap.

**Value:** we rejected a candidate that a cherry-picked demo would have celebrated. Follow-up work needs more varied development instructions and payload forms, task-competence controls, and a new independent confirmation panel. This exposed validation set must not become a source of new patches later presented as independently validated.

Evidence: [frozen protocol](protocols/validation-v1.md), [validation report](VALIDATION_REPORT.md), [paired analysis](validation-v1-analysis.json), [all raw runs](../runs/).

## 3. Structured decoding is already a strong practical baseline

**In simple words:** a model may know the answer and still surround it with unwanted formatting. For JSON, constraining the output format can solve much of that problem without editing weights.

On twelve development JSON/tool examples, a stronger prompt plus a schema gave Qwen 12/12 and Fable 10/12 correct exact responses. The same stronger prompt without the grammar gave 2/12 and 0/12. Schemas specified keys and types already provided in the task; they did not contain the correct values.

On fresh validation JSON-field tasks, original Fable and V-only each included the correct object on all 24 requests but satisfied the plain-response contract on none. These results distinguish content competence from response formatting. The discovery grammar scores have not been independently confirmed on that validation panel.

**Value:** any proposed weight repair should beat or offer a useful tradeoff against prompting and constrained decoding. We should not present an inferior, more expensive weight change as progress merely because it changes weights. Grammar use itself is established practice, not our novel contribution.

Evidence: [development baseline report](BASELINE_REPORT.md), [non-weight fixtures](../datasets/nonweight-controls-v1.json), runs [008](../runs/20261105-008-fable-nonweight-controls/), [009](../runs/20261105-009-qwen-nonweight-controls/), and [010](../runs/20261105-010-lfm3-qad-nonweight-controls/).

## 4. LFM3's fast deployment has checkpoint and template subtleties

The first 40 QAD discovery requests had median server-reported decode throughput **77.46 tokens/s**, ranging from 61.01 to 80.82. This is specific to short prompts, this runtime, and this GPU; it is not a general performance promise.

Some responses include thinking markers or duplicated content despite disabled thinking. We verified the actual rendered prompt and separated a parser-exposed prefilled prefix from tokens newly generated by the model. Switching parsing settings alone did not resolve every observed issue.

Ordinary LFM3 Q8 and QAD Q4 differ on these tiny tasks in both directions. **QAD is a separately trained checkpoint**, so their differences cannot be attributed solely to 4-bit rounding. A controlled quantization study needs a common source checkpoint and a matched quantization path.

Evidence: [journal diagnostics](JOURNAL.md), template-probe runs [002](../runs/20261105-002-lfm3-qad-template-probe/) and [003](../runs/20261105-003-lfm3-q8-template-probe/), [runtime inventory](runtime_inventory.json).

## What we built

- A one-model-at-a-time Vulkan inference runner with exact artifact identities, source/config/data hashes, actual rendered prompts, full request/response evidence, timing, errors, and token accounting.
- Separate-file floating GGUF edits with identity controls, exact BF16 decoding, per-tensor change accounting, untouched-tensor verification, and original-file checksum rechecks.
- Prompt/schema controls, Q/K/V family ablations, frozen validation, paired prompt-level analysis, and explicit negative-result retention.
- Versioned graders and derived corrections. Grader v3 corrects code equivalence/presence errors; the original frozen analysis remains intact. These code checks are static proxies, not functional execution or proof of coding specialization.

Earlier run manifests record the runtime executable hash/version and server logs. Future manifests additionally fingerprint colocated shared libraries; that later addition cannot retroactively establish historical library bytes. Reproducibility is bounded by the recorded provenance.

## The next research decision

First establish what Fable does **better than its parent** on appropriate, varied coding tasks, using meaningful functional checks and a declared extraction policy. Then search for repairs that retain that advantage. Tiny return-expression fixtures cannot demonstrate preservation of coding specialization.

Experiment 2 needs stronger development controls separating missing-information recognition from JSON formatting and a complete two-turn evaluation. Experiment 3 should begin once a useful patch exists; Experiment 4 needs two independently useful patches. Neither prerequisite has passed. Experiment 5's blinded benchmark remains unimplemented and can proceed independently after its case-screening and editor-budget protocol is frozen.

The next potentially publishable result is a hypothesis, not a promised outcome: precision-aware weight comparison, a genuinely generalizing response repair with preserved utility, or a rigorous negative result about weight diagnosis. The current release supplies auditable starting evidence and tools for testing those claims.
