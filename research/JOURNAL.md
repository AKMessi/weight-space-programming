# Append-only research journal

Record observations, hypotheses, decisions, failed attempts, confounds, and follow-ups. Raw per-run evidence belongs in `runs/`; this journal links it into a scientific narrative. All times are UTC where supplied. Prior entries are preserved, with corrections appended.

## 2026-11-05 — Implementation authorization and bootstrap

**Authorization:** the user requested implementation, detailed documentation of every noteworthy result, and commit/push on every coherent change to `AKMessi/weight-space-programming`.

**Observation:** the workspace held planning artifacts but was not yet a Git repository. The supplied remote was accessible over SSH and advertised no refs. A `main` repository was initialized without changing existing planning files.

**Observation:** the inspected system has approximately 9.5 GiB available RAM. The previously inspected llama.cpp CPU runtime identifies as build 10858 / commit `64e9bceb2`; this is not evidence that it is the user's working LFM3 runtime.

**Decision:** start with artifact/runtime inventory, exact graders, and raw inference evidence. Do not assume baseline failures, replace the user's installation, or begin by making uncontrolled weight edits.

**Result status:** no model experiment has been run at this entry.

## 2026-11-05 — Measurement pipeline, runtime discovery, and grader checks

**Observation:** `/home/akmessi/llama-vulkan-build/bin/llama-server` reports build 11942 / `338fce651` and recognizes Vulkan0 as the physical RX 6400. The user's LFM3 QAD and Q8 files exist under `/home/akmessi/models/lfm3`. The first read-only search found no Fable or Qwen model files in the checked locations.

**Implementation:** added a dependency-free local inference runner, immutable input manifests, raw request/response evidence, server lifecycle/logging, exact contract graders, conservative token reservations, pinned artifact downloads, and one-model execution locking. Qwen non-thinking is explicit. All generated fixtures are labeled discovery, and explanation scores are explicitly limited proxies.

**Validation:** eight tests passed, including duplicate JSON keys/non-JSON numbers, bool-versus-integer equality, nested/second-answer cherry picking, whitespace-sensitive copying, missing-field identification, deterministic fixtures, and failed/pending request budget accounting.

**Decision:** first compare the three deployments and the matched Fable F16 parent on 40 synthetic discovery cases, preserving full outputs. Qwen's official Q8 artifact is a quantized baseline, not the floating reference. Fable/parent share sampling. This screen will locate measurement failures and hypotheses, not establish a new method or benchmark score.

**Pending:** verify/download artifact bytes, run runtime compatibility checks, and collect actual responses. No inference result is implied by passing grader tests.

## 2026-11-05 — First real LFM3 QAD discovery screen

**Run:** `runs/20261105-001-lfm3-qad-discovery/`, source commit `5741b71`, seed 17, 40 investigator-visible synthetic discovery cases, maximum 128 output tokens, QAD checksum matched the official pinned artifact. Full offload and native server template/settings are recorded in the run. Generated tokens: 1,209.

**Observed:** all 8 simple JSON format cases, all 4 complete tool cases, and all 4 complete-evidence cases passed their exact contracts. Requested repeats passed 1/4; raw responses in the other 3 repeated the correct requested text around a literal `</think>` marker. Missing/conflicting evidence contracts failed, sometimes with intermediate recognition followed by a contradictory or malformed final response. These are exploratory observations, not estimates of general reliability.

**Confound to investigate:** literal thinking markers appear despite the configured non-thinking request. Inspect native template rendering and runtime parsing before attributing this to model weights. The model can recognize missing information in some intermediate text; this differs from a claim that it cannot detect it at all.

**Grader limitation found:** all 4 code-only requests returned a full one-line function containing the correct return expression. The exact next-line contract correctly fails, but the initial line-based payload proxy incorrectly classifies those expressions as absent. Preserve original raw grades; add a versioned AST-based static payload check and separately derived regrading, never overwrite original responses. Explanation-proxy passes with token cutoffs are not evidence of completed, high-quality explanations.

**Environment action:** created an isolated native-filesystem Python venv with system NumPy access and installed the local `gguf` 0.19.0 reader from `/home/akmessi/llama.cpp/gguf-py` (source repository commit recorded separately). It will be used for read-only metadata inspection before any artifact mutation.

## 2026-11-05 — Versioned grading correction and template diagnostic design

**Measured speed:** first LFM3 screen's server-reported median decode throughput was 77.46 tokens/second (range 61.01–80.82). This is a short-prompt GPU result for these 40 requests, not a general benchmark or contradiction of the user's approximately 50 tok/s observation.

**Observation:** the native QAD GGUF contains an LFM3 architecture and a template that emits `<think></think>` for disabled thinking. The runtime supports per-request reasoning formats and thinking flags. Template text alone does not prove which prompt was actually rendered.

**Correction:** grader v2 adds static AST matching for return expressions, without executing generated code. A derived regrade of run 001 corrects 4 code payload false negatives while preserving strict failures and all original records. Nine tests pass. The runner now records `/apply-template` responses, the grader version, and explicitly whitelisted per-case diagnostic settings.

**Next diagnostic:** a fixed 16-case probe varies disabled thinking/raw parsing, disabled thinking/DeepSeek parsing, disabled thinking/effort-none, and enabled thinking/DeepSeek parsing on the same 4 discovery prompts. Run QAD and ordinary Q8 separately. QAD/Q8 differences are cross-checkpoint observations, not causal evidence of ordinary quantization effects.

## 2026-11-05 — QAD template probe completed

**Run:** `runs/20261105-002-lfm3-qad-template-probe/`, source `491b7c3`, 16 paired-condition requests, 1,782 generated tokens. The rendered disabled-thinking prompts end in `<think></think>`, confirming that the requested switch reaches the template. Explicit parsing and effort-none did not by themselves remove the extra generated closing markers or fix missing/conflicting evidence contracts on these four cases.

**Observed distinction:** raw response mode prepends the generation prompt's `<think></think>` to visible content, despite identical sampled-token counts. That leading prefix is a serialization effect and must not be counted as newly generated thinking. Other closing markers occur after content and require separate investigation.

**Thinking condition:** enabling parsed thinking completed the requested repeat correctly (1/4 exact contracts) but hit the 256-token limit on the other 3 cases, returning empty final content. Those are truncations, not established infinite loops. Increasing the cap may resolve them; no capability or loop-circuit claim follows.

**Next:** compare the ordinary Q8 checkpoint and collect Qwen's first discovery screen; preserve the distinction between different QAD/Q8 trained lineages.

## 2026-11-05 — Ordinary LFM3 Q8 diagnostic and Qwen baseline

**Runs:** `003-lfm3-q8-template-probe` (16 requests, 1,683 tokens) and `004-qwen3-q8-discovery` (40 requests, 713 tokens), under their recorded manifests.

**LFM3 Q8 observation:** under disabled thinking with parsed output, requested repetition and missing-address clarification passed on the one probed example each, where QAD failed. Both checkpoints still exceeded 256 tokens on the conflict example. This is a small paired-checkpoint lead requiring broader replication, not proof that Q4 rounding caused it. Raw serialization mode can expose the prefilled thinking prefix in either checkpoint.

**Qwen observation:** all 8 JSON examples contained the correct decoded object, but only 2/8 satisfied the whole-response contract; fences explain several failures. In this suite, 7/8 copy/repetition cases and 4/4 missing-field cases passed. Tool shape and complete/conflicting-evidence contracts did not. Several code outputs do not contain the requested return expression. Do not interpret every failure here as removable style: underlying task/prompt competence must be established first.

**Next:** collect Fable and same-precision parent before any rollback. Use strong prompt/grammar baselines for the JSON gap before proposing a weight-level solution.

## 2026-11-05 — All target deployments screened; parent comparison underway

**Runs:** `005-fable-f16-discovery` (40 requests, 2,058 tokens) and `006-lfm3-q8-discovery` (40 requests, 1,975 tokens). Their manifests pin the same discovery suite and exact artifact bytes. Fable and its parent's F16 downloads passed published SHA-256 checks.

**Fable observation:** several requests with no tools supplied produce unsolicited tool-call-like syntax (for example, copying text becomes a fictitious `copy(...)` call). Code-only responses contain the correct expression in a fenced full function, while the strict next-line contract fails. Requested-repeat examples sometimes become unrelated processor descriptions. These specific outputs are retained verbatim; a fine-tuning causal claim needs the parent comparison.

**LFM3 full paired lead:** ordinary Q8 passed 8/8 copy/repetition and 4/4 missing-field cases; QAD passed 5/8 and 0/4 on the same exploratory prompts. Both passed 8/8 simple JSON and 4/4 complete tool cases. QAD completed the four complete-evidence contracts while Q8 did not, often exhausting 128 tokens. Thus neither checkpoint dominates all conditions. Broader resampling, longer-cap checks, and lineage controls are needed.

**Decision:** target response-mode switching and evidence decisions as conditional research questions. Do not label all cutoffs as loops. Run the matched F16 parent before attributing Fable's behavior to training or choosing a repair.

## 2026-11-05 — Same-precision parent comparison and first edit protocol

**Run:** `007-parent-f16-discovery`, same suite, seed, F16 precision, and sampling as Fable. Generated tokens: 869. Parent JSON payloads were correct in 8/8 cases (strict format 0/8 due to fences), compared with Fable's 1/8 payloads. Parent exact-copy cases passed 4/4 versus Fable 0/4. Both contained the correct return expressions in all four code-only cases, while failing the requested next-line contract. These tiny tasks do not establish a coding-specialization gain.

**Hypothesis:** at least some Fable response-mode regressions may be recoverable by partial rollback. This is not yet a repair result, and a full/partial return to the parent could erase valuable behavior not represented by this screen.

**Implemented protocol:** separate-file F16 GGUF FFN-output rollback with source/artifact checksums, tokenizer/architecture alignment, per-tensor changed-entry and norm accounting, untouched-tensor verification, exact alpha-zero identity, and original-file rechecks. This is explicitly a non-sparse baseline. Twelve tests pass, including exact endpoint bytes, source immutability, dtype/shape mismatch, and non-finite rejection.

**Planned comparisons:** alpha-zero identity before nonzero edits; then a bounded 25% FFN-output rollback as a discovery candidate. Strong prompt and grammar controls are generated separately without encoding gold values in the schema. No selected checkpoint will be called a validated improvement from this discovery suite alone.

## 2026-11-05 — Identity verified, first nonzero artifact created, and non-weight baselines

**Artifact evidence:** `research/patches/fable-identity.json` records byte-identical alpha-zero output, matching the original whole-file SHA-256. `fable-ffn-rollback-025.json` records a 25% rollback of all 14 FFN output matrices toward the published F16 parent, with 13,764,917 post-cast entries changed. Relative matrix changes are approximately 0.044% each. All 118 untouched tensors and both source file hashes were verified. No behavioral improvement is implied by file integrity.

**Non-weight runs:** `008-fable-nonweight-controls`, `009-qwen-nonweight-controls`, `010-lfm3-qad-nonweight-controls`, 52 development requests each. A schema constrains keys and types already given by the task, never gold values. On the 12 JSON/tool examples, prompt-plus-grammar gave Fable 10/12 and Qwen 12/12 exact correct outputs. Strong prompt without grammar gave 0/12 and 2/12 respectively. These are powerful development baselines that a weight method must be compared with, not omitted.

**Other prompt effects:** the stronger prompt repaired QAD's tested requested-repetition contracts (copy/repeat 8/8 versus 5/8 under the original system prompt), but did not fix its missing-information contracts. On these cases Qwen's missing-field exact responses fell from 4/4 to 1/4 under the stronger system prompt. These are prompt-sensitive exploratory effects with four examples per category, not general rates.

**Causal limitation:** the public Fable card identifies the parent repository but does not pin the original training revision. Our matched-precision parent is the currently pinned public checkpoint. Claims about the exact fine-tuning delta require verifying that it is the actual training parent; checkpoint comparisons remain valid as comparisons of these exact artifacts.

**Next:** run an identity behavior check, evaluate the first rollback on discovery cases, and retain any negative result before considering other sites or doses.

## 2026-11-05 — Audit decision before expanding edits

**Observation:** the first 25% rollback changes many entries but only about 0.044% of each selected matrix's norm, with unusually similar magnitudes across all 14 FFN output matrices. This may reflect small learned updates, dtype/export effects, or another checkpoint difference. It does not justify a neuron/circuit claim.

**Decision:** audit all corresponding floating tensors, including how often rounding Fable values to BF16 reproduces the public parent's values. This diagnoses precision/provenance confounds before expanding the edit search. It does not establish the exact original training checkpoint.

**Implementation:** added paired-run comparison and streaming tensor-delta audit scripts. The existing report generator produces a readable per-group baseline table with strong-prompt and grammar conditions separated.

## 2026-11-05 — First weight intervention negative; attention/provenance lead

**Identity result:** run 011 reproduced the original Fable content exactly on all 10 checked cases, with identical completion-token counts. Source file and identity artifact hashes match.

**Rollback result:** run 012's 25% FFN-output rollback rescued 0 exact contracts and lost 0 across 40 discovery cases; 34/40 outputs were byte-identical. JSON payload proxy improved 1/8 to 2/8, which is not an established useful repair. Preserve it as a negative baseline rather than expanding doses blindly.

**Audit observation:** Q/K/V attention matrices differ by roughly 3.6–4.9% in relative norm between these published checkpoints, while most FFN/convolution/embedding tensors differ by roughly 0.17%. The first one-way BF16 diagnostic does not identify which export is coarser. Add the reverse rounding comparison and on-grid fractions before interpreting these differences as training updates.

**Next predeclared candidates:** (a) a parent with BF16-rounded values stored in the same floating GGUF layout, to isolate value precision from runtime format; (b) a 25% rollback of only attention V matrices at layers 10 and 12 toward precision-aligned parent values. These sites are selected by the weight-delta audit, not by hidden test outcomes. They are discovery baselines, not a claimed new method or identified behavior circuit.

## 2026-11-05 — Precision audit exposes a real checkpoint-comparison confound

**Measured:** Fable's embedding and FFN output values lie entirely on the BF16 value grid despite being stored in an F16 file. The public F16 parent has only about 12.5% of those values on that grid. Rounding the parent's values to BF16 matches approximately 93.7% of Fable's embedding/FFN values. Attention Q/K/V matrices differ substantially beyond that pattern.

**Interpretation:** equal storage dtype does not guarantee equal effective value precision. Some apparent checkpoint differences are plausibly export/precision changes, not useful learned deltas. The approximately 6.3% rounding mismatch may involve double-rounding ties or other provenance differences; do not claim an exact reconstruction from the F16 parent.

**Controls created:** a simulated BF16-value parent stored in its original F16 layout, and a two-site late-V rollback (1,021,626 changed entries; approximately 1.21–1.22% relative norm per selected matrix), each with all untouched tensors and originals verified.

**Next:** evaluate both controls and fetch the separately published, checksummed BF16 parent. Exact BF16 source values can test the rounding/provenance explanation more directly than inferring them from already-rounded F16 values.

## 2026-11-05 — Precision control and late-V candidate evaluated

**Runs 013/014:** the simulated BF16-value parent retained the tested 4/4 exact-copy and 4/4 complete-evidence successes. This suggests value rounding alone does not explain those Fable losses on these cases. Its explanation lengths changed, including three 128-token cutoffs, so precision can still affect generation behavior.

**Late-V candidate:** the two-site 25% attention-V rollback did not rescue the exact response contracts and lost one explanation proxy within its truncated output. Keep this negative result. It is not evidence that attention is irrelevant; it rejects this particular site/dose choice on this discovery screen.

**Next artifact control:** support exact BF16 tensor decoding into the same F16 storage layout. Tests cover known BF16 bit patterns, shape/orientation, dtype preservation, ties-to-even rounding, and rejection of unsupported packed quantization. This avoids treating already-rounded F16 values as an exact source of BF16 values.

## 2026-11-05 — Exact BF16 artifact, template parity, and QKV controls

**Artifact:** the official BF16 parent downloaded and passed its published checksum. Its BF16 linear tensors and F32 normalization/convolution tensors are handled explicitly; “BF16 GGUF” does not mean every stored tensor is BF16. Exact decoding is cross-checked against the installed GGUF library on selected matrices.

**Template control:** Fable and the parent use identical native template hashes, and all 40 rendered discovery prompts match exactly. This rules out a different prompt string as an explanation of their observed difference on these runs; it does not establish historical training provenance.

**Next predeclared inference controls:** decode the published BF16 parent into the same F16 artifact layout; restore all 18 Q/K/V projection matrices in Fable to the published BF16 parent as an upper-bound rollback control; and restore 50% of Q/K/V deltas only at layers 10/12. The all-QKV control may remove specialization and must not be called a skill-preserving repair. Site/dose choices remain exploratory, selected before their inference outcomes.

## 2026-11-05 — Precision alignment changes the interpretation of the deltas

**Measured:** after decoding the official BF16 linear weights and rounding its remaining floating values to BF16, 105/132 tensors match Fable exactly. Of the other 27, 18 are the Q/K/V projection matrices; the remaining 9 contain only 1–3 differing scalar values each. Decoder values matched the installed GGUF library on three checked projection matrices. This is strong numerical evidence for concentrated QKV changes plus precision/export differences, without establishing the historical training revision.

**Correction to interpretation:** the earlier FFN rollback largely perturbed effective precision rather than removing a clean learned FFN update. Its negative result does not identify or exclude FFN behavior circuits. Same-type files were an insufficient control; exact source-value alignment was necessary.

**Runs 015–017:** the exact-source parent control retained the 4/4 copy and 4/4 complete-evidence successes, with different/truncated explanations. Full QKV restoration in Fable also recovered those 8 exact contracts and 8/8 JSON payloads, while retaining the correct four simple return expressions. The late-two-layer 50% QKV candidate did not recover those contracts. None establishes preservation of Fable's useful coding specialization; that needs a fresh, appropriate coding evaluation.

**Next predeclared ablations:** restore Q only, K only, or V only across the six attention layers, each to the exact BF16 parent projection values. This separates projection families with a fixed three-candidate comparison, instead of selecting a flattering subset after test inspection. All outcomes remain discovery data.

## 2026-11-05 — Separate projection families and frozen validation decision

**Runs 018–020:** Q-only and K-only restorations did not recover exact copy on this screen. V-only recovered 4/4 literal-copy contracts and 6/8 JSON payloads, compared with Fable's 0/4 and 1/8. However, its explanation proxy fell from 4/4 to 0/4, so this is a selective behavior change with a regression, not an accepted repair. K-only also lost the simple code expression proxy. These are not clean independent semantic knobs.

**Decision before new evaluation:** freeze a fresh 48-fixture panel and evaluate original Fable, V-only, broad all-QKV, and a uniformly BF16-aligned parent at two seeds. Add Qwen/LFM3 original references at one seed. The purpose is to test the narrow copy-recovery lead and quantify regressions, without further choosing sites/doses. The protocol is `research/protocols/validation-v1.md`; max 480 requests / 61,440 generated tokens. Keep this data out of further patch selection.
