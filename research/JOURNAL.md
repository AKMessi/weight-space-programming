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

## 2026-11-05 — Fresh validation rejects broad copy-repair interpretation

**Completed:** all 10 predeclared validation runs, 480 requests. The frozen analysis is in `research/validation-v1-analysis.json` and `research/VALIDATION_REPORT.md`.

**Primary result:** original Fable and V-only both passed 0/24 literal-copy requests on the new payload forms (12 prompts × 2 seeds). The discovery 4/4 copy recovery therefore did not generalize to this fresh panel. Broad all-QKV rollback and the aligned parent also failed the exact literal-copy contract here. Qwen passed 12/12 reference copies; LFM3 QAD passed 7/12. This does not support shipping the candidate as a general repair.

**Other recorded outcomes:** Fable and V-only both included correct JSON objects on all 24 new JSON-field requests but failed the plain-response contract. Broad QKV rollback/aligned parent restored many dictionary-return expression payloads compared with Fable. These are narrow, synthetic, task-dependent outcomes, not a benchmark of coding specialization.

**Secondary grading issue found by manual audit:** the code-line strict scorer treated equivalent single/double quotes and Python whitespace as different reference strings. Some correct return expressions around thinking markers also escaped the limited payload proxy. Grader v3 uses AST equality for a single return statement and broader static expression-presence detection. It still rejects functions/prose/duplicate outputs as strict next-line responses. Preserve all original v2 validation records and the frozen analysis; publish separate explicitly post-hoc sensitivity regrades. The primary literal-copy metric does not change.

**Decision:** do not tune more doses/sites on this validation panel. Next research should use better task-competence controls, stronger wrapper diversity in development, and a new independent confirmation set. No accepted skill-preserving weight repair has been established.

## 2026-11-05 — First milestone evidence audit and release status

**Integrity:** `scripts/verify_evidence.py` found 30 complete runs, 1,238 unique request records, and 41,908 generated tokens. Run/dataset/case identities and the reservation/settlement ledger match; every request is settled. This is evidence-consistency verification, not a scientific-validity or general-improvement certificate. No inference server remains running.

**Testing and provenance:** 19 tests pass, including prompt-grouped uncertainty and exact nonzero upper bounds after observing zero loss events. Future manifests additionally hash colocated runtime shared libraries because the small server executable loads substantial code dynamically. Earlier manifests retain executable hash/version, template and server logs; current library bytes cannot retroactively prove historical bytes.

**Grader sensitivity release:** derived v3 regrades for all 30 runs are under `research/regrades/`, each tied to the original response hash. The primary literal-copy result is unchanged. On validation, Qwen's strict dictionary-code count changes from 3/12 to 4/12; LFM3's static code-expression presence changes from 0/12 to 12/12. These corrections reinforce the difference between a correct expression appearing somewhere and a complete response satisfying the contract. They do not establish functional coding quality. Original recorded grades and frozen analysis remain intact.

**Release:** `research/FINDINGS.md` summarizes observations and their limits in plain words. `research/PROGRESS.md` now records actual study status rather than the pre-inference placeholder. The five-study design remains a proposal portfolio, with the quantization/composition gates unmet and the blinded benchmark unimplemented. The next useful milestone is establishing Fable's actual coding advantage and stronger development competence controls before further repair search. No accepted patch, new algorithm, neuron-circuit identification, or publication-ready finding is claimed.

**Report verification:** `research/BASELINE_REPORT_GRADER_V3.md` regenerates all 20 discovery/development runs using the committed current grader. The ten frozen validation runs are intentionally excluded from that exploratory report and retain their original analysis. All 30 derived regrade source hashes were independently rechecked. The nine non-projection residual tensors in the precision-alignment audit sum to exactly 12 differing scalar positions.

## 2026-11-05 — User requested completion of the whole portfolio

**Scope:** continue through all five finite pilots, retaining negative/inconclusive outcomes rather than promising successful repairs. `research/protocols/portfolio-completion-v1.md` declares fixture families, bounded candidate search, two-turn scoring, quantization/composition prerequisite failures, and a local blinded diagnosis pilot. Existing validation v1 is not reused for patch selection. Initial additional cap: 250k generated tokens / 150 inference runs within the existing global ceiling.

**Implementation:** grader v4 adds bounded AST interpretation of short Python functions; no generated code is executed by Python. Empty/negative/alternate inputs distinguish correct functionality from expression presence. The runner passes actual assistant replies into follow-up turns, saves each complete request/response, and charges a conservative whole-conversation reservation. A git-status whitespace bug is corrected so the first modified file retains its porcelain status columns. Earlier source was manually committed before runs; this hardens future automated checks.

**Fixed proposals:** A/B attenuate last-layer FFN/attention normalization by 10%; R is a fixed random perturbation matching A's L2 delta. These are small black-box baselines and negative controls, not activation-derived directions or new methods. The protocol explicitly discloses unsupported activation/component-method arms and the limited local editor. Pin additional floating sources before quantization and keep the QAD lineage separate. Experimental results for this phase are not known at this entry.

**Pre-inference artifact checks:** all 15 fixed A/B/R/AB/BA artifacts are verified. AB and BA have identical complete-file hashes for every model. The LFM2 runtime source explicitly maps its output normalization to the legacy token_embd_norm name; that alias is verified rather than assumed. Added bounded dictionary access/list building and sampled host/device memory recording before phase inference. The preservation selection gate also requires no decline in requested explanation success.

### 2026-11-05T11:33:18.017587+00:00 — p2-e1dev-qwen3_q8-s17

Run status: complete; generated tokens: 582. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 1}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:33:36.731740+00:00 — p2-e1dev-qwen3_q8_A-s17

Run status: complete; generated tokens: 684. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-qwen3_q8_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:33:54.336796+00:00 — p2-e1dev-qwen3_q8_B-s17

Run status: complete; generated tokens: 591. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 1}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-qwen3_q8_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:34:11.933442+00:00 — p2-e1dev-qwen3_q8_R-s17

Run status: complete; generated tokens: 591. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 1}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-qwen3_q8_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:34:38.064582+00:00 — p2-e1dev-lfm3_qad-s17

Run status: complete; generated tokens: 1176. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:35:13.885661+00:00 — p2-e1dev-lfm3_qad_A-s17

Run status: complete; generated tokens: 1173. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-lfm3_qad_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:35:52.710604+00:00 — p2-e1dev-lfm3_qad_B-s17

Run status: complete; generated tokens: 1232. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-lfm3_qad_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:36:23.427473+00:00 — p2-e1dev-lfm3_qad_R-s17

Run status: complete; generated tokens: 1221. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-lfm3_qad_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:36:44.926541+00:00 — p2-e1dev-fable_f16-s17

Run status: complete; generated tokens: 2456. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 2, "responses": 6, "success": 2}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 8, "payload_correct": 1, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:37:10.960940+00:00 — p2-e1dev-fable_f16_A-s17

Run status: complete; generated tokens: 2660. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 8, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_f16_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:37:34.884077+00:00 — p2-e1dev-fable_f16_B-s17

Run status: complete; generated tokens: 2582. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 8, "payload_correct": 2, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_f16_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:37:59.060452+00:00 — p2-e1dev-fable_f16_R-s17

Run status: complete; generated tokens: 2616. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 2, "responses": 6, "success": 2}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 8, "payload_correct": 2, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_f16_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:38:11.517497+00:00 — p2-e1dev-parent_uniform_bf16-s17

Run status: complete; generated tokens: 891. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-parent_uniform_bf16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:38:26.026578+00:00 — p2-e1dev-fable_v_only-s17

Run status: complete; generated tokens: 1143. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_v_only-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:38:41.257568+00:00 — p2-e1dev-fable_all_qkv_parent-s17

Run status: complete; generated tokens: 891. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1dev-fable_all_qkv_parent-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:38:59.673283+00:00 — p2-e2dev-original-qwen3_q8-s17

Run status: complete; generated tokens: 632. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:39:20.041251+00:00 — p2-e2dev-original-qwen3_q8_A-s17

Run status: complete; generated tokens: 607. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-qwen3_q8_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:39:40.691480+00:00 — p2-e2dev-original-qwen3_q8_B-s17

Run status: complete; generated tokens: 631. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-qwen3_q8_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:39:58.933668+00:00 — p2-e2dev-strong-qwen3_q8-s17

Run status: complete; generated tokens: 636. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-strong-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:40:21.110252+00:00 — p2-e2dev-grammar-qwen3_q8-s17

Run status: complete; generated tokens: 636. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:40:42.669935+00:00 — p2-e2dev-original-lfm3_qad-s17

Run status: complete; generated tokens: 638. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:41:04.348123+00:00 — p2-e2dev-original-lfm3_qad_A-s17

Run status: complete; generated tokens: 615. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-lfm3_qad_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:41:24.398260+00:00 — p2-e2dev-original-lfm3_qad_B-s17

Run status: complete; generated tokens: 615. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-lfm3_qad_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:41:52.066053+00:00 — p2-e2dev-strong-lfm3_qad-s17

Run status: complete; generated tokens: 1150. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 4, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-strong-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:42:11.529906+00:00 — p2-e2dev-grammar-lfm3_qad-s17

Run status: complete; generated tokens: 618. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:42:20.790559+00:00 — p2-e2dev-original-fable_f16-s17

Run status: complete; generated tokens: 548. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:42:30.189641+00:00 — p2-e2dev-original-fable_f16_A-s17

Run status: complete; generated tokens: 554. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-fable_f16_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:42:43.327279+00:00 — p2-e2dev-original-fable_f16_B-s17

Run status: complete; generated tokens: 546. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-original-fable_f16_B-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:42:54.276098+00:00 — p2-e2dev-strong-fable_f16-s17

Run status: complete; generated tokens: 570. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-strong-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:43:05.142077+00:00 — p2-e2dev-grammar-fable_f16-s17

Run status: complete; generated tokens: 570. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:44:32.824294+00:00 — p2-schema-adversarial-qwen3_q8-s17

Run status: complete; generated tokens: 21. Group counts: `{"schema_check": {"attempted": 3, "cutoff": 0, "payload_correct": 3, "responses": 3, "success": 3}}`. Full immutable requests, raw responses, and settings are in `runs/p2-schema-adversarial-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:49:01.968552+00:00 — p2-e2dev-grammar-v2-qwen3_q8-s17

Run status: complete; generated tokens: 590. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-v2-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:49:23.572134+00:00 — p2-e2dev-grammar-v2-lfm3_qad-s17

Run status: complete; generated tokens: 618. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-v2-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:49:35.257907+00:00 — p2-e2dev-grammar-v2-fable_f16-s17

Run status: complete; generated tokens: 634. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2dev-grammar-v2-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:49:51.386037+00:00 — p2-e1confirm-qwen3_q8-s17

Run status: complete; generated tokens: 653. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 7, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:50:07.320617+00:00 — p2-e1confirm-qwen3_q8-s29

Run status: complete; generated tokens: 688. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-qwen3_q8-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:50:26.050634+00:00 — p2-e1confirm-qwen3_q8_A-s17

Run status: complete; generated tokens: 726. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 3, "responses": 8, "success": 2}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-qwen3_q8_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:50:42.982101+00:00 — p2-e1confirm-qwen3_q8_A-s29

Run status: complete; generated tokens: 785. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 0, "responses": 6, "success": 0}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 2, "responses": 8, "success": 1}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-qwen3_q8_A-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:51:08.884470+00:00 — p2-e1confirm-lfm3_qad-s17

Run status: complete; generated tokens: 1232. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:51:36.014112+00:00 — p2-e1confirm-lfm3_qad-s29

Run status: complete; generated tokens: 1301. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 1, "payload_correct": 6, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-lfm3_qad-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:52:04.183648+00:00 — p2-e1confirm-lfm3_qad_A-s17

Run status: complete; generated tokens: 1304. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-lfm3_qad_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:52:31.807354+00:00 — p2-e1confirm-lfm3_qad_A-s29

Run status: complete; generated tokens: 1364. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 1, "payload_correct": 4, "responses": 6, "success": 2}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-lfm3_qad_A-s29`. These counts are observations; interpretation is deferred to the study report.

## 2026-11-05 — Schema transport correction and completed development selection

**Observed bug:** new v1 type=json_schema fixtures put schema at the top level, which this runtime ignores in favor of response_format.json_schema.schema. Outputs such as action=address despite an enum exposed the problem. Earlier non-weight controls used supported type=json_object with schema and are unaffected. All three malformed study-2 controls remain in Git as generic-JSON evidence; original frozen fixture files were preserved. Corrected v2 files and actual reruns are separate.

**Verified correction:** the adversarial diagnostic passed 3/3 cases (string, integer, boolean enums), even when the user requested a forbidden output. Those diagnostic gold values deliberately appear in the schema and are excluded from capability/improvement scores. Grader v5 adds only diagnosis/confidence scoring; existing functional/conversation semantics remain unchanged. Response-format validation now rejects the malformed wrapper before inference. Thirty tests pass.

**Development decisions frozen:** neither study 1 nor study 2 yielded an eligible new weight repair for any target under the declared criteria. A is retained only as a negative diagnostic in confirmation. Selection files were committed before those runs. No confirmation outcome is being used to choose another candidate.

**Deployment preparation:** pinned Qwen BF16 community export and official ordinary LFM3 F16 both passed their source hashes. Qwen conversion explicitly updates file-type metadata to F16 and records scalar conversion errors; it is not claimed identical to the official Q8 deployment. LFM3 floating/quantized comparison uses the same 18-layer GPU offload to leave VRAM headroom. Host high-water and device-wide sampled VRAM are recorded per run. Preparation/publication is serialized at clean boundaries.

### 2026-11-05T11:55:59.359754+00:00 — p2-e1confirm-fable_f16-s17

Run status: complete; generated tokens: 1333. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 1, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:56:16.505378+00:00 — p2-e1confirm-fable_f16-s29

Run status: complete; generated tokens: 1766. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 1, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 2, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-fable_f16-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:56:31.820789+00:00 — p2-e1confirm-fable_f16_A-s17

Run status: complete; generated tokens: 1239. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 2, "responses": 6, "success": 2}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 1, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-fable_f16_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:56:45.971389+00:00 — p2-e1confirm-fable_f16_A-s29

Run status: complete; generated tokens: 1292. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-fable_f16_A-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:56:58.175061+00:00 — p2-e1confirm-parent_uniform_bf16-s17

Run status: complete; generated tokens: 641. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 4, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-parent_uniform_bf16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:57:08.497262+00:00 — p2-e1confirm-parent_uniform_bf16-s29

Run status: complete; generated tokens: 649. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 4}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 5, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1confirm-parent_uniform_bf16-s29`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:57:31.028264+00:00 — p2-e2confirm-original-qwen3_q8-s17

Run status: complete; generated tokens: 727. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:57:52.363941+00:00 — p2-e2confirm-original-qwen3_q8_A-s17

Run status: complete; generated tokens: 683. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-qwen3_q8_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T11:58:11.540209+00:00 — p2-e2confirm-strong-qwen3_q8-s17

Run status: complete; generated tokens: 698. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-strong-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:02:58.965129+00:00 — p2-e2confirm-grammar-v2-qwen3_q8-s17

Run status: complete; generated tokens: 619. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-grammar-v2-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:03:30.944908+00:00 — p2-e2confirm-original-lfm3_qad-s17

Run status: complete; generated tokens: 1039. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 1, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:12:37.561248+00:00 — p2-e2confirm-original-lfm3_qad_A-s17

Run status: complete; generated tokens: 1197. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 1, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-lfm3_qad_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:13:13.976244+00:00 — p2-e2confirm-strong-lfm3_qad-s17

Run status: complete; generated tokens: 1359. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 4, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 1, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-strong-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:13:35.880224+00:00 — p2-e2confirm-grammar-v2-lfm3_qad-s17

Run status: complete; generated tokens: 654. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-grammar-v2-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:13:45.276614+00:00 — p2-e2confirm-original-fable_f16-s17

Run status: complete; generated tokens: 556. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:13:56.531119+00:00 — p2-e2confirm-original-fable_f16_A-s17

Run status: complete; generated tokens: 558. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-original-fable_f16_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:14:05.682418+00:00 — p2-e2confirm-strong-fable_f16-s17

Run status: complete; generated tokens: 484. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-strong-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:17:15.166624+00:00 — p2-e2confirm-grammar-v2-fable_f16-s17

Run status: complete; generated tokens: 661. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e2confirm-grammar-v2-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:17:33.148439+00:00 — p2-e1-development-strong-qwen3_q8-s17

Run status: complete; generated tokens: 575. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 2}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-strong-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:17:48.539074+00:00 — p2-e1-development-grammar-v2-qwen3_q8-s17

Run status: complete; generated tokens: 575. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 2}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-grammar-v2-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:18:17.702565+00:00 — p2-e1-development-strong-lfm3_qad-s17

Run status: complete; generated tokens: 1227. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-strong-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:18:45.498109+00:00 — p2-e1-development-grammar-v2-lfm3_qad-s17

Run status: complete; generated tokens: 1227. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-grammar-v2-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:18:58.027585+00:00 — p2-e1-development-strong-fable_f16-s17

Run status: complete; generated tokens: 1027. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 0, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-strong-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:19:10.085455+00:00 — p2-e1-development-grammar-v2-fable_f16-s17

Run status: complete; generated tokens: 859. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-development-grammar-v2-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:19:25.951704+00:00 — p2-e1-confirmation-strong-qwen3_q8-s17

Run status: complete; generated tokens: 543. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-strong-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:19:40.968961+00:00 — p2-e1-confirmation-grammar-v2-qwen3_q8-s17

Run status: complete; generated tokens: 543. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-grammar-v2-qwen3_q8-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:20:11.991508+00:00 — p2-e1-confirmation-strong-lfm3_qad-s17

Run status: complete; generated tokens: 1474. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-strong-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:20:47.442050+00:00 — p2-e1-confirmation-grammar-v2-lfm3_qad-s17

Run status: complete; generated tokens: 1474. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 3}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-grammar-v2-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:21:00.049506+00:00 — p2-e1-confirmation-strong-fable_f16-s17

Run status: complete; generated tokens: 946. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 2, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-strong-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:21:12.026065+00:00 — p2-e1-confirmation-grammar-v2-fable_f16-s17

Run status: complete; generated tokens: 914. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "functional_explain": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 3}, "literal_copy": {"attempted": 8, "cutoff": 0, "payload_correct": 2, "responses": 8, "success": 0}, "structured": {"attempted": 8, "cutoff": 0, "payload_correct": 8, "responses": 8, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e1-confirmation-grammar-v2-fable_f16-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:21:25.852132+00:00 — p2-e4engineering-qwen3_q8_AB-s17

Run status: complete; generated tokens: 357. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-qwen3_q8_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:21:41.079115+00:00 — p2-e4ordercheck-qwen3_q8_BA-s17

Run status: complete; generated tokens: 99. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-qwen3_q8_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:22:01.192801+00:00 — p2-e4joint-qwen3_q8_R-s17

Run status: complete; generated tokens: 608. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-qwen3_q8_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:22:18.308955+00:00 — p2-e4joint-qwen3_q8_AB-s17

Run status: complete; generated tokens: 607. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-qwen3_q8_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:22:35.241287+00:00 — p2-e4engineering-lfm3_qad_AB-s17

Run status: complete; generated tokens: 335. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-lfm3_qad_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:22:47.572129+00:00 — p2-e4ordercheck-lfm3_qad_BA-s17

Run status: complete; generated tokens: 113. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-lfm3_qad_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:23:12.056512+00:00 — p2-e4joint-lfm3_qad_R-s17

Run status: complete; generated tokens: 615. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-lfm3_qad_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:23:31.060359+00:00 — p2-e4joint-lfm3_qad_AB-s17

Run status: complete; generated tokens: 615. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 4, "responses": 4, "success": 4}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-lfm3_qad_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:23:44.701509+00:00 — p2-e4engineering-fable_f16_AB-s17

Run status: complete; generated tokens: 1049. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 2, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-fable_f16_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:23:52.279650+00:00 — p2-e4ordercheck-fable_f16_BA-s17

Run status: complete; generated tokens: 155. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-fable_f16_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:24:03.130776+00:00 — p2-e4joint-fable_f16_R-s17

Run status: complete; generated tokens: 548. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-fable_f16_R-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:24:12.409066+00:00 — p2-e4joint-fable_f16_AB-s17

Run status: complete; generated tokens: 546. Group counts: `{"complete": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "conflict": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}, "missing": {"attempted": 4, "cutoff": 0, "payload_correct": 0, "responses": 4, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4joint-fable_f16_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:29:56.381820+00:00 — p2-e3precision-qwen_float-s17

Run status: complete; generated tokens: 356. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:30:15.809409+00:00 — p2-e3precision-qwen_float_A-s17

Run status: complete; generated tokens: 362. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_float_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:30:29.339292+00:00 — p2-e3precision-qwen_Q8_0-s17

Run status: complete; generated tokens: 356. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_Q8_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:30:43.097367+00:00 — p2-e3precision-qwen_Q8_0_A-s17

Run status: complete; generated tokens: 362. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_Q8_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:30:54.613558+00:00 — p2-e3precision-qwen_Q4_0-s17

Run status: complete; generated tokens: 361. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_Q4_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:31:06.487697+00:00 — p2-e3precision-qwen_Q4_0_A-s17

Run status: complete; generated tokens: 370. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-qwen_Q4_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:32:08.692526+00:00 — p2-e3precision-lfm3_float-s17

Run status: complete; generated tokens: 331. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:33:26.812307+00:00 — p2-e3precision-lfm3_float_A-s17

Run status: complete; generated tokens: 331. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_float_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:34:09.561746+00:00 — p2-e3precision-lfm3_Q8_0-s17

Run status: complete; generated tokens: 331. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_Q8_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:34:51.524579+00:00 — p2-e3precision-lfm3_Q8_0_A-s17

Run status: complete; generated tokens: 331. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 4}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_Q8_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:35:26.220345+00:00 — p2-e3precision-lfm3_Q4_0-s17

Run status: complete; generated tokens: 322. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_Q4_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:36:00.839415+00:00 — p2-e3precision-lfm3_Q4_0_A-s17

Run status: complete; generated tokens: 326. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-lfm3_Q4_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:36:13.273227+00:00 — p2-e3precision-fable_float-s17

Run status: complete; generated tokens: 1154. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 1, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:36:27.010618+00:00 — p2-e3precision-fable_float_A-s17

Run status: complete; generated tokens: 1058. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 0, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_float_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:38:00.180921+00:00 — p2-e3precision-fable_Q8_0-s17

Run status: complete; generated tokens: 1141. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 1, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_Q8_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:39:27.365455+00:00 — p2-e3precision-fable_Q8_0_A-s17

Run status: complete; generated tokens: 1058. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 4, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 0, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_Q8_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:40:56.399943+00:00 — p2-e3precision-fable_Q4_0-s17

Run status: complete; generated tokens: 1797. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 2, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 6, "payload_correct": 0, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 0, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_Q4_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:42:24.814086+00:00 — p2-e3precision-fable_Q4_0_A-s17

Run status: complete; generated tokens: 1791. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 2, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 6, "payload_correct": 0, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 5, "payload_correct": 0, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3precision-fable_Q4_0_A-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:42:32.261371+00:00 — p2-e3qadparity-lfm3_qad-s17

Run status: complete; generated tokens: 114. Group counts: `{"diagnosis_probe": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3qadparity-lfm3_qad-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:42:44.371550+00:00 — p2-e3qadparity-qad_rewrite-s17

Run status: complete; generated tokens: 114. Group counts: `{"diagnosis_probe": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e3qadparity-qad_rewrite-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:43:02.343861+00:00 — p2-e5probes-qwen_diagnosis_healthy-s17

Run status: complete; generated tokens: 281. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 12, "responses": 12, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_healthy-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:43:20.279999+00:00 — p2-e5probes-qwen_diagnosis_healthy_permuted-s17

Run status: complete; generated tokens: 281. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 12, "responses": 12, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_healthy_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:44:29.480261+00:00 — p2-e5probes-qwen_diagnosis_output_sign-s17

Run status: complete; generated tokens: 1445. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 9, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_output_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:45:38.948299+00:00 — p2-e5probes-qwen_diagnosis_output_sign_permuted-s17

Run status: complete; generated tokens: 1441. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 10, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_output_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:46:19.334367+00:00 — p2-e5probes-qwen_diagnosis_ffn_sign-s17

Run status: complete; generated tokens: 1245. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 9, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_ffn_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:46:59.325440+00:00 — p2-e5probes-qwen_diagnosis_ffn_sign_permuted-s17

Run status: complete; generated tokens: 1254. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 8, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-qwen_diagnosis_ffn_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:47:30.287644+00:00 — p2-e5probes-lfm3_diagnosis_healthy-s17

Run status: complete; generated tokens: 228. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 12, "responses": 12, "success": 12}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_healthy-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:48:34.973634+00:00 — p2-e5probes-lfm3_diagnosis_healthy_permuted-s17

Run status: complete; generated tokens: 228. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 12, "responses": 12, "success": 12}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_healthy_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:51:08.759320+00:00 — p2-e5probes-lfm3_diagnosis_output_sign-s17

Run status: complete; generated tokens: 1508. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 11, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_output_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:53:23.060749+00:00 — p2-e5probes-lfm3_diagnosis_output_sign_permuted-s17

Run status: complete; generated tokens: 1496. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 10, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_output_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:55:30.925531+00:00 — p2-e5probes-lfm3_diagnosis_ffn_sign-s17

Run status: complete; generated tokens: 1365. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 8, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_ffn_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:57:35.864691+00:00 — p2-e5probes-lfm3_diagnosis_ffn_sign_permuted-s17

Run status: complete; generated tokens: 1344. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 6, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-lfm3_diagnosis_ffn_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:57:44.689119+00:00 — p2-e5probes-fable_diagnosis_healthy-s17

Run status: complete; generated tokens: 248. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 8, "responses": 12, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_healthy-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:57:53.403888+00:00 — p2-e5probes-fable_diagnosis_healthy_permuted-s17

Run status: complete; generated tokens: 248. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 0, "payload_correct": 8, "responses": 12, "success": 8}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_healthy_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:58:09.549323+00:00 — p2-e5probes-fable_diagnosis_output_sign-s17

Run status: complete; generated tokens: 1536. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 12, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_output_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:58:26.042153+00:00 — p2-e5probes-fable_diagnosis_output_sign_permuted-s17

Run status: complete; generated tokens: 1536. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 12, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_output_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:58:39.572383+00:00 — p2-e5probes-fable_diagnosis_ffn_sign-s17

Run status: complete; generated tokens: 1030. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 8, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_ffn_sign-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T12:58:53.211475+00:00 — p2-e5probes-fable_diagnosis_ffn_sign_permuted-s17

Run status: complete; generated tokens: 1030. Group counts: `{"diagnosis_probe": {"attempted": 12, "cutoff": 8, "payload_correct": 0, "responses": 12, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5probes-fable_diagnosis_ffn_sign_permuted-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:06:26.757418+00:00 — p2-e5editor-local_editor-s17

Run status: failed; generated tokens: 67. Group counts: `{"behavior": {"attempted": 3, "cutoff": 0, "payload_correct": 1, "responses": 2, "success": 1}, "weights": {"attempted": 1, "cutoff": 0, "payload_correct": 1, "responses": 1, "success": 1}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5editor-local_editor-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:12:02.834570+00:00 — p2-e5editor-retry1-local_editor-s17

Run status: complete; generated tokens: 532. Group counts: `{"behavior": {"attempted": 12, "cutoff": 0, "payload_correct": 8, "responses": 12, "success": 8}, "weights": {"attempted": 12, "cutoff": 0, "payload_correct": 7, "responses": 12, "success": 7}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e5editor-retry1-local_editor-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:31:35.514922+00:00 — p2-speed-followup-fable_float-s17

Run status: complete; generated tokens: 256. Group counts: `{"decode_measurement": {"attempted": 3, "cutoff": 3, "payload_correct": 0, "responses": 3, "success": 0}, "warmup": {"attempted": 1, "cutoff": 1, "payload_correct": 0, "responses": 1, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-speed-followup-fable_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:31:58.800366+00:00 — p2-speed-followup-fable_Q8_0-s17

Run status: complete; generated tokens: 256. Group counts: `{"decode_measurement": {"attempted": 3, "cutoff": 3, "payload_correct": 0, "responses": 3, "success": 0}, "warmup": {"attempted": 1, "cutoff": 1, "payload_correct": 0, "responses": 1, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-speed-followup-fable_Q8_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:32:14.967534+00:00 — p2-speed-followup-fable_Q4_0-s17

Run status: complete; generated tokens: 256. Group counts: `{"decode_measurement": {"attempted": 3, "cutoff": 3, "payload_correct": 0, "responses": 3, "success": 0}, "warmup": {"attempted": 1, "cutoff": 1, "payload_correct": 0, "responses": 1, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-speed-followup-fable_Q4_0-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:32:31.245979+00:00 — p2-speed-followup-fable_Q4_K_M-s17

Run status: complete; generated tokens: 256. Group counts: `{"decode_measurement": {"attempted": 3, "cutoff": 3, "payload_correct": 0, "responses": 3, "success": 0}, "warmup": {"attempted": 1, "cutoff": 1, "payload_correct": 0, "responses": 1, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-speed-followup-fable_Q4_K_M-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:40:35.170778+00:00 — p2-native-state-qwen_float-s17

Run status: complete; generated tokens: 2. Group counts: `{"raw_state": {"attempted": 2, "cutoff": 2, "payload_correct": 0, "responses": 2, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-native-state-qwen_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:41:10.841644+00:00 — p2-native-state-lfm3_float-s17

Run status: complete; generated tokens: 2. Group counts: `{"raw_state": {"attempted": 2, "cutoff": 2, "payload_correct": 0, "responses": 2, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-native-state-lfm3_float-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:41:31.208602+00:00 — p2-native-state-fable_float-s17

Run status: complete; generated tokens: 2. Group counts: `{"raw_state": {"attempted": 2, "cutoff": 2, "payload_correct": 0, "responses": 2, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-native-state-fable_float-s17`. These counts are observations; interpretation is deferred to the study report.

### Final-state compatibility, timing follow-up, and evidence packaging

All six native final-state contexts preserved the next-token top-1 but failed the predeclared centered-logit agreement gate (maximum error approximately 0.081–0.606, relative error 0.44–3.28%). Raw embeddings, exact chat prompts, native top-20 log probabilities and tokenizations are retained in `research/native-state-probes/`. The cause is unresolved: this is a failure to validate this readout boundary/settings, not proof that activation access or directional editing is impossible. No direction was edited from an unverified state. The separate diagnostic ledger settled 582 prompt/embedding tokens; six generated calibration tokens remain in the ordinary generation ledger.

The controlled four-format Fable timing follow-up produced exactly 64 tokens per call. Post-warmup median decode rates: F16 194.60, Q8_0 13.40, Q4_0 22.07, Q4_K_M 21.52 tokens/s. The locally generated Q8 and Q4_K_M artifacts match their pinned published hashes. A Q8 verbosity-5 startup trace reports all 15/15 layers offloaded. This evidence does not identify the slower kernel or establish a CPU-fallback cause; decode rankings are device/runtime/context-specific. Timing fixtures are measurement-only and have no model-quality score.

A post-hoc within-source storage control reconstructed the same original Fable exactly using a 44,653,069-byte patch from the deterministically BF16-rounded F16 parent. This separates the effect of a value transformation on one source from switching to the separately published BF16 parent, whose aligned patch is 20,308,656 bytes. Both are established XOR/deflate engineering results; neither is a new compression algorithm or an independently useful weight repair.

Final packaging adds recursive rebuilding of ignored GGUF artifacts from committed manifests. Historical manifests are never rewritten; newly rebuilt outputs must match their recorded complete hashes, and changed/missing sources fail closed. A recursive tiny-GGUF test checks restoration, unchanged historical manifests and wrong-source rejection. The batch resume path now verifies exact config/suite/seed/limit/token settings instead of silently accepting a reused name with different inputs. The evidence audit now accounts for real two-turn calls, the known pre-decoding editor rejection, and every settled reservation. No original response or grade has been overwritten.

### Analysis generation correction

The first final-report build stopped while exporting legacy regrades: the earliest original run's manifest predates `grader_version`. Its actual source commit and raw grades exist. This is an analysis compatibility error, not a model failure. `research/portfolio-v1/` is retained as an explicitly incomplete attempt. The corrected builder labels the legacy version as unversioned and writes a fresh `portfolio-v2/`; it never invents historical metadata or changes raw records.

### Composition-cap audit and declared correction

Final comparison audit found that initial AB/BA engineering calls allowed 128 output tokens, whereas the reused original/A/B/R study-1 calls allowed 192. This is a nuisance-variable mismatch even where aggregate successes agree. The original runs and first complete analysis snapshot (`portfolio-v2`) stay available. Six explicitly versioned AB/BA runs will match 192 tokens before final composition interpretation. The joint-evidence calls already all use 128. No candidate or dataset changes are allowed. Final reports move to `portfolio-v3`; v2 is a prior analysis snapshot with this disclosed limitation.

### 2026-11-05T13:51:42.218412+00:00 — p2-e4engineering-v2-qwen3_q8_AB-s17

Run status: complete; generated tokens: 357. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-v2-qwen3_q8_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:52:41.116002+00:00 — p2-e4ordercheck-v2-qwen3_q8_BA-s17

Run status: complete; generated tokens: 99. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-v2-qwen3_q8_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:53:05.104159+00:00 — p2-e4engineering-v2-lfm3_qad_AB-s17

Run status: complete; generated tokens: 335. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 6}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-v2-lfm3_qad_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:53:23.366015+00:00 — p2-e4ordercheck-v2-lfm3_qad_BA-s17

Run status: complete; generated tokens: 113. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 5, "responses": 6, "success": 5}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-v2-lfm3_qad_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:53:39.016942+00:00 — p2-e4engineering-v2-fable_f16_AB-s17

Run status: complete; generated tokens: 1433. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}, "literal_copy": {"attempted": 6, "cutoff": 0, "payload_correct": 6, "responses": 6, "success": 0}, "structured": {"attempted": 6, "cutoff": 6, "payload_correct": 2, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4engineering-v2-fable_f16_AB-s17`. These counts are observations; interpretation is deferred to the study report.

### 2026-11-05T13:53:47.069056+00:00 — p2-e4ordercheck-v2-fable_f16_BA-s17

Run status: complete; generated tokens: 155. Group counts: `{"functional_code": {"attempted": 6, "cutoff": 0, "payload_correct": 3, "responses": 6, "success": 0}}`. Full immutable requests, raw responses, and settings are in `runs/p2-e4ordercheck-v2-fable_f16_BA-s17`. These counts are observations; interpretation is deferred to the study report.

### Matched-cap replay completed; final comparison and artifact audits

All six AB/BA cap-corrected runs completed and were pushed individually. The final bounded continuation contains 137 run attempts (136 complete and one preserved pre-decoding editor failure), 2,761 actual inference requests, and 107,388 generated tokens. Combined with the first milestone: 166 complete runs, one failed attempt, 3,999 calls, and 149,296 generated tokens. No repair has met the declared eligibility/confirmation gates. The separate native diagnostic prompt/embedding ledger totals 582 tokens.

A full integrity pass checked 65 distinct current model-file hashes, including originals and derived artifacts. This snapshot began before the six cap-correction runs; those runs used the same already verified binaries and independently verified their loaded hashes. A fresh final request/ledger audit will include every corrected run. The initial full-file audit is retained, not overwritten. The final comparison audit additionally checks exact initial request/runtime equality across primary weight arms and records per-case strict/payload composition interactions, avoiding conclusions from aggregate zero or floor effects alone.

Reproduction documentation now distinguishes the GGUF Python source from the working Vulkan native build, full model versus cached-base patch size, ordinary PTQ versus QAD lineage, original grading versus post-hoc sensitivity, and all analysis versions. First-milestone findings/status are archived rather than lost. All owned model servers are stopped; no further inference is queued in this bounded portfolio.

### Final release checks

Final request/ledger audit includes all 167 attempts: 166 complete, one preserved failed run, 3,999 calls and 149,296 generated tokens; no pending generated/scored reservations. All 42 primary request/runtime comparison pairs pass equality checks. Three Fable payload finite-difference interactions are retained alongside the strict-score floor; one positive interaction simply reflects two edits losing the same original success and is not an improvement. The final reports explain each case rather than equating interaction sign with utility.

All 34 tests pass. Final PNG/SVG figures were rendered and visually inspected; the diagnosis footnote was moved to avoid overlap. Local links in the main documentation and final report set resolve, and `git diff --check` passes. The machine-readable completion record distinguishes completed bounded pilots from the unimplemented stronger paper-level arms. Every owned model server is stopped. Final source, reports, data-sidecars, figures and audits are being published to the authorized remote; model binaries remain local and ignored.

### 2026-11-05 — Next-experiment planning after the completed portfolio

The user requested a plan for the best next experiments. Reviewed the completed five-pilot design, progress, completion manifest, failed native-readout gate and per-case comparison audit. Wrote `NEXT_EXPERIMENTS.md` with three immediate research bets (verified conditional response repair, compact Fable adapter recovery, and operator-level speed diagnosis) plus two gated extensions (real-patch quantized deployment and evidence-sensitive clarification). No new target inference, model download, training, patch construction or benchmark execution occurred during planning; existing token ledgers and completion results are unchanged.

Primary literature was checked on 2026-11-05: Steer2Edit v3 (2026-11-04), AlphaEdit, BitDelta, DeltaZip, Delta-CoMe, Composable Interventions, RepoBench, the Fable model card and the motivating weight-space report. Preservation projection, component rank-one editing and delta compression are existing techniques; the roadmap does not call them new inventions. The Fable card reports rank-32 LoRA; this motivates approximate adapter recovery but does not establish an exactly rank-32 residual or the historical parent. No original factor-recovery identity is claimed.

Read-only local inspection found both `llama-debug` and `llama-cvector-generator` in the working Vulkan binary directory. The separately inspected llama.cpp source exposes evaluation callbacks and native logit APIs, and upstream debug documentation describes filtered tensor/logit traces. Compatibility with the working build remains untested. Its CMake cache refers to an unavailable source directory (`/media/akmessi/250 GB Volume/ai/llama.cpp`); any necessary instrumentation build must recover and isolate the actual runtime revision rather than modify the working installation. The next proposed gate captures state/logits in one forward execution, starting with CPU, instead of loosening the failed embedding-route tolerance.

Planning caps: at most 575,000 additional generated and 300,000 additional diagnostic/scored tokens, with explicit stage prerequisites and no automatic expenditure. Exact datasets/arms and maximum-token arithmetic must be frozen before execution. No training, paid compute, unattended editor API or subagents are included. Readme/progress links distinguish the new plan from the unchanged completed pilot.

### P3 execution authorized; exact native source recovered

The user authorized execution of the next roadmap and requested persistence to completion. Read the operating rules, original design, next roadmap, completed progress and evidence checkpoint. The working debug binary matches native build 11942 / commit 338fce651. A first git fetch using the abbreviated hash failed because it was not a remote ref; resolving through the official GitHub commit API returned full revision `338fce6511c744893cc0d3d958b275f40552c7a4`, which was fetched and checked out in ignored `local/runtime-source`. These read-only/preparation attempts used no target-model tokens. No installed runtime or original model was modified.

Frozen `p3-native-gate-v1` and 24 visible prefixes before target inference. The new bounded native callback tool uses exact-version headers and existing shared libraries, retains a same-forward readout and native logits, includes no-callback and known-vector causal controls, and charges diagnostic prompt tokens separately. All large raw arrays stay local with published hashes. This avoids comparing two different forward paths as if their states were identical. Further experiment protocols will be frozen sequentially before their own model feedback.

### Native client build correction (zero model calls)

The initial compile failed because `ggml_backend_load_all_from_path` needs explicit `-lggml` linkage in addition to llama/ggml-base. The shell subsequently attempted the gate, which stopped before model loading because no client binary existed. No diagnostic reservation or model call was made. The first CPU attempt directory is preserved with a preflight-failure summary. Added a checked, manifested build step with the missing link dependency and a separately named CPU retry; source/runtime semantics are unchanged.

### Same-forward CPU arithmetic identified; gate v2 declared

V1 captured all 28 CPU requests with equal top-1, but most centered errors narrowly exceeded 0.001. The control-summary serializer then rejected a NumPy boolean; raw responses and reservations were already durable and were pushed with the failed summary. Offline inspection of six saved states plus the exact CPU kernel source identifies the reason: F16 weight vec-dot converts its activation input to F16 before multiplication. Reconstructing that arithmetic reduces observed error from ~0.001–0.0026 to ~3e-6–7e-6. This is not a relaxed threshold. Declared v2 uses the actual CPU operator, preserves v1's incorrect-equation result, accounts for rounding in the causal shift and fixes summary boolean serialization before a fresh gate repeat.

The Fable adapter numerical stage and tests are also frozen. It packages declared Q/K/V deltas as fixed-rank factors or low-bit values, retains only tiny explicit residual corrections, verifies the matching base on merge, and never trains the target. Spectra and packages precede model-quality selection. Two meaningful numerical tests bring the suite to 36 passing tests.

### Measurement gate passed; numerical adapter structure observed

CPU and Vulkan v2 each passed all 26 same-forward boundary checks and four controls. No-callback native logits were bit-identical on the two tested prefixes per backend; fixed-vector interventions had the predicted readout effect. This validates those tested native boundaries/settings, not arbitrary hidden layers or all prompts. The correctly reconstructed CPU input rounding is kept explicit. All original v1 failures remain available.

Across all18 aligned Fable Q/K/V differences, rank32 captures more than99.99% of squared singular-value energy. This supports a compact-adapter hypothesis, not a behavioral preservation claim. The unaligned and within-source-rounded spectra remain separate controls. A fresh pinned repository-disjoint next-line stage and48-case extraction/code/explanation protection set are frozen before candidate quality calls. Nine candidate encodings remain fixed, calibration selects before confirmation, and the adapter stage's maximum-output arithmetic fits80000 tokens. PyArrow23.0.1 was installed only in the existing isolated environment to read the32MB pinned public parquet; no system packages or target training were changed.

### 2026-11-05T17:59:17.992823+00:00 — p3-adapter-repo-dev-fable_f16-s17

Status `complete`, 3405 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-dev-fable_f16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T17:59:47.537754+00:00 — p3-adapter-repo-dev-parent_uniform_bf16-s17

Status `complete`, 2676 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-dev-parent_uniform_bf16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:00:01.096772+00:00 — p3-adapter-repo-cal-fable_f16-s17

Status `complete`, 987 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_f16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:00:13.995850+00:00 — p3-adapter-repo-cal-fable_svd4-s17

Status `complete`, 982 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd4-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:00:28.249715+00:00 — p3-adapter-repo-cal-fable_svd8-s17

Status `complete`, 1023 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd8-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:00:41.159578+00:00 — p3-adapter-repo-cal-fable_svd16-s17

Status `complete`, 984 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:00:54.067305+00:00 — p3-adapter-repo-cal-fable_svd32-s17

Status `complete`, 975 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd32-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:01:07.271768+00:00 — p3-adapter-repo-cal-fable_svd64-s17

Status `complete`, 987 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd64-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:01:20.421068+00:00 — p3-adapter-repo-cal-fable_svd32_bf16-s17

Status `complete`, 1033 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_svd32_bf16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:01:33.449537+00:00 — p3-adapter-repo-cal-fable_int8-s17

Status `complete`, 987 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_int8-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:01:46.480109+00:00 — p3-adapter-repo-cal-fable_int4-s17

Status `complete`, 1005 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_int4-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:01:58.986999+00:00 — p3-adapter-repo-cal-fable_sign-s17

Status `complete`, 908 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-cal-fable_sign-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### Adapter selected before confirmation; fidelity limitation and companion declared

Calibration selected rank16 at921223 bytes:0/24 exact next lines, matching original Fable's0/24, and22/24 extracted first-line matches (91.7%). This follows the frozen rule but cannot prove preserved coding utility at this floor. Both Fable and parent were0/80 on the shortened repository development panel. No repository advantage is established; later reports must retain that limitation.

Before opening confirmation, declared a fixed rank32 numerical-fidelity companion (1.83MB), separate from the selected rank16. It will compare full output/first-line agreement and protected skills, plus held-out native distribution fidelity, using the predeclared reserve rather than silently increasing the adapter stage budget. The original rank16 selection remains primary and cannot be replaced using confirmation feedback.

Frozen the Qwen conditional JSON-switching development/confirmation families, fixed controls and hypothesis gates before its first P3 quality inference. Grader9 adds only the explicit plain/fenced JSON contract; prior semantics are unchanged. Added a separately versioned native client supporting bounded greedy causal generation; the original gate client/source remain preserved. A checked, separate build and untouched-server parity controls precede activation interventions. Thirty-eight tests pass.

### 2026-11-05T18:05:55.794267+00:00 — p3-adapter-repo-confirm-fable_f16-s17

Status `complete`, 11031 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-confirm-fable_f16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:07:17.153640+00:00 — p3-adapter-repo-confirm-parent_uniform_bf16-s17

Status `complete`, 8053 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-confirm-parent_uniform_bf16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:08:55.255912+00:00 — p3-adapter-repo-confirm-fable_svd16-s17

Status `complete`, 11119 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-repo-confirm-fable_svd16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:09:15.235170+00:00 — p3-adapter-protection-fable_f16-s17

Status `complete`, 1679 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-protection-fable_f16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:09:32.910279+00:00 — p3-adapter-protection-fable_svd16-s17

Status `complete`, 1679 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-adapter-protection-fable_svd16-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:11:11.372592+00:00 — p3-reserve-rank32-companion-fable_svd32-s17

Status `complete`, 11161 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-rank32-companion-fable_svd32-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:11:29.704032+00:00 — p3-reserve-rank32-protection-fable_svd32-s17

Status `complete`, 1679 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-rank32-protection-fable_svd32-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:12:54.899732+00:00 — p3-repair-switch-dev-original-qwen_float-s17

Status `complete`, 1266 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-original-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:14:21.932367+00:00 — p3-repair-switch-dev-strong-qwen_float-s17

Status `complete`, 1270 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-strong-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### Component trace mapping and separate fidelity diagnostics frozen

Source inspection found that the LFM2 callback named ffn_out is attached to the normalized FFN input pointer. A debug name alone cannot establish an activation boundary. Native v4 instead selects the real MUL_MAT by its named FFN-down weight operand and saves the corresponding input/output, with algebra and no-op checks required before editing. This is a tracing adaptation; installed runtime/originals remain unchanged. LFM3's optional post-normalization remains a separate architectural issue.

Added exact supplied token IDs to the client so Fable's add-BOS policy is not guessed. Frozen the fixed16 Fable fidelity prefixes (first8 and last8 protection cases) and tokenization provenance before their forwards. Native diagnostic/generated token costs now have explicit stage counters and a generation evidence index for the final audit. The same development switching gate is declared for Fable if Qwen has no natural failure; no Qwen settings will be changed to create one.

### 2026-11-05T18:18:55.699903+00:00 — p3-repair-switch-dev-grammar-qwen_float-s17

Status `complete`, 1270 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-grammar-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### Lossless residual packaging follow-up declared before measuring sizes

Approximate delivery changes some greedy generations even when singular-value energy is nearly retained: primary rank16 matched196/240 complete original outputs and204/240 first lines; the fixed rank32 companion matched218/240 complete outputs and220/240 first lines. All three models were0/240 exact repository next lines on this shortened panel, so no coding advantage is established. The protected code-only functional content was12/12 for original,rank16 andrank32; preservation claims remain bounded by the small task set and output caps.

This motivates a separately declared zero-forward packaging supplement: fixed rank16/32/64 predictors followed by exact lossless residuals to the original Fable file. To avoid BLAS-dependent intermediate hashes, added an exact dyadic integer predictor with explicit overflow/exact-conversion bounds and scalar-reference tests. Source originals and previous approximate artifacts remain unchanged. The combined factor+residual bytes and full target SHA will determine whether this gives a useful exact package, not cherry-picked model outputs. This uses established prediction/residual coding, not a new learning algorithm. Forty tests pass.

### 2026-11-05T18:23:06.360749+00:00 — p3-repair-switch-dev-original-fable_float-s17

Status `complete`, 1396 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-original-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:23:26.065959+00:00 — p3-repair-switch-dev-strong-fable_float-s17

Status `complete`, 1472 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-strong-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:23:45.259614+00:00 — p3-repair-switch-dev-grammar-fable_float-s17

Status `complete`, 1344 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-switch-dev-grammar-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### Exact delivery result and naturally eligible repair subject

The first three lossless capsules reconstructed the complete original Fable hash. Combined factor+residual sizes: rank16 18,116,720 bytes; rank32 5,958,793 bytes; rank64 8,189,207 bytes. Rank32 is3.41x smaller than the previous20.3MB exact patch, but misses the aspirational5MB target. This is an exact cached-base delivery result, not an improved model or standalone6MB model. Deterministic integer prediction avoids BLAS-dependent intermediate identity.

The new switching gate gives Qwen64/64 correct contracts, so it is a correct ineligible control. Original Fable has64/64 correct payloads,19/32 correct plain contracts and32/32 correct fenced contracts: a genuine style failure with content competence. Its stronger prompt worsened plain formatting to0/32; schema control reached32/32 plain and32/32 fenced. Fable is the naturally eligible subject, with grammar a strong practical baseline.

Fixed16-prefix native fidelity retained top1 on all16 contexts for both approximate packages; mean reference-to-candidate KL is0.00607 for rank16 and0.00178 for rank32. These descriptive results do not erase generation differences or the coding score floor. The exact capsule restores the original byte-for-byte instead of claiming the approximation is identical.

Before component tracing/search, froze exact development prefixes, native/server generation parity controls and explicit MLP formulas/budgets. True operation-based input/output mapping is verified algebraically before edits; no misleading debug label is accepted as proof.

### CPU component fallback declared

Observation: Vulkan Fable prefill FFN projections fail the frozen relative-error gate, while the final FFN projection and readout pass. Hypothesis: the matrix-prefill path's reduced-precision accumulation differs from the single-column path. We have not established this mechanism. The predeclared CPU fallback now tests four prefixes before any full CPU trace collection, with unchanged tolerances and source-motivated input rounding. Failed GPU checks remain immutable.

### CPU-only lifecycle correction

The four-request fallback completed but its analysis failed on missing early-layer component captures. The log showed Vulkan compute buffers despite zero GPU layers: default operation offload was still enabled. This is a mixed-backend attempt, not CPU-only evidence. The v5 client exposes explicit op_offload/offload_kqv flags; the fresh v2 fallback disables both. The earlier final-readout CPU-labeled gate likewise used zero layers without these explicit flags, so its readout algebra is valid for that configuration but cannot establish that every graph operation was CPU-only. No requests are discarded or recharged.

Implementation preflight: the first v5 edit command used unavailable `python` instead of `python3`; no files were edited and no new native calls occurred. Subsequent missing-source build/immutable-directory errors were preflight only. Retry uses the same not-yet-created binary and a new build-evidence directory.

### Causal feasibility observed; fixed permanent search opens

On the predeclared first8development pairs, CPU-derived layer6 direction at0.5 and1.0 rescues all4baseline plain-format failures, with16/16payload and8/8fenced correctness each. Dose2.0 damages content/format (2/16payload). These are reused development cases, not confirmation. The12fixed analytic candidates now proceed. Per-step native evidence and generated/scored charges are retained. The test invocation initially tried unavailable pytest; the actual existing unittest suite passes40tests. Two analytic ridge invariants are now added before exporting candidates. No gradients/training.

### 2026-11-05T18:44:27.534436+00:00 — p3-repair-candidate-screen-component_0.005-s17

Status `complete`, 524 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-component_0.005-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:44:42.247760+00:00 — p3-repair-candidate-screen-component_0.01-s17

Status `complete`, 508 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-component_0.01-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:44:54.022441+00:00 — p3-repair-candidate-screen-component_0.025-s17

Status `complete`, 504 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-component_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:45:11.638571+00:00 — p3-repair-candidate-screen-component_0.05-s17

Status `complete`, 1686 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-component_0.05-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:45:23.024441+00:00 — p3-repair-candidate-screen-rank1_0.005-s17

Status `complete`, 524 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-rank1_0.005-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:45:35.025747+00:00 — p3-repair-candidate-screen-rank1_0.01-s17

Status `complete`, 524 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-rank1_0.01-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:45:47.062441+00:00 — p3-repair-candidate-screen-rank1_0.025-s17

Status `complete`, 504 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-rank1_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:46:01.190407+00:00 — p3-repair-candidate-screen-rank1_0.05-s17

Status `complete`, 504 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-rank1_0.05-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:46:11.508111+00:00 — p3-repair-candidate-screen-protected_0.005-s17

Status `complete`, 524 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-protected_0.005-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:46:21.064967+00:00 — p3-repair-candidate-screen-protected_0.01-s17

Status `complete`, 516 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-protected_0.01-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:46:30.126817+00:00 — p3-repair-candidate-screen-protected_0.025-s17

Status `complete`, 504 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-protected_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:46:39.414589+00:00 — p3-repair-candidate-screen-protected_0.05-s17

Status `complete`, 504 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-screen-protected_0.05-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:47:00.380011+00:00 — p3-repair-candidate-dev-component_0.025-s17

Status `complete`, 1352 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-dev-component_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:47:19.365908+00:00 — p3-repair-candidate-dev-protected_0.025-s17

Status `complete`, 1344 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-dev-protected_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:47:37.088181+00:00 — p3-repair-candidate-dev-protected_0.05-s17

Status `complete`, 1344 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-candidate-dev-protected_0.05-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:48:44.300183+00:00 — p3-repair-confirmation-original-fable_float-s17

Status `complete`, 6496 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-confirmation-original-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:49:51.548966+00:00 — p3-repair-confirmation-strong-fable_float-s17

Status `complete`, 6927 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-confirmation-strong-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:50:58.779713+00:00 — p3-repair-confirmation-grammar-fable_float-s17

Status `complete`, 6336 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-confirmation-grammar-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:52:03.406170+00:00 — p3-repair-confirmation-patch-protected_0.025-s17

Status `complete`, 6725 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-confirmation-patch-protected_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:52:23.999178+00:00 — p3-repair-repair-protection-fable_float-s17

Status `complete`, 2056 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-repair-protection-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T18:52:43.904620+00:00 — p3-repair-repair-protection-protected_0.025-s17

Status `complete`, 2059 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-repair-protection-protected_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### Shared-desktop timing limitation and remaining execution order

CPU timing varies materially in the current three-session study; unrelated browser/desktop processes are active on the workstation. We do not stop or modify unrelated user processes. Agent numerical builds/SVD/export/compression work is excluded from timed CPU runs; lightweight documentation/inspection continues. The results describe this shared workstation/runtime, not a controlled idle-device benchmark or general hardware ranking. All measured requests, warmups, actual token counts, caches and per-condition resources remain in the immutable timing evidence. Profiling includes scheduler/initialization work and serializing overhead, so operator totals are not presented as pure per-generated-token wall-time decomposition.

The frozen preservation-aware patch protected_0.025 changes36537values and passed32/32plain plus32/32fenced development. On fresh confirmation it gets63/120plain versus45/120original, but0/120fenced versus7/120original. It fails the conditional-repair preservation requirement. Protected code content stays12/12, while requested explanations are affected and truncated; do not reinterpret the patch as a successful reliable repair. All fixed random/attenuation and sampling checks will still finish. The recovered rank32adapter is frozen independently; its precision study tests limited delivery fidelity rather than repair retention.

Remaining finite queue: finish randomized original/Qwen timings,3hybrid timing arms and fresh function quality; diagnostic repair controls; LFM3QAD same-forward access; clarification development gates/controls and independent field-layout confirmation if gates close; rank32adapter precision allocation/calibration/final comparisons. Then run full evidence/cost/original-identity audits, figures/report and final publication. No target training, paid service, external editor or additional model download is authorized or used.

### 2026-11-05T19:22:20.918065+00:00 — p3-reserve-format-quality-fable_float-s17

Status `complete`, 2381 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:25:24.552893+00:00 — p3-reserve-format-quality-fable_Q8_0-s17

Status `complete`, 2339 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-fable_Q8_0-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:27:23.026755+00:00 — p3-reserve-format-quality-fable_Q4_0-s17

Status `complete`, 2405 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-fable_Q4_0-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:29:22.174876+00:00 — p3-reserve-format-quality-fable_Q4_K_M-s17

Status `complete`, 2380 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-fable_Q4_K_M-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:32:16.613782+00:00 — p3-reserve-format-quality-hybrid_head-s17

Status `complete`, 2410 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-hybrid_head-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:32:33.899835+00:00 — p3-reserve-format-quality-hybrid_head_ffn-s17

Status `complete`, 1979 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-hybrid_head_ffn-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:32:50.744600+00:00 — p3-reserve-format-quality-hybrid_head_shortconv-s17

Status `complete`, 2221 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-quality-hybrid_head_shortconv-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:33:14.093129+00:00 — p3-reserve-format-noop-hybrid_noop-s17

Status `complete`, 381 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-reserve-format-noop-hybrid_noop-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:33:35.272252+00:00 — p3-repair-random-control-random_control-s17

Status `complete`, 528 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-random-control-random_control-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:33:46.587527+00:00 — p3-repair-attenuation-control-attenuation_control-s17

Status `complete`, 528 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-attenuation-control-attenuation_control-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:33:59.810457+00:00 — p3-repair-sampling-stability-fable_float-s17

Status `complete`, 654 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-sampling-stability-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:34:10.368488+00:00 — p3-repair-sampling-stability-fable_float-s29

Status `complete`, 665 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-sampling-stability-fable_float-s29`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:34:20.776836+00:00 — p3-repair-sampling-stability-protected_0.025-s17

Status `complete`, 656 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-sampling-stability-protected_0.025-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:34:31.364806+00:00 — p3-repair-sampling-stability-protected_0.025-s29

Status `complete`, 673 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-repair-sampling-stability-protected_0.025-s29`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:35:55.976017+00:00 — p3-clarify-evidence-dev-original-lfm3_qad-s17

Status `complete`, 1594 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-original-lfm3_qad-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:36:32.932798+00:00 — p3-clarify-evidence-dev-strong-lfm3_qad-s17

Status `complete`, 945 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-strong-lfm3_qad-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:37:11.803525+00:00 — p3-clarify-evidence-dev-grammar-lfm3_qad-s17

Status `complete`, 1341 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-grammar-lfm3_qad-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:37:30.298289+00:00 — p3-clarify-answerability-lfm3_qad-s17

Status `complete`, 468 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-answerability-lfm3_qad-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:38:37.368118+00:00 — p3-clarify-evidence-dev-original-qwen_float-s17

Status `complete`, 1780 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-original-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:39:47.324829+00:00 — p3-clarify-evidence-dev-strong-qwen_float-s17

Status `complete`, 1418 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-strong-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:40:51.037453+00:00 — p3-clarify-evidence-dev-grammar-qwen_float-s17

Status `complete`, 1780 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-grammar-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:41:17.217723+00:00 — p3-clarify-answerability-qwen_float-s17

Status `complete`, 612 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-answerability-qwen_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:41:44.145234+00:00 — p3-clarify-evidence-dev-original-fable_float-s17

Status `complete`, 3026 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-original-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:42:12.580629+00:00 — p3-clarify-evidence-dev-strong-fable_float-s17

Status `complete`, 3139 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-strong-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:42:34.649295+00:00 — p3-clarify-evidence-dev-grammar-fable_float-s17

Status `complete`, 2027 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-evidence-dev-grammar-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.

### 2026-11-05T19:42:47.906400+00:00 — p3-clarify-answerability-fable_float-s17

Status `complete`, 864 generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `runs/p3-clarify-answerability-fable_float-s17`. Interpretation is deferred to the study report; this run alone does not establish a repair.
