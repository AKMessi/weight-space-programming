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
