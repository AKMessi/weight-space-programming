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
