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
