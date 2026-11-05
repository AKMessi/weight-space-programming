# Research progress

Updated 2026-11-05. Measurement pipeline implemented; no inference result yet.

## Active milestone

Build a reproducible local inference/evaluation pipeline and baseline report for Qwen3-0.6B, LFM3-1.7B QAD, and Fable 230M, preserving the matched Fable parent as a control.

## Completed

- Five-study design and model metadata recorded.
- Remote repository checked: accessible over SSH, with no advertised refs at initialization.
- Git repository initialized locally; coherent changes will be committed and pushed.
- Existing Vulkan runtime and LFM3 artifacts located without modifying the installation.
- Raw-evidence inference runner, discovery suite, exact graders, and conservative budget ledger implemented.
- Eight meaningful grader/budget tests passed.

## Next actions

1. Inventory actual model artifacts and working inference runtime.
2. Implement append-only run evidence, deterministic discovery fixtures, exact graders, and budget accounting.
3. Validate graders and run real short baselines before editing weights.
4. Document and investigate observed behaviors; verify artifact editing separately.

## Guardrails

The 1.8M-token envelope is a ceiling for local pilot generation, not an automatic execution target. Experiments remain subject to evidence/compatibility gates. All outputs first collected here are exploratory unless explicitly frozen otherwise.
