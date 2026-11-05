# Research progress

Updated 2026-11-05. **First implementation and experimental milestone completed. Five-study portfolio ongoing; no accepted model repair.**

Read [findings](FINDINGS.md) for the plain-language results and [journal](JOURNAL.md) for the full decision record, negative results, and corrections.

## Completed and verified

- Repository initialized and pushed to the user-authorized GitHub remote; coherent code/evidence changes are committed and pushed.
- Hardware/runtime inventory, pinned downloads/checksums, one-model Vulkan execution, durable raw evidence, exact prompt rendering, conservative token ledger, and source/config/data provenance implemented.
- All three deployments characterized: Qwen3-0.6B Q8, LFM3-1.7B QAD Q4, and Fable 230M F16; ordinary LFM3 Q8 and Fable's pinned public parent included as controls.
- Separate-file floating GGUF edits, bit-identical identity check, BF16 decoding/alignment, per-tensor audits, rollback baselines, and Q/K/V-family ablations implemented and exercised.
- Strong prompt and schema controls measured. All attempted edits and negative results retained.
- Fresh validation protocol/data/arms committed before inference; ten validation runs completed and the promising copy-recovery interpretation rejected.
- Versioned code-grader errors corrected through separate derived regrades; original evidence and frozen analysis preserved.
- **30 complete runs, 1,238 requests, 41,908 generated tokens.** Evidence/ledger consistency verified; every request settled. **19 tests pass.**

## Five-study status

| Study | Current evidence | Gate / next work |
|---|---|---|
| 1. Preserve skills while repairing habits | Fable rollback discovery, controls, precision audit, and frozen copy validation completed | No accepted repair. Establish genuine coding utility; improve varied development cases before new patch search |
| 2. Ask when information is missing | Small baseline and prompt-control screens collected | Underlying recognition versus formatting unresolved; complete two-turn development/evaluation needed |
| 3. Repairs surviving quantization | QAD lineage and effective-precision confounds documented | No useful patch yet; no controlled quantization-repair result |
| 4. Combine repairs | Common artifact/evidence infrastructure available | Fewer than two useful independent patches; composition experiment not started |
| 5. Blind weight diagnosis | Numerical audit tooling available | Blinding, case library, symmetry controls, and editor protocol not implemented |

## Important conclusions

1. F16 storage does not guarantee matching effective precision. After explicit alignment, 105/132 Fable/parent tensors match exactly; the substantial residual differences concentrate in 18 Q/K/V matrices. Historical parent revision remains unverified.
2. V-only rollback recovered the original 4/4 copy examples but 0/24 fresh validation requests, matching original Fable's 0/24. It also regressed a development explanation proxy. Do not ship it as a general repair.
3. Schema-constrained generation is a strong development baseline. A weight method must demonstrate value beyond it where applicable.

## Next milestone

Build and freeze a richer **development** evaluation of coding utility and response switching, with task-family diversity and meaningful functional checks. Determine whether Fable has a measurable advantage worth preserving. Keep `validation-v1-frozen` out of new patch selection; reserve a newly authored confirmation set for a future candidate. Log the protocol before execution.

No unattended experiment is running at this milestone. Future work should resume from the committed state and these gates, rather than repeat the completed screens.

## Budget and scope

The 1.8M generated-token envelope is a ceiling, not an execution target. The completed runs used 41,908 tokens (about 2.3%). The remaining budget is not evidence that additional search is scientifically justified. No target-model training, paid compute, or external-editor API loop was used. Count future scored tokens and editor usage separately if those arms are introduced.
