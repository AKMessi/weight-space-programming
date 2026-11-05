# Research progress: completed pilot and active P3 program

Updated 2026-11-05. **All five bounded pilot tracks executed and reported; no accepted general behavioral repair.** The original paper-level method goals remain unproved and several stronger arms are explicitly unimplemented after failed prerequisites.

The full record now contains **166 complete inference runs and one failed attempt**, **3,999 actual inference requests** (3,998 responses and one known pre-decoding rejection), and **149,296 generated tokens**. This includes the original 30-run milestone and the 137-run continuation phase. The continuation used 107,388 generated tokens and 2,761 calls, within its 150-run / 250,000-token envelope. Real two-turn conversations count as two inference calls, not one.

The separate native-readout diagnostic settled **582 prompt/embedding tokens**. Every generated-token reservation is settled. Twenty unattempted cases in the partial editor trial were subsequently included in the full unchanged 24-case retry; they were not silently discarded.

## Active P3 execution

The user authorized the next roadmap. Exact-version same-forward state/logit gates now pass CPU and Vulkan, including no-op and causal controls. Precision-controlled spectra and all nine recovered Fable packages are built; the rank32 package is about1.83MB, but behavioral fidelity is not yet established. A pinned repository-disjoint evaluation (80 development,24 calibration,240 confirmation cases) and48 protection cases are frozen before quality inference. See [P3 protocols](protocols/p3-native-gate-v2.md), [adapter protocol](protocols/p3-fable-quality-v1.md) and the journal. The historical completion totals below describe the previous portfolio, not the ongoing program.

## Next plan

The [next-experiment roadmap](../NEXT_EXPERIMENTS.md) was written after this completed pilot. It prioritizes a verified causal repair, precision-aware Fable adapter recovery, and the measured speed anomaly, with quantized deployment and clarification as gated extensions. The roadmap is now authorized for execution; the previous completion record remains historical evidence.

## What each study achieved

| Study | Executed evidence | Decision |
|---|---|---|
| 1. Repair habits, preserve useful behavior | Three model baselines, A/B/random perturbations, Fable rollback/parent controls, functional code checks, frozen candidate decisions, fresh two-seed confirmation and matched prompt/schema controls | No eligible or confirmed repair; no demonstrated preservation of repository-level specialization |
| 2. Ask, then answer when evidence arrives | Complete/absent/conflicting evidence, actual two-turn exchanges, fixed edits, prompt/schema controls, deterministic checker, fresh field/layout confirmation | No weight repair; format correctness alone did not prevent invented values |
| 3. Quantization and deployment | Same-source floating/Q8/Q4 original/A arms for all three, untouched-F32 norm verification, separate QAD no-op parity, five exact binary patch roundtrips, controlled four-format timing follow-up | Stress and utility work complete; natural useful-repair retention and precision allocator failed their useful-patch prerequisite |
| 4. Combine changes | Original/A/B/random/AB arms on engineering and joint-evidence panels, exact AB=BA files, six matching order probes per model, corrected matched output caps | Diagnostic composition complete; two independently useful repairs were not available, so useful-repair composition is unproved |
| 5. Blind diagnosis | Nine screened healthy/fault states, eighteen original/permuted target runs, numerical and native parity checks, twelve retained presentations, two evidence arms, full local-editor retry, baselines/confidence/false alarms | Local bounded-summary pilot complete; weights-only editor tied majority accuracy and flagged every healthy presentation. No frontier/full-weight diagnosis or natural-fault claim |

The native final-state compatibility gate also ran on all three floating models: equal next-token top-1 in six contexts, but **0/6 passed the declared centered-logit tolerance**. No directional edit was built from that unverified readout. This does not establish that activation access is impossible.

## Artifacts and checks

- [Five reports and machine-readable results](portfolio-v3/README.md), with immutable derived regrades identifying every changed grade and raw-source checksum.
- [Plain-language findings](FINDINGS.md), [exact checkpoint utility](EXACT_PATCH_DELIVERY.md), [timing/storage/symmetry figures](portfolio-v3/supplement/README.md), and [reproduction guide](../docs/REPRODUCING.md).
- One-model execution, original-file preservation, exact BF16 decoding, streaming GGUF rewrites, controlled quantization, XOR patch reconstruction, recursive artifact rebuilds and full raw evidence.
- **34 tests pass** in the inspection environment. The integrity audit reconciles frozen inputs, cases, follow-up messages, usage, failures and the reservation ledger.
- A full-file integrity pass checked **65 unique current model artifacts**, including originals and derived files. Corrected cap reruns reused those exact artifacts and independently checked each loaded hash. The final request/ledger audit is [final-v2](evidence-integrity-final-v2.json); the full-file snapshot is [final-v1](evidence-integrity-final-v1.json).
- Every coherent source/evidence change is committed and pushed to the user-authorized remote. Model/patch binaries stay ignored; source provenance and exact checksums are public research evidence.

The bounded portfolio is complete as an auditable local pilot. It is not five successful methods or five ready papers. Preservation-constrained directional editing, behavior-guided precision allocation, a validated repair library and a capable interactive external editor remain beyond the evidence here. Those are stronger experiments requiring a separately verified activation path, meaningful task competence and fresh independent evaluation; they are not silently counted as done.

## Corrections retained in the record

The initial Fable FFN difference was confounded by effective precision; an old copy recovery failed fresh validation. New v1 schema wrappers were malformed, then preserved and rerun with corrected v2 transport and an adversarial enforcement diagnostic. Code interpretation/extraction errors were fixed through disclosed derived grades without retuning frozen selections. The editor needed a full-batch 4096-context retry. An early report build encountered legacy unversioned metadata. The composition output-cap mismatch was corrected in six versioned runs. All original data, failures and intermediate analyses remain available.

No target training, paid compute, external editor API or unattended experiment was used. All owned model servers have been shut down. Earlier status is preserved in [the first-milestone record](PROGRESS_FIRST_MILESTONE.md).

## P3 continuation — 2026-11-06

Active execution of NEXT_EXPERIMENTS.md; historical completion totals above remain unchanged. Measurement gate passed same-forward readout checks; real Fable component path now verified on CPU-only after preserving failed Vulkan-prefill and accidental mixed-backend attempts. The CPU-derived layer6 reversible intervention rescues4/4natural plain-format failures at each of0.5/1.0doses in the fixed8-pair development screen;2.0damages output. Permanent12candidate search is active, not confirmed.

Adapter branch: rank16 calibration-selected package921223bytes has85%firstline agreement on240confirmation examples; preregistered rank32companion1830620bytes has91.7%. All original/parent/reconstructions0/240exact next lines under the lossy short-context RepoBench adaptation. Code-content12/12passes are a separate small panel, not proof of specialist advantage. Fixed-prefix meanKL original→rank16 .00607, rank32 .00178. Exact low-rank-predictor+residual capsule rank32 totals5958793bytes and restores the original full-file SHA; base required. No new compression algorithm or paper claim.

Four-format operator profiles complete. Quantized Vulkan matrix-vector paths are slow across shapes, including readout and FFN; no support yet for a specifically3Dshort-conv mechanism. Matched CPU/GPU timing and3fixed hybrids follow. Quantized-patch deployment and evidence-sensitive clarification remain pending; the program is not complete.
