# Research completion status

Updated 2026-11-05. **All five bounded pilot tracks executed and reported; no accepted general behavioral repair.** The original paper-level method goals remain unproved and several stronger arms are explicitly unimplemented after failed prerequisites.

The full record now contains **166 complete inference runs and one failed attempt**, **3,999 actual inference requests** (3,998 responses and one known pre-decoding rejection), and **149,296 generated tokens**. This includes the original 30-run milestone and the 137-run continuation phase. The continuation used 107,388 generated tokens and 2,761 calls, within its 150-run / 250,000-token envelope. Real two-turn conversations count as two inference calls, not one.

The separate native-readout diagnostic settled **582 prompt/embedding tokens**. Every generated-token reservation is settled. Twenty unattempted cases in the partial editor trial were subsequently included in the full unchanged 24-case retry; they were not silently discarded.

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
