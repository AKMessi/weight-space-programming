# Matched comparison audit

All 42 primary comparison pairs passed exact initial-request and runtime-setting checks. Second-turn contexts legitimately contain each model's own prior reply. The initial AB/BA output-cap mismatch is corrected in the final engineering arms.

| Model | Panel | Metric | Nonzero case interactions | Positive / negative | Mean interaction (pp) |
|---|---|---|---:|---:|---:|
| qwen3_q8 | engineering | success | 0/18 | 0 / 0 | +0.00 |
| qwen3_q8 | engineering | payload_correct | 0/18 | 0 / 0 | +0.00 |
| qwen3_q8 | joint_evidence | success | 0/12 | 0 / 0 | +0.00 |
| qwen3_q8 | joint_evidence | payload_correct | 0/12 | 0 / 0 | +0.00 |
| lfm3_qad | engineering | success | 0/18 | 0 / 0 | +0.00 |
| lfm3_qad | engineering | payload_correct | 0/18 | 0 / 0 | +0.00 |
| lfm3_qad | joint_evidence | success | 0/12 | 0 / 0 | +0.00 |
| lfm3_qad | joint_evidence | payload_correct | 0/12 | 0 / 0 | +0.00 |
| fable_f16 | engineering | success | 0/18 | 0 / 0 | +0.00 |
| fable_f16 | engineering | payload_correct | 3/18 | 2 / 1 | +5.56 |
| fable_f16 | joint_evidence | success | 0/12 | 0 / 0 | +0.00 |
| fable_f16 | joint_evidence | payload_correct | 0/12 | 0 / 0 | +0.00 |

Three Fable engineering cases have nonzero **payload** interactions despite zero strict success throughout that panel:

- JSON case 0: only AB includes the correct payload; its strict response still fails.
- `last_or_zero`: A alone has correct functional content; combining B loses that content.
- `positive_sum`: both individual edits lose a baseline success. Their combined failure yields a positive finite-difference interaction, **not an improvement over the original**.

This illustrates why the sign of an interaction is not the sign of a repair gain. These are reused development cases and a coarse grader; they do not identify circuits or establish useful-patch compatibility. Strict-score floors and tiny sample size limit the interpretation. All case-level values and source-response hashes are in [comparison-audit.json](comparison-audit.json).
