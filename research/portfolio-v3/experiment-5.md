# Experiment 5 — Blinded diagnosis and equivalent layouts

Completed a bounded local-editor pilot. The editor receives anonymized summaries/small raw samples or two fixed target outputs in a fresh request per case. It never receives filenames, target hashes, a healthy reference, screening scores or expected labels. This is not frontier-LLM full-weight tool use and does not replicate the motivating report.
The full batch uses context 4096 after a tokenization preflight (maximum prompt 2581 tokens). The original 2K attempt completed three cases and rejected a fourth before decoding; its raw evidence and known-zero failed-request settlement remain in the ledger. The entire unchanged batch was repeated, rather than combining contexts or dropping the long case.

| Model | State | Healthy payload / 12 | Variant payload / 12 | Clear fault? | Numerical parity | Runtime parity |
|---|---|---:|---:|---|---|---|
| qwen | healthy | 12 | 12 | True | True | False |
| qwen | output_sign | 12 | 0 | True | True | False |
| qwen | ffn_sign | 12 | 0 | True | True | False |
| lfm3 | healthy | 12 | 12 | True | True | True |
| lfm3 | output_sign | 12 | 0 | True | True | False |
| lfm3 | ffn_sign | 12 | 0 | True | True | False |
| fable | healthy | 8 | 8 | True | True | True |
| fable | output_sign | 8 | 0 | True | True | False |
| fable | ffn_sign | 8 | 0 | True | True | True |

## Diagnosis results

| Evidence | Correct | Healthy false alarms | Layout agreement | Confidence Brier |
|---|---:|---:|---:|---:|
| weights | 7/12 | 5/5 | 3/3 | 0.24479166666666666 |
| behavior | 8/12 | 0/5 | 3/3 | 0.40335 |

Deterministic baselines: {"behavior_probe": {"correct": 12, "n": 12}, "weight_anomaly": {"correct": 8, "n": 12}}. The weight baseline flags a negative output-normalization mean; the behavior baseline uses exact supplied-fact answer checks on the two exposed probes.
Majority-class baseline: 7/12; class counts {'healthy': 5, 'degraded': 7}. Accuracy above 50% is not automatically better than this baseline.

Faults were screened before editor inference using a declared 25pp payload drop. Failed numerical/runtime layouts and perturbations without that decline were explicitly excluded from binary accuracy. Original/permuted presentations are correlated, and there are at most nine independent model-state combinations. High agreement can come from a constant diagnosis; read it with accuracy and false alarms.

Output-normalization sign reversal is conspicuous; FFN sign reversal is a different stress case. These injected changes are not natural failures or unique behavioral circuits. Summaries partly discard layout information by construction, while raw samples change under permutation; this measures only the limited evidence given. No diagnostic repair was executed, no external API editor was used, and confidence statistics are descriptive on this tiny set. Full case mapping, exclusions, frozen prompts and all editor outputs are retained.

Code scoring uses post-hoc derived grader v7 (v6 functional rules); original recorded grades, frozen selections and raw outputs are preserved. The [derived sidecars](regrades/) list every change with its source checksum. This correction is sensitivity analysis, not an independent prespecified confirmation.
