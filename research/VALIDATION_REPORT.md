# Frozen validation v1 results

Small frozen synthetic validation, not a general benchmark or proof of useful specialization preservation.

Protocol, fixtures, and candidate artifacts were committed before validation inference. No new dose/site was chosen on this panel. Counts aggregate decoding seeds; uncertainty groups seeds by prompt.

| Model/arm | Group | Exact contract | Payload proxy | Cutoffs |
|---|---|---:|---:|---:|
| fable_f16 | dictionary_code | 0/24 | 0/24 | 5 |
| fable_f16 | evidence_decision | 0/24 | 0/24 | 0 |
| fable_f16 | literal_copy | 0/24 | 1/24 | 20 |
| fable_f16 | new_json_fields | 0/24 | 24/24 | 0 |
| fable_v_only | dictionary_code | 0/24 | 3/24 | 0 |
| fable_v_only | evidence_decision | 0/24 | 0/24 | 0 |
| fable_v_only | literal_copy | 0/24 | 5/24 | 11 |
| fable_v_only | new_json_fields | 0/24 | 24/24 | 0 |
| fable_all_qkv_parent | dictionary_code | 0/24 | 20/24 | 0 |
| fable_all_qkv_parent | evidence_decision | 0/24 | 0/24 | 0 |
| fable_all_qkv_parent | literal_copy | 0/24 | 10/24 | 0 |
| fable_all_qkv_parent | new_json_fields | 0/24 | 24/24 | 0 |
| parent_uniform_bf16 | dictionary_code | 0/24 | 20/24 | 0 |
| parent_uniform_bf16 | evidence_decision | 0/24 | 0/24 | 0 |
| parent_uniform_bf16 | literal_copy | 0/24 | 10/24 | 0 |
| parent_uniform_bf16 | new_json_fields | 0/24 | 24/24 | 0 |
| qwen3_q8 | dictionary_code | 3/12 | 4/12 | 0 |
| qwen3_q8 | evidence_decision | 3/12 | 3/12 | 0 |
| qwen3_q8 | literal_copy | 12/12 | 12/12 | 0 |
| qwen3_q8 | new_json_fields | 8/12 | 8/12 | 0 |
| lfm3_qad | dictionary_code | 0/12 | 0/12 | 0 |
| lfm3_qad | evidence_decision | 3/12 | 3/12 | 1 |
| lfm3_qad | literal_copy | 7/12 | 8/12 | 0 |
| lfm3_qad | new_json_fields | 12/12 | 12/12 | 0 |

## Paired literal-copy result

- fable_v_only: mean paired change +0.0%; conditional prompt-bootstrap interval [+0.0%, +0.0%]; 0/12 prompts rescued on at least one seed and 0 with a loss.
- fable_all_qkv_parent: mean paired change +0.0%; conditional prompt-bootstrap interval [+0.0%, +0.0%]; 0/12 prompts rescued on at least one seed and 0 with a loss.
- parent_uniform_bf16: mean paired change +0.0%; conditional prompt-bootstrap interval [+0.0%, +0.0%]; 0/12 prompts rescued on at least one seed and 0 with a loss.

## Interpretation limits

- Twelve prompts per category and a few authored forms are insufficient for broad generalization or a tight no-regression claim.
- A degenerate bootstrap interval does not eliminate uncertainty outside these fixtures; see the exact event bounds and assumptions in the JSON analysis.
- Dictionary code payload is a static return-expression proxy, not an execution benchmark or proof of the fine-tune's coding advantage.
- V-only already regressed the discovery explanation proxy. Even if copying improves here, it is not an accepted overall repair.
- All-QKV is a broad rollback that may erase specialization. Qwen and LFM3 are reference deployments with different sizes/sampling/precision, not controlled architecture comparisons.
- Do not reuse this panel to select a new patch and then call it held-out.

[Machine-readable paired analysis](validation-v1-analysis.json). Raw per-run evidence is under `runs/20261105-validation-*`.

## Post-analysis grading addendum

The frozen records use grader v2. Its code-line exact/reference comparison was too literal about equivalent quote and whitespace choices. Grader v3 accepts AST-equivalent single return statements and recognizes reference expressions around inline/thinking delimiters as a presence proxy. All v2 records and the primary analysis above are retained. Separate v3 regrades are explicitly post-hoc sensitivity results; they do not change the literal-copy primary outcome or establish functional code quality.
