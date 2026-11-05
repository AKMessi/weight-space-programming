# Experiment 1 — Repair habits while preserving utility

Completed bounded pilot. Development selection and confirmation were separated. Twelve functional fixtures per seed include code and explanation modes; the six underlying functions are correlated pairs. These tasks do not establish repository-level coding specialization.

## Frozen confirmation
| Arm | Group | Strict | Payload / function | Cutoffs |
|---|---|---:|---:|---:|
| qwen3_q8 | functional_code | 12/12 | 12/12 | 0 |
| qwen3_q8 | functional_explain | 8/12 | 12/12 | 0 |
| qwen3_q8 | literal_copy | 0/16 | 15/16 | 0 |
| qwen3_q8 | structured | 8/16 | 16/16 | 0 |
| qwen3_q8_A | functional_code | 12/12 | 12/12 | 0 |
| qwen3_q8_A | functional_explain | 6/12 | 12/12 | 0 |
| qwen3_q8_A | literal_copy | 0/16 | 16/16 | 0 |
| qwen3_q8_A | structured | 3/16 | 5/16 | 0 |
| lfm3_qad | functional_code | 8/12 | 8/12 | 0 |
| lfm3_qad | functional_explain | 6/12 | 11/12 | 1 |
| lfm3_qad | literal_copy | 0/16 | 16/16 | 0 |
| lfm3_qad | structured | 16/16 | 16/16 | 0 |
| lfm3_qad_A | functional_code | 8/12 | 10/12 | 0 |
| lfm3_qad_A | functional_explain | 5/12 | 9/12 | 1 |
| lfm3_qad_A | literal_copy | 0/16 | 16/16 | 0 |
| lfm3_qad_A | structured | 16/16 | 16/16 | 0 |
| fable_f16 | functional_code | 0/12 | 6/12 | 0 |
| fable_f16 | functional_explain | 5/12 | 7/12 | 0 |
| fable_f16 | literal_copy | 0/16 | 2/16 | 0 |
| fable_f16 | structured | 0/16 | 10/16 | 0 |
| fable_f16_A | functional_code | 0/12 | 6/12 | 0 |
| fable_f16_A | functional_explain | 5/12 | 6/12 | 0 |
| fable_f16_A | literal_copy | 0/16 | 1/16 | 0 |
| fable_f16_A | structured | 0/16 | 16/16 | 0 |
| aligned Fable parent | functional_code | 8/12 | 8/12 | 0 |
| aligned Fable parent | functional_explain | 10/12 | 10/12 | 0 |
| aligned Fable parent | literal_copy | 0/16 | 9/16 | 0 |
| aligned Fable parent | structured | 0/16 | 0/16 | 0 |

## Matched prompt / real-schema controls
| Arm | Group | Strict | Payload / function | Cutoffs |
|---|---|---:|---:|---:|
| qwen3_q8/strong | functional_code | 3/6 | 6/6 | 0 |
| qwen3_q8/strong | functional_explain | 0/6 | 6/6 | 0 |
| qwen3_q8/strong | literal_copy | 0/8 | 8/8 | 0 |
| qwen3_q8/strong | structured | 0/8 | 8/8 | 0 |
| qwen3_q8/grammar-v2 | functional_code | 3/6 | 6/6 | 0 |
| qwen3_q8/grammar-v2 | functional_explain | 0/6 | 6/6 | 0 |
| qwen3_q8/grammar-v2 | literal_copy | 0/8 | 8/8 | 0 |
| qwen3_q8/grammar-v2 | structured | 8/8 | 8/8 | 0 |
| lfm3_qad/strong | functional_code | 3/6 | 5/6 | 0 |
| lfm3_qad/strong | functional_explain | 6/6 | 6/6 | 0 |
| lfm3_qad/strong | literal_copy | 0/8 | 8/8 | 0 |
| lfm3_qad/strong | structured | 8/8 | 8/8 | 0 |
| lfm3_qad/grammar-v2 | functional_code | 3/6 | 5/6 | 0 |
| lfm3_qad/grammar-v2 | functional_explain | 6/6 | 6/6 | 0 |
| lfm3_qad/grammar-v2 | literal_copy | 0/8 | 8/8 | 0 |
| lfm3_qad/grammar-v2 | structured | 8/8 | 8/8 | 0 |
| fable_f16/strong | functional_code | 0/6 | 3/6 | 0 |
| fable_f16/strong | functional_explain | 3/6 | 3/6 | 0 |
| fable_f16/strong | literal_copy | 0/8 | 2/8 | 0 |
| fable_f16/strong | structured | 0/8 | 8/8 | 0 |
| fable_f16/grammar-v2 | functional_code | 0/6 | 3/6 | 0 |
| fable_f16/grammar-v2 | functional_explain | 3/6 | 3/6 | 0 |
| fable_f16/grammar-v2 | literal_copy | 0/8 | 2/8 | 0 |
| fable_f16/grammar-v2 | structured | 8/8 | 8/8 | 0 |

## Decisions

- qwen3_q8: development eligible=False; confirmation paired rescues/losses=2/7; functional losses=0; explanation losses=4; confirmed pilot repair=False.
- lfm3_qad: development eligible=False; confirmation paired rescues/losses=0/0; functional losses=2; explanation losses=1; confirmed pilot repair=False.
- fable_f16: development eligible=False; confirmation paired rescues/losses=0/0; functional losses=2; explanation losses=1; confirmed pilot repair=False.

A retained diagnostic candidate cannot become an independently selected success merely because its confirmation looks attractive. Fable superiority over the parent is not presumed. The interpreter covers a bounded Python subset; syntax/unsupported-language rejections remain in the raw grading records and need manual audit before broad capability claims. Explanation length is a proxy.

All discovery, prior Q/K/V restorations, rejected candidates and exact selection rules are preserved in the journal, run directories and `research/selections/e1.json`. The new schema v1 controls were malformed and superseded by separately recorded v2 controls. No new editing-method claim or victory over unimplemented activation methods is made.
