# Experiment 2 — Ask, then recover after evidence arrives

Completed two-turn pilot with actual prior model replies in the second-turn context. Twelve conversations = four requested-field families × three evidence conditions. These are correlated synthetic tasks, not 12 independently sampled domains.

| Arm | Strict joint / 12 | Semantic joint / 12 | First correct | Second correct | Supported answers / 4 | Unsupported answer actions / 8 | Invented non-null / 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_q8 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| qwen3_q8_A | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| qwen3_q8/strong | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| qwen3_q8/grammar-v2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| lfm3_qad | 4 | 4 | 4 | 7 | 4 | 8 | 6 |
| lfm3_qad_A | 4 | 4 | 4 | 6 | 4 | 8 | 7 |
| lfm3_qad/strong | 4 | 4 | 4 | 12 | 4 | 8 | 8 |
| lfm3_qad/grammar-v2 | 4 | 4 | 4 | 12 | 4 | 8 | 8 |
| fable_f16 | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| fable_f16_A | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| fable_f16/strong | 0 | 0 | 1 | 0 | 1 | 7 | 7 |
| fable_f16/grammar-v2 | 4 | 4 | 4 | 12 | 4 | 8 | 8 |

Deterministic structured-record checker: 12/12 complete conversations. This is a useful task-specific baseline, not evidence of general understanding.

## Interpretation

Schema correctness is insufficient for evidence-sensitive decisions. An answer action with null is a wrong decision but is separated from inventing a non-null value. A correct second turn cannot erase a bad first decision. Coverage and unsupported answering must be read together. The fixed A/B edits are black-box baselines; an activation-derived evidence direction was not implemented. No extra inference calls are hidden in the one-call arms.

- qwen3_q8: confirmed pilot repair=False; complete-case losses=0; paired rescues/losses=0/0.
- lfm3_qad: confirmed pilot repair=False; complete-case losses=0; paired rescues/losses=0/0.
- fable_f16: confirmed pilot repair=False; complete-case losses=0; paired rescues/losses=0/0.
