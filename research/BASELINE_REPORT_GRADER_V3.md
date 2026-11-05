# Exploratory baseline report

Generated 2026-11-05T11:12:39.641102+00:00 with grader 3.
Source commit: `55420ff5b7d111c41aeb3d56ad442e60964a339f`.

**Discovery only.** Small investigator-visible synthetic tasks; no confidence, novelty, general benchmark, or model-improvement claim.

Code payload checks are static expression proxies; explanation passes do not establish explanation quality. Cutoff explanations are not complete answers.

| Run/model | Group | Exact/proxy success | Payload | Cutoffs |
|---|---|---:|---:|---:|
| 20261105-001-lfm3-qad-discovery/lfm3_qad | coding | 0/4 | 4/4 | 0 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | copy | 5/8 | 8/8 | 0 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | evidence_complete | 4/4 | 4/4 | 0 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | evidence_conflict | 0/4 | 0/4 | 1 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | explanation | 4/4 | 4/4 | 4 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | format | 8/8 | 8/8 | 0 |
| 20261105-001-lfm3-qad-discovery/lfm3_qad | tool | 4/4 | 4/4 | 0 |
| 20261105-002-lfm3-qad-template-probe/lfm3_qad | false_effort_none | 0/4 | 2/4 | 1 |
| 20261105-002-lfm3-qad-template-probe/lfm3_qad | false_parsed | 0/4 | 2/4 | 1 |
| 20261105-002-lfm3-qad-template-probe/lfm3_qad | false_raw | 0/4 | 2/4 | 1 |
| 20261105-002-lfm3-qad-template-probe/lfm3_qad | true_parsed | 1/4 | 1/4 | 3 |
| 20261105-003-lfm3-q8-template-probe/lfm3_q8 | false_effort_none | 0/4 | 2/4 | 1 |
| 20261105-003-lfm3-q8-template-probe/lfm3_q8 | false_parsed | 2/4 | 3/4 | 1 |
| 20261105-003-lfm3-q8-template-probe/lfm3_q8 | false_raw | 0/4 | 2/4 | 1 |
| 20261105-003-lfm3-q8-template-probe/lfm3_q8 | true_parsed | 1/4 | 1/4 | 3 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | coding | 0/4 | 0/4 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | copy | 7/8 | 7/8 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | evidence_missing | 4/4 | 4/4 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | explanation | 1/4 | 1/4 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | format | 2/8 | 8/8 | 0 |
| 20261105-004-qwen3-q8-discovery/qwen3_q8 | tool | 0/4 | 0/4 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | coding | 0/4 | 4/4 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | copy | 0/8 | 4/8 | 3 |
| 20261105-005-fable-f16-discovery/fable_f16 | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | explanation | 4/4 | 4/4 | 4 |
| 20261105-005-fable-f16-discovery/fable_f16 | format | 0/8 | 1/8 | 0 |
| 20261105-005-fable-f16-discovery/fable_f16 | tool | 0/4 | 0/4 | 0 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | coding | 0/4 | 4/4 | 0 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | copy | 8/8 | 8/8 | 0 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | evidence_complete | 0/4 | 0/4 | 3 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | evidence_conflict | 0/4 | 0/4 | 4 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | evidence_missing | 4/4 | 4/4 | 0 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | explanation | 4/4 | 4/4 | 4 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | format | 8/8 | 8/8 | 0 |
| 20261105-006-lfm3-q8-discovery/lfm3_q8 | tool | 4/4 | 4/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | coding | 0/4 | 4/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | copy | 4/8 | 4/8 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | evidence_complete | 4/4 | 4/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | explanation | 4/4 | 4/4 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | format | 0/8 | 8/8 | 0 |
| 20261105-007-parent-f16-discovery/parent_f16 | tool | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | coding/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | copy/strong_prompt | 0/8 | 4/8 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | evidence_complete/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | evidence_conflict/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | evidence_missing/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | explanation/strong_prompt | 4/4 | 4/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | format/strong_prompt | 0/8 | 0/8 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | format/strong_prompt_plus_schema | 6/8 | 6/8 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | tool/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-008-fable-nonweight-controls/fable_f16 | tool/strong_prompt_plus_schema | 4/4 | 4/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | coding/strong_prompt | 0/4 | 4/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | copy/strong_prompt | 8/8 | 8/8 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | evidence_complete/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | evidence_conflict/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | evidence_missing/strong_prompt | 1/4 | 1/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | explanation/strong_prompt | 1/4 | 1/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | format/strong_prompt | 2/8 | 8/8 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | format/strong_prompt_plus_schema | 8/8 | 8/8 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | tool/strong_prompt | 0/4 | 4/4 | 0 |
| 20261105-009-qwen-nonweight-controls/qwen3_q8 | tool/strong_prompt_plus_schema | 4/4 | 4/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | coding/strong_prompt | 0/4 | 4/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | copy/strong_prompt | 8/8 | 8/8 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | evidence_complete/strong_prompt | 4/4 | 4/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | evidence_conflict/strong_prompt | 0/4 | 0/4 | 2 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | evidence_missing/strong_prompt | 0/4 | 0/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | explanation/strong_prompt | 4/4 | 4/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | format/strong_prompt | 8/8 | 8/8 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | format/strong_prompt_plus_schema | 8/8 | 8/8 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | tool/strong_prompt | 4/4 | 4/4 | 0 |
| 20261105-010-lfm3-qad-nonweight-controls/lfm3_qad | tool/strong_prompt_plus_schema | 4/4 | 4/4 | 0 |
| 20261105-011-fable-identity-check/fable_identity | coding | 0/1 | 1/1 | 0 |
| 20261105-011-fable-identity-check/fable_identity | copy | 0/2 | 1/2 | 1 |
| 20261105-011-fable-identity-check/fable_identity | evidence_complete | 0/1 | 0/1 | 0 |
| 20261105-011-fable-identity-check/fable_identity | evidence_conflict | 0/1 | 0/1 | 0 |
| 20261105-011-fable-identity-check/fable_identity | evidence_missing | 0/1 | 0/1 | 0 |
| 20261105-011-fable-identity-check/fable_identity | explanation | 1/1 | 1/1 | 1 |
| 20261105-011-fable-identity-check/fable_identity | format | 0/2 | 0/2 | 0 |
| 20261105-011-fable-identity-check/fable_identity | tool | 0/1 | 0/1 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | coding | 0/4 | 4/4 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | copy | 0/8 | 4/8 | 3 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | explanation | 4/4 | 4/4 | 4 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | format | 0/8 | 2/8 | 0 |
| 20261105-012-fable-rollback-025-discovery/fable_rollback_025 | tool | 0/4 | 0/4 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | coding | 0/4 | 4/4 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | copy | 4/8 | 4/8 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | evidence_complete | 4/4 | 4/4 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | explanation | 4/4 | 4/4 | 3 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | format | 0/8 | 8/8 | 0 |
| 20261105-013-parent-bf16-values-discovery/parent_bf16_values | tool | 0/4 | 0/4 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | coding | 0/4 | 4/4 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | copy | 0/8 | 4/8 | 3 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | explanation | 3/4 | 3/4 | 4 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | format | 0/8 | 0/8 | 0 |
| 20261105-014-fable-late-v-025-discovery/fable_late_v_025 | tool | 0/4 | 0/4 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | coding | 0/4 | 4/4 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | copy | 4/8 | 4/8 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | evidence_complete | 4/4 | 4/4 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | explanation | 1/4 | 1/4 | 4 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | format | 0/8 | 8/8 | 0 |
| 20261105-015-parent-exact-bf16-values/parent_exact_bf16_values | tool | 0/4 | 0/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | coding | 0/4 | 4/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | copy | 4/8 | 4/8 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | evidence_complete | 4/4 | 4/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | explanation | 3/4 | 3/4 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | format | 0/8 | 8/8 | 0 |
| 20261105-016-fable-all-qkv-parent/fable_all_qkv_parent | tool | 0/4 | 0/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | coding | 0/4 | 4/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | copy | 0/8 | 4/8 | 3 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | explanation | 3/4 | 3/4 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | format | 0/8 | 0/8 | 0 |
| 20261105-017-fable-late-qkv-050/fable_late_qkv_050 | tool | 0/4 | 0/4 | 0 |
| 20261105-018-fable-q-only/fable_q_only | coding | 0/4 | 4/4 | 0 |
| 20261105-018-fable-q-only/fable_q_only | copy | 0/8 | 4/8 | 0 |
| 20261105-018-fable-q-only/fable_q_only | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-018-fable-q-only/fable_q_only | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-018-fable-q-only/fable_q_only | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-018-fable-q-only/fable_q_only | explanation | 0/4 | 0/4 | 3 |
| 20261105-018-fable-q-only/fable_q_only | format | 0/8 | 4/8 | 0 |
| 20261105-018-fable-q-only/fable_q_only | tool | 0/4 | 3/4 | 0 |
| 20261105-019-fable-k-only/fable_k_only | coding | 0/4 | 0/4 | 0 |
| 20261105-019-fable-k-only/fable_k_only | copy | 0/8 | 4/8 | 0 |
| 20261105-019-fable-k-only/fable_k_only | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-019-fable-k-only/fable_k_only | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-019-fable-k-only/fable_k_only | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-019-fable-k-only/fable_k_only | explanation | 0/4 | 0/4 | 4 |
| 20261105-019-fable-k-only/fable_k_only | format | 0/8 | 4/8 | 0 |
| 20261105-019-fable-k-only/fable_k_only | tool | 0/4 | 0/4 | 0 |
| 20261105-020-fable-v-only/fable_v_only | coding | 0/4 | 4/4 | 0 |
| 20261105-020-fable-v-only/fable_v_only | copy | 4/8 | 4/8 | 0 |
| 20261105-020-fable-v-only/fable_v_only | evidence_complete | 0/4 | 0/4 | 0 |
| 20261105-020-fable-v-only/fable_v_only | evidence_conflict | 0/4 | 0/4 | 0 |
| 20261105-020-fable-v-only/fable_v_only | evidence_missing | 0/4 | 0/4 | 0 |
| 20261105-020-fable-v-only/fable_v_only | explanation | 0/4 | 0/4 | 1 |
| 20261105-020-fable-v-only/fable_v_only | format | 0/8 | 6/8 | 0 |
| 20261105-020-fable-v-only/fable_v_only | tool | 0/4 | 0/4 | 0 |

## 20261105-001-lfm3-qad-discovery

- Artifact SHA-256: `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`
- Raw-response SHA-256: `b2c391cc1f69c0c2dd0f66fc44912ece644473df1c49722b59f4f46f8b3ecd53`
- Requests: 40; generated tokens: 1209.
- Median server decode speed: 77.46 tokens/s on these requests.
- [Raw evidence](../runs/20261105-001-lfm3-qad-discovery/responses.jsonl).

## 20261105-002-lfm3-qad-template-probe

- Artifact SHA-256: `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`
- Raw-response SHA-256: `b0ec3266240dcc4a2f58cd35363fc6ec381bd3abc1b0db225a5aa78047167235`
- Requests: 16; generated tokens: 1782.
- Median server decode speed: 77.84 tokens/s on these requests.
- [Raw evidence](../runs/20261105-002-lfm3-qad-template-probe/responses.jsonl).

## 20261105-003-lfm3-q8-template-probe

- Artifact SHA-256: `8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6`
- Raw-response SHA-256: `7f487594c6ef55acb9e6918c73719b1b65735d76c4f9f8cecfad4bc618a7db20`
- Requests: 16; generated tokens: 1683.
- Median server decode speed: 51.92 tokens/s on these requests.
- [Raw evidence](../runs/20261105-003-lfm3-q8-template-probe/responses.jsonl).

## 20261105-004-qwen3-q8-discovery

- Artifact SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Raw-response SHA-256: `eb6d70f9ec8617ce93413e4dcbaca502bf36b6bcd632ea9671f08140a7f52559`
- Requests: 40; generated tokens: 713.
- Median server decode speed: 86.48 tokens/s on these requests.
- [Raw evidence](../runs/20261105-004-qwen3-q8-discovery/responses.jsonl).

## 20261105-005-fable-f16-discovery

- Artifact SHA-256: `1e1624de4d7ebe413d82dbe2b5f48fcf83e6c46b1b5414c8aef0596009380cc6`
- Raw-response SHA-256: `41a522fea0255632380db2a2faddf7940e23190940d1533840aee3bc6c78d3a9`
- Requests: 40; generated tokens: 2058.
- Median server decode speed: 169.90 tokens/s on these requests.
- [Raw evidence](../runs/20261105-005-fable-f16-discovery/responses.jsonl).

## 20261105-006-lfm3-q8-discovery

- Artifact SHA-256: `8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6`
- Raw-response SHA-256: `671d345491528eb6999cabc5da6fdef4a3e8564f8c532abd14b2f36d6421fb85`
- Requests: 40; generated tokens: 1975.
- Median server decode speed: 52.15 tokens/s on these requests.
- [Raw evidence](../runs/20261105-006-lfm3-q8-discovery/responses.jsonl).

## 20261105-007-parent-f16-discovery

- Artifact SHA-256: `4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`
- Raw-response SHA-256: `e98792d62071106c87b6af6eda295723ededc63b3686a8a0540058b441a67476`
- Requests: 40; generated tokens: 869.
- Median server decode speed: 157.78 tokens/s on these requests.
- [Raw evidence](../runs/20261105-007-parent-f16-discovery/responses.jsonl).

## 20261105-008-fable-nonweight-controls

- Artifact SHA-256: `1e1624de4d7ebe413d82dbe2b5f48fcf83e6c46b1b5414c8aef0596009380cc6`
- Raw-response SHA-256: `c20cd98e09dda0d6e812db4abb2921eaa006a6d8a4da31ca281ed72d11b4d22d`
- Requests: 52; generated tokens: 1283.
- Median server decode speed: 161.67 tokens/s on these requests.
- [Raw evidence](../runs/20261105-008-fable-nonweight-controls/responses.jsonl).

## 20261105-009-qwen-nonweight-controls

- Artifact SHA-256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Raw-response SHA-256: `9fb4d440d65aa5a30c2cf2241d8726655c0be4c21277cbde108b3f44665d8026`
- Requests: 52; generated tokens: 978.
- Median server decode speed: 73.87 tokens/s on these requests.
- [Raw evidence](../runs/20261105-009-qwen-nonweight-controls/responses.jsonl).

## 20261105-010-lfm3-qad-nonweight-controls

- Artifact SHA-256: `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`
- Raw-response SHA-256: `d8f27d348bf86bb2edf6a81536e30266a5585e73951553f252130921d6a12878`
- Requests: 52; generated tokens: 1538.
- Median server decode speed: 74.03 tokens/s on these requests.
- [Raw evidence](../runs/20261105-010-lfm3-qad-nonweight-controls/responses.jsonl).

## 20261105-011-fable-identity-check

- Artifact SHA-256: `1e1624de4d7ebe413d82dbe2b5f48fcf83e6c46b1b5414c8aef0596009380cc6`
- Raw-response SHA-256: `be4c1490657171a4fe21d0c79fd8a8f0dad595fad7da56324a74c048987da0d0`
- Requests: 10; generated tokens: 501.
- Median server decode speed: 172.67 tokens/s on these requests.
- [Raw evidence](../runs/20261105-011-fable-identity-check/responses.jsonl).

## 20261105-012-fable-rollback-025-discovery

- Artifact SHA-256: `886dec26e4f87801838fb95142ede66355c2cf0e96fd019101450bc104621b3f`
- Raw-response SHA-256: `95468fd7af48451754c11e163e7cf296a19297f52b2337c0a0aa7e3bd26c11f5`
- Requests: 40; generated tokens: 2066.
- Median server decode speed: 170.62 tokens/s on these requests.
- [Raw evidence](../runs/20261105-012-fable-rollback-025-discovery/responses.jsonl).

## 20261105-013-parent-bf16-values-discovery

- Artifact SHA-256: `65f67bce2ad8a35f59a3325d297944ec423927a6ade01303e36c9806ce174454`
- Raw-response SHA-256: `0f088c2e5ad42bba2614934683bfe5015397dee61be36b3fb0ae7be3a16eca93`
- Requests: 40; generated tokens: 1038.
- Median server decode speed: 162.57 tokens/s on these requests.
- [Raw evidence](../runs/20261105-013-parent-bf16-values-discovery/responses.jsonl).

## 20261105-014-fable-late-v-025-discovery

- Artifact SHA-256: `c0d216d9ebd9ca6ba181fa47263172a3e3dc9193eea97ce1d9dbdd16cbade57d`
- Raw-response SHA-256: `11f0095114a12ad9f1048ce5133260cad3262c21f008ce7625bd9fda8df6c8ca`
- Requests: 40; generated tokens: 1953.
- Median server decode speed: 168.48 tokens/s on these requests.
- [Raw evidence](../runs/20261105-014-fable-late-v-025-discovery/responses.jsonl).

## 20261105-015-parent-exact-bf16-values

- Artifact SHA-256: `30c3b7c42253cbdf8458621cd22c0fae81f1674dc9b8213644dd3d9ea25ebd06`
- Raw-response SHA-256: `d1dc76c16a7bd666dd4aee237016b0e639062dead2a8aed24f6f00766dafe8cf`
- Requests: 40; generated tokens: 1034.
- Median server decode speed: 162.50 tokens/s on these requests.
- [Raw evidence](../runs/20261105-015-parent-exact-bf16-values/responses.jsonl).

## 20261105-016-fable-all-qkv-parent

- Artifact SHA-256: `de39e08dfa54e8d23c4f8964d8a05de7f788656a7821b2dd7ed8b08d7aa2d230`
- Raw-response SHA-256: `157579d642603e530867b2096317ea589a0422fee022fe60d3a75fbe143a183c`
- Requests: 40; generated tokens: 812.
- Median server decode speed: 155.66 tokens/s on these requests.
- [Raw evidence](../runs/20261105-016-fable-all-qkv-parent/responses.jsonl).

## 20261105-017-fable-late-qkv-050

- Artifact SHA-256: `f8067cc049c562a3c0693183b98c120aa48a4dd74174b5e4ce06181df927abeb`
- Raw-response SHA-256: `c2e546d15cff87c7b1cdda2327d8b6dd93a54d047363d1c73342cd98a2b726ed`
- Requests: 40; generated tokens: 1679.
- Median server decode speed: 170.73 tokens/s on these requests.
- [Raw evidence](../runs/20261105-017-fable-late-qkv-050/responses.jsonl).

## 20261105-018-fable-q-only

- Artifact SHA-256: `0c88fb78554e210d5876bd45f82261d5590f0426fd2e7ae95ed59a966647286b`
- Raw-response SHA-256: `23a283acd96be749217ae4e18bf544ea02189592afe13e811835e42a246c35b4`
- Requests: 40; generated tokens: 1595.
- Median server decode speed: 165.82 tokens/s on these requests.
- [Raw evidence](../runs/20261105-018-fable-q-only/responses.jsonl).

## 20261105-019-fable-k-only

- Artifact SHA-256: `fa5c4ed0dae3a5fd79e53c37a22a905f5a76e3e2de04994422fd82b62742387a`
- Raw-response SHA-256: `ae26c8003b1a6f0ef4d72900db093943644fa2de4d88668854ad78d84d22049f`
- Requests: 40; generated tokens: 1614.
- Median server decode speed: 168.45 tokens/s on these requests.
- [Raw evidence](../runs/20261105-019-fable-k-only/responses.jsonl).

## 20261105-020-fable-v-only

- Artifact SHA-256: `a7e5ac40b01b5d100f2d4e59ca1f795aaf5c430a83f855ed08b0cd1eaf6e44a7`
- Raw-response SHA-256: `924a3dbcab5daa40dba35c61f5c9b9b4079b80518770faf30d020d97af2b9d2e`
- Requests: 40; generated tokens: 880.
- Median server decode speed: 164.81 tokens/s on these requests.
- [Raw evidence](../runs/20261105-020-fable-v-only/responses.jsonl).
