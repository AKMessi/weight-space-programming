# Exploratory baseline report

Generated 2026-11-05T10:38:41.705354+00:00 with grader 2.
Source commit: `f33c63247eb2186ec1a54e98dc83f8cd0c5dd83d`.

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

## 20261105-001-lfm3-qad-discovery

- Artifact SHA-256: `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`
- Raw-response SHA-256: `b2c391cc1f69c0c2dd0f66fc44912ece644473df1c49722b59f4f46f8b3ecd53`
- Requests: 40; generated tokens: 1209.
- Median server decode speed: 77.46 tokens/s on these requests.
- [Raw evidence](../runs/20261105-001-lfm3-qad-discovery/responses.jsonl).

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
