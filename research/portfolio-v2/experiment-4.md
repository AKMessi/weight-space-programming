# Experiment 4 — Patch interactions

Completed diagnostic composition pilot. A/B were not both independently established useful repairs; the useful-repair composition claim fails its prerequisite. These measurements test numerical commutation and behavioral interaction of fixed interventions.

| Model | Panel | Base | A | B | R | AB | Interaction (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_q8 | engineering (n=18) | 5 | 6 | 5 | 5 | 6 | +0.0 |
| qwen3_q8 | joint_evidence (n=12) | 0 | 0 | 0 | 0 | 0 | +0.0 |
| lfm3_qad | engineering (n=18) | 12 | 12 | 11 | 12 | 11 | +0.0 |
| lfm3_qad | joint_evidence (n=12) | 4 | 4 | 4 | 4 | 4 | +0.0 |
| fable_f16 | engineering (n=18) | 0 | 0 | 0 | 0 | 0 | +0.0 |
| fable_f16 | joint_evidence (n=12) | 0 | 0 | 0 | 0 | 0 | +0.0 |

AB/BA file identity and six-request behavioral identity are reported in the JSON. Parameter commutation follows from disjoint edited tensors; additive behavior does not follow. Nonzero interactions on a small shared panel are exploratory, not a learned interference predictor. R has a matched L2 delta but is not a semantic repair.

The engineering subset reuses the declared development cases from Experiment 1; the joint panel reuses Experiment 2 development conversations. This reuse reduces compute and is explicit. Neither panel is presented as fresh confirmation for composition. A larger independently useful patch library is required for the original paper-level hypothesis.

Code scoring uses post-hoc derived grader v7 (v6 functional rules); original recorded grades, frozen selections and raw outputs are preserved. The [derived sidecars](regrades/) list every change with its source checksum. This correction is sensitivity analysis, not an independent prespecified confirmation.
