# Prior-work scope check

Checked primary paper versions during P3 execution,2026-11-06local. This records attribution and limits, not new model evidence.

[Steer2Edit v3](https://arxiv.org/html/2602.09870v3) derives component rank-one edits, normalized output/input directions, cosine importance and elastic-net magnitudes. Our implementation uses its MLP-neuron rule with disclosed fixed64-site/global-norm constraints and teacher-forced contrast construction. Attention editing and the paper's full task evaluations are not reproduced. The ridge-protected comparator is an analytic adaptation, not a new claimed invention. Our failure on fresh instruction families is evidence about this setup, not a refutation of the paper.

[Delta-CoMe v2](https://arxiv.org/html/2406.08903v2) assigns different precision to groups of singular vectors, with calibrated quantization. We have not implemented its complete method. Our SVD/delta-packing/capsule comparisons cannot establish superiority to it. The exact capsule tests full-file identity, a different objective from approximate task-preserving compression.

[BitDelta](https://arxiv.org/html/2402.10193) includes sign quantization followed by scale distillation. Our uncalibrated sign/mean-magnitude packing is a limited baseline and must not be called a full BitDelta reproduction. Model distillation was not performed under the current no-training scope.

The practical outputs can support a reproducible local case study: precision/provenance-sensitive merged-adapter recovery, exact delivery, measured backend-format behavior, and a failed conditional-repair transfer. They do not establish a novel general editing/compression algorithm or a publication-ready comparative result. Independent additional fine-tunes, devices and relevant complete method baselines are necessary before broader claims.

Additional confounds: bare-versus-fenced teacher-forced contrasts differ in token count/position as well as formatting. Fresh repair confirmation changes schema, fact layout and contract wording together. Clarification holds field/layout families out together. These tests establish transfer limits for their joint changes, not the separate cause of each failure. Position-matched contrasts and crossed-factor tests remain follow-ups; confirmation data must not be reused to claim a new independent success.
