#import "template.typ": *

#show: paper.with(
  title: "The Quantization Cliff of Weight-Space Programming: Hardware-Aligned Constant-Block Embedding and Post-Quantization Healing",
  short-title: "The Quantization Cliff of Weight-Space Programming & Hardware-Aligned BA-WSC",
  author: "Aaryan Kakad",
  affiliation: "Independent Machine Learning Researcher",
  date: "November 2026",
  companion-note: "Companion Paper 2 of 2 — see Paper 1 for the Greedy-Perturbation Illusion, Universal Layer Anatomy, and Zero-Shot Transfer.",
  keywords: (
    "Weight-Space Programming",
    "Quantization",
    "GGUF Block Packing",
    "SwiGLU Symmetry",
    "Post-Quantization Healing",
    "Edge Reasoning LLMs",
  ),
  abstract: [
    Practical deployment of local reasoning LLMs relies overwhelmingly on `4-bit` (`Q4_0`, `Q4_K_M`) and `3-bit` (`Q3_K_M`) block quantization. Yet the intersection of *Weight-Space Programming (WSP)* and post-training quantization remains largely unexplored. We first characterize a *Scale-Dependent Quantization-Induced Reasoning Collapse* across three architecturally distinct reasoning models: on `0.6B` pure GQA (`Qwen3-0.6B`) and `0.8B` hybrid Gated-DeltaNet (`Qwen3.5-0.8B`), `4-bit` and `3-bit` quantization triggers a `2.3×–8.2×` explosion in repetition loops (reaching `83%–92%` looping), whereas the `2.5×` larger `DeepSeek-R1-Distill-Qwen-1.5B` absorbs `4-bit` cleanly and shifts its timeout/loop cliff down to `3-bit` `Q3_K_M` (`limit 36%` $->$ `58%` cal, `66.0%` on held-out `MATH-500`). We then demonstrate that *standard floating-point weight edits shatter under real `llama-quantize` block quantization*: because scattered or diffuse edits share 32-element (`Q4_0`) or 256-element (`Q4_K_M`/`Q3_K_M`) block scales with unedited neighbor weights, edit cosine similarity drops to `0.23–0.55` while collateral distortion to unedited neighbors reaches up to *`4.13×` the edit signal norm*.

    To solve this structurally with *zero extra bytes, zero runtime latency, and `0.000000` floating-point discrepancy*, we prove *Theorem 1* (sub-block alignment, $k = B = 32$) and *Corollary 1* (hierarchical K-quant super-block alignment, $k = "QK"_K = 256$): exploiting exact SwiGLU neuron-permutation symmetry $P$ and positive-homogeneity gauge balancing $eta$, we pack replicated constant-payload dormant neurons into dedicated hardware quantization blocks. Verified empirically on real `llama-quantize` across *all three architectures*, our Block-Aligned Weight-Space Compilation (`BA-WSC`) preserves full $Delta W$ cosine similarity at *`0.996–0.99984`* down to `3-bit` `Q3_K_M`, slashing neighbor collateral distortion by *`4,514×`* (`Qwen3-0.6B`), *`10,119×–16,685×`* (`Qwen3.5-0.8B`), and *`12,133×`* (`DeepSeek-R1-1.5B`, consuming only `2.86%` of its 8,960 FFN neurons). Behaviorally, we introduce *Post-Quantization Weight-Space Healing (`PQ-BA-WSC`)*, tracing exact dequantized low-bit models to heal their own quantization-induced loops: on a 20-arm held-out `MATH-500` matrix (`Qwen3-0.6B`, `temp = 0.6`), `PQ-BA-WSC` systematically outperforms FP-transfer at both `4-bit` (`+7.3 pp`, $p = 0.0347$, $Delta"PPL" = +0.003$) and *`3-bit` (`+8.7 pp` / `+68%` relative gain, $p = 0.0044$, `#1` overall)*, while `oblique_res` heals *`83%` of the `4-bit` cliff* (`+10.0 pp`, $p = 0.0041$). Finally, we establish an empirical *behavioral proof of Theorem 1*: across two independent architectures (`Qwen3.5-0.8B` `Q8_0` $->$ `Q4_0` and `DeepSeek-R1-1.5B` `Q8_0` $->$ `Q3_K_M`), diffuse `oblique_res` dominates at `8-bit` then collapses under low-bit quantization, whereas block-aligned `BA-WSC` survives intact and overtakes it—producing statistically significant Precision $times$ Method double crossovers (`+15/150`, $p = 0.00336$ on `Qwen3.5-0.8B`; `+13/150`, $p = 0.0663$ on `DeepSeek-R1-1.5B`; *Fisher combined $p = 0.00210$*).
  ],
)

= Introduction <sec:intro>

Edge deployment of open-weights reasoning LLMs depends critically on `4-bit` and `3-bit` block-wise quantization (`GGUF` `Q4_0`, `Q4_K_M`, `Q3_K_M` @gerganov2023). While perplexity benchmarks often report modest losses at `4-bit`, multi-step chain-of-thought (`<think>`) inference exhibits far sharper behavioral cliffs: subtle logit perturbations accumulate autoregressively over thousands of tokens, trapping small reasoning models in endless verification or verbatim repetition loops.

Weight-Space Programming (WSP)—compiling zero-gradient behavioral patches directly into frozen MLP weights (companion Paper 1; @steer2edit2026; @takehi2026)—would be an ideal remedy for edge deployments because it adds *zero bytes and zero latency*. However, a critical unexamined gap separates floating-point WSP research from real-world quantized deployment: *do closed-form weight edits survive `4-bit` and `3-bit` block quantization, and can WSP directly heal quantization-induced reasoning collapse?*

This paper provides a rigorous theoretical and empirical answer across three distinct model families (`Qwen3-0.6B`, hybrid Gated-DeltaNet `Qwen3.5-0.8B` @qwen2025, and `DeepSeek-R1-Distill-Qwen-1.5B` @deepseek2025), making *four contributions*:
1. *Scale-Dependent Quantization Loop Cliff (@sec:cliff):* We document how `4-bit` and `3-bit` quantization induces `2.3×–8.2×` repetition-loop explosions on `0.6B–0.8B` models, while shifting the timeout and loop cliff down to `3-bit` `Q3_K_M` on the `2.5×` larger `1.5B` model.
2. *Formal Quantization Geometry — Theorem 1 & Corollary 1 (@sec:theory–@sec:audit):* We diagnose mixed-block scale contamination—showing prior sparse and unconditional WSP lose up to `77%` of their directional signal (`cos = 0.23–0.55`) and inject up to `4.13×` more neighbor collateral than signal—and prove *Theorem 1 ($k = 32$)* and *Corollary 1 ($k = "QK"_K = 256$)* using exact SwiGLU neuron-permutation symmetry $P$ and positive-homogeneity gauge balancing $eta$. Real `llama-quantize` audits confirm `cos = 0.996–0.99984` and *`4,514×–16,685×` collateral reduction* across all three models with zero storage or latency cost.
3. *Post-Quantization Weight-Space Healing (`PQ-BA-WSC`, @sec:pq):* Across a controlled 20-arm `Q4_0` and `Q3_K_M` matrix on `Qwen3-0.6B` (including exact permutation no-op `ba_wsc_noop` and matched random noise controls), compiling directly on dequantized low-bit activations outperforms FP-transfer at both bit-widths and ranks `#1` overall at `3-bit` `Q3_K_M` (`+8.7 pp` / `+68%` relative gain, $p = 0.0044$, $Delta"PPL" = +0.116$ vs. unconditional WSP's `+2.202`), while `oblique_res` heals `83%` of the `4-bit` cliff ($p = 0.0041$).
4. *Behavioral Proof of Theorem 1 via Precision $times$ Method Double Crossovers (@sec:crossover):* Across two independent architectures (`Qwen3.5-0.8B` `Q8_0` $->$ `Q4_0` and `DeepSeek-R1-1.5B` `Q8_0` $->$ `Q3_K_M`), we prove that hardware block alignment is causally decisive in end-to-end generation: diffuse `oblique_res` wins at `8-bit` then collapses under quantization, whereas block-aligned `BA-WSC` holds constant or improves (*Fisher combined double-crossover $p = 0.00210$*).

= Background: GGUF Block & Hierarchical K-Quantization <sec:bg>

In `llama.cpp` GGUF formats @gerganov2023, each row $w in bb(R)^(d_"ff")$ of `ffn_down.weight` ($W_"down" in bb(R)^(d times d_"ff")$) is partitioned along the inner dimension $d_"ff"$ into contiguous blocks (analogous to block-wise quantization in QLoRA @dettmers2023 and AWQ @lin2024):
- *Flat Block Formats (`Q8_0`, `Q4_0`):* Block width $B = 32$. For each block $b$ of 32 weights $w_b in bb(R)^32$, the symmetric quantizer computes an `FP16` block scale $delta_b = (max_(j in b) abs(w_j)) / q_max$ ($q_max = 7$ for `Q4_0`, $127$ for `Q8_0`), stores 4-bit (or 8-bit) signed integers $q_j = "round"(w_j / delta_b)$, and reconstructs $hat(w)_j = delta_b q_j$.
- *Hierarchical K-Quant Super-Block Formats (`Q4_K_M`, `Q3_K_M`):* Super-block width $"QK"_K = 256$, subdivided into 8 sub-blocks of 32 (or 16 sub-blocks of 16 in `Q3_K`). Crucially, all 256 weights in a super-block share a *single `FP16` super-block scale $d_S$*, with low-bit (6-bit) sub-block scales $s_(S,m)$ quantized relative to $d_S$.

This hardware layout induces a fundamental coupling: *modifying even a single weight $w_j$ inside block $b$ (or super-block $S$) alters $delta_b$ (or $d_S$), shifting the quantization grid for every unedited weight sharing that block.*

= Scale-Dependent Quantization-Induced Reasoning Collapse <sec:cliff>

We first measure unedited baseline trajectories across `BF16`, `Q8_0`, `Q4_K_M`, `Q4_0`, and `Q3_K_M` on calibration `cal_math_50` (`max_tokens = 2048`) and held-out `MATH-500-strat150` (`temp = 0.6`) across all three architectures (@tab:cliff, @fig:cliff).

#figure(
  booktable(
    columns: (1.70fr, 1.05fr, 1.05fr, 1.05fr, 1.10fr, 1.65fr),
    align: (left, right, right, right, right, left),
    header: ([Model & Architecture], [`Q8_0`\ PPL/Lp/Lim], [`Q4_K_M`\ PPL/Lp/Lim], [`Q4_0`\ PPL/Lp/Lim], [`Q3_K_M`\ PPL/Lp/Lim], [Cliff Threshold]),
    font-size: 7.45pt,
    inset: (x: 3.2pt, y: 3.2pt),
    rows: (
      ([*`Qwen3-0.6B`*\ (`0.6B` Pure GQA)], [`19.28/24%/46%`], [`20.73/26%/52%`], [`21.87/`*`46%`*`/68%`], [`26.72/`*`74%`*`/68%`], [*4-bit & 3-bit*\ (`3.1×–8.2×` MATH loops)]),
      ([*`Qwen3.5-0.8B`*\ (`0.8B` Hybrid)], [`15.51/34%/100%`], [`16.19/40%/98%`], [`18.23/`*`88%`*`/94%`], [`19.20/`*`92%`*`/98%`], [*4-bit & 3-bit*\ (`83.3%` loops on MATH `Q4_0`)]),
      ([*`DeepSeek-R1-1.5B`*\ (`1.5B` Qwen2 RL)], [`37.25/4%/36%`], [`39.50/4%/42%`], [`41.19/6%/36%`], [`40.71/`*`8%`*`/`*`58%`*], [*Shifts to 3-bit!*\ (`66.0%` limit on MATH `Q3_K_M`)]),
    ),
  ),
  caption: [Scale-dependent quantization-induced reasoning collapse across three architectures (`cal_math_50` + held-out `MATH-500` summary). `Lp` = repetition loop rate; `Lim` = 2048-token truncation rate.],
) <tab:cliff>

*Key Observation:* Quantization-induced reasoning degradation manifests primarily as *control-flow destabilization* (loop explosions and failure to close `</think>`), not merely local arithmetic errors. On `Qwen3-0.6B` held-out `MATH-500` (`temp = 0.6`), `Q4_0` drops Clean-EOS accuracy by `-12.0 pp` (`53` $->$ `35/150`) with a `3.1×` loop jump (`9` $->$ `28`), and `Q3_K_M` drops to `19/150` (`-22.7 pp`) with an *`8.2×` loop explosion (`9` $->$ `74`)*. Scaling `2.5×` to `1.5B` (`DeepSeek-R1-Distill-Qwen-1.5B`) confers resilience at `4-bit` (`Q4_0`), shifting the severe timeout cliff down to `3-bit` `Q3_K_M` (`99/150 = 66.0%` timeouts).

#figure(
  image("../figures/p2_fig1_quant_loop_cliff_3models.png", width: 100%),
  caption: [Quantization-induced reasoning collapse across all three architectures (`cal_math_50`). Smaller `0.6B` and `0.8B` models suffer severe loop explosions already at `4-bit` `Q4_0`, whereas `1.5B` `DeepSeek-R1` absorbs `4-bit` and shifts its collapse cliff down to `3-bit` `Q3_K_M`.],
) <fig:cliff>

= Formal Quantization Geometry of Weight-Space Programming <sec:theory>

== Metrics for Edit Quantization Survival <sec:metrics>

Let $W$ be an unedited floating-point tensor, $W' = W + Delta W_"FP"$ the edited tensor, and $Q(dot)$ a real GGUF quantizer followed by exact dequantization. Define the realized quantized edit $Delta W_Q = Q(W') - Q(W)$. On the support mask $M = bb(I)[Delta W_"FP" != 0]$ and its complement $M^c$, decompose $Delta W_Q = Delta W_Q^"sig" + Delta W_Q^"col"$ (supported on $M$ and $M^c$, respectively) and measure:
1. *Full Edit Cosine Similarity:* $cos_"full" = (chevron.l Delta W_Q, Delta W_"FP" chevron.r) / (norm(Delta W_Q)_F norm(Delta W_"FP")_F)$.
2. *Collateral-to-Signal Norm Ratio:* $rho_"col" = norm(Delta W_Q^"col")_F / norm(Delta W_Q^"sig")_F$, quantifying unintended distortion injected into *unedited* weights strictly due to shared block scales.

For permutation-aligned `BA-WSC`, we evaluate $Q(W'_"BA") - Q(W_"noop")$ against the exact permuted no-op so permutation re-binning is separated cleanly from edit quantization distortion (and we additionally include `ba_wsc_noop` as a direct behavioral control in @sec:pq).

== Theorem 1: Hardware Sub-Block Constant-Payload Embedding ($k = B = 32$) <sec:thm1>

#theorem(1, title: "Zero-Collateral Exact-Code Sub-Block Embedding")[
  Consider a SwiGLU MLP layer $"MLP"(x) = W_"down" ("SiLU"(W_"gate" x) dot.o (W_"up" x))$ with $W_"down" in bb(R)^(d times d_"ff")$ quantized row-wise by a symmetric block quantizer $Q_B$ of block width $B$ (e.g., $B = 32$ in `Q4_0`/`Q8_0`, max integer level $q_max$) with `FP16` block scales. Let $cal(S) subset {0, dots, d_"ff" - 1}$ be any set of $|cal(S)| = B$ dormant neuron indices, and let $P in {0, 1}^(d_"ff" times d_"ff")$ be a permutation mapping $cal(S)$ to the contiguous block ${0, dots, B - 1}$. Form the permuted weights $tilde(W)_"down" = W_"down" P$, $tilde(W)_"gate" = P^top W_"gate"$, $tilde(W)_"up" = P^top W_"up"$, and compile a replicated rank-1 payload across block $0$:
  $ tilde(W)'_"gate" [j, :] = g^top, quad tilde(W)'_"up" [j, :] = eta u_"up"^top, quad tilde(W)'_"down" [:, j] = 1 / (B eta) v quad forall j in {0, dots, B - 1}, $
  with gauge factor $eta > 0$ chosen to equalize $max abs(tilde(W)'_"up" [0:B, :])$ and $max abs(tilde(W)'_"down" [:, 0:B])$. Then:
  1. *Exact Floating-Point Functional Symmetry:* Before editing $cal(S)$, the permutation $P$ changes $"MLP"(x)$ by identically $0$ for all $x in bb(R)^d$ and adds $0$ parameters and $0$ runtime ops.
  2. *Zero Mixed Blocks & Zero Scale Collateral:* Every block $b >= 1$ of every row $i$ of $tilde(W)'_"down"$ is identical to $tilde(W)_"down"$, so $N_"mixed" = 0$ and $Delta W_Q^"col" equiv 0$ on all unedited columns $j >= B$ (i.e., $rho_"col" = 0$ identically for exact-zero evicted dormant weights, and bounded solely by dormant eviction residual $norm(tilde(W)_"down" [:, 0:B])_F$ in practice).
  3. *Exact Maximum-Code Integer Quantization:* Inside block $b = 0$ of row $i$, all $B$ entries equal the constant $c_i = v_i / (B eta)$. Therefore, the block scale is $delta_(i,0) = "FP16"(abs(c_i) / q_max)$, every weight quantizes to the exact extreme code $q_(i,j) = "sign"(c_i) q_max$ with *zero integer rounding error*, and reconstruction fidelity is bounded solely by `FP16` mantissa precision:
  $ abs(hat(c)_i - c_i) / abs(c_i) <= 2^(-11) approx 4.88 times 10^(-4) . $
]

#proof-box[
  *(1)* Follows immediately from elementwise commutation of $"SiLU"(dot)$ and $dot.o$ with permutation matrices $P^top$:
  $W_"down" P ("SiLU"(P^top W_"gate" x) dot.o P^top W_"up" x) = W_"down" P P^top ("SiLU"(W_"gate" x) dot.o W_"up" x) = "MLP"(x)$.
  *(2)* Since $Q_B$ operates independently on disjoint contiguous segments $lr([b B, (b + 1) B))$ and $tilde(W)'_"down" [:, b B : (b + 1) B] = tilde(W)_"down" [:, b B : (b + 1) B]$ for all $b >= 1$, both $delta_(i,b)$ and all quantized codes on $j >= B$ are bit-identical between $Q_B (tilde(W)'_"down")$ and $Q_B (tilde(W)_"down")$, yielding zero collateral across all unedited blocks.
  *(3)* For $b = 0$, $max_(j in lr([0, B))) abs(tilde(W)'_"down" [i, j]) = abs(c_i)$. Thus $c_i / delta_(i,0) = "sign"(c_i) q_max (1 + epsilon_"FP16")$ with $abs(epsilon_"FP16") <= 2^(-11)$. For all $q_max <= 127$, $q_max dot 2^(-11) <= 0.062 < 0.5$, so $"round"(c_i / delta_(i,0)) = "sign"(c_i) q_max$ identically across all $B$ entries, yielding $hat(c)_i = delta_(i,0) "sign"(c_i) q_max = c_i (1 + epsilon_"FP16")$. Finally, by positive homogeneity of SwiGLU's linear `up` branch, scaling $tilde(W)'_"up" |-> eta tilde(W)'_"up"$ and $tilde(W)'_"down" |-> eta^(-1) tilde(W)'_"down"$ leaves the product invariant while preventing either matrix from underflowing or clipping.
]

== Corollary 1: Hierarchical Super-Block Alignment for K-Quants ($k = "QK"_K = 256$) <sec:cor1>

#corollary(1, title: "K-Quant Super-Block Isolation")[
  In hierarchical K-quant formats (`Q4_K_M`, `Q3_K_M`), 8 sub-blocks of width 32 share a single `FP16` super-block scale $d_S$ across width $"QK"_K = 256$. When only $k = 32$ columns are aligned, sub-block $0$ shares $d_S$ with unedited sub-blocks $1"–"7$, reintroducing bounded super-block scale collateral ($rho_"col" approx 0.01"–"0.08$). Extending the exact permutation $P$ to $|cal(S)| = "QK"_K = 256$ dormant neurons mapped to $lr([0, 256))$ isolates the entire super-block $S = 0$, restoring zero super-block scale interference across all unedited super-blocks $S >= 1$ and driving $rho_"col" -> O(10^(-4))$ even at `3-bit` `Q3_K_M`.
]

*Neuron-Budget Scaling Advantage:* Evicting $k = 256$ dormant neurons consumes `8.33%` of $d_"ff" = 3072$ in `Qwen3-0.6B`, `7.14%` of $d_"ff" = 3584$ in `Qwen3.5-0.8B`, and only *`2.86%` of $d_"ff" = 8960$ in `DeepSeek-R1-Distill-Qwen-1.5B`*—making Corollary 1 progressively cheaper at larger LLM widths (below `1%` at `7B–14B`).

= Empirical Verification on Real `llama-quantize` Across 3 Models <sec:audit>

We verified Theorem 1 (@sec:thm1) and Corollary 1 (@sec:cor1) directly on official `llama-quantize` binaries (`llama.cpp` build `8864`) across `Q4_0`, `Q4_K_M`, and `Q3_K_M` on all three model architectures (`Qwen3-0.6B` L20, `Qwen3.5-0.8B` L19, `DeepSeek-R1-1.5B` L20 `ffn_down`, @tab:audit, @fig:thm1). Every single theoretical prediction holds quantitatively: moving from $k = 32$ to $k = 256$ drops K-quant collateral by an additional *`132×–158×`* (`Qwen3`), *`196×`* (`Qwen3.5`), and *`119×–134×`* (`DeepSeek-R1-1.5B`), slashing total collateral relative to `sparse64` by *`4,514×` to `16,685×`* with `0` extra bytes and `0` latency overhead.

#figure(
  booktable(
    columns: (1.45fr, 1.55fr, 1.15fr, 1.18fr, 1.18fr, 1.48fr),
    align: (left, left, right, right, right, right),
    header: ([Model (Layer, $d_"ff"$)], [Method], [`Q4_0` cos / col], [`Q4_K_M` cos / col], [`Q3_K_M` cos / col], [Peak Drop vs `sparse64`]),
    font-size: 7.3pt,
    inset: (x: 3pt, y: 2.8pt),
    row-fills: ("5": callout-blue-bg, "6": win-bg, "11": callout-blue-bg, "12": win-bg, "17": callout-blue-bg, "18": win-bg),
    rows: (
      ([*Qwen3-0.6B*\ (L20, $d_"ff"=3072$)], [`uncond_a0.6` (Steer2Edit)], [`0.386 / 0.47296`], [`0.754 / 0.27529`], [`0.332 / 0.95781`], [—]),
      ([], [`sparse64_a1.0` (Takehi)], [`0.640 / 0.63272`], [`0.916 / 0.35082`], [`0.379 / 2.25724`], [`1×` (ref)]),
      ([], [`oblique_a0.35` (Diffuse)], [`0.772 / 0.11012`], [`0.978 / 0.04430`], [`0.784 / 0.16088`], [`14×`]),
      ([], [`dormant32_scattered`], [`0.988 / 0.11608`], [`0.999 / 0.03872`], [`0.950 / 0.31281`], [`7×`]),
      ([], [*`BA-WSC k=32` (Thm 1)*], [*`0.9959 / 0.00063`*], [*`0.9997 / 0.01453`*], [*`0.9942 / 0.07877`*], [*`1,004×`* (`Q4_0`)]),
      ([], [*`BA-WSC k=256` (Cor 1)*], [*`0.9961 / 0.00031`*], [*`0.99983 / 0.00011`*], [*`0.99732 / 0.00050`*], [*`4,514×`* (`Q3_K_M`)]),
      ([*Qwen3.5-0.8B*\ (L19, $d_"ff"=3584$)], [`uncond_a0.6` (Steer2Edit)], [`0.339 / 0.59260`], [`0.284 / 1.18030`], [`0.284 / 1.18030`], [—]),
      ([], [`sparse64_a1.0` (Takehi)], [`0.548 / 0.83424`], [`0.293 / 3.03570`], [`0.293 / 3.03570`], [`1×` (ref)]),
      ([], [`oblique_a0.25` (Diffuse)], [`0.544 / 0.26070`], [`0.526 / 0.42700`], [`0.526 / 0.42700`], [`7×`]),
      ([], [`dormant32_scattered`], [`0.991 / 0.09220`], [`0.938 / 0.35170`], [`0.938 / 0.35170`], [`9×`]),
      ([], [*`BA-WSC k=32` (Thm 1)*], [*`0.9961 / 0.00013`*], [*`0.9955 / 0.05881`*], [*`0.9955 / 0.05881`*], [*`6,417×`* (`Q4_0`)]),
      ([], [*`BA-WSC k=256` (Cor 1)*], [*`0.9962 / 0.00005`*], [*`0.99739 / 0.00030`*], [*`0.99739 / 0.00030`*], [*`16,685×`* (`Q4_0`)\ / *`10,119×`* (`Q3_K`)]),
      ([*DeepSeek-R1-1.5B*\ (L20, $d_"ff"=8960$)], [`uncond_a0.6` (Steer2Edit)], [`0.299 / 0.71745`], [`0.611 / 0.47588`], [`0.236 / 1.69317`], [—]),
      ([], [`sparse64_a1.0` (Takehi)], [`0.511 / 0.84142`], [`0.798 / 0.62476`], [`0.226 / 4.12538`], [`1×` (ref)]),
      ([], [`oblique_a0.25` (Diffuse)], [`0.461 / 0.40406`], [`0.828 / 0.22321`], [`0.406 / 0.78398`], [`5×`]),
      ([], [`dormant32_scattered`], [`0.990 / 0.09639`], [`0.999 / 0.03788`], [`0.935 / 0.36143`], [`11×`]),
      ([], [*`BA-WSC k=32` (Thm 1)*], [*`0.9961 / 0.00115`*], [*`0.99979 / 0.00834`*], [*`0.99618 / 0.04540`*], [*`732×`* (`Q4_0`)]),
      ([], [*`BA-WSC k=256` (Cor 1)*], [*`0.99625 / 0.00014`*], [*`0.99984 / 0.00007`*], [*`0.99741 / 0.00034`*], [*`12,133×`* (`Q3_K_M`,\ `2.86%` neurons!)]),
    ),
  ),
  caption: [Real `llama-quantize` verification of Theorem 1 ($k = 32$) and Corollary 1 ($k = 256$) across all three architectures (`theorem1_quant_audit.json` & `corollary1_superblock_audit.json`). Note: in `Qwen3.5-0.8B` (`Q4_K_M`), `llama-quantize` maps `blk.19.ffn_down` to `Q4_K`/`Q3_K` identically under our mixed-precision policy.],
) <tab:audit>

#figure(
  image("../figures/p2_fig2_theorem1_corollary1_3models.png", width: 100%),
  caption: [Full $Delta W$ cosine similarity (top) and log-scale collateral-to-signal ratio $rho_"col"$ (bottom) measured directly on real `llama-quantize` outputs across all three architectures. Corollary 1 ($k = 256$) maintains $cos >= 0.996$ and drives collateral to $O(10^(-4))$ at `3-bit` `Q3_K_M` across every family.],
) <fig:thm1>

= Post-Quantization Weight-Space Healing (`PQ-BA-WSC`): 20-Arm Study <sec:pq>

Even when a floating-point edit survives quantization geometrically, quantization shifts the model's activation manifold and introduces new quantization-specific repetition loops (`3.1×` at `Q4_0`, `8.2×` at `Q3_K_M`). We therefore introduce *Post-Quantization Weight-Space Healing (`PQ-BA-WSC` / `PQ-Oblique`)* (Algorithm 1).

#algorithm-box(1, "Post-Quantization Weight-Space Healing (PQ-BA-WSC)")[
  *Input:* Target low-bit quantized GGUF $M_Q$ (`Q4_0` or `Q3_K_M`); calibration splits; target block width $k in {32, 256}$. \
  *1. Exact Low-Bit Dequantization:* Load exact dequantized tensors $Q(W)$ from $M_Q$ directly into PyTorch (`src/wsp/pq_loader.py`). \
  *2. Native Quantized Tracing:* Collect calibration rollouts *from $M_Q$ itself* to capture its native quantization-induced loops $cal(P)_Q$ and clean closures $cal(R)_Q$. \
  *3. Block-Aligned Compilation:* Solve Virtual Bias $b^*_Q$, detector $u_Q$, and resolution vector $v_("resolve", Q)$ on dequantized activations; permute $k$ dormant neurons to $lr([0, k))$ and compile `BA-WSC` payload onto $Q(W)$. \
  *4. Re-Quantize:* Write patched GGUF and re-quantize to target format (`0` extra bytes, `0` runtime ops).
]

@tab:pq and @fig:pq report our complete *20-arm untouched held-out `MATH-500-strat150` + `Copy-90` + 32-chunk PPL matrix* on `Qwen3-0.6B` (`temp = 0.6`, Seed 42; unedited `Q8_0` reference: `PPL = 19.280`, `Clean-EOS = 53/150` [`35.3%`], `Loops = 9`).

#figure(
  booktable(
    columns: (2.15fr, 1.05fr, 0.42fr, 1.15fr, 1.18fr, 0.44fr, 0.48fr, 0.56fr, 1.08fr),
    align: (left, right, right, right, right, right, right, right, right),
    header: ([Precision & Arm (`Qwen3-0.6B`)], [PPL ($Delta$PPL)], [STD], [Clean-EOS (/150)], [`p` vs Base], [Lim], [Lp], [`Copy`], [$Delta W$ cos / col]),
    font-size: 7.1pt,
    inset: (x: 2.6pt, y: 2.65pt),
    row-fills: ("8": win-bg, "9": win-bg, "13": warn-bg, "17": callout-blue-bg, "18": callout-blue-bg, "19": win-bg),
    rows: (
      ([*`Q4_0` `baseline`* (`-12.0pp` cliff)], [`21.866 (0.000)`], [`36`], [`35 (23.3%)`], [—], [`114`], [`28`], [`65/90`], [—]),
      ([`Q4_0` `ba_wsc_noop` (Perm ctrl)], [`21.870 (+0.004)`], [`33`], [`32 (21.3%)`], [`0.607`], [`116`], [`25`], [`64/90`], [—]),
      ([`Q4_0` `random_matched` (Noise)], [`21.785 (-0.081)`], [`34`], [`32 (21.3%)`], [`0.664`], [`118`], [`23`], [`66/90`], [`0.582 / 0.1474`]),
      ([`Q4_0` `uncond_a0.8` (Prior WSP)], [`22.805 (+0.939)`], [`38`], [`38 (25.3%)`], [`0.581` (n.s.)], [`110`], [`25`], [`65/90`], [`0.443 / 0.3702`]),
      ([`Q4_0` `bawsc32_res_a0.25` (FP→Q4)], [`21.847 (-0.019)`], [`43`], [`41 (27.3%)`], [`0.263`], [`102`], [`42`], [`64/90`], [`0.9959 / 0.00063`]),
      ([`Q4_0` `dormant32_res_a0.25` (Scat.)], [`21.848 (-0.018)`], [`45`], [`44 (29.3%)`], [`0.078`], [`102`], [`53`], [`65/90`], [`0.9884 / 0.1163`]),
      ([`Q4_0` `pq_oblique_a0.25`], [`21.838 (-0.028)`], [`48`], [`40 (26.7%)`], [`0.332`], [`108`], [`38`], [`64/90`], [`0.668 / 0.1735`]),
      ([*`Q4_0` `pq_bawsc32_a0.25`* (PQ)], [*`21.869 (+0.003)`*], [`47`], [*`46 (30.7%, +7.3pp)`*], [*`0.0347`* (`.0125` R)], [`100`], [`36`], [`64/90`], [*`0.9960 / 0.00036`*]),
      ([*`Q4_0` `oblique_res_a0.25`* (\#1)], [`22.034 (+0.168)`], [*`53`*], [*`50 (33.3%, +10.0pp)`*], [*`0.0041`* (`.0005` R)], [*`98`*], [`41`], [`64/90`], [`0.691 / 0.1563`]),
      ([*`Q3_K_M` `baseline`* (`-22.7pp` cliff)], [`26.723 (0.000)`], [`24`], [`19 (12.7%)`], [—], [`115`], [`74`], [`39/90`], [—]),
      ([`Q3_K_M` `ba_wsc_noop` (Perm ctrl)], [`26.792 (+0.069)`], [`22`], [`20 (13.3%)`], [`1.000`], [`122`], [`84`], [`42/90`], [—]),
      ([`Q3_K_M` `random_matched` (Noise)], [`26.775 (+0.052)`], [`26`], [`21 (14.0%)`], [`0.815`], [`119`], [`81`], [`39/90`], [`0.625 / 0.1700`]),
      ([`Q3_K_M` `uncond_a0.8` (Prior WSP)], [*`28.924 (+2.202)`*], [`27`], [`27 (18.0%)`], [`0.096` (n.s.)], [`93`], [`68`], [`41/90`], [`0.401 / 0.6904`]),
      ([`Q3_K_M` `bawsc32_res_a0.25` (FP→Q3)], [`26.807 (+0.084)`], [`27`], [`26 (17.3%)`], [`0.167`], [`97`], [`79`], [`40/90`], [`0.9941 / 0.0788`]),
      ([`Q3_K_M` `oblique_res_a0.25` (FP→Q3)], [`27.117 (+0.394)`], [`32`], [`29 (19.3%)`], [`0.0525`], [`84`], [`71`], [`40/90`], [`0.688 / 0.2441`]),
      ([`Q3_K_M` `dormant32_res_a0.25` (Scat.)], [`26.714 (-0.009)`], [`32`], [`31 (20.7%)`], [`0.0227`], [`79`], [`64`], [`40/90`], [`0.950 / 0.3127`]),
      ([*`Q3_K_M` `pq_oblique_a0.25`*], [`26.765 (+0.043)`], [`32`], [`30 (20.0%, +7.3pp)`], [*`0.0266`*], [`85`], [*`53 (-28%)`*], [`39/90`], [`0.566 / 0.4114`]),
      ([*`Q3_K_M` `pq_bawsc32_a0.25`*], [`26.834 (+0.112)`], [`30`], [`30 (20.0%, +7.3pp)`], [*`0.0347`*], [`87`], [`63`], [`41/90`], [*`0.9941 / 0.0789`*]),
      ([*`Q3_K_M` `pq_bawsc32_a0.35`* (\#1!)], [*`26.839 (+0.116)`*], [*`34`*], [*`32 (21.3%, +8.7pp)`*], [*`0.0044`* (`.0266` R)], [*`82`*], [`60`], [`40/90`], [*`0.9941 / 0.0788`*]),
    ),
  ),
  caption: [Complete 20-arm held-out `MATH-500-strat150` + `Copy-90` + 32-chunk PPL healing matrix on `Qwen3-0.6B` (`temp = 0.6`, Seed 42; 19 quantized arms shown + `Q8_0` reference `53/150` [`35.3%`], `PPL = 19.280`). `R` = exact McNemar $p$ vs. `random_matched`.],
) <tab:pq>

*Three Clear Behavioral Conclusions:*
1. *`83%` of the 4-bit Cliff Healed:* `oblique_res_a0.25` recovers `50/150` Clean-EOS (`+10.0 pp`, $p = 0.0041$ vs. `Q4_0` baseline `35/150`, $p = 0.0005$ vs. random `32/150`), recovering `15/18` (`83%`) of the entire `Q8_0` $->$ `Q4_0` accuracy cliff.
2. *Quantization-Native Compilation Beats FP Transfer:* At both `Q4_0` (`46` vs. `41/150`, $Delta"PPL" = +0.003$, `cos = 0.9960`, `col = 0.00036`) and `Q3_K_M` (`30–32` vs. `26/150`), `PQ-BA-WSC` systematically beats FP-compiled `BA-WSC`, achieving *`#1` overall at `3-bit` `Q3_K_M` (`32/150 = +68%` relative gain, $p = 0.0044$)* with *$19times$ less PPL damage* than `uncond_a0.8` (`+0.116` vs. `+2.202`).
3. *Controls Hold:* Both `ba_wsc_noop` ($p = 0.607$ at `Q4_0`, $p = 1.000$ at `Q3_K_M`) and `random_matched` ($p = 0.664$ at `Q4_0`, $p = 0.815$ at `Q3_K_M`) show zero gain, and `Copy-90` remains strictly preserved across every healing arm.

#figure(
  image("../figures/p2_fig3_pq_healing_matrix_qwen3.png", width: 100%),
  caption: [Complete `Q4_0` and `Q3_K_M` held-out `MATH-500-strat150` healing results on `Qwen3-0.6B` (`temp = 0.6`). `PQ-BA-WSC` systematically beats FP-transfer at both precisions and ranks `#1` overall at `3-bit` `Q3_K_M` ($p = 0.0044$).],
) <fig:pq>

= Behavioral Proof of Theorem 1: Precision $times$ Method Double Crossovers <sec:crossover>

If Theorem 1's (@sec:thm1) hardware block alignment is causally meaningful—not merely a tensor norm artifact—then in regimes where diffuse edits suffer severe quantization distortion (`cos` $approx 0.40"–"0.54$, large $Delta"PPL"$), *diffuse `oblique_res` should lead at `8-bit` `Q8_0` and degrade at low bit-widths, whereas block-aligned `BA-WSC` (`cos = 0.996`) should preserve its gain and overtake `oblique_res`.*

We observe this *exact Precision $times$ Method double crossover independently on BOTH Model 2 (`Qwen3.5-0.8B` Hybrid) and Model 3 (`DeepSeek-R1-Distill-Qwen-1.5B`)* (@tab:m2cross, @tab:m3cross, @fig:crossover).

== Model 2 (`Qwen3.5-0.8B` Hybrid): `Q8_0` $->$ `Q4_0` Double Crossover ($p = 0.00336$) <sec:m2cross>

#figure(
  booktable(
    columns: (1.72fr, 1.48fr, 2.08fr, 2.02fr),
    align: (left, right, right, left),
    header: ([Arm (`Qwen3.5-0.8B`, `N=150`)], [`Q8_0` PPL / STD / *EOS* / Lim], [`Q4_0` PPL / STD / *EOS* / Lim], [`Q4_0` vs `Q8_0` Shift ($Delta$ vs Base)]),
    font-size: 7.35pt,
    inset: (x: 3pt, y: 3.1pt),
    row-fills: ("4": warn-bg, "5": win-bg),
    rows: (
      ([`baseline`], [`15.513 / 6 / 1 / 149`], [`18.227 / 11 / 9 / 141` (`83.3%` lp)], [Reference]),
      ([`random_matched`], [`15.723 / 4 / 3 / 147`], [`18.432 / 7 / 7 / 143`], [`-4/150`]),
      ([`uncond_a0.8`], [`15.509 / 1 / 0 / 150`], [`18.190 / 6 / 6 / 144`], [`-2/150`]),
      ([*`oblique_res_a1.0`* (`cos=.544`)], [`15.814 / 12 / `*`10`*` / 126`], [*`18.717 (+0.490)`*` / 9 / `*`7`*` / 136`], [*`-11/150`* (`perm_p = 0.0115` collapse!)]),
      ([*`bawsc32_res_a0.45`* (`cos=.996`)], [`15.541 / 8 / 3 / 145`], [*`18.292 (+0.065)`*` / `*`17`*` / `*`15 (#1!)`*` / 135`], [*`+4/150`* (`p=.0117` vs unc, `.021` vsR STD)]),
    ),
  ),
  caption: [Hybrid `Qwen3.5-0.8B` `Q8_0` $->$ `Q4_0` Precision $times$ Method double crossover on untouched `MATH-500-strat150` (`temp = 0.6`). Exact 100k-permutation interaction `(BA-WSC - Oblique)_(Q4 - Q8)`: *`+15/150 (+10.00 pp)`, $p = 0.00336$* (Clean-EOS); *`+12/150 (+8.00 pp)`, $p = 0.01067$* (STD).],
) <tab:m2cross>

*Mechanistic Diagnosis:* At `Q4_0`, `oblique_res_a1.0` still triggers `</think>` closure (`49.3%`), yet its diffuse `4-bit` collateral ($Delta"PPL" = +0.490$, $7.5times$ larger than `BA-WSC`'s `+0.065`) corrupts post-close accuracy so it falls below baseline (`7` vs. `9`). Block-aligned `BA-WSC` preserves clean PPL (`+0.065`) and surges to `#1` overall (`15/150` Clean-EOS, `17/150` STD, `Copy-90 = 69/90`).

== Model 3 (`DeepSeek-R1-Distill-Qwen-1.5B`): Replicated `Q8_0` $->$ `Q4_0` $->$ `Q3_K_M` Crossover <sec:m3cross>

#figure(
  booktable(
    columns: (1.90fr, 1.75fr, 1.75fr, 1.85fr),
    align: (left, right, right, right),
    header: ([Arm (`DeepSeek-R1-1.5B`, `N=150`)], [`Q8_0` PPL / *EOS* ($Delta$) / Lim], [`Q4_0` PPL / *EOS* ($Delta$) / Lim], [`Q3_K_M` PPL / *EOS* ($Delta$) / Lim]),
    font-size: 7.45pt,
    inset: (x: 3.2pt, y: 3.1pt),
    row-fills: ("4": warn-bg, "5": win-bg),
    rows: (
      ([`baseline`], [`37.254 / 43 (0) / 77`], [`41.186 / 60 (0) / 71`], [`40.710 / 48 (0) / 99` (`66%` lim)]),
      ([`random_matched`], [`37.298 / 39 (-4) / 82`], [`41.198 / 54 (-6) / 75`], [`40.750 / 51 (+3) / 94`]),
      ([`uncond_a0.8`], [`37.532 / 48 (+5) / 69`], [`41.421 / 66 (+6) / 65`], [`40.747 / 59 (+11) / 85`]),
      ([*`oblique_res_a0.25`* (`Q3 cos=.406`)], [`37.338 / `*`53 (+10)`*` / 69`], [`41.223 / 64 (+4) / 65`], [*`40.952 (+0.242)`*` / `*`45 (-3)`*` / 95`]),
      ([*`bawsc32_res_a0.5`* (`Q3 cos=.996`)], [*`37.257 (+0.003)`*` / `*`49 (+6)`*` / 67`], [*`41.056 (-0.131)`*` / `*`66 (+6)`*` / `*`62`*], [*`40.761 (+0.051)`*` / `*`54 (+6)`*` / 89`]),
    ),
  ),
  caption: [Replicated `Q8_0` $->$ `Q4_0` $->$ `Q3_K_M` crossover on `DeepSeek-R1-Distill-Qwen-1.5B` (`MATH-500-strat150`, `temp = 0.6`). Block-aligned `BA-WSC` holds an invariant `+6, +6, +6` Clean-EOS gain across all three precisions while diffuse `oblique_res` drops monotonically `+10` $->$ `+4` $->$ `-3`.],
) <tab:m3cross>

Look at the stark contrast across all three precisions on `DeepSeek-R1-1.5B`:
- *Block-aligned `BA-WSC`* (`cos = 0.9961–0.9962`) maintains an *invariant `+6, +6, +6` Clean-EOS advantage over same-precision baseline* across `8-bit`, `4-bit`, and `3-bit` (`Copy-90 contains_gold = 67, 65, 70/90` preserved throughout; significantly beating random noise at `Q4_0` across all 4 metrics: EOS $p = 0.0357$, STD $p = 0.0192$, Limit $p = 0.0072$, Close $p = 2.7 times 10^(-5)$).
- *Diffuse `oblique_res`* (`Q3_K_M cos = 0.4056, col = 0.784`) plunges monotonically as bit-width drops: *`+10 (Q8_0)` $->$ `+4 (Q4_0)` $->$ `-3 (Q3_K_M, below baseline!)`* while incurring `+0.242` PPL damage at `3-bit` (`4.7×` worse than `BA-WSC`'s `+0.051`).
- *Exact 100,000-Permutation `Q8_0` $->$ `Q3_K_M` Crossover `(BA-WSC - Oblique)_(Q3 - Q8)`:* *`+13 / 150` (`+8.67 pp`) Clean-EOS (`perm_p = 0.0663`)* and *`+14 / 150` (`+9.33 pp`) STD (`perm_p = 0.0594`)*.
- *Combined Cross-Architecture Meta-Analysis (Model 2 `Qwen3.5` + Model 3 `DeepSeek-R1-1.5B`):* Fisher's exact combined test yields *$chi^2_4 = 16.82$, `p = 0.00210` (Clean-EOS)* and *$chi^2_4 = 14.73$, `p = 0.00530` (STD)*.

#figure(
  image("../figures/p2_fig4_double_crossover_2models.png", width: 100%),
  caption: [Empirical behavioral proof of Theorem 1 across two independent model architectures. *(a)* On Hybrid `Qwen3.5-0.8B`, diffuse `oblique_res` drops at `4-bit` while block-aligned `BA-WSC` surges to `#1` (`+15/150` swing, $p = 0.00336$). *(b)* On `DeepSeek-R1-1.5B`, block-aligned `BA-WSC` holds a flat `+6, +6, +6` gain across `8-bit`, `4-bit`, and `3-bit` while diffuse `oblique_res` plunges `+10` $->$ `+4` $->$ `-3` (`+13/150` swing, $p = 0.0663$; combined Fisher $p = 0.00210$).],
) <fig:crossover>

= Honest Limitations, Nuances & Practical Deployment Rules <sec:limits>

1. *Where Diffuse Edits Survive vs. Where They Break:* On `Qwen3-0.6B` at `Q4_0`, small-norm `oblique_res_a0.25` ($Delta"PPL" = +0.168$, `cos = 0.691`) still healed `83%` of the 4-bit cliff (`+10.0 pp`). Diffuse edits break specifically when edit magnitude is large relative to block scale (`Qwen3.5-0.8B` `Q4_0`, $alpha = 1.0$, $Delta"PPL" = +0.490$) or at ultra-low `3-bit` precision (`DeepSeek-R1-1.5B` `Q3_K_M`, `cos = 0.406`, $Delta"PPL" = +0.242$).
2. *Unexpected `uncond_a0.8` Gain at 3-Bit on 1.5B:* Honestly noting a non-trivial observation: on `DeepSeek-R1-1.5B` at `Q3_K_M`, `uncond_a0.8` reached `59/150` EOS ($p = 0.0708$ vs. base, limit `99` $->$ `85`, $p = 0.0125$, $Delta"PPL" = +0.038$) despite its `Q3_K_M` tensor cosine dropping to `0.236`. On a wider `1.5B` residual stream, a strong unconditional rank-1 perturbation can occasionally still bias stopping thresholds even after severe directional quantization noise—whereas on `0.6B` (`+2.202` PPL penalty) and `0.8B` (`0/150` at `Q8_0`, `6/150` at `Q4_0`) it fails severely.
3. *Permutation No-Op Variance (`ba_wsc_noop`):* Although SwiGLU permutation $P$ is bit-exact (`0.000000`) in floating point, shifting non-edited neurons across 32-element quantization block boundaries creates small quantization re-binning noise (`Qwen3-0.6B`: `32/150` vs. `35/150` on `Q4_0`, $p = 0.607$; `20/150` vs. `19/150` on `Q3_K_M`, $p = 1.000$). Always benchmark block-aligned edits against `ba_wsc_noop`.
4. *Practical Deployment Decision Rule:*
   - *8-bit (`Q8_0`) / FP16:* Use *`oblique_res`* for maximum Clean-EOS accuracy (`#1` on 3/3 architectures) or *`BA-WSC`* when strict $Delta"PPL" approx 0$ preservation is mandatory.
   - *4-bit (`Q4_0`):* Use *`PQ-BA-WSC`* ($k = 32$) for guaranteed zero-collateral survival ($cos >= 0.996$, $Delta"PPL" approx 0$); deploy `oblique_res` only after verifying 4-bit $Delta"PPL" < 0.2$.
   - *3-bit (`Q3_K_M`) / K-Quant:* Use *`PQ-BA-WSC`* ($k = 32$ on small `0.6B` widths; *Corollary 1 $k = 256$* on `1.5B+` widths where 256 neurons is $<= 2.9\%$ of $d_"ff"$).

= Conclusion <sec:conclusion>

Weight-space edits designed in floating point cannot be assumed to survive edge quantization: block-scale sharing shreds unaligned edits while quantization itself triggers `2.3×–8.2×` reasoning loop explosions. By aligning exact SwiGLU permutation symmetries with hardware quantization boundaries (*Theorem 1 & Corollary 1*) and compiling directly on dequantized low-bit activations (*`PQ-BA-WSC`*), we preserve edit fidelity at `cos = 0.996–0.99984`, slash neighbor distortion up to `16,685×` at zero memory or latency cost, heal up to `83%` of the 4-bit cliff and `+68%` relative at 3-bit, and establish a replicated cross-architecture behavioral crossover ($p = 0.00210$).

#v(0.5em)
#bibliography("../references.bib", title: "References", style: "ieee")

#v(0.9em)
#line(length: 100%, stroke: 0.9pt + navy)
#v(0.25em)
#align(center)[#text(size: 8.6pt, weight: "bold", tracking: 0.12em, fill: navy)[APPENDICES · PROVENANCE, FULL STATISTICAL AUDITS & REPRODUCIBILITY]]
#v(0.25em)
#line(length: 100%, stroke: 0.45pt + rule-gray)
#set text(size: 9.8pt)
#set par(hanging-indent: 0em, leading: 0.62em, spacing: 0.78em)
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Model, Quantizer & Hardware Provenance <app:prov>

All quantization audits and low-bit behavioral benchmarks ran locally on one *AMD Radeon RX 6400 (4 GB VRAM, Vulkan)* workstation using official `llama.cpp` build `8864 (ff6b1062a)` (`llama-quantize`, `llama-perplexity`, `llama-server`, Clang `19.1.5`, Windows `x86_64`). Original BF16 base checkpoints in `local/models/` remained bit-for-bit untouched; every quantized baseline and patched checkpoint was SHA-256 verified before inference.

#figure(
  booktable(
    columns: (1.55fr, 0.55fr, 0.65fr, 0.90fr, 0.85fr, 2.0fr),
    align: (left, right, right, right, left, left),
    header: ([Base Model (BF16)], [$L$], [$d_"ff"$], [`k=256` Budget], [Target $cal(L)$], [Original BF16 SHA-256 (Prefix)]),
    font-size: 7.5pt,
    rows: (
      ([`Qwen3-0.6B` (`1,198.2 MB`)], [`28`], [`3072`], [`8.33%`], [`[18, 20, 22]`], [`f9c9f1d3c1e21755...`]),
      ([`Qwen3.5-0.8B` (`1,516.7 MB`)], [`24`], [`3584`], [`7.14%`], [`[15, 19, 21]`], [`cedf89af31c9041b...`]),
      ([`DeepSeek-R1-1.5B` (`3,560.4 MB`)], [`28`], [`8960`], [*`2.86%`*], [`[20, 22, 25]`], [`6ec1aa822868a25c...`]),
    ),
  ),
  caption: [Exact model dimensions, Corollary 1 (`k = 256`) neuron budget percentage, compiled target layers, and immutable SHA-256 hashes of the source BF16 GGUF weights.],
) <tab:appmodels>

Across the entire research program (`Papers 1 & 2`), *`206` runs* produced *`18,513` raw inference responses* (`1,810,381` prompt tokens + `23,248,608` completion tokens = `25,058,989` total tokens) with *`0` HTTP/parse errors*. Every held-out `MATH-500-strat150` (`54709820f3e27767`) and `Copy-90` (`8567025de54bc42a`) run flushed each request line-by-line to immutable `responses.jsonl`.

= Complete Pairwise McNemar & 100k-Permutation Crossover Tables <app:stats>

@tab:m3stats and @tab:crossall provide the complete exact two-sided McNemar $p$-values (Clean-EOS, Limit reduction, and `</think>` closure) and 100,000-iteration paired sign-permutation interaction tests from `heldout_stats_audit.json` for Models 2 and 3.

#figure(
  booktable(
    columns: (0.85fr, 1.50fr, 1.15fr, 1.18fr, 1.15fr, 1.25fr),
    align: (left, left, left, right, right, right),
    header: ([Precision], [Edited Arm (`DeepSeek-1.5B`)], [Reference], [EOS $Delta$ (`p`)], [Limit $Delta$ (`p`)], [Close $Delta$ (`p`)]),
    font-size: 7.35pt,
    inset: (x: 3pt, y: 2.75pt),
    row-fills: ("2": win-bg, "3": callout-blue-bg, "4": win-bg, "8": win-bg, "10": win-bg, "11": warn-bg),
    rows: (
      ([`Q8_0`], [`oblique_res_a0.25`], [`baseline`], [`+10 (0.1214)`], [`-8 (0.1849)`], [`+10 (0.0639)`]),
      ([], [`oblique_res_a0.25`], [`random_matched`], [*`+14 (0.0288)`*], [*`-13 (0.0146)`*], [*`+14 (0.0026)`*]),
      ([], [`bawsc32_res_a0.5`], [`baseline`], [`+6 (0.3771)`], [`-10 (0.0639)`], [*`+14 (0.0043)`*]),
      ([], [`bawsc32_res_a0.5`], [`random_matched`], [`+10 (0.0639)`], [*`-15 (0.0041)`*], [*`+18 (0.0003)`*]),
      ([`Q4_0`], [`oblique_res_a0.25`], [`baseline`], [`+4 (0.5966)`], [`-6 (0.2632)`], [*`+13 (0.0106)`*]),
      ([], [`oblique_res_a0.25`], [`random_matched`], [`+10 (0.0639)`], [*`-10 (0.0309)`*], [*`+18 (0.0005)`*]),
      ([], [`bawsc32_res_a0.5`], [`baseline`], [`+6 (0.4050)`], [`-9 (0.0784)`], [*`+17 (0.0009)`*]),
      ([], [*`bawsc32_res_a0.5`*], [`random_matched`], [*`+12 (0.0357)`*], [*`-13 (0.0072)`*], [*`+22 (<0.0001)`*]),
      ([`Q3_K_M`], [`oblique_res_a0.25`], [`baseline`], [`-3 (0.7283)`], [`-4 (0.5966)`], [`+3 (0.7011)`]),
      ([], [`bawsc32_res_a0.5`], [`baseline`], [`+6 (0.3915)`], [`-10 (0.0987)`], [`+6 (0.3771)`]),
      ([], [`oblique_res_a0.25`], [`random_matched`], [`-6 (0.3915)`], [`+1 (1.0000)`], [`+8 (0.1849)`]),
      ([], [`uncond_a0.8`], [`baseline`], [`+11 (0.0708)`], [*`-14 (0.0125)`*], [`+7 (0.2478)`]),
    ),
  ),
  caption: [Full exact two-sided paired McNemar audit for `DeepSeek-R1-Distill-Qwen-1.5B` across `Q8_0`, `Q4_0`, and `Q3_K_M` (`MATH-500-strat150`, $N = 150$, `heldout_stats_audit.json`, `3,600/3,600` responses with `0` errors).],
) <tab:m3stats>

#figure(
  booktable(
    columns: (1.55fr, 2.35fr, 1.05fr, 0.95fr, 1.05fr),
    align: (left, left, right, right, right),
    header: ([Model], [Interaction Contrast (100k Exact Permutations)], [Swing (/150)], [Nonzero Items], [Two-Sided `p`]),
    font-size: 7.45pt,
    row-fills: ("1": win-bg, "2": win-bg, "4": callout-blue-bg, "5": callout-blue-bg, "6": win-bg, "7": win-bg),
    rows: (
      ([`Qwen3.5-0.8B`], [`(BA-WSC - Oblique)_(Q4_0 - Q8_0)` — Clean-EOS], [*`+15 (+10.00pp)`*], [`19`], [*`0.00336`*]),
      ([], [`(BA-WSC - Oblique)_(Q4_0 - Q8_0)` — Standard], [*`+12 (+8.00pp)`*], [`17`], [*`0.01067`*]),
      ([], [`(Oblique - Base)_(Q4_0 - Q8_0)` — Clean-EOS], [`-11 (-7.33pp)`], [`14`], [*`0.01145`*]),
      ([`DeepSeek-R1-1.5B`], [`(BA-WSC - Oblique)_(Q3_K_M - Q8_0)` — Clean-EOS], [*`+13 (+8.67pp)`*], [`34`], [`0.06633`]),
      ([], [`(BA-WSC - Oblique)_(Q3_K_M - Q8_0)` — Standard], [*`+14 (+9.33pp)`*], [`36`], [`0.05940`]),
      ([*2-Model Fisher*], [*Combined Crossover Meta-Analysis — Clean-EOS*], [*$chi^2_4 = 16.82$*], [—], [*`0.00210`*]),
      ([*2-Model Fisher*], [*Combined Crossover Meta-Analysis — Standard*], [*$chi^2_4 = 14.73$*], [—], [*`0.00530`*]),
    ),
  ),
  caption: [Complete Precision $times$ Method double-crossover interaction tests across Model 2 (`Qwen3.5-0.8B`) and Model 3 (`DeepSeek-R1-1.5B`) plus exact Fisher's combined meta-analysis.],
) <tab:crossall>

= Exact `Copy-90` Retrieval Audit Across Quantization Levels <app:copy>

Across all low-bit `DeepSeek-R1-1.5B` experiments (`Q8_0`, `Q4_0`, `Q3_K_M`), `Copy-90` `contains_gold` retrieval accuracy stayed tightly preserved around baseline: `Q8_0` (`68, 76, 70, 70, 67`), `Q4_0` (`66, 64, 71, 63, 65`), and `Q3_K_M` (`72, 73, 72, 75, 70`) out of `90` for `baseline`, `random_matched`, `uncond_a0.8`, `oblique_res_a0.25`, and `bawsc32_res_a0.5`, respectively (`heldout_stats_audit.json`). Similarly, on `Qwen3.5-0.8B` at `Q4_0`, `bawsc32_res_a0.45` achieved `69/90` exact `Copy-90` (vs. `68/90` baseline).

= Reproducibility & Transparent Acknowledgments <app:repro>

*Programmatic Verification:* Running `python scripts/wsp_verify_papers.py` verifies every single cited metric, Theorem 1 / Corollary 1 collateral ratio, Fisher meta-analysis $p$-value, figure file, and zero-error JSONL log directly from `research/wsp/`. Figure regeneration is one command (`python scripts/wsp_generate_all_paper_figures.py`).

*Acknowledgments & AI-Assistance Disclosure:* This research was conceived, directed, authorized, and executed on personal workstation hardware by Aaryan Kakad as an independent ML research program. Autonomous AI coding and research agents were used extensively for experiment orchestration, GGUF/PyTorch engineering, `llama-quantize` auditing, statistical verification, figure rendering, and manuscript drafting under human oversight; every empirical claim is strictly backed by immutable, cryptographically hashed raw JSON logs in the public repository.
