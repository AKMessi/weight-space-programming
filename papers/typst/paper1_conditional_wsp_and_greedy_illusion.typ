#import "template.typ": *

#show: paper.with(
  title: "The Greedy-Perturbation Illusion and Zero-Gradient Control-Flow Compilation in Weight Space",
  short-title: "The Greedy-Perturbation Illusion and Zero-Gradient Weight-Space Compilation",
  author: "Aaryan Kakad",
  affiliation: "Independent Machine Learning Researcher",
  date: "November 2026",
  companion-note: "Companion Paper 1 of 2 — see Paper 2 for Quantization Survival (Theorem 1, Corollary 1) and Post-Quantization Healing.",
  keywords: (
    "Weight-Space Programming",
    "Reasoning LLMs",
    "Greedy-Perturbation Illusion",
    "Mechanistic Interpretability",
    "SwiGLU Compilation",
    "Zero-Gradient Editing",
  ),
  abstract: [
    Small reasoning language models (`0.6B–1.5B`) routinely exhaust their token budget through recursive overthinking and repetition loops inside `<think>` traces. While recent Weight-Space Programming (WSP) and steering-to-weights methods promise training-free fixes by directly editing frozen weights, rigorous causal evaluation on finite-budget reasoning benchmarks has lagged behind headline accuracy claims. We first expose two severe methodological traps that invalidate naive weight-editing evaluations: *(1) The Greedy-Perturbation Illusion*, wherein injecting pure zero-information isotropic Gaussian noise into frozen MLP weights boosts held-out `MATH-500` accuracy on `Qwen3-0.6B` by `+6.0` percentage points (`34.7%` $->$ `40.7%`, spanning a `6.0 pp` chaotic band across 6 zero-information draws) under greedy decoding (`temp = 0.0`), yet collapses completely to `-0.33 pp` ($p = 1.000$) under multi-seed official stochastic sampling (`temp = 0.6`, $N = 300$) and degrades OOD `GSM8K` by `-7.0 pp`; and *(2) The Final-Answer Echo Trap*, wherein high-dose weight edits score `+8.7 pp` ($p = 0.029$) under standard `\boxed{}` grading at zero perplexity loss while doubling repetition loops (`29` $->$ `59`, $p = 0.0001$) and plunging below random noise (`32.0%`, $p = 0.011$) under strict Clean-EOS auditing.

    To address overthinking causally under strict stochastic controls, we build a `0.00e+00`-diff GGUF-to-PyTorch mechanistic tracer and uncover a *universal late-MLP reasoning-control surge* (`10×–35×` jump in pre-readout MLP attribution) alongside a stable post-RMSNorm *Virtual Constant-1 Bias Register* $b^*$ (out-of-sample $"std"(b^(*top) x) = 0.020"–"0.074$) across three distinct architectures: pure GQA Transformers (`Qwen3-0.6B`), hybrid Gated-DeltaNet + Full-Attention (`Qwen3.5-0.8B`), and RL-distilled Qwen2 (`DeepSeek-R1-Distill-Qwen-1.5B`). Leveraging this geometry, we introduce two zero-gradient, zero-extra-byte, zero-latency compilers: *(i) Null-Space-Gated Oblique Surgery* (`oblique_res`), which steers stalled post-SwiGLU activations toward `</think>` resolution orthogonally to preservation subspaces, and *(ii) Nonlinear Dormant-SwiGLU `IF-THEN` Compilation* (`BA-WSC`), which compiles affine threshold gates into bias-free dormant SwiGLU neurons via $b^*$ at virtually zero perplexity penalty ($Delta"PPL" in [-0.018, +0.028]$). Across $N = 600$ untouched held-out `MATH-500` stochastic rollouts spanning all three architectures (`Q8_0`, `temp = 0.6`), `oblique_res` ranks `#1` on *3/3 architectures*, improving strict Clean-EOS accuracy by *`+6.00 pp` over baseline* (`+61 / -25`, exact McNemar $p = 0.000130$, Fisher $p = 0.000586$) and *`+6.50 pp` over norm-matched random noise* (`+65 / -26`, $p = 5.31 times 10^(-5)$), while cutting generation wall-clock time by `20%–30%`. Calibrated solely on math, exact-copy, and WikiText-2 with *zero science data*, our edits transfer out-of-distribution to `GSM8K-100` (`+9.0 pp` vs. baseline $p = 0.049$, `+16.0 pp` vs. random $p = 0.0001$) and zero-shot to non-math science `ARC-Challenge-100` (`oblique_res +15.0 pp`, $p = 0.0003$; `BA-WSC +11.0 pp`, $p = 0.0074$), demonstrating compilation of a domain-general metacognitive resolution circuit.
  ],
)

= Introduction <sec:intro>

Chain-of-thought reasoning models trained via reinforcement learning or distillation (`Qwen3`, `Qwen3.5`, `DeepSeek-R1-Distill`) achieve remarkable accuracy per parameter, yet suffer from severe inference-time pathologies under practical token budgets (e.g., `2048` tokens): models routinely compute the correct answer early inside `<think>`, then enter recursive self-doubt ("Wait, let me double-check...") or verbatim repetition loops until forced truncation. Existing mitigations fall into three costly or brittle categories: (1) post-training or RL fine-tuning, which requires gradient updates, large datasets, and risks catastrophic forgetting; (2) decoding heuristics such as presence penalties, DRY repetition samplers, and concise system prompts, which we show fail or actively harm mathematical reasoning (`DRY` drops `Qwen3-0.6B` development accuracy from `36.7%` to `23.3%`, $p = 0.039$); and (3) runtime activation steering hooks, which require interceptor code and break native quantized inference engines such as `llama.cpp` @gerganov2023.

*Weight-Space Programming (WSP)* offers a fourth path: compile behavioral control flow directly into frozen model weights via closed-form linear algebra—with *zero gradients, zero extra parameters, and zero runtime latency overhead*. However, systematic investigation across small local reasoning models revealed a sobering methodological reality: naive weight-space edits are remarkably easy to mistake for genuine breakthroughs when evaluated under standard greedy math protocols.

This paper makes *four primary contributions*:
1. *Exposure and Formal Explanation of Two Evaluation Traps (@sec:traps):* We document the *Greedy-Perturbation Illusion*—demonstrating that zero-information isotropic Gaussian noise produces a `+6.0 pp` Clean-EOS jump on `MATH-500` at `temp = 0.0` via asymmetric reservoir trajectory rerolling (Proposition 1), then vanishes completely to `-0.33 pp` ($p = 1.000$) under official multi-seed stochastic sampling (`temp = 0.6`). We further expose the *Final-Answer Echo Trap*, proving standard `\boxed{}` graders mask severe post-answer repetition loops, and establish *Strict Clean-EOS* auditing (Definition 1) alongside an inside-`<think>` SwiGLU `W_up` AND-gate.
2. *Universal Cross-Architecture Reasoning-Control Anatomy & Virtual Bias Register $b^*$ (@sec:method):* Using a `0.00e+00`-diff GGUF-to-PyTorch bridge, we map every decoder layer across pure GQA (`Qwen3-0.6B`), hybrid Gated-DeltaNet (`Qwen3.5-0.8B`), and RL-distilled Qwen2 (`DeepSeek-R1-1.5B`), revealing the exact same `10×–35×` late-MLP pathology surge and an out-of-sample constant-1 direction $b^*$ ($"std" = 0.020"–"0.074$) enabling exact affine thresholds inside bias-free SwiGLU MLPs.
3. *Two Closed-Form Conditional Compilers (@sec:oblique–@sec:bawsc):* We derive *Null-Space-Gated Oblique Surgery* (`oblique_res`) and *Nonlinear Dormant-SwiGLU `IF-THEN` Compilation* (`BA-WSC`), rigorously benchmarking both against prior unconditional subspace excision (`uncond`), Frobenius-matched random noise (`random_matched`), and non-weight decoding controls.
4. *Cross-Architecture & Zero-Shot Domain Generalization Proof (@sec:results):* Under strict anti-leakage splits and stochastic sampling (`temp = 0.6`), `oblique_res` beats both baseline and matched random noise across *3/3 distinct architectures* ($N = 600$ pooled `MATH-500` rollouts: `+6.00 pp`, $p = 0.000130$ vs. baseline; `+6.50 pp`, $p = 5.31 times 10^(-5)$ vs. random), while transferring out-of-distribution to `GSM8K-100` (`+9 pp` vs. baseline, `+16 pp` vs. random, `30%` faster) and *zero-shot to non-math science `ARC-Challenge-100`* (`+15 pp`, $p = 0.0003$; `BA-WSC +11 pp`, $p = 0.0074$ at `-0.018` $Delta"PPL"$).

_Companion Paper 2_ analyzes the quantization geometry of weight-space edits, proves Theorem 1 and Corollary 1 for `4-bit` and `3-bit` hardware block-aligned survival (cutting collateral distortion up to `16,685×`), and establishes Post-Quantization Weight-Space Healing (`PQ-BA-WSC`) and the low-bit behavioral double crossover ($p = 0.00210$).

= Related Work <sec:related>

*Steering-to-Weights and Weight-Space Editing.* Task Arithmetic @ilharco2023 edits capabilities via linear combinations of fine-tuned checkpoints, while knowledge-editing algorithms such as ROME @meng2022 locate and update factual associations inside middle MLP layers via rank-one closed-form projections. Recent zero-gradient behavioral methods including Steer2Edit @steer2edit2026 and sparse FFN column editing @takehi2026 translate activation contrast vectors directly into static weight updates in $W_"down"$. While inspiring, unconditional rank-1 projections apply uniformly to *every* token—incurring heavy language-modeling perplexity degradation (`+1.338` WikiText-2 PPL on `Qwen3-0.6B`) and collapsing completely on severely stalling hybrid models (`0/150` on `Qwen3.5-0.8B`). Our work replaces unconditional ablation with (a) null-space-gated oblique updates targeting `</think>` resolution and (b) thresholded nonlinear `IF-THEN` circuits compiled into dormant SwiGLU neurons via $b^*$.

*Overthinking in Small Reasoning Models.* Open-weights reasoning families (`DeepSeek-R1-Distill` @deepseek2025, `Qwen3` and `Qwen3.5` @qwen2025) exhibit acute overthinking under tight token budgets, repeatedly verifying already-solved steps until context exhaustion. Decoding-level mitigations (DRY repetition samplers, presence penalties, logit biasing) operate externally and can disrupt valid mathematical syntax (repeated variables and equations). We demonstrate directly that DRY harms `Qwen3-0.6B` math accuracy ($p = 0.039$), whereas localized late-MLP weight compilation improves both accuracy and wall-clock throughput across domains.

*Evaluation Rigor in Model Editing.* Prior weight-editing evaluations frequently report single-run greedy decoding (`temp = 0.0`) without norm-matched random weight perturbations. Our Greedy-Perturbation Illusion shows formally and empirically why single-seed greedy decoding on finite-budget reasoning tasks creates systematic false positives.

= Two Methodological Traps in Reasoning Weight Editing <sec:traps>

== The Greedy-Perturbation Illusion <sec:greedy>

During initial held-out evaluation on `Qwen3-0.6B-BF16` (`MATH-500-strat150`, `max_tokens = 2048`, greedy `temp = 0.0`), we included a mandatory negative control: `random_matched_oblique`, which adds isotropic zero-mean Gaussian noise to layers `[18, 20, 22]` `ffn_down` scaled to match the exact per-layer Frobenius norm of `oblique_res_a0.25`.

Unexpectedly, *pure random noise raised strict Clean-EOS accuracy by `+6.0 pp`* (`52/150` [`34.7%`] $->$ `61/150` [`40.7%`], exact McNemar $p = 0.0636$), outperforming prior unconditional subspace excision (`uncond_a0.8`: `57/150` [`38.0%`]). To determine whether this was a lucky seed or a structural pathology of greedy decoding, we evaluated *six zero-information draws* at `temp = 0.0` (`BF16 baseline`, `Q8_0 baseline`, `Q8_0 + rand_s999/1000/1001`, and `BF16 + rand_s999`, @tab:greedy). Across zero-information perturbations alone, greedy decoding spans a *`52–61/150` (`6.0 pp`) chaotic band* (mean `55.7`, SD `3.4`).

#figure(
  booktable(
    columns: (2.8fr, 0.9fr, 1.3fr, 0.8fr, 0.8fr),
    align: (left, right, right, right, right),
    header: ([Zero-Information Draw (`MATH-500-strat150`, `temp = 0.0`)], [STD (/150)], [Clean-EOS (/150)], [Limit], [Loops]),
    row-fills: ("6": warn-bg),
    rows: (
      ([`BF16 baseline`], [`53`], [`52 (34.7%)`], [`97`], [`29`]),
      ([`Q8_0 + rand_s1000`], [`53`], [`52 (34.7%)`], [`95`], [`25`]),
      ([`Q8_0 + rand_s999`], [`57`], [`55 (36.7%)`], [`93`], [`26`]),
      ([`Q8_0 baseline`], [`57`], [`57 (38.0%)`], [`92`], [`21`]),
      ([`Q8_0 + rand_s1001`], [`60`], [`57 (38.0%)`], [`91`], [`30`]),
      ([*`BF16 + rand_s999`* (`random_matched_oblique`)], [*`61`*], [*`61 (40.7%, +6.0 pp)`*], [*`88`*], [`30`]),
    ),
  ),
  caption: [Six zero-information perturbations of `Qwen3-0.6B` under greedy decoding (`temp = 0.0`, `MATH-500-strat150`). Pure Gaussian noise spans a `6.0 pp` chaotic band and appears to gain `+6.0 pp` over `BF16` baseline without suppressing repetition loops (`29` $->$ `30`).],
) <tab:greedy>

#proposition(1, title: "Asymmetric Reservoir Reroll Bias under Finite-Budget Greedy Decoding")[
  Let a reasoning model evaluated at `temp = 0.0` under token cap $L_max$ partition $N$ problems into $N_"trunc"$ truncated failures (hitting $L_max$), $N_"short"$ short correct trajectories whose top-1 margins are perturbation-robust, and $N_"fragile"$ near-cliff correct trajectories ($N_"fragile" << N_"trunc"$). When a small isotropic weight perturbation $epsilon Delta W$ flips at least one argmax token early in decoding with probability $p_"div"$, it acts as an unguided trajectory reroll. If the latent stochastic success probability on rerolled truncated items is $p_"escape" > 0$, the expected greedy accuracy shift is:
  $ bb(E)[Delta "Acc"_"greedy"] approx 1/N ( N_"trunc" p_"div" p_"escape" - N_"fragile" p_"div" p_"break" ). $
  Whenever $N_"trunc" >> N_"fragile"$ (as in small reasoning LLMs where baseline truncation reaches `50%–95%`), zero-information noise exhibits a strictly positive net greedy accuracy bias without improving the underlying stochastic policy.
]

*Empirical Verification of Proposition 1:*
1. *Item-level mechanism:* Inspecting all 14 items rescued by `BF16 + rand_s999` confirmed that *`14/14` (`100%`) had hit the `2048`-token cap at baseline* (5 looping), whereas the 5 broken items had a mean baseline length of `1,710` tokens (already near the `2048` cliff). Moreover, `random_matched_oblique` produced *zero loop suppression* (`29` $->$ `30` loops, $p = 1.000$).
2. *Complete collapse under stochastic sampling:* Under official recommended stochastic decoding (`temp = 0.6, top_p = 0.95, top_k = 20`, Seeds 17 & 42, $N = 300$ rollouts on `Qwen3-0.6B-Q8_0`), the baseline itself stabilizes (`52/150` on Seed 17, `53/150` on Seed 42; pooled `105/300 = 35.0%`), and *`random_s999` collapses from `+6.0 pp` down to `-0.33 pp`* (`104/300 = 34.7%`, 95% CI `[-4.3, +3.7]`, exact McNemar $p = 1.000$, problem-clustered permutation `cl_p = 1.000`, @fig:illusion). Across `GSM8K-100` (`temp = 0.6`) and `DeepSeek-R1-1.5B` (`temp = 0.6`), matched random noise actively degrades accuracy (`-7.0 pp` on `GSM8K`; `-2.7 pp` at `Q8_0` and `-4.0 pp` at `Q4_0` on `DeepSeek-R1-1.5B`).

== The Final-Answer Echo Trap & Strict Clean-EOS Auditing <sec:echo>

Standard mathematical evaluators extract the final `\boxed{...}` substring anywhere in the generated text. During `Qwen3-0.6B` development (`dev_math_60`), high-dose `bawsc32_res_a0.70` achieved the highest standard accuracy (`32/60 = 53.3%`, $p = 0.013$) at `-0.025` $Delta"PPL"$. On held-out `MATH-500-strat150` (`BF16`), it likewise scored *`66/150` (`44.0%`, `+8.7 pp`, $p = 0.0294$)* at $Delta"PPL" = -0.022$.

Inspecting raw transcripts exposed a severe hidden pathology (Figure 1(b)): after successfully closing `</think>` and outputting `\boxed{answer}`, the high-dose edit continued firing on post-`</think>` tokens, causing the model to echo the final answer indefinitely until hitting the `2048`-token cutoff. Repetition loops *more than doubled (`29` $->$ `59`, $p = 0.0001$)*.

#definition(1, title: "Strict Clean-EOS Accuracy")[
  Across all experiments, our primary evaluation metric requires correctness, explicit reasoning closure, and natural end-of-sequence termination:
  $ "Clean-EOS"(y) equiv bb(I)[ "Correct"(y) and "HasThinkClose"(y) and ("FinishReason"(y) = #raw("stop")) ]. $
]

Under Strict Clean-EOS, `bawsc32_res_a0.70` collapses from `44.0%` down to *`48/150` (`32.0%`)—statistically worse than matched random noise ($p = 0.0106$)*. Reducing the dose to $alpha = 0.25$ eliminates post-answer echoing completely (`65/65 = 100%` Clean-EOS alignment on `BF16`, loops reduced `29` $->$ `22`). Furthermore, in an architectural ablation (`bawsc32_andgate_a0.70`, Appendix C), replacing the constant `W_up` factor with a linear classifier distinguishing inside-`<think>` from post-`</think>` activations (AUROC `0.9999–1.0000`) drives post-`</think>` circuit RMS down to `0.0008` and cuts `a = 0.70` development loops from `20` $->$ `12`.

#figure(
  image("../figures/p1_fig1_greedy_illusion_and_stochastic.png", width: 100%),
  caption: [*(a)* On `Qwen3-0.6B` held-out `MATH-500-strat150`, matched random Gaussian noise gains `+6.0 pp` at greedy `temp = 0.0` (pink band: 6 zero-information draws spanning `34.7%–40.7%`), yet collapses to `-0.33 pp` (`cl_p = 1.000`) under multi-seed stochastic sampling (`temp = 0.6`, $N = 300$), where `oblique_res` significantly beats both baseline (`cl_p = 0.0148`) and random noise (`cl_p = 0.0146`). *(b)* Standard `\boxed{}` grading masks the Final-Answer Echo Trap in high-dose `bawsc32` ($alpha = 0.70$): `44.0%` standard vs. `32.0%` Strict Clean-EOS (`59` loops).],
) <fig:illusion>

= Universal Cross-Architecture Anatomy & Closed-Form Weight Compilers <sec:method>

== Exact GGUF-to-PyTorch Tracing & Universal Late-MLP Surge <sec:anatomy>

To eliminate readout discrepancies between inference engines and PyTorch, we built a streaming loader (`src/wsp/gguf_to_torch.py`) that populates PyTorch `Qwen3`, `Qwen2`, and `Qwen3_5ForConditionalGeneration` modules directly from `BF16` GGUF tensors with *`0.00e+00` maximum absolute weight discrepancy*, supporting native AVX2 `float32` CPU tracing with `skip_lm_head=True` (`412.1 s` for all 28 layers of the 1.5B model in `5.2 GB` RAM).

For each model, we collect strictly calibration-split trajectories (`cal_math_50`, `cal_gsm8k_48`, `cal_copy_36`, `cal_wikitext_64`) and partition tokens into four disjoint sets:
- *Pathological ($cal(P)$):* Stalled/looping tail tokens from non-correct truncated calibration rollouts.
- *Healthy ($cal(H)$):* Clean reasoning tokens from correct, cleanly stopped calibration rollouts.
- *Resolution ($cal(R)$):* The final 12 tokens immediately preceding and including clean `</think>` closures.
- *Preservation ($cal(S)_"pres"$):* Healthy reasoning + exact-copy + WikiText-2 validation tokens.

_Leak-Free Calibration Enrichment for Severely Stalling Models:_ On `Qwen3.5-0.8B`, where unprompted baseline rollouts timed out on `96%–100%` of math calibration items leaving only $n_"resolve" = 2$, we ran Prompt-to-Weight Calibration Enrichment using a concise system prompt strictly on `cal_math_50[:30]` and `cal_gsm8k_48[:25]` to harvest genuine `</think>` resolution states while keeping pathological tails strictly unprompted and all held-out benchmarks completely untouched.

At every decoder layer $ell$, we measure MLP output attribution along the residual pathology direction:
$ "MLPProj"(ell) = (mu_(y, cal(P))^((ell)) - mu_(y, cal(H))^((ell)))^top (mu_(h, cal(P))^((ell)) - mu_(h, cal(H))^((ell))) / norm(mu_(h, cal(P))^((ell)) - mu_(h, cal(H))^((ell)))_2 . $

*Universal Cross-Architecture Finding (@fig:anatomy):* Despite fundamentally different sequence mixers and parameter counts, all three model families exhibit the exact same late-MLP localization profile:
1. *`Qwen3-0.6B` (28L Pure GQA + QK-Norm):* Flat across L0–L15 ($<= 0.65$), surging across *L16–L22* (`1.12` $->$ `6.98`, readout L27 = `65.87`). Compiled target layers: `[18, 20, 22]`.
2. *`Qwen3.5-0.8B` (24L Hybrid: 18 Gated-DeltaNet + 6 Full-Attention):* Flat across L0–L13 (`0.004–0.023`), then *surging `10×` precisely at Full-Attention Layer 15 (`0.1723`)*, with secondary peaks at *Full-Attention Layer 19 (`0.1372`)* and DeltaNet Layer 21 (`0.1670`, readout L23 = `0.5682`). Compiled target layers: `[15, 19, 21]`.
3. *`DeepSeek-R1-Distill-Qwen-1.5B` (28L RL-Distilled Qwen2):* Flat across L0–L18 ($<= 0.56$, L18 = `0.038`), then *surging `35×` at L19 (`1.316`) $->$ L22 (`2.672`) $->$ L25 (`5.528`) $->$ L26 (`7.776`)* (readout L27 = `40.126`). Compiled target layers: `[20, 22, 25]`.

#figure(
  image("../figures/p1_fig2_universal_layer_anatomy.png", width: 100%),
  caption: [Pre-readout MLP attribution $"MLPProj"(ell)$ to the pathology direction across all three architectures. Red bars denote compiled target layers; orange bars in `Qwen3.5-0.8B` mark earlier Full-Attention blocks (L3, L7, L11). Inset callouts report out-of-sample stability of the Virtual Constant-1 Bias Register $b^*$.],
) <fig:anatomy>

== Invention 1: The Virtual Constant-1 Bias Register ($b^*$) <sec:vb>

Modern decoder MLPs employ bias-free SwiGLU layers:
$ "MLP"(x) = W_"down" ( "SiLU"(W_"gate" x) dot.o (W_"up" x) ), quad x = "RMSNorm"(h) in bb(R)^d . $
Compiling a conditional rule $"IF" (u^top x > tau) "THEN" Delta h$ into a single SwiGLU neuron requires an additive threshold $-tau$ inside $"SiLU"(dot)$ and a constant input in the `up` branch—neither of which exists explicitly when bias vectors are absent.

We observe that because post-RMSNorm hidden states $x$ lie on a radius-$sqrt(d)$ sphere with consistent anisotropic mean structure across token domains, there exists a *Virtual Constant-1 Bias Register* $b^* in bb(R)^d$ solving the closed-form ridge problem on calibration preservation tokens $X_"pres"$:
$ b^* = (X_"pres"^top X_"pres" + lambda_b I)^(-1) X_"pres"^top bold(1) . $
Across unseen text, math, and copy domains, $b^(*top) x approx 1$ holds tightly on every tested architecture:
- *`Qwen3-0.6B`:* out-of-sample $"std"(b^(*top) x) = 0.048"–"0.074$ (in-sample calibration `0.0068–0.0084`).
- *`Qwen3.5-0.8B` (Hybrid):* $"std"(b^(*top) x) = 0.0208"–"0.0252$ (`0.0227–0.0252` on target layers `[15, 19, 21]`).
- *`DeepSeek-R1-Distill-Qwen-1.5B`:* $"std"(b^(*top) x) = 0.0198"–"0.0250$ (`0.0227–0.0238` on target layers `[20, 22, 25]`).

Consequently, $w_"gate"^top x = gamma (u - tau b^*)^top x approx gamma (u^top x - tau)$ implements a clean, bias-free affine threshold gate with zero architectural modification.

== Invention 2: Null-Space-Gated Oblique Surgery (`oblique_res`) <sec:oblique>

Let $Z_"pres" in bb(R)^(n_"pres" times d_"ff")$ denote post-SwiGLU activations $z = "SiLU"(W_"gate" x) dot.o (W_"up" x)$ on preservation tokens, and let $V_r in bb(R)^(d_"ff" times r)$ ($r = 64$) be its top-$r$ right singular basis. Define the preservation null-space projector $P_perp = I - V_r V_r^top$. We solve for the calibrated null-space detector $d in bb(R)^(d_"ff")$:
$ d_"raw" = P_perp (Z_"pres"^top Z_"pres" + lambda I)^(-1) (mu_(z, cal(P)) - mu_(z, cal(H))), quad d = d_"raw" / ((mu_(z, cal(P)) - mu_(z, cal(H)))^top d_"raw") , $
and form the *resolution-directed* rank-1 update directly inside $W_"down"$:
$ Delta W_"down" = alpha v_"resolve" d^top, quad v_"resolve" = mu_(h, cal(R)) - mu_(h, cal(P)) . $
Unlike unconditional subspace excision (`uncond`, which subtracts a pathology direction uniformly across all inputs), `oblique_res` vanishes on the dominant preservation subspace ($norm(Z_"pres" d)_2 approx 0$) while pushing stalled trajectories directly toward `</think>` closure. On `Qwen3-0.6B`, this reduces WikiText-2 PPL penalty by nearly *$10times$* (`+0.141` vs. `+1.338`).

== Invention 3: Nonlinear Dormant-SwiGLU `IF-THEN` Compilation (`BA-WSC`) <sec:bawsc>

For ultra-low perplexity overhead and low-bit quantization survival (proved formally in companion Paper 2), we compile a nonlinear thresholded `IF-THEN` circuit directly into the $k = 32$ lowest-RMS *dormant SwiGLU neurons* $cal(S)$ measured across all preservation tokens (`dormant_32 / median` activation contribution `0.32–0.62`, Appendix A):
1. Fit a dual-negative regularized linear detector $u in bb(R)^d$ separating $X_cal(P)$ from $X_cal(H) union X_"pres"$ (`AUROC = 0.994–1.000` on target layers across all 3 models), and set threshold $tau = "Quantile"_q (X_"pres" u)$ ($q in {0.90, 0.95}$) so the gate remains silent on `90%–95%` of normal tokens.
2. Apply an exact SwiGLU neuron permutation $P$ moving $cal(S)$ to contiguous indices $lr([0, k))$ (`0.000000` FP output change; isolates edited slots into a dedicated 32-element GGUF quantization block—Theorem 1 in Paper 2).
3. Compile replicated thresholded circuits across all $j in lr([0, k))$ using $b^*$ and positive-homogeneity gauge factor $eta$:
$ W'_"gate" [j, :] = gamma (u - tau b^*)^top, quad W'_"up" [j, :] = eta (b^*)^top, quad W'_"down" [:, j] = alpha / (k eta s_cal(P)) v_"resolve" , $
where $s_cal(P) = bb(E)_(x tilde cal(P))["SiLU"(gamma (u^top x - tau))]$. Because $"SiLU"(z) -> 0$ exponentially for $z < 0$, preservation tokens incur virtually zero residual perturbation (selectivity ratio `60×–80×`, $Delta"PPL" in [-0.018, +0.028]$).

#algorithm-box(1, "Zero-Gradient Weight-Space Compilation Pipeline (oblique_res & BA-WSC)")[
  *Input:* Frozen BF16 GGUF model $M$; calibration splits (`cal_math_50`, `cal_gsm8k_48`, `cal_copy_36`, `cal_wikitext_64`); disjoint `dev_math_60`; scalar dose grid $cal(A)$. \
  *1. Exact Trace:* Load $M$ into PyTorch (`max_abs_diff = 0.00e+00`) and record post-RMSNorm $x^((ell))$, post-SwiGLU $z^((ell))$, and MLP output $y^((ell))$ across $cal(P), cal(H), cal(R), cal(S)_"pres"$. \
  *2. Automated Layer Localization:* Compute $"MLPProj"(ell)$ for all $ell in [0, L-1]$; select 3 pre-readout surge layers $cal(L)_"target"$. \
  *3. Closed-Form Solve:* For each $ell in cal(L)_"target"$, compute $v_"resolve" = mu_(h, cal(R)) - mu_(h, cal(P))$ and compile either `oblique_res` ($Delta W_"down" = alpha v_"resolve" d^top$) or `BA-WSC` (virtual bias $b^*$, permutation $P$, $k=32$ dormant neurons). \
  *4. Anti-Leakage Dev Freeze:* Select single scalar dose $alpha^*$ per method on disjoint `dev_math_60` under Strict Clean-EOS + PPL gate, commit manifests to git, and evaluate untouched held-out test suites (`temp = 0.6`).
]

= Experimental Protocol & Controls <sec:protocol>

- *Strict Anti-Leakage Splits:* Calibration (`cal_math_50`, `cal_gsm8k_48`, `cal_copy_36`, `cal_wikitext_64`) uses training/validation splits only (SHA-256 manifests in Appendix B). Target layers and scalar doses $alpha$ are selected strictly on disjoint `dev_math_60` (`dev_math_60[:24]` for Models 2 & 3) and committed to git before touching any held-out item.
- *Held-Out Evaluation Suite:* Stratified `MATH-500-strat150` (`max_tokens = 2048`), OOD `GSM8K-100` (`max_tokens = 2048`), zero-shot non-math `ARC-Challenge-100` (`max_tokens = 1024`), `Copy-90` (exact copy, key-value retrieval, cycle induction), and 32-chunk `WikiText-2` test PPL (`ctx = 512`).
- *Decoding, Hardware & Auditing:* Official stochastic sampling (`temp = 0.6, top_p = 0.95, top_k = 20`) on `llama.cpp` build `8864 (ff6b1062a)` Vulkan (`AMD Radeon RX 6400 4 GB`). Every request flushes immediately to immutable `responses.jsonl`; across all `206` program runs (`18,513` responses, `25.06M` tokens), total HTTP/parse errors equal *`0`*. During Model 3 baseline auditing, we stripped markdown bold (`**Answer:** 34`) in `math_grader.py` and verified *`0 / 9,154` changed grades* across all Model 1 and Model 2 runs.
- *Statistical Testing:* Exact two-sided paired McNemar tests (`mc_p`) on identical held-out items; for multi-seed pooled $N = 300$ runs (`Qwen3-0.6B` Seeds 17+42), we additionally run *100,000-iteration problem-clustered paired sign-permutation tests (`cl_p`)* and *20,000-sample problem-clustered bootstrap 95% CIs*. Cross-architecture aggregation uses exact pooled McNemar and Fisher's combined $chi^2$ test.

= Main Experimental Results <sec:results>

== Non-Weight Decoding Controls Fail (`Qwen3-0.6B` Dev Screen) <sec:devctrl>

Before editing weights, we benchmarked non-weight decoding and prompting baselines on `dev_math_60` (`Qwen3-0.6B-BF16`, @tab:devctrl). Repetition samplers (`DRY`) eliminate n-gram loops only by breaking valid mathematical repetition (symbols and equations), significantly harming accuracy (`36.7%` $->$ `23.3%`, $p = 0.039$). Concise prompting provides a non-significant shift ($p = 0.508$).

#figure(
  booktable(
    columns: (2.1fr, 1.3fr, 1.1fr, 0.65fr, 0.65fr, 2.0fr),
    align: (left, right, right, right, right, left),
    header: ([Control / Method (`dev_math_60`)], [16-Chunk PPL], [STD (/60)], [Limit], [Loops], [Finding]),
    row-fills: ("2": warn-bg, "6": win-bg, "7": win-bg),
    rows: (
      ([`baseline_fp`], [`21.160`], [`22 (36.7%)`], [`36`], [`11`], [Unedited reference]),
      ([`ctrl_dry_sampler`], [`21.160`], [*`14 (23.3%)`*], [`40`], [`0`], [*Significantly harms ($p = 0.039$)*]),
      ([`ctrl_presence_pen`], [`21.160`], [`21 (35.0%)`], [`37`], [`6`], [No gain]),
      ([`ctrl_concise_prompt`], [`21.160`], [`25 (41.7%)`], [`32`], [`5`], [Not significant ($p = 0.508$)]),
      ([`uncond_a0.8` (Prior WSP)], [`22.648 (+1.488)`], [`27 (45.0%)`], [`33`], [`11`], [Heavy PPL cost]),
      ([*`oblique_res_a0.25` (Frozen)*], [`21.328 (+0.168)`], [*`28 (46.7%)`*], [*`29`*], [*`5`*], [Selected ($9times$ lower $Delta$PPL)]),
      ([*`bawsc32_res_a0.25` (Frozen)*], [*`21.139 (-0.021)`*], [`26 (43.3%)`], [`31`], [`8`], [Selected (zero PPL cost)]),
    ),
  ),
  caption: [Non-weight decoding controls vs. weight-space compilation on `Qwen3-0.6B-BF16` development screen (`dev_math_60`).],
) <tab:devctrl>

== Multi-Seed Stochastic Proof on `Qwen3-0.6B` (`MATH-500`, $N = 300$) <sec:m1stoch>

On `Qwen3-0.6B-Q8_0` (`MATH-500-strat150`, Seeds 17 + 42, `temp = 0.6`, $N = 300$ rollouts, @tab:m1stoch), *`oblique_res_a0.25`* achieves statistically significant Clean-EOS gains over *both* baseline and matched random noise (`+5.67 pp`, `cl_p = 0.0148` vs. base, `cl_p = 0.0146` vs. random). Prior unconditional WSP (`uncond_a0.8`) pays nearly $10times$ the PPL penalty (`+1.338`) yet fails to reach statistical significance against either baseline (`cl_p = 0.107`) or random noise (`cl_p = 0.088`). `BA-WSC` incurs negative PPL change (`-0.018`) while delivering statistically significant gains on `</think>` closure ($p = 0.020$), timeout reduction ($p = 0.025$), and token efficiency (`-51.4 tok/prob`, $p = 0.0071$).

#figure(
  booktable(
    columns: (1.75fr, 0.72fr, 0.52fr, 0.95fr, 1.38fr, 0.75fr, 0.75fr, 0.95fr, 0.82fr, 0.85fr),
    align: (left, right, right, right, right, right, right, right, right, right),
    header: ([Arm ($N = 300$)], [$Delta$PPL], [STD], [Clean-EOS], [$Delta$ vs Base [95% CI]], [`cl_p` Base], [`cl_p` Rand], [Close], [Limit], [Mean Tok]),
    font-size: 7.4pt,
    inset: (x: 3pt, y: 3.2pt),
    row-fills: ("5": win-bg),
    rows: (
      ([`baseline`], [`0.000`], [`109`], [`105 (35.0%)`], [—], [—], [`1.000`], [`44.7%`], [`189`], [`1784.9`]),
      ([`random_s999`], [`-0.023`], [`109`], [`104 (34.7%)`], [`-0.33pp [-4.3,+3.7]`], [`1.000`], [—], [`43.0%`], [`191`], [`1774.1`]),
      ([`uncond_a0.8`], [`+1.338`], [`121`], [`116 (38.7%)`], [`+3.67pp [-0.3,+7.7]`], [`0.107`], [`0.088`], [`49.0%`], [`176`], [`1741.7`]),
      ([`bawsc32_res_a0.25`], [*`-0.018`*], [`118`], [`114 (38.0%)`], [`+3.00pp [-1.7,+7.7]`], [`0.255`], [`0.154`], [*`50.0%`*\ (`p=.020`)], [*`173`*\ (`p=.025`)], [*`1733.5`*\ (`p=.007`)]),
      ([*`oblique_res_a0.25`*], [`+0.141`], [*`125`*], [*`122 (40.7%)`*], [*`+5.67pp [+1.3,+10.0]`*], [*`0.0148`*], [*`0.0146`*], [*`52.3%`*\ (`p=.0017`)], [*`171`*\ (`p=.015`)], [*`1722.5`*\ (`p=.0008`)]),
    ),
  ),
  caption: [Multi-seed stochastic evaluation (`temp = 0.6`, Seeds 17 + 42, $N = 300$) on `Qwen3-0.6B-Q8_0` held-out `MATH-500-strat150`. Confidence intervals and `cl_p` use problem-clustered bootstrap (20k) and sign-permutation (100k) tests.],
) <tab:m1stoch>

== Out-of-Distribution (`GSM8K-100`) & Zero-Shot Non-Math Science (`ARC-Challenge-100`) <sec:ood>

Evaluating the exact same frozen `Qwen3-0.6B-Q8_0` edits (`temp = 0.6`, Seed 42) on untouched OOD `GSM8K-100` and *zero-shot science `ARC-Challenge-100`* (zero science data in calibration or development) proves the compiled circuit is domain-general (@tab:ood, Figure 3(a)). On `ARC-Challenge-100`, both conditional compilers cut token exhaustion by `58%–65%` (`26` $->$ `11` and `26` $->$ `9`) and raise strict Clean-EOS accuracy by `+11.0 pp` ($p = 0.0074$) and `+15.0 pp` ($p = 0.0003$), while `oblique_res` simultaneously cuts `GSM8K-100` wall-clock time by *`30%`* (`1062 s` $->$ `744 s`) and beats matched random noise by `+16.0 pp` ($p = 0.0001$).

#figure(
  booktable(
    columns: (1.55fr, 1.85fr, 1.35fr, 1.65fr, 1.05fr, 0.90fr),
    align: (left, right, right, right, right, right),
    header: ([Arm (`Qwen3-0.6B-Q8_0`)], [`GSM8K-100` Clean-EOS], [`GSM8K` Lim / Tok / Wall], [`ARC-100` Clean-EOS], [`ARC` Lim / Tok], [`Copy-90` (BF16)]),
    font-size: 7.5pt,
    inset: (x: 3.4pt, y: 3.3pt),
    row-fills: ("4": win-bg, "5": win-bg),
    rows: (
      ([`baseline`], [`71/100 (71%)`], [`19 / 1234 / 1062s`], [`56/100 (56%)`], [`26 / 597`], [`64/90 (71.1%)`]),
      ([`random_s999`], [`64/100 (-7pp, p=.230)`], [`31 / 1321 / 985s`], [`62/100 (p=.180 n.s.)`], [`23 / 582`], [`64/90 (71.1%)`]),
      ([`uncond_a0.8`], [`73/100 (p=.839 n.s.)`], [`18 / 1274 / 823s`], [`64/100 (p=.039)`], [`19 / 548`], [`66/90 (73.3%)`]),
      ([*`bawsc32_res_a0.25`*], [`75/100 (p=.027 vsR)`], [`15 / 1167 / 894s`], [*`67/100 (+11pp, p=0.0074)`*], [*`11 (-58%) / 536`*], [*`65/90 (72.2%)`*]),
      ([*`oblique_res_a0.25`*], [*`80/100 (+9pp, p=0.049)`*\ (`+16pp vsR p=0.0001`)], [*`8 (-58%) / 1096 / 744s (-30%)`*], [*`71/100 (+15pp, p=0.0003)`*], [*`9 (-65%) / 497`*], [`64/90 (71.1%)`]),
    ),
  ),
  caption: [Out-of-distribution (`GSM8K-100`) and zero-shot non-math science (`ARC-Challenge-100`) transfer of math-calibrated `Qwen3-0.6B-Q8_0` edits (`temp = 0.6`, Seed 42), with exact `Copy-90` retrieval strictly preserved.],
) <tab:ood>

== Three-Architecture Stochastic Replication (`Q8_0`, `MATH-500`, `temp = 0.6`) <sec:crossarch>

We replicate the entire end-to-end protocol—automated layer profiling, strict dev dose freezing (full dev screens in Appendix C), and untouched held-out `MATH-500-strat150` + `Copy-90` + 32-chunk PPL—on *Hybrid Gated-DeltaNet `Qwen3.5-0.8B`* (`layers=[15,19,21]`, frozen `oblique` $alpha=1.0$, `bawsc` $alpha=0.45$) and *1.5B RL-Distilled `DeepSeek-R1-Distill-Qwen-1.5B`* (`layers=[20,22,25]`, frozen `oblique` $alpha=0.25$, `bawsc` $alpha=0.50$, where dev `bawsc` jumped `12/24` $->$ `19/24 = 79.2%` at $Delta"PPL" = +0.010$). @tab:crossarch, Figure 3(b), and @fig:pareto summarize all three architectures side by side.

#figure(
  booktable(
    columns: (1.52fr, 1.30fr, 1.12fr, 0.45fr, 1.15fr, 0.48fr, 0.76fr, 0.74fr, 1.32fr),
    align: (left, left, right, right, right, right, right, right, left),
    header: ([Model (`Q8_0`, `t=0.6`)], [Arm], [PPL ($Delta$PPL)], [STD], [Clean-EOS], [Lim], [Wall], [`Copy-90`], [McNemar `p` (Base / Rand)]),
    font-size: 7.25pt,
    inset: (x: 2.8pt, y: 2.9pt),
    row-fills: ("5": win-bg, "10": win-bg, "15": win-bg),
    rows: (
      ([*Qwen3-0.6B* ($N=300$)\ #emph[Pure GQA, 28L]], [`baseline`], [`19.306 (0.000)`], [`109`], [`105 (35.0%)`], [`189`], [—], [`64/90`], [—]),
      ([], [`random_s999`], [`19.282 (-0.023)`], [`109`], [`104 (34.7%)`], [`191`], [—], [`64/90`], [`1.000` / —]),
      ([], [`uncond_a0.8`], [`20.644 (+1.338)`], [`121`], [`116 (38.7%)`], [`176`], [—], [`66/90`], [`0.107` / `0.088` (`cl_p`)]),
      ([], [`bawsc32_res_a0.25`], [*`19.287 (-0.018)`*], [`118`], [`114 (38.0%)`], [`173`], [—], [`65/90`], [`0.255` / `0.154` (`cl_p`)]),
      ([], [*`oblique_res_a0.25`*], [`19.446 (+0.141)`], [*`125`*], [*`122 (40.7%)`*], [*`171`*], [—], [`64/90`], [*`0.0148` / `0.0146`* (`cl_p`)]),
      ([*Qwen3.5-0.8B* ($N=150$)\ #emph[Hybrid DeltaNet, 24L]], [`baseline`], [`15.513 (0.000)`], [`6`], [`1 (0.7%)`], [`149`], [`6420s`], [`66/90`], [—]),
      ([], [`random_matched`], [`15.723 (+0.210)`], [`4`], [`3 (2.0%)`], [`147`], [`6277s`], [`67/90`], [`0.500` / —]),
      ([], [`uncond_a0.8`], [`15.509 (-0.004)`], [`1`], [`0 (0.0%)`], [`150`], [`6291s`], [`66/90`], [`1.000` / `0.250`]),
      ([], [`bawsc32_res_a0.45`], [*`15.541 (+0.028)`*], [`8`], [`3 (2.0%)`], [`145`], [`6270s`], [`66/90`], [`0.625` / `1.000` (cls `.031`)]),
      ([], [*`oblique_res_a1.0`*], [`15.814 (+0.301)`], [*`12`*], [*`10 (6.7%, 10×)`*], [*`126`*], [*`5650s (-12%)`*], [`66/90`], [*`0.0039` / `0.0391`*]),
      ([*DeepSeek-R1-1.5B*\ ($N=150$, #emph[Qwen2 RL, 28L])], [`baseline`], [`37.254 (0.000)`], [`45`], [`43 (28.7%)`], [`77`], [`4403s`], [`68/90`#super[\*]], [—]),
      ([], [`random_matched`], [`37.298 (+0.044)`], [`40`], [`39 (26.0%)`], [`82`], [`3956s`], [`76/90`#super[\*]], [`0.541` / —]),
      ([], [`uncond_a0.8`], [`37.532 (+0.278)`], [`49`], [`48 (32.0%)`], [`69`], [`3550s`], [`70/90`#super[\*]], [`0.533` / `0.136`]),
      ([], [`bawsc32_res_a0.5`], [*`37.257 (+0.003)`*], [`51`], [`49 (32.7%)`], [*`67`*], [*`3445s (-22%)`*], [`67/90`#super[\*]], [`0.377` / `0.064` (cls `.0043`)]),
      ([], [*`oblique_res_a0.25`*], [`37.338 (+0.084)`], [*`55`*], [*`53 (35.3%)`*], [`69`], [*`3527s (-20%)`*], [`70/90`#super[\*]], [`0.121` / *`0.0288`* (STD `.020`)]),
    ),
  ),
  caption: [Three-architecture held-out `MATH-500-strat150` replication (`Q8_0`, `temp = 0.6`). `oblique_res` ranks `#1` on *3/3 architectures*, while `BA-WSC` preserves $Delta"PPL" approx 0$ across every family. \*On `DeepSeek-R1-1.5B`, whose chat template forces `<think>\n` and markdown bold, `Copy-90` reports `contains_gold` retrieval accuracy (strict bare-string match: `16, 19, 19, 9, 17` out of 90).],
) <tab:crossarch>

*Cross-Architecture Pooled Meta-Analysis ($N = 600$ Untouched Held-Out `MATH-500` Rollouts at `Q8_0`):*
- *`oblique_res` vs. Baseline:* *`+61 / -25`* (net `+36`, `+6.00 pp`), exact pooled McNemar *$p = 0.000130$*, 3-model Fisher $chi^2_6 = 23.73$, *$p = 0.000586$*.
- *`oblique_res` vs. Matched Random Noise:* *`+65 / -26`* (net `+39`, `+6.50 pp`), exact pooled McNemar *$p = 5.31 times 10^(-5)$*, 3-model Fisher $chi^2_6 = 22.04$, *$p = 0.00119$*.
- *`BA-WSC` (near-zero $Delta"PPL"$) vs. Matched Random Noise:* *`+48 / -28`* (net `+20`, `+3.33 pp`), exact pooled McNemar *$p = 0.0286$* (vs. baseline: `+52 / -35`, `+2.83 pp`, $p = 0.0857$), while significantly increasing `</think>` closure on *3/3 architectures* ($p = 0.020, 0.031, 0.0043$).

#figure(
  image("../figures/p1_fig3_ood_and_cross_arch.png", width: 100%),
  caption: [*(a)* Domain-general transfer of `Qwen3-0.6B` math-calibrated edits to OOD `GSM8K-100` (`+9 pp` vs. base, `+16 pp` vs. random) and zero-shot non-math science `ARC-Challenge-100` (`+15 pp` Oblique, `+11 pp` BA-WSC) with `Copy-90` preserved. *(b)* Consistent `Q8_0` stochastic superiority of `oblique_res` (`#1` on 3/3 architectures; pooled $p = 0.000130$ vs. base, $p = 5.31 times 10^(-5)$ vs. random) while matched random noise never achieves significant gain.],
) <fig:cross>

#figure(
  image("../figures/p1_fig4_pareto_ppl_vs_acc.png", width: 92%),
  caption: [Pareto frontier of Clean-EOS accuracy gain ($Delta$ pp) vs. WikiText-2 $Delta"PPL"$ across all three architectures at `Q8_0`. `BA-WSC` sits directly on the $Delta"PPL" approx 0$ axis across all 3 models, while `oblique_res` dominates the top-left high-gain/low-distortion frontier.],
) <fig:pareto>

= Honest Limitations & Failure Modes <sec:limits>

1. *`BA-WSC` vs. `oblique_res` at High Precision (`Q8_0`):* At 8-bit `Q8_0`, linear null-space `oblique_res` consistently outperforms nonlinear `BA-WSC` in raw Clean-EOS accuracy across all 3 models (`40.7%` vs. `38.0%` on `Qwen3`; `6.7%` vs. `2.0%` on `Qwen3.5`; `35.3%` vs. `32.7%` on `DeepSeek-R1-1.5B`). `BA-WSC`'s decisive structural advantages are (a) virtually zero perplexity drift ($Delta"PPL" in [-0.018, +0.028]$) and (b) *quantization survival under 4-bit and 3-bit compression*, where diffuse `oblique_res` breaks while block-aligned `BA-WSC` survives (proved theoretically and empirically in companion Paper 2).
2. *Dose Sensitivity & Echo Monitoring:* Over-driving `BA-WSC` ($alpha = 0.70$ on `Qwen3-0.6B`) triggers the Final-Answer Echo Trap unless paired with inside-`<think>` `W_up` AND-gating or strict Clean-EOS development screening.
3. *Single-Seed Statistical Power on Models 2 & 3:* Within our single 4 GB RX 6400 hardware envelope, `Qwen3.5-0.8B` and `DeepSeek-R1-1.5B` held-out `MATH-500` runs used $N = 150$ (Seed 42) per precision (whereas `Qwen3-0.6B` used 2 seeds, $N = 300$). Consequently, `DeepSeek-R1-1.5B` `Q8_0` `oblique_res` (`+6.7 pp` vs. baseline) reaches $p = 0.0288$ vs. random noise and $p = 0.121$ vs. baseline individually, achieving $p = 0.000130$ / $5.31 times 10^(-5)$ when pooled across all 3 models ($N = 600$).
4. *Model Scale:* All experiments target `0.6B–1.5B` edge reasoning models. Because our closed-form solvers require zero backpropagation and scale as $O(d_"ff" d_"model")$ per layer, scaling to `7B–32B` models is straightforward future work.

= Conclusion <sec:conclusion>

Evaluating weight-space reasoning edits under finite-budget greedy decoding or permissive `\boxed{}` graders can turn pure isotropic Gaussian noise or severe post-answer echo loops into illusory `+6–9 pp` accuracy gains. Under strict multi-seed stochastic sampling, matched random noise controls, and Clean-EOS auditing across three distinct architectures (`Qwen3-0.6B`, hybrid Gated-DeltaNet `Qwen3.5-0.8B`, and `DeepSeek-R1-Distill-Qwen-1.5B`), our closed-form *Null-Space-Gated Oblique Surgery* and *Virtual-Bias `BA-WSC`* compilers translate universal late-MLP mechanistic localization into genuine, zero-gradient, zero-extra-byte, zero-latency reasoning improvements that transfer zero-shot across domains (`ARC-Challenge +15 pp`, `p = 0.0003`) while accelerating inference by `20%–30%`.

#v(0.6em)
#bibliography("../references.bib", title: "References", style: "ieee")

#pagebreak()
#line(length: 100%, stroke: 0.9pt + navy)
#v(0.25em)
#align(center)[#text(size: 8.6pt, weight: "bold", tracking: 0.12em, fill: navy)[APPENDICES · ARCHITECTURAL SPECS, PROVENANCE & DEV SCREENS]]
#v(0.25em)
#line(length: 100%, stroke: 0.45pt + rule-gray)
#set text(size: 9.8pt)
#set par(hanging-indent: 0em, leading: 0.62em, spacing: 0.78em)
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Model Architectural Specifications & Cryptographic Artifact Manifests <app:models>

All experiments ran locally on a single workstation equipped with an *AMD Radeon RX 6400 (4 GB VRAM, Vulkan backend)* and CPU PyTorch (`float32` AVX2), using `llama.cpp` build `8864 (ff6b1062a)` compiled with Clang `19.1.5` (`Windows x86_64`), launched via `llama-server -ngl 99 -dev Vulkan0 --cache-ram 0 --reasoning-format none`. Original model files in `local/models/` were preserved bit-for-bit untouched; all edited and quantized checkpoints were written to isolated artifact directories and SHA-256 verified before every inference run.

#figure(
  booktable(
    columns: (1.55fr, 1.35fr, 0.50fr, 0.62fr, 0.65fr, 0.78fr, 0.85fr, 1.85fr),
    align: (left, left, right, right, right, right, left, left),
    header: ([Model (BF16 Source)], [Sequence Mixer], [$L$], [$d_"model"$], [$d_"ff"$], [Heads (KV)], [Target $cal(L)$], [Original BF16 SHA-256 (Prefix)]),
    font-size: 7.3pt,
    inset: (x: 3pt, y: 3pt),
    rows: (
      ([`Qwen3-0.6B`\ (`1,198.2 MB`)], [Pure GQA + QK-Norm], [`28`], [`1024`], [`3072`], [`16 (8)`], [`[18, 20, 22]`], [`f9c9f1d3c1e21755...`]),
      ([`Qwen3.5-0.8B`\ (`1,516.7 MB`)], [18 Gated-DeltaNet +\ 6 Full-Attn (int=4)], [`24`], [`1024`], [`3584`], [`8 (2)`], [`[15, 19, 21]`], [`cedf89af31c9041b...`]),
      ([`DeepSeek-R1-1.5B`\ (`3,560.4 MB`)], [RL-Distilled Qwen2\ GQA Transformer], [`28`], [`1536`], [`8960`], [`12 (2)`], [`[20, 22, 25]`], [`6ec1aa822868a25c...`]),
    ),
  ),
  caption: [Exact GGUF architectural dimensions, compiled target layers, and immutable cryptographic SHA-256 digests of the three unmodified base models.],
) <tab:arch>

#figure(
  booktable(
    columns: (1.65fr, 0.65fr, 1.05fr, 0.90fr, 0.95fr, 1.15fr),
    align: (left, right, right, right, right, right),
    header: ([Model], [Layer], [`MLPProj`], [`AUROC_x`], [`vb_std` (cal)], [`dormant32 / median`]),
    font-size: 7.8pt,
    rows: (
      ([`Qwen3-0.6B`], [`L18`], [`1.5161`], [`0.9993`], [`0.0068`], [`0.3292`]),
      ([], [`L20`], [`3.2877`], [`0.9985`], [`0.0077`], [`0.3356`]),
      ([], [`L22`], [`6.9774`], [`0.9960`], [`0.0084`], [`0.3174`]),
      ([`Qwen3.5-0.8B`], [`L15` (FA)], [`0.1723`], [`0.9999`], [`0.0228`], [`0.5138`]),
      ([], [`L19` (FA)], [`0.1372`], [`1.0000`], [`0.0252`], [`0.6233`]),
      ([], [`L21` (DN)], [`0.1670`], [`1.0000`], [`0.0227`], [`0.6185`]),
      ([`DeepSeek-R1-1.5B`], [`L20`], [`1.4808`], [`0.9968`], [`0.0238`], [`0.6137`]),
      ([], [`L22`], [`2.6715`], [`1.0000`], [`0.0227`], [`0.5598`]),
      ([], [`L25`], [`5.5279`], [`0.9941`], [`0.0238`], [`0.5571`]),
    ),
  ),
  caption: [Mechanistic profile diagnostics at all 9 compiled target layers across the three models (`layer_profile.json`). Note: `Qwen3-0.6B` out-of-sample `vb_std` across held-out domains is `0.048–0.074`.],
) <tab:layerdiag>

= Anti-Leakage Dataset Splits & Total Program Accounting <app:datasets>

Across the complete WSP research program (spanning Papers 1 and 2), we executed *`206` verified runs* comprising *`18,513` raw inference responses*, *`1,810,381` prompt tokens*, and *`23,248,608` generated completion tokens* (`25,058,989` total tokens) with *`0` HTTP/parse errors* (`Qwen3-0.6B`: 115 study runs / `10,998` responses; `Qwen3.5-0.8B`: 31 study runs / `2,664` responses; `DeepSeek-R1-1.5B`: 36 study runs / `3,744` responses; quantization/calibration baselines: 24 runs / `1,107` responses).

#figure(
  booktable(
    columns: (2.0fr, 1.0fr, 0.70fr, 1.55fr, 2.0fr),
    align: (left, left, right, left, left),
    header: ([Dataset File (`datasets/wsp/`)], [Role], [Items], [SHA-256 Prefix], [Usage Policy]),
    font-size: 7.5pt,
    rows: (
      ([`cal_math_50.jsonl`], [Calibration], [`50`], [`a1d6d012924b2e5e`], [Trace $cal(P), cal(H), cal(R)$ (train split)]),
      ([`cal_gsm8k_48.jsonl`], [Calibration], [`48`], [`d473132b82d0f9b5`], [Trace $cal(P), cal(H), cal(R)$ (train split)]),
      ([`cal_copy_36.jsonl`], [Calibration], [`36`], [`a8025b6ede31658a`], [Preservation $cal(S)_"pres"$ + dev check]),
      ([`cal_wikitext_64.jsonl`], [Calibration], [`64`], [`4e5a6c9b994bc1a8`], [Preservation $cal(S)_"pres"$ + $b^*$ fit]),
      ([`dev_math_60.jsonl`], [Development], [`60`], [`dd9729dce118e168`], [Dose selection $alpha^*$ only (frozen pre-test)]),
      ([`test_math500_strat150.jsonl`], [Held-Out Test], [`150`], [`54709820f3e27767`], [Primary confirmatory benchmark]),
      ([`test_gsm8k_250.jsonl` (`[:100]`)], [OOD Test], [`100`], [`adad9d21c6985d0c`], [Zero-shot math generalization]),
      ([`test_arc_challenge_250.jsonl` (`[:100]`)], [Zero-Shot OOD], [`100`], [`2f4bc98b186b6cff`], [Zero-shot non-math science (0 cal items)]),
      ([`test_copy_90.jsonl`], [Held-Out Safety], [`90`], [`8567025de54bc42a`], [Exact copy / KV / cycle preservation]),
    ),
  ),
  caption: [Pinned anti-leakage dataset splits and exact SHA-256 digests (`datasets/wsp/manifest.json`).],
) <tab:datasets>

= Development Dose-Selection Screens & Echo-Gate Ablations <app:dev>

All scalar doses $alpha^*$ evaluated on held-out `MATH-500-strat150` were frozen strictly on `dev_math_60` (`[:24]` for Models 2 & 3) prior to running held-out inference (@tab:dev23). Additionally, on `Qwen3-0.6B`, replacing the constant `W_up` branch in high-dose `bawsc32` ($alpha = 0.70$) with an inside-`<think>` vs. post-`</think>` linear classifier (`bawsc32_andgate_a0.70`) achieved `AUROC = 0.9999–1.0000`, cut post-`</think>` circuit RMS down to `0.0008–0.00155`, preserved `PPL = 21.140` ($Delta"PPL" = -0.020$), and reduced development loops from `20` $->$ `12` (`dev_noecho_results.json`).

#figure(
  booktable(
    columns: (1.65fr, 1.45fr, 0.90fr, 1.05fr, 0.75fr, 1.15fr),
    align: (left, left, right, right, right, left),
    header: ([Model (`dev_math_60[:24]`)], [Candidate Arm], [Dev PPL], [Dev Correct (/24)], [Limit], [Selection Decision]),
    font-size: 7.5pt,
    row-fills: ("5": win-bg, "7": win-bg, "10": win-bg, "13": win-bg),
    rows: (
      ([`Qwen3.5-0.8B` (`Q8_0`)], [`baseline`], [`20.477`], [`1 (4.2%)`], [`24`], [Reference (`4.2%` close)]),
      ([], [`ctrl_concise_prompt`], [`20.477`], [`2 (8.3%)`], [`21`], [Prompt control]),
      ([], [`uncond_a0.8`], [`20.474`], [`2 (8.3%)`], [`22`], [Prior baseline]),
      ([], [`oblique_res_a0.45 / 0.70`], [`20.673 / 20.832`], [`7 / 7`], [`21 / 19`], [Lower close (`17–29%`)]),
      ([], [*`oblique_res_a1.00`*], [`21.100`], [`6 (25.0%)`], [*`16`*], [*Frozen (`62.5%` close!)*]),
      ([], [`bawsc32_res_a0.25`], [`20.511`], [`3 (12.5%)`], [`23`], [Under-dosed]),
      ([], [*`bawsc32_res_a0.45`*], [*`20.510`*], [*`4 (16.7%)`*], [`21`], [*Frozen (best BA-WSC)*]),
      ([`DeepSeek-R1-1.5B` (`Q8_0`)], [`baseline`], [`49.003`], [`12 (50.0% EOS)`], [—], [Reference]),
      ([], [`uncond_a0.8`], [`49.353`], [`14 (58.3% EOS)`], [—], [Prior baseline]),
      ([], [*`oblique_res_a0.25`*], [`49.063`], [*`15 (62.5% EOS)`*], [—], [*Frozen (best oblique)*]),
      ([], [`oblique_res_a0.50`], [`49.339`], [`14 (58.3% EOS)`], [—], [Higher PPL]),
      ([], [`bawsc32_res_a0.25`], [`48.989`], [`15 (62.5% EOS)`], [—], [Lower EOS]),
      ([], [*`bawsc32_res_a0.50`*], [*`49.013`*], [*`19 (79.2% EOS)`*], [—], [*Frozen (`+29.2 pp` dev!)*]),
    ),
  ),
  caption: [Strict pre-test development dose-selection screens for Models 2 and 3 (`dev_dose_selection.json`). Selected doses were committed before running any held-out `MATH-500` evaluation.],
) <tab:dev23>

= Reproducibility & Transparent Acknowledgments <app:repro>

*End-to-End Verification:* Running `python scripts/wsp_verify_papers.py` programmatically checks all 44 numerical claims across Papers 1 and 2 directly against raw JSON logs in `research/wsp/`, verifies all 8 300-DPI figures, recomputes pooled McNemar and Fisher meta-analyses, and confirms `0` errors across raw `responses.jsonl` traces.

*Acknowledgments & AI-Assistance Disclosure:* This research was conceived, directed, authorized, and executed on personal workstation hardware by Aaryan Kakad as an independent ML research program. Autonomous AI coding and research agents were used extensively for experiment orchestration, mechanistic PyTorch/GGUF software engineering, statistical auditing, figure generation, and manuscript drafting under human oversight and verification; all empirical claims are grounded strictly in immutable, cryptographically hashed raw JSON logs reproducible from the public repository.
