# Weight-Space Programming & Quantization Survival: Complete Technical Briefing

**Repository:** `AKMessi/weight-space-programming`  
**Hardware Envelope:** Single consumer AMD Radeon RX 6400 (4 GB VRAM, Vulkan via `llama.cpp`) + CPU PyTorch (zero gradients, zero backpropagation, zero cloud GPUs, zero external API calls)  
**Status:** **ALL THREE MODELS (`Qwen3-0.6B`, `Qwen3.5-0.8B`, `DeepSeek-R1-Distill-Qwen-1.5B`) 100% COMPLETE** across all held-out benchmarks, quantization matrices, 8 publication 300-DPI figures (`papers/figures/`), and both programmatically verified research papers (`papers/`).

---

## 1. How We Turned a Stalled Pilot Repo into a Rigorous Research Program

### 1.1 Starting Point: Why the Original Pilot Closed With Zero Confirmed Repairs (Nov 5–6, 2026)

When this research continuation began (up through commit `0fd8294`), the repository contained an honest, thoroughly audited bounded pilot (`FIVE_EXPERIMENTS.md`, `research/PROGRESS.md`, `research/FINDINGS.md`) executed on `Qwen3-0.6B`, `LFM3-1.7B`, and `lfm2.5-230m-fable-5`. That pilot explicitly closed with **no accepted general behavioral repair and no paper-ready editing method**.

Forensic inspection of the pilot logs revealed **four structural bottlenecks** that had doomed earlier attempts:

1. **Non-selective edit primitives:** Early edits applied fixed scalar attenuation to final-layer RMSNorm weights (`attn_norm` / `ffn_norm`). These edits had no state-conditional gate, lacked directional precision, and remained stored in `FP32` inside quantized GGUF files — meaning quantization stress tests could not even observe whether a real matrix edit survives 4-bit or 3-bit rounding.
2. **Broken activation readout bridge:** Attempting to extract native hidden states through `llama.cpp` endpoints failed the predeclared centered-logit parity check (`0/6` contexts passed; relative logit discrepancy `0.44%–3.28%`). Without exact internal activation access, no directional or subspace compiler could be built.
3. **Synthetic micro-panels dominated by trivial non-weight baselines:** The pilot evaluated 6–12 item synthetic JSON/tool-calling/copy fixtures where a grammar-constrained decoder solved `8/8` and a deterministic schema checker solved `12/12`. There was no genuine, hard neural failure mode where prompting and decoders actually fail.
4. **Uncontrolled evaluation variance:** Earlier demos lacked Frobenius-norm-matched random weight perturbations under stochastic sampling, leaving evaluation vulnerable to false-positive trajectory rerolls.

### 1.2 The Architectural & Methodological Pivot (Commit `fd6cb3f` Onward)

Instead of polishing weak synthetic demos, we archived the old pilot cleanly (`ac9977f`) and rebuilt the entire program around four engineering and scientific pillars:

1. **Exact `0.00e+00`-Diff GGUF-to-PyTorch Bridge (`src/wsp/gguf_to_torch.py`):** Loads arbitrary `BF16` (and exact dequantized `Q4_0`/`Q3_K_M`) GGUF checkpoints directly into PyTorch for exact internal residual/SwiGLU hook tracing (`src/wsp/compiler.py`) — bypassing the broken `llama.cpp` readout entirely while running cleanly on CPU (`412s` for all 28 layers of a 1.5B model using native AVX2 `float32` with `skip_lm_head=True`).
2. **A Real, Hard Reasoning Pathology:** **Overthinking Budget Exhaustion & Quantization-Induced Repetition Loop Explosion** in small reasoning LLMs (`<think> ... </think>`), where standard prompting, presence penalties, and DRY repetition samplers fail or actively degrade accuracy (`DRY` drops `Qwen3-0.6B` dev accuracy from `22/60` to `14/60`, `p = 0.039`).
3. **Strict Anti-Leakage Benchmark Protocol:**
   - **Calibration (`cal_*`):** Strictly training splits only (`cal_math_50` from MATH-train, `cal_gsm8k_48` from GSM8K-train, `cal_copy_36`, `cal_wikitext_64` from WikiText-2 validation).
   - **Development (`dev_math_60`):** Disjoint MATH-train slice used solely to select layer targets and scalar doses $\alpha$, frozen in git before touching held-out data.
   - **Untouched Held-Out Confirmation (`test_*`):** Stratified `MATH-500-strat150` (proportional across Levels 1–5), OOD `GSM8K-100`, zero-shot non-math science `ARC-Challenge-100` (zero science data ever seen in calibration or dev), `Copy-90` (exact copy, key-value retrieval, cycle induction), and 32-chunk `WikiText-2` test perplexity.
   - **Mandatory Controls:** Every held-out claim is tested against (a) unedited baseline, (b) **Frobenius-norm-matched isotropic Gaussian noise** in the exact same layers (`random_matched`), (c) **prior unconditional subspace excision** (`uncond`, Takehi / Steer2Edit style), and (d) exact no-op permutation (`ba_wsc_noop`) in quantization studies, using exact paired two-sided McNemar tests and 100,000-iteration problem-clustered sign-permutation tests.
4. **Three Architecturally Distinct Models:**
   - **Model 1:** `Qwen3-0.6B` (28L Pure GQA Transformer + QK-Norm, $d=1024, d_{\text{ff}}=3072$)
   - **Model 2:** `Qwen3.5-0.8B` (24L Hybrid **18 Gated-DeltaNet + 6 Full-Attention**, $d=1024, d_{\text{ff}}=3584$)
   - **Model 3:** `DeepSeek-R1-Distill-Qwen-1.5B` (28L Pure Qwen2 RL-distilled, $d=1536, d_{\text{ff}}=8960$, $2.5\times$ larger)

This transformed the repository into **two cohesive, publication-grade research papers** built on **four closed-form inventions**, **two newly identified evaluation traps**, **one formal hardware-alignment theorem + corollary**, and **over 15,000 logged, checksum-verified inference traces**.

---

## 2. Two Methodological & Evaluation Discoveries

### 2.1 Discovery 1: The Greedy-Perturbation Illusion

Under finite-budget **greedy decoding (`temp=0.0`, 2048 max tokens)** on held-out `MATH-500-strat150` (`Qwen3-0.6B-BF16`), injecting **zero-information isotropic Gaussian noise** (`random_matched_oblique`) into L18/20/22 `ffn_down` increased strict Clean-EOS accuracy by **`+6.0 percentage points`** (`52/150 [34.7%] → 61/150 [40.7%]`, McNemar `p = 0.0636`), outperforming prior unconditional WSP (`57/150`)!

**Verified across 6 zero-information greedy draws (`Qwen3-0.6B`, `MATH-500-strat150`, `temp=0.0`):**
- `BF16 baseline`: `52/150 (34.7%)`
- `Q8_0 + rand_s1000`: `52/150 (34.7%)`
- `Q8_0 + rand_s999`: `55/150 (36.7%)`
- `Q8_0 baseline`: `57/150 (38.0%)`
- `Q8_0 + rand_s1001`: `57/150 (38.0%)`
- `BF16 + rand_s999`: `61/150 (40.7%)`
- **Chaotic zero-info greedy band:** `52–61/150` (`6.0 pp` width; mean `55.7`, SD `3.4`).

**Exact Mathematical Mechanism (Asymmetric Reservoir Reroll Bias):**
Inspecting all 14 items rescued by `random_matched_oblique` revealed that **`14/14 (100%)` had hit the 2048-token cutoff at baseline** (5 looping), while the 5 broken items had a mean baseline length of `1,710` tokens (already near the cutoff cliff). Because the reservoir of baseline truncated failures ($N_{\text{trunc}} = 97$) is nearly twice as large as the reservoir of fragile near-cliff successes ($N_{\text{fragile}} \ll 52$), any tiny weight perturbation acts as a single-shot deterministic trajectory reroll with positive expected bias:
$$\mathbb{E}[\Delta \text{Acc}_{\text{greedy}}] \approx \frac{1}{N}\left(N_{\text{trunc}}\, p_{\text{escape}} - N_{\text{fragile}}\, p_{\text{break}}\right) > 0.$$
Crucially, `random_matched_oblique` did **not** suppress repetition loops (`29 → 30`, `p = 1.000`).

**Resolution under Official Multi-Seed Stochastic Sampling (`temp=0.6, top_p=0.95, top_k=20`, Seeds 17 + 42, $N=300$ rollouts):**
- `baseline`: `105/300 (35.0%)` Clean-EOS (Seed 17: `52`, Seed 42: `53` — $\pm 1$ item stability).
- `random_s999`: **`104/300 (34.7%, Δ = -0.33 pp [-4.3, +3.7], McNemar p = 1.000, clustered-perm p = 1.000)`**.
- On OOD `GSM8K-100` (`temp=0.6`), `random_s999` actually **harmed** performance (`71% → 64%`, `-7.0 pp`; timeouts surged `19 → 31`, `p = 0.029`).

**Methodological Mandate:** Any weight-editing study on finite-budget reasoning benchmarks evaluated solely at `temp=0.0` without Frobenius-matched random noise controls cannot distinguish true mechanistic editing from asymmetric greedy reroll noise.

### 2.2 Discovery 2: The Final-Answer Echo Trap & Strict Clean-EOS Auditing

Standard math graders extract the last `\boxed{}` match anywhere in the output. On `Qwen3-0.6B-BF16` (`MATH-500-strat150`), high-dose `bawsc32_res_a0.70` achieved:
- `WikiText-2 PPL = 19.2836` (`ΔPPL = -0.022` — zero perplexity penalty),
- **Standard `\boxed{}` accuracy = `66/150 (44.0%, +8.7 pp, p = 0.0294)`** — appearing to be the single best checkpoint!

Raw trace inspection exposed a fatal pathology: once the model closed `</think>` and printed `\boxed{answer}`, residual circuit firing kept triggering post-answer repetition until the 2048-token cap:
- Repetition loops **more than doubled** (`29 → 59`, `p = 0.0001`).
- Under **Strict Clean-EOS grading** (`correct AND has_think_close AND finish_reason == "stop"`), `bawsc32_res_a0.70` plummeted from `44.0%` to **`48/150 (32.0%)` — statistically WORSE than random noise (`p = 0.0106`)**!

We solved this two ways: (1) enforcing **Strict Clean-EOS** as our primary metric across all experiments (at proper dose `a=0.25`, `bawsc32` achieves `65/65 = 100%` Clean-EOS alignment with zero echo), and (2) implementing a **SwiGLU `W_up` inside-`<think>` AND-gate ablation** (`bawsc32_andgate_a0.70`), which uses `W_up` to suppress firing once post-`</think>` representations emerge (`post-think activation RMS = 0.0008`, cutting `a=0.70` dev loops from `20 → 12`).

---

## 3. Four Closed-Form Weight-Space Programming Inventions

All four methods require **zero gradient computation, zero backpropagation, zero fine-tuning, zero additional parameters, and zero inference-time latency overhead**. Every edit is compiled directly in closed form from ~60–80 calibration trajectories and written in place into a GGUF checkpoint.

### 3.1 Universal Cross-Architecture Anatomy & Invention A: Virtual Constant-1 Bias Register ($b^*$)

Tracing all layers across all three model families (`Qwen3-0.6B`, Hybrid `Qwen3.5-0.8B`, `DeepSeek-R1-1.5B`) revealed a **universal late-MLP reasoning-control anatomy**:
- Early and middle layers have near-zero MLP projection onto the pathology difference vector $\mu_{\text{path}} - \mu_{\text{healthy}}$ (`Qwen3` L0–15 flat; `Qwen3.5` L0–13 `0.004–0.023`; `DeepSeek-R1-1.5B` L0–18 $\le 0.56$, L18 = `0.038`).
- Attribution then surges **$10\times\text{--}35\times$** in late pre-readout MLP layers:
  - `Qwen3-0.6B` (28L): L16–22 (`1.12 → 6.98`, readout L27 = `65.87`).
  - `Qwen3.5-0.8B` (24L Hybrid): **automatically pinpoints Full-Attention L15 (`0.1723`, $10\times$ jump) and Full-Attention L19 (`0.1372`)** plus DeltaNet L21 (`0.1670`, readout L23 = `0.5682`).
  - `DeepSeek-R1-1.5B` (28L): **$35\times$ jump at L19 (`1.316`) → L22 (`2.672`) → L25 (`5.528`) → L26 (`7.776`)** (readout L27 = `40.126`).

Modern LLMs use bias-free SwiGLU MLPs: $\text{MLP}(x) = W_{\text{down}}\!\left(\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\right)$, where $x = \text{RMSNorm}(h) \in \mathbb{R}^{d}$. Without an additive bias term, a single SwiGLU neuron cannot natively express a shifted threshold $\text{SiLU}(u^\top x - \tau)$ required for an `IF (u^\top x > \tau) THEN \Delta h` control-flow gate.

**Invention A (Virtual Constant-1 Bias Register $b^*$):** Because post-RMSNorm activations $x$ lie near a structured low-curvature manifold with stable directional means across tokens, we solve the closed-form ridge regression on calibration preservation tokens $X_{\text{pres}}$:
$$b^* = \left(X_{\text{pres}}^\top X_{\text{pres}} + \lambda I\right)^{-1} X_{\text{pres}}^\top \mathbf{1}.$$
Out-of-sample across held-out domains, $b^{*\top} x \approx 1$ holds remarkably tightly across all three architectures:
- `Qwen3-0.6B`: OOS $\text{std}(b^{*\top} x) = 0.048\text{--}0.074$
- `Qwen3.5-0.8B` (Hybrid): OOS $\text{std}(b^{*\top} x) = 0.0208\text{--}0.0252$
- `DeepSeek-R1-1.5B`: OOS $\text{std}(b^{*\top} x) = 0.0198\text{--}0.0250$

Thus, setting a gate row to $w_{\text{gate}} = \gamma (u - \tau b^*)$ compiles an exact affine threshold $w_{\text{gate}}^\top x \approx \gamma (u^\top x - \tau)$ inside a strictly bias-free SwiGLU layer!

### 3.2 Invention B: Null-Space-Gated Oblique Surgery (`oblique_res`)

Prior unconditional WSP methods (e.g., Steer2Edit / Takehi subspace excision) project out a pathology direction indiscriminately across all tokens, incurring heavy WikiText-2 perplexity degradation (`ΔPPL = +1.338` on `Qwen3-0.6B`).

We instead compute a **closed-form rank-1 oblique update in post-SwiGLU space** $z = \text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x) \in \mathbb{R}^{d_{\text{ff}}}$:
1. Extract the top-$r$ ($r=64$) right singular subspace $V_r$ of preservation activations $Z_{\text{pres}}$ (healthy reasoning + exact copy + WikiText-2) and form the null-space projector $P_\perp = I - V_r V_r^\top$.
2. Solve the ridge-regularized pathological detector in the null space:
   $$d_{\text{raw}} = P_\perp \left(Z_{\text{pres}}^\top Z_{\text{pres}} + \lambda I\right)^{-1} (\mu_{z,\text{path}} - \mu_{z,\text{healthy}}), \qquad d = \frac{d_{\text{raw}}}{\mu_{z,\text{path}}^\top d_{\text{raw}} - \mu_{z,\text{healthy}}^\top d_{\text{raw}}}.$$
3. Target the **resolution transition vector** $v_{\text{resolve}} = \mu_{h,\text{resolve}} - \mu_{h,\text{path}}$ (measured from the final 12 tokens preceding clean `</think>` closures in calibration rollouts) rather than blindly ablating toward generic text:
   $$\Delta W_{\text{down}} = \alpha \, v_{\text{resolve}} \, d^\top.$$

By construction, $\Delta W_{\text{down}} z \approx 0$ on preservation text/copying (cutting PPL cost by **$10\times$**, from `+1.338` to `+0.141` on `Qwen3-0.6B`) while steering stalled reasoning states directly toward `</think>` resolution.

### 3.3 Invention C: Block-Aligned Gauge-Balanced Weight-Space Compilation (`BA-WSC`) — Theorem 1 & Corollary 1

While `oblique_res` writes a diffuse rank-1 matrix across all $d_{\text{ff}}$ columns of $W_{\text{down}}$, we discovered that **4-bit (`Q4_0`) and 3-bit (`Q3_K_M`) block quantization severely corrupts diffuse and scattered sparse edits**:
- GGUF block quantizers partition each row of $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ into contiguous blocks of size $B = 32$ (`Q4_0`, `Q8_0`) or hierarchical super-blocks of $QK_K = 256$ containing 8 sub-blocks of 32 (`Q4_K_M`, `Q3_K_M`).
- When an edit touches scattered columns (like Takehi's `sparse64`) or adds small diffuse entries (`uncond`, `oblique`), two destructive effects occur: (1) small edit deltas fall below the block's existing dynamic range and round to zero (**signal loss**, $\cos(\Delta W_Q, \Delta W_{\text{FP}}) = 0.23\text{--}0.54$), and (2) larger sparse deltas inflate the block scale $\delta_b = \max_{j \in b}|W_{ij}|/q_{\max}$, shifting the rounding grid for all 31 innocent unedited neighbor weights in that block (**collateral distortion**, reaching up to **$4.1\times$ larger than the edit signal itself!**).

**Construction of `BA-WSC` (Nonlinear Dormant-SwiGLU `IF-THEN` Compiler):**
1. **Dormant Neuron Discovery:** Measure per-neuron RMS across all preservation tokens and identify the $k$ lowest-RMS dormant SwiGLU slots $\mathcal{S}$ ($|\mathcal{S}| = k$, where $k = 32$ for `Q4_0` or $k = 256$ for K-quant super-blocks).
2. **Exact SwiGLU Permutation Symmetry $P$:** Construct permutation $P$ moving $\mathcal{S}$ to contiguous hardware-aligned slots $\{0, 1, \dots, k-1\}$, and permute:
   $$W_{\text{down}}' = W_{\text{down}} P, \qquad W_{\text{gate}}' = P^\top W_{\text{gate}}, \qquad W_{\text{up}}' = P^\top W_{\text{up}}.$$
   Because $\text{MLP}(x) = W_{\text{down}} P \, (\text{SiLU}(P^\top W_{\text{gate}} x) \odot P^\top W_{\text{up}} x) = W_{\text{down}}(\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)$, this transformation produces **$0.000000$ floating-point functional change** and adds **0 bytes and 0 runtime ops**.
3. **Replicated Constant-Payload `IF-THEN` Compilation with Gauge Balancing:** Into all $k$ aligned slots $j \in \{0, \dots, k-1\}$, compile identical copies of the thresholded detector + action:
   $$W_{\text{gate}}'[j, :] = \gamma(u - \tau b^*)^\top, \qquad W_{\text{up}}'[j, :] = \eta \, (b^*)^\top, \qquad W_{\text{down}}'[:, j] = \frac{\alpha}{k \, \eta \, s_{\text{path}}} \, v_{\text{resolve}},$$
   where exact SwiGLU positive-homogeneity gauge factor $\eta$ balances $\max|W_{\text{up}}'|$ and $\max|W_{\text{down}}'|$ so both sit cleanly inside quantized dynamic range.

**Theorem 1 (Sub-Block Alignment, $k = B = 32$):** Under any symmetric or affine per-block quantizer of block size $B = 32$ (e.g., `Q4_0`, `Q8_0`), the compiled block $b=0$ of every row $i$ of $W_{\text{down}}'$ is constant-valued ($W_{\text{down}}'[i, j] = c_i\ \forall j \in [0, 32)$). Consequently:
1. **Zero Mixed Blocks ($N_{\text{mixed}} = 0$):** Every unedited weight in blocks $b \ge 1$ retains its exact pre-edit block scale and integer code — **strictly zero scale-induced neighbor collateral**.
2. **Exact Integer Representation:** Inside block $b = 0$, the block scale $\delta_{i,0} = |c_i|/q_{\max}$ maps every element to the exact extreme quantization level $\text{sign}(c_i) q_{\max}$, so edit fidelity is bounded solely by the FP16 block-scale precision ($\cos \ge 0.996$ even at 4-bit!).

**Corollary 1 (Hierarchical Super-Block Alignment, $k = QK_K = 256$):** In GGUF K-quants (`Q4_K_M`, `Q3_K_M`), 8 sub-blocks of 32 share a single FP16 super-block scale $d$ (width $QK_K = 256$). Aligning only $k = 32$ leaves 7 unedited sub-blocks sharing $d$ with the edited sub-block, producing residual collateral (`0.015–0.079`). Extending the exact permutation $P$ to $k = 256$ isolates the entire super-block, eliminating super-block scale interference and cutting collateral by a further **$119\times\text{--}196\times$** (up to **$16,685\times$** below Takehi-style `sparse64`), at a neuron budget of only `8.3%` in 0.6B (`d_ff=3072`) and just **`2.86%` in 1.5B (`d_ff=8960`)**!

### 3.4 Invention D: Post-Quantization Weight-Space Healing (`PQ-BA-WSC` / `PQ-Oblique`)

When a model is quantized to 4-bit (`Q4_0`) or 3-bit (`Q3_K_M`), its activation manifold shifts and new quantization-specific repetition loops emerge (`3.1×–8.2×` higher loop rates). Compiling an edit on FP16 trajectories and transferring it to 3-bit leaves substantial healing on the table.

**Post-Quantization Healing (`src/wsp/pq_loader.py` + `scripts/wsp_post_quant_heal.py`):**
1. Dequantize the target `Q4_0` or `Q3_K_M` GGUF directly into PyTorch (`load_dequantized_gguf`).
2. Trace the quantized model's *own* calibration rollouts (capturing its actual low-bit loop states and residual shifts).
3. Compile a block-aligned `PQ-BA-WSC` (or `PQ-Oblique`) patch directly on the dequantized weights and re-quantize to the target format.

---

## 4. Complete Verified Quantitative Evidence Across All 3 Architectures

Every number below is read directly from immutable JSON logs committed in `research/wsp/` and `runs/wsp_baselines/`.

### 4.1 Paper 1 — Evaluation Controls, Multi-Seed Stochastic Gains, Zero-Shot OOD Transfer & Hybrid Replication

#### A. Non-Weight Controls vs. Weight Compilers (`Qwen3-0.6B` Strict Dev `dev_math_60`, `BF16`, Greedy)
| Method (`dev_math_60`) | WikiText-2 PPL (16-ch) | STD Correct | Limit | Loops | Notes |
|---|---:|---:|---:|---:|---|
| `baseline_fp` | `21.160` | `22/60 (36.7%)` | `36` | `11` | Reference |
| `ctrl_dry_sampler` | `21.160` | `14/60 (23.3%)` | `40` | `0` | **Significantly harms (`p = 0.039`)** |
| `ctrl_presence_pen` | `21.160` | `21/60 (35.0%)` | `37` | `6` | No gain |
| `ctrl_concise_prompt` | `21.160` | `25/60 (41.7%)` | `32` | `5` | Not significant (`p = 0.508`) |
| `uncond_a0.8` (Prior) | `22.648 (+1.488)` | `27/60 (45.0%)` | `33` | `11` | Heavy PPL cost |
| `oblique_res_a0.25` (Ours A) | `21.328 (+0.168)` | **`28/60 (46.7%)`** | **`29`** | **`5`** | Selected for held-out |
| `bawsc32_res_a0.25` (Ours C) | **`21.139 (-0.021)`** | `26/60 (43.3%)` | `31` | `8` | Zero PPL cost; clean EOS |
| `bawsc32_res_a0.70` (Echo trap) | `21.135 (-0.025)` | `32/60 (53.3%)` | `32` | **`20`** | Echo trap exposed! |

#### B. Model 1 (`Qwen3-0.6B-Q8_0`) Untouched Held-Out Multi-Seed Stochastic `MATH-500` ($N=300$ rollouts, Seeds 17+42, `temp=0.6`)
*(Exact pooled two-sided McNemar `mc_p` and 100,000-perm problem-clustered `cl_p`; 20,000-sample clustered bootstrap 95% CIs)*

| Arm (`MATH-500-strat150`, 2 seeds) | $\Delta$PPL | STD (/300) | **Clean-EOS (/300)** | $\Delta$ vs Base [95% CI] | `cl_p` vs Base | `cl_p` vs Rand | `</think>` Close | Limit | Mean Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `0.000` | `109` | `105 (35.0%)` | — | — | `1.000` | `44.7%` | `189` | `1784.9` |
| `random_s999` (Neg. Control) | `-0.023` | `109` | **`104 (34.7%)`** | `-0.33pp [-4.3,+3.7]` | `1.000` | — | `43.0%` | `191` | `1774.1` |
| `uncond_a0.8` (Prior WSP) | `+1.338` | `121` | `116 (38.7%)` | `+3.67pp [-0.3,+7.7]` | `0.107` (n.s.) | `0.088` (n.s.) | `49.0%` | `176` | `1741.7` |
| `bawsc32_res_a0.25` (Ours C) | **`-0.018`** | `118` | `114 (38.0%)` | `+3.00pp [-1.7,+7.7]` | `0.255` | `0.154` | **`50.0% (p=.020)`** | **`173 (p=.025)`** | **`1733.5 (p=.007)`** |
| **`oblique_res_a0.25` (Ours A)** | `+0.141` | **`125`** | **`122 (40.7%)`** | **`+5.67pp [+1.3,+10.0]`** | **`0.0148`** | **`0.0146`** | **`52.3% (p=.0017)`** | **`171 (p=.015)`** | **`1722.5 (p=.0008)`** |

#### C. Model 1 (`Qwen3-0.6B-Q8_0`) Untouched OOD Transfer (`GSM8K-100`) & Zero-Shot Non-Math Science (`ARC-Challenge-100`, `temp=0.6`, Seed 42)
*(ZERO science questions ever appeared in calibration or development!)*

| Arm | `GSM8K-100` Clean-EOS | `GSM8K` Limit / Tok / Wall | `ARC-100` Clean-EOS | `ARC` Limit / Tok | `Copy-90` (BF16) |
|---|---:|---:|---:|---:|---:|
| `baseline` | `71/100 (71%)` | `19 / 1234 / 1062s` | `56/100 (56%)` | `26 / 597` | `64/90 (71.1%)` |
| `random_s999` | `64/100 (-7pp)` | `31 / 1321 / 985s` | `62/100 (p=0.180 n.s.)` | `23 / 582` | `64/90 (71.1%)` |
| `uncond_a0.8` | `73/100 (p=0.839)` | `18 / 1274 / 823s` | `64/100 (p=0.039)` | `19 / 548` | `66/90 (73.3%)` |
| `bawsc32_res_a0.25` | `75/100 (p=.027 vsR)` | `15 / 1167 / 894s` | **`67/100 (+11pp, p=0.0074)`** | **`11 (-58%) / 536`** | **`65/90 (72.2%)`** |
| **`oblique_res_a0.25`** | **`80/100 (+9pp, p=0.049; +16pp vsR p=0.0001)`** | **`8 (-58%) / 1096 / 744s (-30% faster!)`** | **`71/100 (+15pp, p=0.0003)`** | **`9 (-65%) / 497`** | `64/90 (71.1%)` |

#### D. Model 2 (`Qwen3.5-0.8B` Hybrid Gated-DeltaNet, `Q8_0`) Cross-Architecture Replication (`MATH-500-strat150`, `temp=0.6`, Seed 42)
- `baseline`: PPL `15.513`, Clean-EOS `1/150 (0.7%)`, Limit `149`, Close `0.7%`, MeanTok `2044.0`, `Copy-90 = 66/90`.
- `random_matched`: PPL `15.723`, Clean-EOS `3/150 (2.0%, p=0.500 n.s.)`, Limit `147`, `Copy-90 = 67/90`.
- `uncond_a0.8`: PPL `15.509`, **Clean-EOS `0/150 (0.0%, limit = 150 — complete failure)`**, `Copy-90 = 66/90`.
- `bawsc32_res_a0.45`: PPL `15.541 (+0.028)`, STD `8/150`, Clean-EOS `3/150`, Close `4.7% (p=0.0312 vs base)`, `Copy-90 = 66/90`.
- **`oblique_res_a1.0`:** PPL `15.814 (+0.301)`, STD `12/150`, **Clean-EOS `10/150 (6.7% — 10× jump!)`**, beating baseline (`+9/-0, p = 0.0039`), beating `random_matched` (`+8/-1, p = 0.0391`), beating `uncond_a0.8` (`+10/-0, p = 0.0020`), cutting limits `149 → 126 (-23, p < 0.0001)` and tokens `-199 tok/prob`, with **`Copy-90 = 66/90` (0.0 pp loss)**.

#### E. Model 3 (`DeepSeek-R1-Distill-Qwen-1.5B`) Complete 15-Arm Held-Out Results (`MATH-500-strat150`, `temp=0.6`, Seed 42, `error_count = 0` across all 3,600 requests)
- **8-Bit `Q8_0` (`#1` on 3/3 architectures for Paper 1!):**
  - `baseline`: PPL `37.254`, STD `45/150`, Clean-EOS `43/150 (28.7%)`, Limit `77`, Wall `4403s`, `Copy contains_gold = 68/90`.
  - `random_matched`: PPL `37.298`, Clean-EOS **`39/150 (26.0%, -2.7 pp)`**, Limit `82 (+5)`.
  - `uncond_a0.8`: PPL `37.532 (+0.278)`, Clean-EOS `48/150` (`p = 0.136` vs rand n.s.).
  - `bawsc32_res_a0.5` *(Ours C)*: **PPL `37.257 (+0.003)`**, Clean-EOS `49/150 (+6 vs base, +10 vs rand)`, Close `p = 0.0043` vs base / `p = 0.0003` vs rand, Limit `67 (-10)`, **`-22%` faster wall-clock (`3445s`)**.
  - **`oblique_res_a0.25` *(Ours A, `#1`)*:** PPL `37.338 (+0.084)`, STD `55/150`, **Clean-EOS `53/150 (35.3%, +6.7 pp vs base, +9.3 pp vs random: EOS p = 0.0288, STD p = 0.0201, Limit p = 0.0146, Close p = 0.0026)`**, **`-20%` faster (`3527s`)**, `Copy gold = 70/90`.
  - **3-Model Pooled `Q8_0` Meta-Analysis (`N = 600` rollouts across `Qwen3-0.6B`, `Qwen3.5-0.8B`, `DeepSeek-R1-1.5B`):** `oblique_res` reaches **`+61 / -25` (`+6.00 pp`, exact McNemar `p = 0.00013`, Fisher `p = 0.00059`)** vs baseline and **`+65 / -26` (`+6.50 pp`, `p = 5.3 × 10⁻⁵`)** vs matched random noise!
- **4-Bit `Q4_0` & 3-Bit `Q3_K_M` Replicated Precision × Method Crossover:**
  - `Q4_0`: `bawsc32_res_a0.5` overtakes `oblique_res` (`66/150` vs `64/150`, `ΔPPL = -0.131`, lowest timeouts `62`, beating random noise across all 4 metrics: EOS `p = 0.0357`, STD `p = 0.0192`, Limit `p = 0.0072`, Close `p = 2.7 × 10⁻⁵`).
  - `Q3_K_M` (`baseline`: `48/150`, `lim = 99/150 = 66%`): Diffuse `oblique_res_a0.25` (`cos = 0.406`) **collapses below baseline (`45/150`, `ΔPPL = +0.242`)**, while Theorem 1 block-aligned **`bawsc32_res_a0.5` (`cos = 0.9962`)** holds a flat `+6, +6, +6` gain across 8/4/3-bit (`54/150` EOS, `55/150` STD, `ΔPPL = +0.051`, `+9` over oblique)! *(Replicated `Q8 → Q3` crossover: `+13/150`, `perm_p = 0.0663` EOS / `+14/150`, `p = 0.0594` STD; **Fisher combined meta-analysis with Model 2 `Qwen3.5-0.8B`: $\chi^2_4 = 16.82$, `p = 0.00210`**!)*

---

### 4.2 Paper 2 — Quantization Cliff, Theorem 1 & Corollary 1 Verification, Healing Matrix & 4-Bit Crossover

#### A. Scale-Dependent Quantization-Induced Loop Explosion (`cal_math_50`)
- **`Qwen3-0.6B` (0.6B Pure GQA):** `BF16` loops `20%` (PPL `19.31`) → `Q4_0` loops **`46%` (`2.3×` greedy; `3.1×` stochastic on MATH-500, `-12.0 pp` cliff)** → `Q3_K_M` loops **`74%` (`3.7×` greedy; `8.2×` stochastic on MATH-500, PPL `26.72`)**.
- **`Qwen3.5-0.8B` (0.8B Hybrid DeltaNet):** `Q8_0` stochastic loops `34%` (PPL `15.51`) → `Q4_0` loops **`88%` (`2.6×` cal; `83.3%` on MATH-500)** → `Q3_K_M` loops **`92%` (`2.7×`)**.
- **`DeepSeek-R1-Distill-Qwen-1.5B` (1.5B RL-Distilled Qwen2):** Absorbs 4-bit `Q4_0` cleanly (`limit 36%`, `loops 6%`), shifting the cliff down to **3-bit `Q3_K_M`** (`limit` jumps `36% → 58%`, `</think>` close drops `66% → 48%`, loops `2×`, mean tokens `+214`).

#### B. Three-Architecture Master Verification of Theorem 1 ($k=32$) & Corollary 1 ($k=256$) on Real `llama-quantize`
*(Measured directly on dequantized `ffn_down` weight tensors after running official `llama-quantize`)*

| Model (Layer, $d_{\text{ff}}$) | Method | `Q4_0` cos / col | `Q4_K_M` cos / col | `Q3_K_M` cos / col | Peak Collat. Reduction vs `sparse64` |
|---|---|---:|---:|---:|---:|
| **Qwen3-0.6B** (L20, `3072`) | `uncond_a0.6` | `0.386 / 0.473` | `0.754 / 0.275` | `0.332 / 0.958` | — |
| | `sparse64_a1.0` (Takehi) | `0.640 / 0.633` | `0.916 / 0.351` | `0.379 / 2.257` | `1×` (ref) |
| | `dormant32_scattered` | `0.988 / 0.116` | `0.999 / 0.039` | `0.950 / 0.313` | `7×` |
| | **`BA-WSC k=32` (Thm 1)** | **`0.9959 / 0.00063`** | **`0.9997 / 0.01453`** | **`0.9942 / 0.07877`** | **`1,004×` (`Q4_0`)** |
| | **`BA-WSC k=256` (Cor 1)** | **`0.9961 / 0.00031`** | **`0.99983 / 0.00011`** | **`0.99732 / 0.00050`** | **`4,514×` (`Q3_K_M`)** |
| **Qwen3.5-0.8B** (L19, `3584`) | `uncond_a0.6` | `0.339 / 0.593` | `0.284 / 1.180` | `0.284 / 1.180` | — |
| | `sparse64_a1.0` (Takehi) | `0.548 / 0.834` | `0.293 / 3.036` | `0.293 / 3.036` | `1×` (ref) |
| | `oblique_a0.25` | `0.544 / 0.261` | `0.526 / 0.427` | `0.526 / 0.427` | `7×` |
| | **`BA-WSC k=32` (Thm 1)** | **`0.9961 / 0.00013`** | **`0.9955 / 0.05881`** | **`0.9955 / 0.05881`** | **`6,417×` (`Q4_0`)** |
| | **`BA-WSC k=256` (Cor 1)** | **`0.9962 / 0.00005`** | **`0.99739 / 0.00030`** | **`0.99739 / 0.00030`** | **`16,685×` (`Q4_0`) / `10,119×` (`Q3_K`)** |
| **DeepSeek-R1-1.5B** (L20, `8960`) | `uncond_a0.6` | `0.299 / 0.717` | `0.611 / 0.476` | `0.236 / 1.693` | — |
| | `sparse64_a1.0` (Takehi) | `0.511 / 0.841` | `0.798 / 0.625` | `0.226 / 4.125` | `1×` (ref) |
| | `oblique_a0.25` | `0.461 / 0.404` | `0.828 / 0.223` | `0.406 / 0.784` | `5×` |
| | **`BA-WSC k=32` (Thm 1)** | **`0.9961 / 0.00115`** | **`0.99979 / 0.00834`** | **`0.99618 / 0.04540`** | **`732×` (`Q4_0`)** |
| | **`BA-WSC k=256` (Cor 1, 2.86% neurons!)** | **`0.99625 / 0.00014`** | **`0.99984 / 0.00007`** | **`0.99741 / 0.00034`** | **`12,133×` (`Q3_K_M`)** |

#### C. Model 1 (`Qwen3-0.6B`) 20-Arm 4-Bit `Q4_0` & 3-Bit `Q3_K_M` Survival & Healing Matrix (`MATH-500-strat150`, `temp=0.6`, Seed 42)
*(Unedited 8-bit `Q8_0` reference: PPL `19.280`, Clean-EOS `53/150 [35.3%]`, Loops `9`)*

| Precision & Arm | PPL ($\Delta$PPL) | STD | **Clean-EOS (/150)** | McNemar `p` vs Quant Base | Limit | Loops | `Copy-90` | $\Delta W$ `cos` / `col` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Q4_0` `baseline`** | `21.866 (0.000)` | `36` | `35 (23.3%)` | — | `114` | `28` | `65/90` | — |
| `Q4_0` `ba_wsc_noop` (Perm ctrl) | `21.870 (+0.004)` | `33` | `32 (21.3%)` | `0.607` | `116` | `25` | `64/90` | — |
| `Q4_0` `random_matched` | `21.785 (-0.081)` | `34` | `32 (21.3%)` | `0.664` | `118` | `23` | `66/90` | `0.582 / 0.147` |
| `Q4_0` `uncond_a0.8` (Prior) | `22.805 (+0.939)` | `38` | `38 (25.3%)` | `0.581` (n.s.) | `110` | `25` | `65/90` | `0.443 / 0.370` |
| `Q4_0` `bawsc32_res` (FP→Q4) | `21.847 (-0.019)` | `43` | `41 (27.3%)` | `0.263` | `102` | `42` | `64/90` | `0.9959 / 0.00063` |
| **`Q4_0` `pq_bawsc32_a0.25` (Ours)** | **`21.869 (+0.003)`** | `47` | **`46 (30.7%, +7.3pp)`** | **`0.0347` (`p=.0125` vsR)** | `100` | `36` | `64/90` | **`0.9960 / 0.00036`** |
| **`Q4_0` `oblique_res_a0.25` (Ours)** | `22.034 (+0.168)` | **`53`** | **`50 (33.3%, +10.0pp)`** | **`0.0041` (`p=.0005` vsR)** | **`98`** | `41` | `64/90` | `0.691 / 0.156` |
| **`Q3_K_M` `baseline`** | `26.723 (0.000)` | `24` | `19 (12.7%)` | — | `115` | `74` | `39/90` | — |
| `Q3_K_M` `ba_wsc_noop` | `26.792 (+0.069)` | `22` | `20 (13.3%)` | `1.000` | `122` | `84` | `42/90` | — |
| `Q3_K_M` `random_matched` | `26.775 (+0.052)` | `26` | `21 (14.0%)` | `0.815` | `119` | `81` | `39/90` | `0.625 / 0.170` |
| `Q3_K_M` `uncond_a0.8` (Prior) | **`28.924 (+2.202)`** | `27` | `27 (18.0%)` | `0.096` (n.s.) | `93` | `68` | `41/90` | `0.401 / 0.690` |
| `Q3_K_M` `bawsc32_res` (FP→Q3) | `26.807 (+0.084)` | `27` | `26 (17.3%)` | `0.167` | `97` | `79` | `40/90` | `0.9941 / 0.0788` |
| `Q3_K_M` `pq_oblique_a0.25` | `26.765 (+0.043)` | `32` | `30 (20.0%, +7.3pp)` | **`0.0266`** | `85` | **`53 (-28%)`** | `39/90` | `0.566 / 0.411` |
| `Q3_K_M` `pq_bawsc32_a0.25` | `26.834 (+0.112)` | `30` | `30 (20.0%, +7.3pp)` | **`0.0347`** | `87` | `63` | `41/90` | `0.9941 / 0.0789` |
| **`Q3_K_M` `pq_bawsc32_a0.35` (#1!)** | **`26.839 (+0.116)`** | **`34`** | **`32 (21.3%, +8.7pp / +68% rel.)`** | **`0.0044` (`p=.0266` vsR)** | **`82`** | `60` | `40/90` | **`0.9941 / 0.0788`** |

**Key Takeaways:** (1) `oblique_res` heals **83% of the entire 4-bit `Q4_0` cliff** (`35 → 50/150` vs 8-bit `53/150`, `p = 0.0041`). (2) Compiling directly on quantized activations (**`PQ-BA-WSC`**) beats FP-transfer at BOTH 4-bit (`46` vs `41/150`) and 3-bit (`30–32` vs `26/150`), winning **#1 overall at 3-bit `Q3_K_M`** (`+68%` relative gain, `p = 0.0044`, at $19\times$ lower PPL penalty than `uncond`)!

#### D. Model 2 (`Qwen3.5-0.8B` Hybrid) Behavioral Proof of Theorem 1: The Precision × Method Double Crossover (`p = 0.00336`!)

Evaluating the exact same 5 frozen arms on 100% untouched `MATH-500-strat150` across `Q8_0` and `Q4_0`:
- **At 8-bit `Q8_0`:** Diffuse `oblique_res_a1.0` dominates (`10/150` Clean-EOS vs `bawsc32_res_a0.45` `3/150`, `p = 0.0391`).
- **At 4-bit `Q4_0` (`83.3%` baseline loops):**
  - Diffuse `oblique_res_a1.0` (`cos = 0.544`, `col = 0.261`) suffers heavy PPL distortion (`ΔPPL = +0.490`) and **collapses to `7/150` Clean-EOS** (`9/150` STD, `p = 0.774` vs baseline `9/150`; paired `Q4_0 - Q8_0` drop relative to baseline = **`-11/150`, exact permutation `p = 0.01145`**).
  - Theorem 1 block-aligned **`bawsc32_res_a0.45`** (`cos = 0.9961`, `col = 0.00013`) preserves its circuit cleanly (`ΔPPL = +0.065` — $7.5\times$ smaller shift!) and **jumps to #1 overall at 4-bit: `15/150 (10.0%)` Clean-EOS / `17/150 (11.3%)` STD** (beating `uncond 6/150` at `p = 0.0117` EOS / `p = 0.0010` STD, `random 7/150` at `p = 0.0213` STD, and `oblique 7/150` by `+8/150`, with `Copy-90 = 69/90`).
- **Formal Paired Interaction Test (`heldout_stats_audit.json`, 100,000 exact sign permutations over the 150 identical held-out problems):**
  - **Clean-EOS Double Crossover `(BA-WSC - Oblique)_(Q4_0 - Q8_0)`:** **`+15/150 (+10.00 pp)` swing, `p = 0.00336`!**
  - **STD Accuracy Double Crossover:** **`+12/150 (+8.00 pp)` swing, `p = 0.01067`!**

---

## 5. Honest Limitations, Negative Results & Boundary Conditions

A credible ML researcher will immediately ask where our methods fail or have tradeoffs. Every negative result is logged and reported transparently:

1. **Precision-regime boundary between `oblique_res` and `BA-WSC`:** At high precision (`Q8_0`), `oblique_res` is stronger than `BA-WSC` on `Qwen3-0.6B` (`40.7%` vs `38.0%`, where `bawsc32_a0.25` reaches significance on `</think>` close `p=0.020`, limit `p=0.025`, token length `p=0.0071`, and zero-shot `ARC +11pp p=0.0074`, but `cl_p = 0.255` on pooled MATH EOS) and on `Qwen3.5-0.8B` (`10` vs `3/150`, `p=0.039`). Conversely, when strong edits meet 4-bit/3-bit quantization (`Qwen3.5 Q4_0` or `Qwen3 Q3_K_M`), block-aligned `BA-WSC` / `PQ-BA-WSC` wins decisively.
2. **High-dose echo sensitivity:** Over-driving `BA-WSC` without post-`</think>` gating (`a=0.70` on `Qwen3-0.6B`) causes post-answer repetition loops (`32.0%` Clean-EOS), proving strict Clean-EOS validation is mandatory during dev dose selection.
3. **Permutation quantization variance (`ba_wsc_noop`):** While SwiGLU permutation $P$ is $0.000000$-diff in floating point, re-binning neighbor neurons into shifted 32-element quantization blocks introduces small non-zero quantization noise (`Q4_0`: `32/150` vs `35/150`, `p = 0.607`; `Q3_K_M`: `20/150` vs `19/150`, `p = 1.000`). Including `ba_wsc_noop` as a mandatory control isolates true circuit gains from permutation noise.
4. **Neuron budget tradeoff of Corollary 1 ($k=256$):** Evicting 256 dormant neurons uses `8.3%` of `d_ff=3072` in 0.6B (where $k=32$ is preferable for behavioral compilation if minimal eviction is desired), versus only **`2.86%` of `d_ff=8960` in 1.5B** — meaning Corollary 1 becomes progressively cheaper at larger model scales.
5. **Model scale:** All experiments run on 0.6B–1.5B reasoning models due to the 4 GB VRAM hardware envelope (scaling the closed-form linear algebra to 7B–32B is $O(d_{\text{ff}} d_{\text{model}})$ per layer and requires zero backprop, left as future work). Additionally, at 3-bit `Q3_K_M` on 1.5B `DeepSeek-R1`, `uncond_a0.8` unexpectedly reached `59/150` EOS (`p = 0.071` vs base) despite `cos = 0.236` directional distortion—reported transparently in Paper 2 Section 8.

---

## 6. Two Complete Verified Papers & Reproducibility Index

- **Paper 1:** [papers/paper1_conditional_wsp_and_greedy_illusion.md](../papers/paper1_conditional_wsp_and_greedy_illusion.md) — *The Greedy-Perturbation Illusion and Zero-Gradient Control-Flow Compilation in Weight Space* (4 embedded 300-DPI figures).
- **Paper 2:** [papers/paper2_quantization_cliff_and_ba_wsc.md](../papers/paper2_quantization_cliff_and_ba_wsc.md) — *The Quantization Cliff of Weight-Space Programming: Hardware-Aligned Constant-Block Embedding and Post-Quantization Healing* (4 embedded 300-DPI figures).
- **Programmatic Paper Verification:** `scripts/wsp_verify_papers.py` (verifies every number, statistical test, figure, and `0`-error log integrity across all 3 models).

### Key Code & Immutable Evidence Paths
- Core library: `src/wsp/compiler.py`, `src/wsp/gguf_patch.py`, `src/wsp/gguf_to_torch.py`, `src/wsp/pq_loader.py`, `src/wsp/llama_eval.py`, `src/wsp/math_grader.py`
- Model 1 (`Qwen3-0.6B`) logs: `research/wsp/qwen3_0.6b/{layer_profile,theorem1_quant_audit,corollary1_superblock_audit,dev_results,test_primary_math150,stochastic_pooled_stats,ood_gsm8k_arc_results,quant_matrix_stoch_results}.json`
- Model 2 (`Qwen3.5-0.8B`) logs: `research/wsp/qwen35_0.8b/{layer_profile,theorem1_quant_audit,corollary1_superblock_audit,dev_dose_selection,test_heldout_results,heldout_stats_audit}.json`
- Model 3 (`DeepSeek-R1-1.5B`) logs: `research/wsp/deepseek_r1_1.5b/{baseline_regraded,layer_profile,theorem1_quant_audit,corollary1_superblock_audit,dev_dose_selection}.json`
- Complete chronological research journal: `research/JOURNAL.md`
- Visual summaries: `research/wsp/figures/wsp_simple_infographic.png`, `research/wsp/figures/wsp_research_infographic.png`
