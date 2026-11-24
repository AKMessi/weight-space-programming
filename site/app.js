/* All numbers below match immutable JSON audits in research/wsp/ and scripts/wsp_verify_papers.py */

const HERO_DATA = {
  qwen3: {
    high: {
      desc: "Pure GQA Transformer (28 layers, d_ff = 3,072). Multi-seed stochastic MATH-500 (Q8_0, temp = 0.6, N = 300 rollouts).",
      verdict: "Oblique surgery ranks first (+5.67 pp, clustered p = 0.0148 vs baseline, p = 0.0146 vs random noise).",
      maxPct: 48,
      bars: [
        { name: "Unedited baseline", pct: 35.0, detail: "105/300 · ΔPPL 0.000", cls: "is-base" },
        { name: "Matched random noise", pct: 34.7, detail: "104/300 · p = 1.000", cls: "is-random" },
        { name: "Prior unconditional WSP", pct: 38.7, detail: "116/300 · ΔPPL +1.338", cls: "is-uncond" },
        { name: "BA-WSC (Ours, 0 PPL)", pct: 38.0, detail: "114/300 · ΔPPL -0.018", cls: "is-bawsc", nameCls: "is-green" },
        { name: "Oblique surgery (Ours, #1)", pct: 40.7, detail: "122/300 · p = 0.0148", cls: "is-oblique", nameCls: "is-winner" },
      ],
    },
    low: {
      desc: "Pure GQA Transformer at 3-bit Q3_K_M (8.2x baseline loop explosion). Held-out MATH-500 (temp = 0.6, N = 150).",
      verdict: "Quantization-native PQ-BA-WSC ranks first overall (+8.7 pp / +68% rel., p = 0.0044, 19x less PPL loss than prior WSP).",
      maxPct: 30,
      bars: [
        { name: "Q3_K_M baseline (74 loops)", pct: 12.7, detail: "19/150 · ΔPPL 0.000", cls: "is-base" },
        { name: "Permutation no-op control", pct: 13.3, detail: "20/150 · p = 1.000", cls: "is-uncond" },
        { name: "Matched random noise", pct: 14.0, detail: "21/150 · p = 0.815", cls: "is-random" },
        { name: "Prior unconditional WSP", pct: 18.0, detail: "27/150 · ΔPPL +2.202", cls: "is-uncond" },
        { name: "FP Oblique transferred", pct: 19.3, detail: "29/150 · ΔPPL +0.394", cls: "is-oblique" },
        { name: "PQ-BA-WSC heal (Ours, #1)", pct: 21.3, detail: "32/150 · p = 0.0044", cls: "is-bawsc", nameCls: "is-green" },
      ],
    },
  },
  qwen35: {
    high: {
      desc: "Hybrid 18 Gated-DeltaNet + 6 Full-Attention (24 layers, d_ff = 3,584). Held-out MATH-500 (Q8_0, temp = 0.6, N = 150).",
      verdict: "Oblique surgery delivers a 10x Clean-EOS jump (1 to 10/150, p = 0.0039 vs baseline, p = 0.0391 vs random); prior WSP gets 0/150.",
      maxPct: 12,
      bars: [
        { name: "Unedited baseline", pct: 0.7, detail: "1/150 · 149 timeouts", cls: "is-base" },
        { name: "Matched random noise", pct: 2.0, detail: "3/150 · p = 0.500", cls: "is-random" },
        { name: "Prior unconditional WSP", pct: 0.0, detail: "0/150 · 150 timeouts", cls: "is-uncond" },
        { name: "BA-WSC (Ours, 0 PPL)", pct: 2.0, detail: "3/150 · ΔPPL +0.028", cls: "is-bawsc", nameCls: "is-green" },
        { name: "Oblique surgery (Ours, #1)", pct: 6.7, detail: "10/150 · p = 0.0039", cls: "is-oblique", nameCls: "is-winner" },
      ],
    },
    low: {
      desc: "Hybrid Qwen3.5-0.8B compressed to 4-bit Q4_0 (83.3% baseline loops). Held-out MATH-500 (temp = 0.6, N = 150).",
      verdict: "Exact behavioral double crossover: diffuse oblique drops below baseline (7/150, ΔPPL +0.490) while block-aligned BA-WSC surges to #1 (15/150, crossover p = 0.00336).",
      maxPct: 14,
      bars: [
        { name: "Q4_0 baseline (83% loops)", pct: 6.0, detail: "9/150 · ΔPPL 0.000", cls: "is-base" },
        { name: "Matched random noise", pct: 4.7, detail: "7/150 · ΔPPL +0.205", cls: "is-random" },
        { name: "Prior unconditional WSP", pct: 4.0, detail: "6/150 · ΔPPL -0.037", cls: "is-uncond" },
        { name: "Diffuse Oblique (breaks)", pct: 4.7, detail: "7/150 · ΔPPL +0.490", cls: "is-oblique" },
        { name: "Block-aligned BA-WSC (#1)", pct: 10.0, detail: "15/150 · ΔPPL +0.065", cls: "is-bawsc", nameCls: "is-green" },
      ],
    },
  },
  deepseek: {
    high: {
      desc: "RL-distilled Qwen2 (28 layers, d_ff = 8,960, 2.5x larger). Held-out MATH-500 (Q8_0, temp = 0.6, N = 150).",
      verdict: "Oblique surgery ranks #1 (+6.7 pp vs base, +9.3 pp vs random p = 0.0288, 20% faster); BA-WSC holds ΔPPL +0.003 and 22% faster wall time.",
      maxPct: 44,
      bars: [
        { name: "Unedited baseline", pct: 28.7, detail: "43/150 · 4,403s wall", cls: "is-base" },
        { name: "Matched random noise", pct: 26.0, detail: "39/150 · -2.7 pp", cls: "is-random" },
        { name: "Prior unconditional WSP", pct: 32.0, detail: "48/150 · ΔPPL +0.278", cls: "is-uncond" },
        { name: "BA-WSC (Ours, 0 PPL)", pct: 32.7, detail: "49/150 · ΔPPL +0.003", cls: "is-bawsc", nameCls: "is-green" },
        { name: "Oblique surgery (Ours, #1)", pct: 35.3, detail: "53/150 · p = 0.0288 vs R", cls: "is-oblique", nameCls: "is-winner" },
      ],
    },
    low: {
      desc: "DeepSeek-R1-1.5B at its 3-bit Q3_K_M cliff (66% baseline timeouts). Held-out MATH-500 (temp = 0.6, N = 150).",
      verdict: "Block-aligned BA-WSC holds an invariant +6/+6/+6 gain across 8-bit, 4-bit, and 3-bit while diffuse oblique (cos = 0.406) drops below baseline (-3).",
      maxPct: 46,
      bars: [
        { name: "Q3_K_M baseline (66% lim)", pct: 32.0, detail: "48/150 · ΔPPL 0.000", cls: "is-base" },
        { name: "Matched random noise", pct: 34.0, detail: "51/150 · ΔPPL +0.040", cls: "is-random" },
        { name: "Diffuse Oblique (breaks)", pct: 30.0, detail: "45/150 · ΔPPL +0.242", cls: "is-oblique" },
        { name: "Block-aligned BA-WSC", pct: 36.0, detail: "54/150 · ΔPPL +0.051", cls: "is-bawsc", nameCls: "is-green" },
      ],
    },
  },
};

const TRAP_DATA = {
  greedy: {
    note: "On greedy MATH-500 (temp = 0.0), six zero-information draws span a 6.0 pp chaotic band because 14 of 14 rescued problems were baseline 2048-token timeouts getting randomly rerolled. Under official sampling (temp = 0.6, N = 300), random noise falls to -0.33 pp (p = 1.000).",
    pairs: [
      {
        name: "Pure Gaussian noise (random_matched)",
        delta: "+6.0 pp fake greedy -> -0.33 pp real (p = 1.000)",
        deltaCls: "is-bad",
        aLabel: "Greedy t=0.0",
        aVal: 40.7,
        aColor: "#EF4444",
        bLabel: "Stoch t=0.6",
        bVal: 34.7,
        bColor: "#94A3B8",
      },
      {
        name: "Unedited baseline",
        delta: "34.7% greedy vs 35.0% multi-seed stochastic",
        deltaCls: "is-neutral",
        aLabel: "Greedy t=0.0",
        aVal: 34.7,
        aColor: "#64748B",
        bLabel: "Stoch t=0.6",
        bVal: 35.0,
        bColor: "#475569",
      },
      {
        name: "Oblique surgery (oblique_res, Ours)",
        delta: "Holds real +5.67 pp stochastic gain (cl_p = 0.0148)",
        deltaCls: "is-good",
        aLabel: "Greedy t=0.0",
        aVal: 41.3,
        aColor: "#3B82F6",
        bLabel: "Stoch t=0.6",
        bVal: 40.7,
        bColor: "#1446C8",
      },
    ],
  },
  echo: {
    note: "Standard boxed-answer extraction hides severe post-answer echoing. High-dose BA-WSC (alpha = 0.70) doubled repetition loops (29 to 59, p = 0.0001) and fell to 32.0% under Strict Clean-EOS. Reducing the dose to alpha = 0.25 restored 100% Clean-EOS alignment (65/65).",
    pairs: [
      {
        name: "High-dose BA-WSC (alpha = 0.70, Echo Trap)",
        delta: "44.0% standard -> 32.0% Clean-EOS (59 loops, p = 0.011 vs R)",
        deltaCls: "is-bad",
        aLabel: "Boxed grade",
        aVal: 44.0,
        aColor: "#F59E0B",
        bLabel: "Clean-EOS",
        bVal: 32.0,
        bColor: "#C81E1E",
      },
      {
        name: "Clean-dose BA-WSC (alpha = 0.25, Ours)",
        delta: "100% Clean-EOS alignment (22 loops, ΔPPL -0.022)",
        deltaCls: "is-good",
        aLabel: "Boxed grade",
        aVal: 43.3,
        aColor: "#10B981",
        bLabel: "Clean-EOS",
        bVal: 43.3,
        bColor: "#047857",
      },
      {
        name: "Unedited BF16 baseline",
        delta: "35.3% standard vs 34.7% Clean-EOS (29 loops)",
        deltaCls: "is-neutral",
        aLabel: "Boxed grade",
        aVal: 35.3,
        aColor: "#64748B",
        bLabel: "Clean-EOS",
        bVal: 34.7,
        bColor: "#475569",
      },
    ],
  },
};

const ANATOMY_DATA = {
  qwen3: {
    stat: "Compiled layers [18, 20, 22] · Virtual bias std = 0.048 to 0.074 (OOS)",
    caption: "Qwen3-0.6B (28L Pure GQA + QK-Norm): flat through L0 to L15 (<= 0.65), surging across L16 to L22 (1.12 to 6.98; readout L27 = 65.87 excluded for scale).",
    vals: [0.2268, 0.3142, 0.1247, 0.2514, 0.1525, 0.1537, 0.2516, 0.2528, 0.3371, 0.4612, 0.6477, 0.3765, 0.28, 0.0383, 0.16, 0.1375, 1.1238, 1.2717, 1.5161, 2.2873, 3.2877, 4.6985, 6.9774, 3.8357, 2.9672, -0.1792, 4.6017],
    targets: [18, 20, 22],
    special: [],
  },
  qwen35: {
    stat: "Compiled layers [15, 19, 21] (Full-Attn L15 & L19!) · Virtual bias std = 0.021 to 0.025",
    caption: "Qwen3.5-0.8B (24L Hybrid: 18 Gated-DeltaNet + 6 Full-Attention): flat across L0 to L13, surging 10x directly at Full-Attention L15 (0.1723) and L19 (0.1372). Amber bars mark earlier Full-Attention layers.",
    vals: [0.064, 0.0398, 0.0232, 0.0135, 0.0228, 0.0211, 0.0232, -0.0033, 0.0044, 0.0115, 0.0046, 0.0042, 0.0146, 0.0172, 0.0923, 0.1723, 0.1129, 0.121, 0.0706, 0.1372, 0.1271, 0.167, 0.1474],
    targets: [15, 19, 21],
    special: [3, 7, 11],
  },
  deepseek: {
    stat: "Compiled layers [20, 22, 25] (35x late surge) · Virtual bias std = 0.020 to 0.025",
    caption: "DeepSeek-R1-Distill-Qwen-1.5B (28L Qwen2 RL, d_ff = 8,960): flat through L0 to L18 (L18 = 0.038), then surging 35x across L19 (1.316) through L26 (7.776; readout L27 = 40.13 excluded).",
    vals: [1.087, 0.2639, 0.5071, 0.1812, 0.5607, -0.3178, 0.4206, 0.1842, 0.1319, 0.3691, 0.1685, -0.2328, 0.0082, -0.0303, 0.1058, 0.4252, 0.3488, 0.012, 0.038, 1.3161, 1.4808, 2.4666, 2.6715, 1.9446, 2.853, 5.5279, 7.7758],
    targets: [20, 22, 25],
    special: [],
  },
};

const LAB_AUDIT = {
  qwen3: {
    dff: 3072,
    scattered: { cos: "0.379", col: "2.25724", drop: "1x (sparse64 ref)", budget: "64 / 3,072 (2.08%)", good: false, explain: "In Q3_K_M on Qwen3-0.6B L20, scattered edits contaminate shared super-block scales so neighbor collateral is 2.26x larger than the intended edit signal." },
    thm1: { cos: "0.9959", col: "0.00063", drop: "1,004x lower (Q4_0)", budget: "32 / 3,072 (1.04%)", good: true, explain: "Permuting 32 dormant neurons into Block 0 isolates Q4_0 completely: exact extreme integer codes (+/-7), zero rounding error, and 1,004x lower collateral." },
    cor1: { cos: "0.99732", col: "0.00050", drop: "4,514x lower (Q3_K_M)", budget: "256 / 3,072 (8.33%)", good: true, explain: "Packing 256 dormant neurons fills an entire K-quant super-block, eliminating shared super-block scale drift and cutting Q3_K_M collateral by 4,514x at zero byte cost." },
  },
  qwen35: {
    dff: 3584,
    scattered: { cos: "0.293", col: "3.03570", drop: "1x (sparse64 ref)", budget: "64 / 3,584 (1.79%)", good: false, explain: "On Hybrid Qwen3.5-0.8B L19, sparse64 loses over 70% of directional fidelity under K-quantization and injects 3.04x more collateral noise than signal." },
    thm1: { cos: "0.9961", col: "0.00013", drop: "6,417x lower (Q4_0)", budget: "32 / 3,584 (0.89%)", good: true, explain: "Sub-block alignment on Q4_0 drives neighbor collateral down to 0.00013 (6,417x reduction) while preserving 0.9961 cosine fidelity." },
    cor1: { cos: "0.99739", col: "0.00030", drop: "16,685x (Q4) / 10,119x (Q3)", budget: "256 / 3,584 (7.14%)", good: true, explain: "Super-block isolation hits 0.00005 collateral on Q4_0 (16,685x drop) and 0.00030 on Q3_K_M (10,119x drop) with 0.000000 floating-point discrepancy." },
  },
  deepseek: {
    dff: 8960,
    scattered: { cos: "0.226", col: "4.12538", drop: "1x (sparse64 ref)", budget: "64 / 8,960 (0.71%)", good: false, explain: "On DeepSeek-R1-1.5B L20 at 3-bit Q3_K_M, sparse64 collapses to 0.226 cosine and triggers 4.13x neighbor collateral distortion." },
    thm1: { cos: "0.9961", col: "0.00115", drop: "732x lower (Q4_0)", budget: "32 / 8,960 (0.36%)", good: true, explain: "Just 32 dormant neurons (0.36% of FFN width) isolate Q4_0 completely at 0.9961 cosine with zero extra storage or latency." },
    cor1: { cos: "0.99741", col: "0.00034", drop: "12,133x lower (Q3_K_M)", budget: "256 / 8,960 (2.86%)", good: true, explain: "Wider MLPs make Corollary 1 almost free: 256 dormant neurons consume just 2.86% of DeepSeek-R1-1.5B's 8,960 FFN width while slashing 3-bit collateral 12,133x." },
  },
};

const CROSS_DATA = {
  qwen35: {
    caption: "Qwen3.5-0.8B Hybrid: diffuse oblique drops at 4-bit (10 to 7/150, ΔPPL +0.490) while block-aligned BA-WSC surges to #1 (3 to 15/150). Exact 100k-permutation crossover: +15 items (+10.00 pp), p = 0.00336.",
    xLabels: ["8-bit Q8_0", "4-bit Q4_0"],
    yLabel: "Clean-EOS solved (/150)",
    yMin: 0,
    yMax: 17,
    series: [
      { label: "Block-aligned BA-WSC (cos = 0.996)", color: "#10B981", pts: [3, 15], dash: false },
      { label: "Diffuse Oblique (cos = 0.544)", color: "#60A5FA", pts: [10, 7], dash: false },
      { label: "Quantized baseline", color: "#94A3B8", pts: [1, 9], dash: true },
      { label: "Matched random noise", color: "#F87171", pts: [3, 7], dash: true },
    ],
  },
  deepseek: {
    caption: "DeepSeek-R1-1.5B: block-aligned BA-WSC holds a flat +6, +6, +6 gain across 8-bit, 4-bit, and 3-bit while diffuse oblique slides +10 to +4 to -3 (+13 item swing, p = 0.0663; 2-model Fisher p = 0.00210).",
    xLabels: ["8-bit Q8_0", "4-bit Q4_0", "3-bit Q3_K_M"],
    yLabel: "Delta Clean-EOS vs baseline (/150)",
    yMin: -8,
    yMax: 12,
    zeroLine: true,
    series: [
      { label: "Block-aligned BA-WSC (Q3 cos = 0.996)", color: "#10B981", pts: [6, 6, 6], dash: false },
      { label: "Diffuse Oblique (Q3 cos = 0.406)", color: "#60A5FA", pts: [10, 4, -3], dash: false },
      { label: "Matched random noise", color: "#F87171", pts: [-4, -6, 3], dash: true },
    ],
  },
};

const LEDGER_TABLES = {
  p1: {
    headers: [
      { text: "Model (Q8_0, temp = 0.6)" },
      { text: "Benchmark (N)" },
      { text: "Baseline", num: true },
      { text: "Random noise", num: true },
      { text: "BA-WSC (0 PPL)", num: true },
      { text: "Oblique surgery", num: true },
      { text: "Exact significance" },
    ],
    rows: [
      { cells: ["Qwen3-0.6B (Pure GQA)", "MATH-500 (300)", "35.0%", "34.7% (-0.3 pp)", "38.0% (-0.018 PPL)", "<strong>40.7% (+5.7 pp)</strong>", "cl_p = 0.0148 vs Base, 0.0146 vs Rand"] },
      { cells: ["Qwen3-0.6B (OOD Math)", "GSM8K-100 (100)", "71.0%", "64.0% (-7.0 pp)", "75.0% (p = 0.027 vs R)", "<strong>80.0% (+9.0 pp)</strong>", "p = 0.049 vs Base, 0.0001 vs Rand (-30% wall)"] },
      { cells: ["Qwen3-0.6B (Zero-shot Sci)", "ARC-100 (100)", "56.0%", "62.0% (n.s.)", "<strong>67.0% (+11.0 pp)</strong>", "<strong>71.0% (+15.0 pp)</strong>", "p = 0.0003 (Oblique), p = 0.0074 (BA-WSC)"] },
      { cells: ["Qwen3.5-0.8B (Hybrid)", "MATH-500 (150)", "0.7% (1)", "2.0% (3)", "2.0% (+0.028 PPL)", "<strong>6.7% (10, 10x)</strong>", "p = 0.0039 vs Base, 0.0391 vs Rand"] },
      { cells: ["DeepSeek-R1-1.5B (Qwen2)", "MATH-500 (150)", "28.7% (43)", "26.0% (39)", "32.7% (+0.003 PPL)", "<strong>35.3% (53, +6.7 pp)</strong>", "p = 0.0288 vs Rand (-20% wall)"] },
      { cells: ["<strong>Pooled 3-architecture</strong>", "<strong>MATH-500 (600)</strong>", "24.8%", "24.3%", "27.7% (p = 0.029 vs R)", "<strong>30.8% (+6.00 pp)</strong>", "<strong>+61/-25 (p = 0.000130), +65/-26 (p = 5.3e-5)</strong>"], cls: "row-win" },
    ],
  },
  p2heal: {
    headers: [
      { text: "Precision & arm (Qwen3-0.6B, N = 150)" },
      { text: "PPL (delta)", num: true },
      { text: "Clean-EOS", num: true },
      { text: "p vs quant base", num: true },
      { text: "Timeouts", num: true },
      { text: "Loops", num: true },
      { text: "Copy-90", num: true },
    ],
    rows: [
      { cells: ["Q4_0 baseline (4-bit cliff: -12.0 pp)", "21.866 (0.000)", "35 (23.3%)", "ref", "114", "28", "65/90"] },
      { cells: ["Q4_0 permutation no-op (ba_wsc_noop)", "21.870 (+0.004)", "32 (21.3%)", "0.607", "116", "25", "64/90"] },
      { cells: ["Q4_0 matched random noise", "21.785 (-0.081)", "32 (21.3%)", "0.664", "118", "23", "66/90"] },
      { cells: ["Q4_0 prior unconditional WSP", "22.805 (+0.939)", "38 (25.3%)", "0.581", "110", "25", "65/90"], cls: "row-warn" },
      { cells: ["Q4_0 PQ-BA-WSC heal (quant-native)", "<strong>21.869 (+0.003)</strong>", "<strong>46 (30.7%, +7.3 pp)</strong>", "<strong>0.0347</strong>", "100", "36", "64/90"], cls: "row-win" },
      { cells: ["Q4_0 Oblique surgery (#1, 83% cliff healed)", "22.034 (+0.168)", "<strong>50 (33.3%, +10.0 pp)</strong>", "<strong>0.0041</strong>", "<strong>98</strong>", "41", "64/90"], cls: "row-win" },
      { cells: ["Q3_K_M baseline (3-bit cliff: -22.7 pp)", "26.723 (0.000)", "19 (12.7%)", "ref", "115", "74", "39/90"] },
      { cells: ["Q3_K_M prior unconditional WSP", "<strong>28.924 (+2.202)</strong>", "27 (18.0%)", "0.096", "93", "68", "41/90"], cls: "row-warn" },
      { cells: ["Q3_K_M PQ-Oblique heal (-28% loops)", "26.765 (+0.043)", "30 (20.0%, +7.3 pp)", "0.0266", "85", "<strong>53</strong>", "39/90"] },
      { cells: ["Q3_K_M PQ-BA-WSC heal a = 0.35 (#1 at 3-bit)", "<strong>26.839 (+0.116)</strong>", "<strong>32 (21.3%, +8.7 pp)</strong>", "<strong>0.0044</strong>", "<strong>82</strong>", "60", "40/90"], cls: "row-win" },
    ],
  },
  t1: {
    headers: [
      { text: "Model (target layer, d_ff)" },
      { text: "Method" },
      { text: "Q4_0 cos / col", num: true },
      { text: "Q4_K_M cos / col", num: true },
      { text: "Q3_K_M cos / col", num: true },
      { text: "Peak collateral drop", num: true },
    ],
    rows: [
      { cells: ["Qwen3-0.6B (L20, 3,072)", "sparse64 (Prior WSP)", "0.640 / 0.63272", "0.916 / 0.35082", "0.379 / 2.25724", "1x ref"] },
      { cells: ["Qwen3-0.6B (L20, 3,072)", "BA-WSC k = 32 (Theorem 1)", "0.9959 / 0.00063", "0.9997 / 0.01453", "0.9942 / 0.07877", "1,004x (Q4_0)"] },
      { cells: ["Qwen3-0.6B (L20, 3,072)", "BA-WSC k = 256 (Corollary 1)", "<strong>0.9961 / 0.00031</strong>", "<strong>0.99983 / 0.00011</strong>", "<strong>0.99732 / 0.00050</strong>", "<strong>4,514x (Q3_K_M)</strong>"], cls: "row-win" },
      { cells: ["Qwen3.5-0.8B (L19, 3,584)", "sparse64 (Prior WSP)", "0.548 / 0.83424", "0.293 / 3.03570", "0.293 / 3.03570", "1x ref"] },
      { cells: ["Qwen3.5-0.8B (L19, 3,584)", "BA-WSC k = 32 (Theorem 1)", "0.9961 / 0.00013", "0.9955 / 0.05881", "0.9955 / 0.05881", "6,417x (Q4_0)"] },
      { cells: ["Qwen3.5-0.8B (L19, 3,584)", "BA-WSC k = 256 (Corollary 1)", "<strong>0.9962 / 0.00005</strong>", "<strong>0.99739 / 0.00030</strong>", "<strong>0.99739 / 0.00030</strong>", "<strong>16,685x (Q4_0)</strong>"], cls: "row-win" },
      { cells: ["DeepSeek-R1-1.5B (L20, 8,960)", "sparse64 (Prior WSP)", "0.511 / 0.84142", "0.798 / 0.62476", "0.226 / 4.12538", "1x ref"] },
      { cells: ["DeepSeek-R1-1.5B (L20, 8,960)", "BA-WSC k = 32 (Theorem 1)", "0.9961 / 0.00115", "0.99979 / 0.00834", "0.99618 / 0.04540", "732x (Q4_0)"] },
      { cells: ["DeepSeek-R1-1.5B (L20, 8,960)", "BA-WSC k = 256 (Corollary 1)", "<strong>0.99625 / 0.00014</strong>", "<strong>0.99984 / 0.00007</strong>", "<strong>0.99741 / 0.00034</strong>", "<strong>12,133x (2.86% FFN)</strong>"], cls: "row-win" },
    ],
  },
};

const GALLERY_FIGURES = [
  { src: "./assets/wsp_simple_infographic.png", title: "Plain-English visual summary across all three architectures" },
  { src: "./papers/figures/p1_fig1_greedy_illusion_and_stochastic.png", title: "Paper 1 Fig. 1: Greedy-Perturbation Illusion & Echo Trap" },
  { src: "./papers/figures/p1_fig2_universal_layer_anatomy.png", title: "Paper 1 Fig. 2: Universal late-MLP surge & Virtual Bias b*" },
  { src: "./papers/figures/p1_fig3_ood_and_cross_arch.png", title: "Paper 1 Fig. 3: OOD GSM8K, zero-shot ARC & 3-model Q8_0" },
  { src: "./papers/figures/p1_fig4_pareto_ppl_vs_acc.png", title: "Paper 1 Fig. 4: Accuracy vs. WikiText-2 PPL Pareto frontier" },
  { src: "./papers/figures/p2_fig1_quant_loop_cliff_3models.png", title: "Paper 2 Fig. 1: Scale-dependent quantization loop cliff" },
  { src: "./papers/figures/p2_fig2_theorem1_corollary1_3models.png", title: "Paper 2 Fig. 2: Real llama-quantize proof of Theorem 1 & Cor. 1" },
  { src: "./papers/figures/p2_fig3_pq_healing_matrix_qwen3.png", title: "Paper 2 Fig. 3: 20-arm Post-Quantization Healing matrix" },
  { src: "./papers/figures/p2_fig4_double_crossover_2models.png", title: "Paper 2 Fig. 4: Replicated Precision x Method double crossovers" },
];

// State
let heroModel = "qwen3";
let heroPrec = "high";
let labMode = "scattered";
let labModel = "qwen3";

function setActiveTab(container, attr, value) {
  container.querySelectorAll(".seg-btn").forEach((btn) => {
    const active = btn.dataset[attr] === value;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", String(active));
  });
}

/* ---------- 1. Hero Workbench ---------- */
function renderHero() {
  const entry = HERO_DATA[heroModel][heroPrec];
  document.getElementById("wb-arch-desc").textContent = entry.desc;
  document.getElementById("wb-verdict").textContent = entry.verdict;
  const wrap = document.getElementById("wb-bars");
  wrap.innerHTML = entry.bars
    .map((b) => {
      const width = Math.max(1.2, Math.min(100, (b.pct / entry.maxPct) * 100));
      return `
        <div class="bar-row">
          <div class="bar-name ${b.nameCls || ""}">${b.name}</div>
          <div class="bar-track"><div class="bar-fill ${b.cls}" style="width:${width.toFixed(1)}%"></div></div>
          <div class="bar-metric"><strong>${b.pct.toFixed(1)}%</strong> · ${b.detail}</div>
        </div>
      `;
    })
    .join("");
}

/* ---------- 2. Evaluation Traps Stage ---------- */
function renderTrap(mode) {
  const d = TRAP_DATA[mode];
  document.getElementById("trap-note").textContent = d.note;
  const stage = document.getElementById("trap-stage");
  stage.innerHTML = d.pairs
    .map((p) => {
      const wa = Math.min(100, (p.aVal / 50) * 100);
      const wb = Math.min(100, (p.bVal / 50) * 100);
      return `
        <div class="trap-pair">
          <div class="trap-pair-head">
            <span class="trap-pair-name">${p.name}</span>
            <span class="trap-pair-delta ${p.deltaCls}">${p.delta}</span>
          </div>
          <div class="dual-track">
            <span>${p.aLabel}</span>
            <div class="dual-bar-bg"><div class="dual-bar-fill" style="width:${wa.toFixed(1)}%;background-color:${p.aColor}"></div></div>
            <strong style="text-align:right">${p.aVal.toFixed(1)}%</strong>
          </div>
          <div class="dual-track">
            <span>${p.bLabel}</span>
            <div class="dual-bar-bg"><div class="dual-bar-fill" style="width:${wb.toFixed(1)}%;background-color:${p.bColor}"></div></div>
            <strong style="text-align:right">${p.bVal.toFixed(1)}%</strong>
          </div>
        </div>
      `;
    })
    .join("");
}

/* ---------- 3. Universal Layer Anatomy SVG ---------- */
function renderAnatomy(key) {
  const d = ANATOMY_DATA[key];
  document.getElementById("anatomy-stat").textContent = d.stat;
  document.getElementById("anatomy-caption").textContent = d.caption;

  const svg = document.getElementById("anatomy-svg");
  const W = 960;
  const H = 250;
  const padL = 52;
  const padR = 18;
  const padT = 22;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxV = Math.max(...d.vals) * 1.1;
  const minV = Math.min(0, ...d.vals) * 1.15;
  const span = maxV - minV;
  const yZero = padT + (maxV / span) * plotH;

  const n = d.vals.length;
  const step = plotW / n;
  const barW = Math.max(10, step * 0.68);

  let out = "";
  // Horizontal grid lines
  [0, 0.5, 1].forEach((f) => {
    const val = maxV * f;
    const y = padT + ((maxV - val) / span) * plotH;
    out += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="${f === 0 ? "#94A3B8" : "#E2E8F0"}" stroke-width="${f === 0 ? 1.2 : 1}"/>`;
    out += `<text x="${padL - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#475569" font-size="11.5" font-family="Inter, sans-serif">${val.toFixed(maxV < 1 ? 2 : 1)}</text>`;
  });

  d.vals.forEach((v, i) => {
    const x = padL + i * step + (step - barW) / 2;
    const yVal = padT + ((maxV - v) / span) * plotH;
    const y = Math.min(yVal, yZero);
    const h = Math.max(1.5, Math.abs(yVal - yZero));
    let fill = "#94A3B8";
    if (d.targets.includes(i)) fill = "#1446C8";
    else if (d.special.includes(i)) fill = "#D97706";
    else if (i >= n - 9) fill = "#64748B";

    out += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${fill}"><title>Layer ${i}: MLPProj = ${v.toFixed(4)}</title></rect>`;
    if (i % 2 === 0 || d.targets.includes(i)) {
      out += `<text x="${(x + barW / 2).toFixed(1)}" y="${H - 12}" text-anchor="middle" fill="${d.targets.includes(i) ? "#1446C8" : "#475569"}" font-weight="${d.targets.includes(i) ? "700" : "400"}" font-size="11" font-family="Inter, sans-serif">L${i}</text>`;
    }
  });

  svg.innerHTML = out;
}

/* ---------- 4. Theorem 1 & Corollary 1 Hardware Block Lab ---------- */
const SCATTERED_SLOTS = {
  0: [4, 19],
  1: [11],
  2: [2, 24],
  3: [15],
  4: [7, 29],
  5: [13],
  6: [6, 21],
  7: [27],
};

function renderBlockLab() {
  const grid = document.getElementById("superblock-grid");
  let html = "";
  for (let b = 0; b < 8; b++) {
    let cls = "";
    let status = "Pristine block";
    if (labMode === "scattered") {
      cls = "is-contaminated";
      status = "Mixed scale";
    } else if (labMode === "thm1") {
      if (b === 0) {
        cls = "is-isolated";
        status = "Dedicated edit block";
      } else {
        status = "Shared K-super-block only";
      }
    } else {
      cls = "is-isolated";
      status = "Full super-block (S = 0)";
    }

    let cells = "";
    for (let j = 0; j < 32; j++) {
      let cCls = "";
      if (labMode === "scattered") {
        cCls = (SCATTERED_SLOTS[b] || []).includes(j) ? "is-edit-bad" : "is-col";
      } else if (labMode === "thm1") {
        cCls = b === 0 ? "is-edit" : "";
      } else {
        cCls = "is-edit";
      }
      cells += `<span class="w-cell ${cCls}"></span>`;
    }

    html += `
      <div class="subblock ${cls}">
        <div class="subblock-title"><span>Block ${b} (32 w)</span><span>${status}</span></div>
        <div class="subblock-cells">${cells}</div>
      </div>
    `;
  }
  grid.innerHTML = html;

  const stats = LAB_AUDIT[labModel][labMode];
  const roCos = document.getElementById("ro-cos");
  const roCol = document.getElementById("ro-col");
  const roDrop = document.getElementById("ro-drop");
  roCos.textContent = stats.cos;
  roCol.textContent = stats.col;
  roDrop.textContent = stats.drop;
  document.getElementById("ro-budget").textContent = stats.budget;
  document.getElementById("ro-explain").textContent = stats.explain;

  roCos.className = `readout-value ${stats.good ? "is-good" : "is-bad"}`;
  roCol.className = `readout-value ${stats.good ? "is-good" : "is-bad"}`;
  roDrop.className = `readout-value ${stats.good ? "is-good" : "is-bad"}`;
}

/* ---------- 5. Double-Crossover SVG ---------- */
function renderCrossover(key) {
  const d = CROSS_DATA[key];
  document.getElementById("cross-caption").textContent = d.caption;
  const svg = document.getElementById("crossover-svg");
  const W = 540;
  const H = 280;
  const padL = 52;
  const padR = 26;
  const padT = 38;
  const padB = 38;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const nX = d.xLabels.length;

  const xPos = (idx) => (nX === 2 ? padL + (idx === 0 ? plotW * 0.16 : plotW * 0.84) : padL + (idx / (nX - 1)) * plotW * 0.82 + plotW * 0.09);
  const yPos = (val) => padT + ((d.yMax - val) / (d.yMax - d.yMin)) * plotH;

  let out = "";
  // Legend row at top
  d.series.forEach((s, idx) => {
    const lx = padL + (idx % 2) * 245;
    const ly = 12 + Math.floor(idx / 2) * 15;
    out += `<line x1="${lx}" y1="${ly}" x2="${lx + 18}" y2="${ly}" stroke="${s.color}" stroke-width="2.4" ${s.dash ? 'stroke-dasharray="4 3"' : ""}/>`;
    out += `<text x="${lx + 24}" y="${ly + 4}" fill="#CBD5E1" font-size="11" font-family="Inter, sans-serif">${s.label}</text>`;
  });

  // Grid & y ticks
  const ticks = d.zeroLine ? [-5, 0, 5, 10] : [0, 5, 10, 15];
  ticks.forEach((t) => {
    const y = yPos(t);
    out += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="${t === 0 ? "#64748B" : "#1E2E4E"}" stroke-width="${t === 0 ? 1.2 : 1}" ${t === 0 && d.zeroLine ? 'stroke-dasharray="5 4"' : ""}/>`;
    out += `<text x="${padL - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#94A3B8" font-size="11.5" font-family="Inter, sans-serif">${t > 0 && d.zeroLine ? "+" + t : t}</text>`;
  });

  // X labels
  d.xLabels.forEach((lbl, i) => {
    const x = xPos(i);
    out += `<text x="${x.toFixed(1)}" y="${H - 12}" text-anchor="middle" fill="#E2E8F0" font-size="12" font-weight="600" font-family="Inter, sans-serif">${lbl}</text>`;
  });

  // Lines & points
  d.series.forEach((s) => {
    const coords = s.pts.map((v, i) => `${xPos(i).toFixed(1)},${yPos(v).toFixed(1)}`).join(" ");
    out += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.dash ? 1.7 : 2.8}" ${s.dash ? 'stroke-dasharray="5 4"' : ""} points="${coords}"/>`;
    s.pts.forEach((v, i) => {
      const cx = xPos(i);
      const cy = yPos(v);
      out += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${s.dash ? 4 : 5.2}" fill="${s.color}" stroke="#0B1324" stroke-width="1.5"/>`;
      if (!s.dash) {
        let dy = -11;
        if (key === "deepseek" && s.color === "#60A5FA" && i >= 1) dy = 18;
        else if (key === "qwen35" && s.color === "#60A5FA" && i === 1) dy = 18;
        else if (v < 0) dy = 17;
        const labelVal = d.zeroLine && v > 0 ? `+${v}` : `${v}`;
        out += `<text x="${cx.toFixed(1)}" y="${(cy + dy).toFixed(1)}" text-anchor="middle" fill="${s.color}" font-size="11.5" font-weight="700" font-family="Inter, sans-serif">${labelVal}</text>`;
      }
    });
  });

  svg.innerHTML = out;
}

/* ---------- 6. Evidence Ledger Tables ---------- */
function renderLedger(key) {
  const t = LEDGER_TABLES[key];
  const thead = t.headers.map((h) => `<th class="${h.num ? "num" : ""}">${h.text}</th>`).join("");
  const tbody = t.rows
    .map((r) => {
      const tds = r.cells
        .map((c, i) => `<td class="${t.headers[i].num ? "num" : ""}">${c}</td>`)
        .join("");
      return `<tr class="${r.cls || ""}">${tds}</tr>`;
    })
    .join("");
  document.getElementById("ledger-table-wrap").innerHTML = `
    <table class="data-table">
      <thead><tr>${thead}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

/* ---------- 7. Figure Gallery & Lightbox ---------- */
function openLightbox(src, title) {
  const lb = document.getElementById("lightbox");
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-img").alt = title;
  document.getElementById("lightbox-title").textContent = title;
  lb.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  document.getElementById("lightbox").hidden = true;
  document.body.style.overflow = "";
}

function initGallery() {
  const grid = document.getElementById("figure-gallery");
  grid.innerHTML = GALLERY_FIGURES.map(
    (f) => `
      <button class="gallery-item" type="button" data-fig="${f.src}" data-title="${f.title}">
        <img src="${f.src}" alt="${f.title}" loading="lazy">
        <span class="gallery-item-title">${f.title}</span>
      </button>
    `
  ).join("");

  document.querySelectorAll("[data-fig]").forEach((btn) => {
    btn.addEventListener("click", () => openLightbox(btn.dataset.fig, btn.dataset.title));
  });
}

/* ---------- Boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  // Header scroll state
  const header = document.getElementById("site-header");
  const onScroll = () => header.classList.toggle("is-scrolled", window.scrollY > 18);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  // Mobile nav
  const navToggle = document.getElementById("nav-toggle");
  const nav = document.getElementById("primary-nav");
  navToggle.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  nav.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      nav.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  });

  // Hero controls
  const heroModelTabs = document.getElementById("hero-model-tabs");
  heroModelTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    heroModel = btn.dataset.model;
    setActiveTab(heroModelTabs, "model", heroModel);
    renderHero();
  });

  const heroPrecTabs = document.getElementById("hero-prec-tabs");
  heroPrecTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    heroPrec = btn.dataset.prec;
    setActiveTab(heroPrecTabs, "prec", heroPrec);
    renderHero();
  });

  // Traps toggle
  const trapTabs = document.getElementById("trap-tabs");
  trapTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    setActiveTab(trapTabs, "trap", btn.dataset.trap);
    renderTrap(btn.dataset.trap);
  });

  // Anatomy toggle
  const anatomyTabs = document.getElementById("anatomy-tabs");
  anatomyTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    setActiveTab(anatomyTabs, "anat", btn.dataset.anat);
    renderAnatomy(btn.dataset.anat);
  });

  // Block lab toggles
  const labModeTabs = document.getElementById("lab-mode-tabs");
  labModeTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    labMode = btn.dataset.mode;
    setActiveTab(labModeTabs, "mode", labMode);
    renderBlockLab();
  });

  const labModelTabs = document.getElementById("lab-model-tabs");
  labModelTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    labModel = btn.dataset.lmodel;
    setActiveTab(labModelTabs, "lmodel", labModel);
    renderBlockLab();
  });

  // Crossover toggle
  const crossTabs = document.getElementById("cross-tabs");
  crossTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    setActiveTab(crossTabs, "cross", btn.dataset.cross);
    renderCrossover(btn.dataset.cross);
  });

  // Ledger toggle
  const ledgerTabs = document.getElementById("ledger-tabs");
  ledgerTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    setActiveTab(ledgerTabs, "ledger", btn.dataset.ledger);
    renderLedger(btn.dataset.ledger);
  });

  // Lightbox
  document.getElementById("lightbox-backdrop").addEventListener("click", closeLightbox);
  document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });

  // Copy buttons
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.getElementById(btn.dataset.copy);
      if (!target) return;
      const orig = btn.textContent;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Select text to copy";
      }
      setTimeout(() => {
        btn.textContent = orig;
      }, 1600);
    });
  });

  // Initial renders
  renderHero();
  renderTrap("greedy");
  renderAnatomy("qwen3");
  renderBlockLab();
  renderCrossover("qwen35");
  renderLedger("p1");
  initGallery();
});
