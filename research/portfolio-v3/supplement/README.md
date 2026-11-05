# Deployment and integrity follow-ups

These are disclosed follow-ups to the bounded portfolio; quality decisions and primary diagnosis exclusions stay frozen.

## Fable: the larger file is faster on this device

| Format | File MB (decimal) | Median decode tokens/s | Range across three post-warmup calls |
|---|---:|---:|---:|
| fable_float | 461.9 | 194.60 | 193.89–195.05 |
| fable_Q8_0 | 246.6 | 13.40 | 13.40–13.40 |
| fable_Q4_0 | 149.1 | 22.07 | 22.05–22.08 |
| fable_Q4_K_M | 153.4 | 21.52 | 21.47–21.54 |

Each format received the same 28-token rendered story prompt, seed 17, context 2048, four calls and 64 actual generated tokens per call; the first call was warmup. Three calls in one session are descriptive repeats, not independent machines or robust uncertainty bounds. Ignore-EOS was requested, so this is a decode microbenchmark with no quality score. Generation content differs between formats. F16 is about 14.5× faster than Q8 here while requiring a larger file and more VRAM. The locally exported Q8 and Q4_K_M hashes match the pinned published Fable artifacts.

The separate Q8 verbosity-5 startup trace reports 15/15 layers offloaded. This does not identify which execution path causes the difference. No CPU-fallback or quantization-causes-slowdown mechanism is established. The matched repeated timing result is specific to this runtime/device and short-context workload; cache policy is disabled as recorded in each request.

## Exact checkpoint delivery

A named-tensor XOR/deflate patch restores the complete original Fable GGUF from its exact matching aligned base in 20,308,656 bytes. The same-source BF16-rounded F16 parent needs 44,653,069 bytes; the original unrounded F16 parent needs 167,671,933 bytes. All reconstruct the same complete 461,883,712-byte target hash. The same-source rounding comparison reduces patch cost by 3.75×; changing to the separately published aligned BF16 base reduces it further. Source provenance and double-rounding effects prevent assigning the full 8.26× source-choice difference to rounding alone.

These sizes exclude the matching base. This is an exact delivery tool and one measured fine-tune case, using established delta encoding; no new compression algorithm, standalone 20 MB model or validated behavioral repair is claimed. See [utility documentation](../../EXACT_PATCH_DELIVERY.md) and the raw roundtrip manifests.

## Permutation audit

| Model | State | Original payload / 12 | Permuted payload / 12 | Exact text pairs / 12 |
|---|---|---:|---:|---:|
| qwen | healthy | 12 | 12 | 10 |
| qwen | output_sign | 0 | 0 | 0 |
| qwen | ffn_sign | 0 | 0 | 0 |
| lfm3 | healthy | 12 | 12 | 12 |
| lfm3 | output_sign | 0 | 0 | 3 |
| lfm3 | ffn_sign | 0 | 0 | 3 |
| fable | healthy | 8 | 8 | 12 |
| fable | output_sign | 0 | 0 | 10 |
| fable | ffn_sign | 0 | 0 | 12 |

All nine gated-MLP numerical checks passed at the declared tolerance; three of nine full-runtime layouts passed strict text parity. Equal coarse task scores cannot establish functional equivalence. The six excluded layouts remain excluded from primary blinded-editor accuracy. This exposes a practical difference between mathematical symmetry and identical native generation; it does not prove a new capability loss or explain the numerical cause.

## Native final-state gate

| Model | Context | Same top-1? | Max centered-logit error | Relative error | Passed? |
|---|---|---|---:|---:|---|
| qwen | raw-state-0 | True | 0.0810 | 0.44% | False |
| qwen | raw-state-1 | True | 0.2429 | 1.33% | False |
| lfm3 | raw-state-0 | True | 0.4731 | 1.46% | False |
| lfm3 | raw-state-1 | True | 0.6062 | 2.25% | False |
| fable | raw-state-0 | True | 0.3249 | 3.28% | False |
| fable | raw-state-1 | True | 0.2184 | 2.13% | False |

Six contexts retained top-1 but none met both predeclared error tolerances. No edit was derived from these unverified readouts. The two-context gate is a compatibility test at this boundary/backend, not proof that hidden-state access is impossible. A separately validated CPU/native or instrumented path is future method work, not a completed comparison in this portfolio.

All code scores shown here use the disclosed derived v7 sensitivity policy; every raw grade remains unchanged. Data and source checksums are in [results.json](results.json).
