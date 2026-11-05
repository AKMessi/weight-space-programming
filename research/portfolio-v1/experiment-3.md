# Experiment 3 — Quantization and patch deployment

Completed same-lineage numerical/deployment stress pilot. Natural useful-repair retention requires an independently useful repair on this exact source artifact; that prerequisite is not established by these diagnostic A edits. No precision allocator or repair-retention success is claimed.

| Model | Precision | Original strict / 18 | A strict / 18 | Gain | Bytes original | A bytes | Median original decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen | float | 5 | 6 | +5.6% | 1198182848 | 1198182848 | 61.12 |
| qwen | Q8_0 | 5 | 6 | +5.6% | 639447488 | 639447488 | 90.57 |
| qwen | Q4_0 | 5 | 5 | +0.0% | 381566400 | 381566400 | 108.65 |
| lfm3 | float | 10 | 10 | +0.0% | 3466626336 | 3466626336 | 14.52 |
| lfm3 | Q8_0 | 10 | 10 | +0.0% | 1846238496 | 1846238496 | 21.86 |
| lfm3 | Q4_0 | 12 | 11 | -5.6% | 1049615648 | 1049615648 | 24.27 |
| fable | float | 0 | 0 | +0.0% | 461883712 | 461883712 | 170.65 |
| fable | Q8_0 | 0 | 0 | +0.0% | 246597952 | 246597952 | 13.25 |
| fable | Q4_0 | 0 | 0 | +0.0% | 149080384 | 149080384 | 21.76 |

QAD no-op rewrite: exact content/token parity=True on 6 greedy schema probes. Every quantized tensor byte and original metadata item was separately verified. This does not equate QAD to ordinary Q4.

The A intervention edits a normalization vector retained in F32 by these quantizers. Quantization manifests verify those bytes. The comparison tests propagation of an identical small parameter edit through a quantized network; it cannot show a sparse floating patch being rounded away, or demonstrate a novel mixed-precision allocation method.

Qwen uses a pinned community floating export as its own source, not an assumed reconstruction of the official Q8 file. LFM3 uses official ordinary F16; all its precision arms use 18 GPU layers for memory headroom. Actual templates, backend/settings, resource samples, quantizer logs and source hashes are retained. Speeds are short-task observations with background preparation/download activity, not isolated benchmarks. Ratios of tiny or zero behavioral gains are intentionally omitted.
