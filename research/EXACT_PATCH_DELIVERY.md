# Exact GGUF checkpoint patches

The deployment utility reconstructs a **byte-identical target GGUF** from a matching base plus a compressed named-tensor XOR patch. It handles reordered tensors and changed metadata/layout, copies unchanged tensor bytes, and checks the complete output SHA-256. Originals are never rewritten. It uses established delta encoding and deflate compression; there is no new learning or compression-algorithm claim.

## Measured results

| Target | Base | Target bytes | Whole-target zlib bytes | Exact patch bytes | Changed tensors |
|---|---|---:|---:|---:|---:|
| Qwen diagnostic A | Original official Q8 | 639,446,688 | 610,730,163 | 19,270 | 1 |
| LFM3 QAD diagnostic A | Original QAD Q4 | 1,049,615,680 | 995,496,702 | 19,315 | 1 |
| Original Fable | Public F16 parent | 461,883,712 | 356,510,251 | 167,671,933 | 132 |
| Original Fable | Same F16 parent after deterministic BF16 rounding | 461,883,712 | 356,510,251 | 44,653,069 | 92 |
| Original Fable | Precision-aligned BF16-value parent in F16 layout | 461,883,712 | 356,510,251 | 20,308,656 | 27 |

All five reconstructions matched their complete target-file hashes. The two Fable patches reconstruct the **same** original Fable checkpoint, SHA-256 `1e1624de4d7ebe413d82dbe2b5f48fcf83e6c46b1b5414c8aef0596009380cc6`.

The aligned-base patch is approximately **20.3 MB instead of 167.7 MB**, a factor of **8.26** smaller. This connects the earlier numerical precision audit to a measurable storage cost. After alignment, 105 tensors can be copied exactly; the substantial differences are concentrated in the 18 Q/K/V projections, with tiny residual differences in nine other tensors.

Patch sizes exclude the base. A patch requires the exact matching base hash; the saving is useful when that base is already cached or shared across multiple variants. The two A edits are research diagnostics and have no confirmed general utility. Delivering them exactly is an engineering result, not evidence of an improved model.

## A source-provenance control

The F16 and BF16 source files are separately published artifacts; their exact historical conversion lineage is not independently established. The size comparison alone must not be attributed exclusively to rounding. A separately declared post-hoc follow-up uses **the same original F16 parent with deterministic BF16 value rounding**. It tests how much of the cost reduction is obtained without switching source checkpoints. The measured patch is **44,653,069 bytes** and restored the exact same Fable hash. Rounding this one source reduces patch cost by **3.75×**; changing to the separately published aligned BF16 source reduces it further. The control is recorded separately in [exact-patch-rounding-control.json](exact-patch-rounding-control.json); the original four results remain unchanged. The remaining difference cannot be attributed exclusively to precision rather than conversion provenance or double rounding.

## Using the utility

Run from this repository using the recorded GGUF inspection environment:

```bash
PYTHONPATH=src /home/akmessi/.local/share/weight-space-programming/venv/bin/python -m weightlab.patchcodec pack BASE.gguf TARGET.gguf artifacts/change.wpatch --manifest research/patches/UNIQUE-pack.json
PYTHONPATH=src /home/akmessi/.local/share/weight-space-programming/venv/bin/python -m weightlab.patchcodec apply BASE.gguf artifacts/change.wpatch artifacts/restored.gguf --manifest research/patches/UNIQUE-apply.json
```

Every output and manifest name must be new. Source and target tensor names, logical shapes, and encodings must match; this is not a cross-architecture or cross-quantization patch format. Output binaries stay under ignored `artifacts/` or `local/`. A wrong base is rejected, and reconstruction is accepted only after the target checksum matches.

The existing local Fable patch is `artifacts/patches/fable-aligned.wpatch`; its base is specified in [the exact manifest](patches/storage-fable-aligned.json). A verified reconstruction is `artifacts/patches/fable-aligned-restored.gguf`. Binaries remain local, while code, checksums, sizes, provenance, timings and failure records are committed.

## Limits

This is one fine-tuned-model case plus two small diagnostic edits. No comparison with specialized binary-delta tools, deduplicating stores, or state-of-the-art compression is claimed. Packing/reconstruction timings were collected alongside other local work, not as isolated performance benchmarks. Archive metadata can include timestamps; exact reconstructed model identity is the invariant. The [machine-readable results](exact-patch-storage.json) and [roundtrip manifests](patches/) contain the evidence.
