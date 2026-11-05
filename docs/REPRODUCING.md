# Reproducing the portfolio

## Inspect results without inference

The repository contains the frozen suites, source/config/artifact hashes, actual rendered prompts, complete native requests/responses, errors, timings, graders, selections, reports and derived sensitivity grades. Model binaries are intentionally ignored.

```bash
PYTHONPATH=src python3 scripts/verify_evidence.py --output local/offline-audit.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The offline audit validates all committed run/ledger identities, case ordering, complete conversations including actual prior assistant messages, recorded usage and the known-zero pre-decoding context rejection. Optional GGUF/NumPy tests require the inspection environment below. The audit is a consistency test, not a proof of scientific conclusions. Use a new output name every time. Original audit snapshots remain in `research/`.

## Environment

Inference uses Python's standard library and the existing Vulkan `llama-server`: version 0.7.0-dev, build 11942, commit `338fce651`. The complete executable and colocated-library fingerprints are recorded in new run manifests. The earlier `/home/akmessi/llama.cpp` build was CPU-only and is not the working inference build. Its local `gguf-py` source at `64e9bceb2c3a856efed96feda784a50947049feb` supplied the GGUF Python tools. These are two different source/build identities.

The inspected environment is `/home/akmessi/.local/share/weight-space-programming/venv/bin/python`: Python 3.14.4, NumPy 2.5.3, GGUF 0.19.0 and Matplotlib 3.11.2. `research/environment.json` preserves the initial environment; `research/environment-final.json` records the final package versions. A new environment can install this package with its editing extra and Matplotlib; use the recorded NumPy/GGUF implementation for exact artifact replication. Rebuilding another native revision is a new condition and does not guarantee identical bytes or outputs.

## Recreate ignored artifacts

Do not rerun preparation scripts over committed historical manifests: they deliberately refuse to overwrite artifacts. The rebuild command recursively follows those manifests, creates missing ignored GGUF files, and stores **new local rebuild logs** without changing the historical recipes. It accepts an existing file only if its entire hash matches. It can relocate an unavailable absolute source path to `local/models/<pinned basename>`. Downloads require immutable repo revisions and expected checksums.

```bash
PYTHONPATH=src /path/to/inspection-venv/bin/python -m weightlab.rebuild --config configs/portfolio_precision.json --download --quantizer /path/to/recorded/llama-quantize --output-config local/rebuilt-precision.json
PYTHONPATH=src /path/to/inspection-venv/bin/python -m weightlab.rebuild --config configs/portfolio.json --download --output-config local/rebuilt-portfolio.json
```

Use `--keys qwen_float_A` to rebuild just a chosen artifact and its dependencies. The full set occupies tens of GB; one selected pair is sufficient for a focused reproduction. Quantized rebuilds check the quantizer executable hash and require the final model hash to match. Matching bytes are the acceptance criterion; a mismatching conversion is retained with an identity-error record and rejected.

Adapt a **copied** config's runtime binary to your machine. Ordinary LFM3 floating/Q8/Q4 runs used 18 GPU layers; other panels use the layer setting in their own manifests. The runtime, template, context, sampling, repetition penalties and output cap all matter. Do not conflate the community Qwen floating export with the official Q8 checkpoint, or ordinary LFM3 Q4 with QAD.

## Repeat a chosen inference condition

All original run directories are immutable, including failed attempts. Use a new run name and a copied runtime config. Source changes must be committed before inference.

```bash
PYTHONPATH=src python3 -m weightlab --config local/rebuilt-portfolio.json run qwen3_q8 datasets/portfolio/e1-confirmation.json runs/reproduction-qwen-e1-s17 --seed 17 --max-tokens 192
```

Copy a raw run's own manifest/suite to match its exact settings. Study 1 uses 192 output tokens; study 2 uses 128 per actual turn; study 3 uses 128; corrected study-4 engineering/order uses 192 and joint evidence uses 128. Diagnosis target probes and editor settings have their own manifests; the full editor retry uses context 4096. The CLI does not automatically commit a reproduction run. The original portfolio batch wrapper publishes each completed or failed run, as authorized for this repository, and rejects resumed names with different inputs.

The append-only budget ledger includes committed historical costs; new runs add to that ledger. Its 1.8M generated-token ceiling is a cap, not a target. Native final-state diagnostic prefill/embedding usage has a separate scored-token ledger. Chat input token usage is retained in every raw response and is not charged as generated output.

## Rebuild analysis and figures

```bash
PYTHONPATH=src:scripts /path/to/inspection-venv/bin/python scripts/portfolio_report.py --output local/reproduced-report
PYTHONPATH=src:scripts python3 scripts/comparison_audit.py --output local/reproduced-report/comparison-audit.json
PYTHONPATH=src /path/to/inspection-venv/bin/python scripts/portfolio_supplements.py --output local/reproduced-report/supplement
```

The primary report currently checks actual local model sizes, so restore the referenced artifacts first; alternatively the committed results provide sizes. Reports refuse existing output directories. Relative report links assume the published `research/` location; local copies may need adjusted navigation. Derived regrading never overwrites original raw grades. Grader v6 changed code extraction/signature sensitivity; v7 only added measurement-only fixtures. Frozen selections were not retuned on corrected confirmation outcomes.

`portfolio-v1` is an explicitly incomplete analysis attempt caused by legacy unversioned metadata. `portfolio-v2` is an earlier complete snapshot before the study-4 cap mismatch was corrected. `portfolio-v3` contains the matched-cap analysis. These are analysis versions, not independent replications. Known schema-wrapper and editor-context corrections are described in the protocol and journal.

For exact binary patch creation/reconstruction, use [the codec guide](../research/EXACT_PATCH_DELIVERY.md). All five full-file reconstructions passed their target SHA-256 checks; the codec is not a cross-architecture or cross-quantization format. Packing archive bytes may differ because of timestamps; the restored target hash is the invariant.
