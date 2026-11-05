# Running the measurement pipeline

The first implementation uses Python's standard library and an existing local llama.cpp server. It does not require installing PyTorch to collect exploratory deployment evidence.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m weightlab fetch
PYTHONPATH=src python3 -m weightlab suite datasets/discovery-v1.json
PYTHONPATH=src python3 -m weightlab run lfm3_qad datasets/discovery-v1.json runs/UNIQUE-RUN-ID --max-tokens 128
```

Run directories cannot be reused. Source/config changes must be committed before inference. The runner verifies each model checksum and records the source commit, binary hash/version, server arguments, server properties/template, dataset hash, exact requests, full responses, grading, token usage, and errors. A server log records startup, offload, and backend details. One execution lock prevents concurrent runs.

The budget ledger reserves the requested maximum before sending a request, then settles actual usage after evidence is saved. Failed/interrupted requests remain conservatively charged. Each response and ledger entry is flushed to disk. Source files, compact run evidence, and journal entries are committed and pushed after each coherent change/run.

## Interpretation

`discovery-v1` is an investigator-visible synthetic screen. Its results are exploratory, not benchmark scores or held-out validation. `explanation_proxy` checks a simple output contract and is not a judgment of reasoning quality. JSON payload correctness is recorded separately from whole-response contract correctness. Qwen is explicitly in non-thinking mode in this first screen. Sampling is model-specific, while Fable and its parent use matched settings.

Model paths in `configs/local_models.json` describe this device. Adapt a copied config on another machine. Downloads are pinned and checksummed; model binaries are ignored by Git. The runner starts its own loopback server and terminates only the process it started.

## Separate-file weight arithmetic

`weightlab.edit` requires NumPy and GGUF (the isolated inspection environment is recorded in `research/environment.json`). It implements a **non-sparse baseline**: partial rollback of selected floating GGUF tensors, with FFN outputs selected by default. `--pattern` and `--layers` select other explicitly declared sites. Exact BF16 parent decoding and an explicit uniform-BF16 value control are supported. It verifies source checksums, architecture/tokenizer compatibility, selected shapes/dtypes, untouched tensor hashes, and post-cast values. An alpha-zero operation must preserve the entire file hash.

```bash
PYTHONPATH=src /path/to/inspection-venv/bin/python -m weightlab.edit --alpha 0 --output artifacts/fable-identity.gguf --manifest research/patches/fable-identity.json
```

Originals are read-only, outputs must be new files under ignored `artifacts/` or `local/`, and manifests are retained even after a failed edit. This is infrastructure and a comparison baseline, not evidence that rollback preserves specialization or improves the model.

## Evidence checks and grading corrections

```bash
PYTHONPATH=src python3 scripts/verify_evidence.py
PYTHONPATH=src python3 scripts/regrade.py runs/RUN/responses.jsonl research/regrades/NEW-UNIQUE-NAME.json
```

The verifier checks complete runs, frozen suite hashes, exact case identities, unique request IDs, response counts, and the token-reservation ledger. It verifies consistency, not scientific validity. Regrading creates a derived artifact and never overwrites raw responses. The original frozen validation uses recorded grader v2; v3 sensitivity results are separate.

Future manifests also fingerprint colocated shared libraries because the small server executable can load a larger inference implementation dynamically. Earlier manifests record the binary hash, build/version, arguments, template, and server logs; a later current-library snapshot is supplementary and cannot retroactively prove historical library bytes.
