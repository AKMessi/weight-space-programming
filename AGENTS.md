# Research operating instructions

The user authorized implementation, local experiments, and commits/pushes to `git@github.com:AKMessi/weight-space-programming.git` on 2026-11-05. Commit and push each coherent change, including experimental evidence and negative results. Do not force-push or rewrite evidence. No sub-agents are requested.

## Scientific record

- Read `FIVE_EXPERIMENTS.md` and `research/PROGRESS.md` before continuing.
- Every experimental run needs an immutable manifest, dataset/config hashes, model artifact identity, runtime version, exact requests, raw responses, timing, errors, and grading results. Flush evidence after each request.
- Append discoveries, failures, confounds, decisions, and follow-ups to `research/JOURNAL.md`. Clearly distinguish observations from hypotheses and user-reported speed from measured speed.
- Mark discovery data as discovery. Never present development tuning or injected faults as held-out/natural evidence. Do not inspect confirmatory outcomes while selecting patches.
- Preserve model originals. Write edits to a separate ignored artifact directory and verify identity/rollback before causal claims.
- Count generated/scored tokens and failed attempts. No target-model training, paid compute, or unattended external API usage without a newly declared scope.
- Never commit credentials, environment contents, private unrelated files, model binaries, caches, or unbounded logs. Commit compact research evidence and exact provenance; keep large artifacts local with hashes.
- Use the existing working runtime when available; isolate any needed builds/environments. Load one model at a time and monitor memory.
- A useful negative result is evidence. Do not invent or overstate discoveries, significance, novelty, or completion.

## Current execution scope

Implement the common measurement/recording pipeline, characterize all three models, and proceed through gated experiments. Read-only inspection is permitted. Model artifacts and runtime binaries are local/ignored. Publishing code, documentation, and research results to the authorized repository is explicitly requested.
