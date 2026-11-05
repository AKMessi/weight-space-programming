"""Reproducible exploratory report; versioned regrading does not mutate raw evidence."""
import argparse
import json
from pathlib import Path
import statistics

from weightlab.evidence import file_sha256, git_state, utc_now
from weightlab.grading import GRADER_VERSION, grade
from weightlab.runner import summarize

p = argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('runs', nargs='+')
a = p.parse_args()
lines = ['# Exploratory baseline report', '', f'Generated {utc_now()} with grader {GRADER_VERSION}.',
         f'Source commit: `{git_state()["commit"]}`.', '',
         '**Discovery only.** Small investigator-visible synthetic tasks; no confidence, novelty, general benchmark, or model-improvement claim.', '',
         'Code payload checks are static expression proxies; explanation passes do not establish explanation quality. Cutoff explanations are not complete answers.', '',
         '| Run/model | Group | Exact/proxy success | Payload | Cutoffs |',
         '|---|---|---:|---:|---:|']
details = []
for folder in a.runs:
    root = Path(folder)
    manifest = json.loads((root/'manifest.json').read_text())
    original_rows = [json.loads(l) for l in (root/'responses.jsonl').read_text().splitlines()]
    rows = []
    for row in original_rows:
        if 'content' in row:
            row['grading'] = grade(row['case'], row['content'])
        rows.append(row)
    summary = summarize(rows)
    for group,g in summary['groups'].items():
        lines.append(f'| {root.name}/{manifest["model_key"]} | {group} | {g["success"]}/{g["attempted"]} | {g["payload_correct"]}/{g["attempted"]} | {g["cutoff"]} |')
    speeds = [r['timings']['predicted_per_second'] for r in rows if r.get('timings',{}).get('predicted_per_second')]
    details.extend(['', f'## {root.name}', '',
                    f'- Artifact SHA-256: `{manifest["verified_model_sha256"]}`',
                    f'- Raw-response SHA-256: `{file_sha256(root/"responses.jsonl")}`',
                    f'- Requests: {summary["attempt_count"]}; generated tokens: {summary["generated_tokens"]}.',
                    f'- Median server decode speed: {statistics.median(speeds):.2f} tokens/s on these requests.' if speeds else '- Decode speed unavailable.',
                    f'- [Raw evidence](../{root.as_posix()}/responses.jsonl).'])
lines.extend(details)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text('\n'.join(lines)+'\n')
