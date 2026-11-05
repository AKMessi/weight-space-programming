"""P3 run publisher with per-stage maximum-output reservations."""
import json
from pathlib import Path
from weightlab.runner import execute
from weightlab.evidence import digest,append_jsonl,utc_now
from portfolio_batch import publish

CAPS={'repair':200000,'adapter':80000,'speed':15000,'precision':100000,'clarify':120000,'reserve':50000}

def stage_used(stage):
    prefix=f'runs/p3-{stage}-';charges={}
    for line in Path('research/budget.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['request_id'].startswith(prefix):charges[row['request_id']]=row['tokens']
    return sum(charges.values())

def run_one(config,key,suite_path,label,stage,max_tokens=128,seed=17,limit=None):
    suite=json.loads(Path(suite_path).read_text());out=Path(f'runs/p3-{stage}-{label}-{key}-s{seed}')
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
        if old['status']!='complete':raise RuntimeError('Failed runs require a versioned retry')
        expected={'config_sha256':digest(config),'suite_sha256':digest(suite),'model_key':key,'seed':seed,'max_tokens':max_tokens,'limit':limit}
        if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Resume inputs changed')
        print('Already complete',out,flush=True);return old
    cases=suite['cases'][:limit] if limit else suite['cases']
    maximum=sum(max_tokens*(1+len(c.get('followups',[]))) for c in cases)
    if stage_used(stage)+maximum>CAPS[stage]:raise RuntimeError('P3 stage token ceiling')
    error=None
    try:summary=execute(config,key,suite,out,max_tokens,seed,limit)
    except BaseException as exc:
        error=exc;summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {'status':'preflight_failed','error':repr(exc)}
    append_jsonl('research/p3-events.jsonl',{'utc':utc_now(),'stage':stage,'run':str(out),'summary':summary})
    with Path('research/JOURNAL.md').open('a') as f:f.write(f'\n### {utc_now()} — {out.name}\n\nStatus `{summary.get("status")}`, {summary.get("generated_tokens",0)} generated tokens. Immutable raw inputs/responses/config/runtime and resource samples are in `{out}`. Interpretation is deferred to the study report; this run alone does not establish a repair.\n')
    paths=[out,'research/budget.jsonl','research/p3-events.jsonl','research/JOURNAL.md']
    publish([p for p in paths if Path(p).exists()],f'Record P3 {stage} {label} {key} {summary.get("status")}')
    if error:raise error
    return summary
