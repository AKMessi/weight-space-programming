"""Analyze only the frozen v1 panel after all predeclared arms finish."""
import json
import math
from pathlib import Path
import random
import statistics

from weightlab.evidence import file_sha256, git_state, utc_now, write_json


def upper_event_probability(events, n, alpha=.05):
    """Exact one-sided binomial upper bound; requires independent prompt events."""
    if not 0 <= events <= n or n <= 0:
        raise ValueError('Invalid event count')
    if events == n:
        return 1.0
    lo,hi=0.0,1.0
    for _ in range(70):
        mid=(lo+hi)/2
        cdf=sum(math.comb(n,k)*mid**k*(1-mid)**(n-k) for k in range(events+1))
        if cdf>alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2


def paired_summary(original, comparison, ids, seeds):
    differences=[]
    losses=rescues=0
    for key in ids:
        pairs=[(original[(key,s)]['grading']['success'],comparison[(key,s)]['grading']['success']) for s in seeds]
        differences.append(statistics.mean(int(b)-int(a) for a,b in pairs))
        losses+=any(a and not b for a,b in pairs)
        rescues+=any(not a and b for a,b in pairs)
    rng=random.Random(20261105)
    samples=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    return {'prompts':len(ids),'seeds':seeds,'mean_paired_gain':statistics.mean(differences),
            'prompt_bootstrap_percentile_95':[samples[249],samples[9749]],
            'prompts_with_any_rescue':rescues,'prompts_with_any_loss':losses,
            'one_sided_95_upper_loss_event_probability':upper_event_probability(losses,len(ids)),
            'limits':'Intervals assume prompt sampling within these fixed authored families; seeds are grouped. Degenerate bootstrap intervals are not population certainty. Binomial bound assumes independent prompt events; template correlation can weaken it.'}


def main():
    suite=json.loads(Path('datasets/validation-v1-frozen.json').read_text())
    expected={c['id']:c for c in suite['cases']}
    specs={'fable_f16':[17,29],'fable_v_only':[17,29],
           'fable_all_qkv_parent':[17,29],'parent_uniform_bf16':[17,29],
           'qwen3_q8':[17],'lfm3_qad':[17]}
    report={'utc':utc_now(),'git':git_state(),'protocol':'research/protocols/validation-v1.md',
            'dataset_sha256':file_sha256('datasets/validation-v1-frozen.json'),
            'scope':'Small frozen synthetic validation, not a general benchmark or proof of useful specialization preservation.',
            'models':{},'sources':[],'paired_vs_fable':{}}
    data={}
    for model,seeds in specs.items():
        rows={}
        for seed in seeds:
            root=Path(f'runs/20261105-validation-{model}-s{seed}')
            summary=json.loads((root/'summary.json').read_text())
            if summary.get('status')!='complete':raise ValueError(f'Incomplete run: {root}')
            records=[json.loads(l) for l in (root/'responses.jsonl').read_text().splitlines()]
            if len(records)!=len(expected):raise ValueError('Missing/extra records')
            for row in records:
                key=row['case']['id']
                if row['case']!=expected[key] or 'grading' not in row:raise ValueError('Case/evidence mismatch')
                if (key,seed) in rows:raise ValueError('Duplicate request')
                rows[(key,seed)]=row
            report['sources'].append({'run':str(root),'responses_sha256':file_sha256(root/'responses.jsonl')})
        data[model]=rows
        groups={}
        for group in sorted({c['group'] for c in suite['cases']}):
            subset=[r for r in rows.values() if r['case']['group']==group]
            groups[group]={'requests':len(subset),'unique_prompts':len(subset)//len(seeds),
                           'exact_successes':sum(r['grading']['success'] for r in subset),
                           'payload_proxy_successes':sum(r['grading']['payload_correct'] for r in subset),
                           'cutoffs':sum(r['finish_reason']=='length' for r in subset)}
        report['models'][model]={'seeds':seeds,'groups':groups}
    for model in ['fable_v_only','fable_all_qkv_parent','parent_uniform_bf16']:
        report['paired_vs_fable'][model]={}
        for group in sorted({c['group'] for c in suite['cases']}):
            ids=[c['id'] for c in suite['cases'] if c['group']==group]
            report['paired_vs_fable'][model][group]=paired_summary(data['fable_f16'],data[model],ids,[17,29])
    if Path('research/validation-v1-analysis.json').exists():raise FileExistsError('Analysis already exists; version a new analysis instead')
    write_json('research/validation-v1-analysis.json',report)
    lines=['# Frozen validation v1 results','',report['scope'],'',
           'Protocol, fixtures, and candidate artifacts were committed before validation inference. No new dose/site was chosen on this panel. Counts aggregate decoding seeds; uncertainty groups seeds by prompt.', '',
           '| Model/arm | Group | Recorded exact/reference metric | Payload proxy | Cutoffs |',
           '|---|---|---:|---:|---:|']
    for model,result in report['models'].items():
        for group,g in result['groups'].items():
            lines.append(f'| {model} | {group} | {g["exact_successes"]}/{g["requests"]} | {g["payload_proxy_successes"]}/{g["requests"]} | {g["cutoffs"]} |')
    lines.extend(['','## Paired literal-copy result',''])
    for model,groups in report['paired_vs_fable'].items():
        g=groups['literal_copy'];ci=g['prompt_bootstrap_percentile_95']
        lines.append(f'- {model}: mean paired change {g["mean_paired_gain"]:+.1%}; conditional prompt-bootstrap interval [{ci[0]:+.1%}, {ci[1]:+.1%}]; {g["prompts_with_any_rescue"]}/{g["prompts"]} prompts rescued on at least one seed and {g["prompts_with_any_loss"]} with a loss.')
    lines.extend(['','## Interpretation limits','',
        '- Twelve prompts per category and a few authored forms are insufficient for broad generalization or a tight no-regression claim.',
        '- A degenerate bootstrap interval does not eliminate uncertainty outside these fixtures; see the exact event bounds and assumptions in the JSON analysis.',
        '- Dictionary code payload is a static return-expression proxy, not an execution benchmark or proof of the fine-tune\'s coding advantage.',
        '- Frozen grader v2 used an overly literal code reference-line comparison. Later grader v3 accepts equivalent quotes/spacing and improves expression-presence detection. Keep this recorded analysis unchanged; report v3 corrections as separate post-hoc sensitivity results. Literal-copy primary results are unaffected.',
        '- V-only already regressed the discovery explanation proxy. Even if copying improves here, it is not an accepted overall repair.',
        '- All-QKV is a broad rollback that may erase specialization. Qwen and LFM3 are reference deployments with different sizes/sampling/precision, not controlled architecture comparisons.',
        '- Do not reuse this panel to select a new patch and then call it held-out.',
        '', '[Machine-readable paired analysis](validation-v1-analysis.json). Raw per-run evidence is under `runs/20261105-validation-*`.'])
    Path('research/VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['paired_vs_fable']['fable_v_only']['literal_copy'],indent=2))


if __name__=='__main__':main()
