"""Audit matched request/runtime settings and individual composition interactions."""
import json
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.grading import grade_turns
from portfolio_batch import ROOTS


def load(label,key,seed=17):
    path=Path(f'runs/p2-{label}-{key}-s{seed}')
    records={r['case']['id']:r for r in map(json.loads,(path/'responses.jsonl').read_text().splitlines())}
    return path,json.loads((path/'manifest.json').read_text()),records


def match(a,b,ids=None):
    pa,ma,ra=a;pb,mb,rb=b
    assert ma['runtime']==mb['runtime'],(pa,pb,'runtime mismatch')
    identifiers=list(ids) if ids is not None else list(ra)
    assert set(identifiers)<=ra.keys() and set(identifiers)<=rb.keys()
    for identifier in identifiers:
        x,y=ra[identifier],rb[identifier]
        assert x['case']==y['case'],(pa,pb,identifier,'case mismatch')
        assert x['request']==y['request'],(pa,pb,identifier,'initial-request mismatch')
        # Follow-up messages legitimately depend on each model's actual first answer.
        for sx,sy in zip(x['steps'][1:],y['steps'][1:]):
            assert {k:v for k,v in sx['request'].items() if k!='messages'}=={k:v for k,v in sy['request'].items() if k!='messages'}
    return {'a':str(pa),'b':str(pb),'matched_cases':len(identifiers),'exact_initial_request_and_runtime':True,
            'a_responses_sha256':file_sha256(pa/'responses.jsonl'),'b_responses_sha256':file_sha256(pb/'responses.jsonl')}


def main():
    out=Path('research/portfolio-v3/comparison-audit.json')
    if out.exists():raise FileExistsError(out)
    checks=[];interactions=[]
    engineering={c['id'] for c in json.loads(Path('datasets/portfolio/engineering.json').read_text())['cases']}
    for root in ROOTS:
        for seed in [17,29]:checks.append(match(load('e1confirm',root,seed),load('e1confirm',root+'_A',seed)))
        checks.append(match(load('e2confirm-original',root),load('e2confirm-original',root+'_A')))
        for panel in ['engineering','joint_evidence']:
            arms={}
            for suffix in ['', '_A','_B','_R','_AB']:
                label=('e4engineering-v2' if suffix=='_AB' else 'e1dev') if panel=='engineering' else ('e4joint' if suffix in ['_R','_AB'] else 'e2dev-original')
                arms[suffix or 'base']=load(label,root+suffix)
            ids=engineering if panel=='engineering' else set(arms['base'][2])
            for arm in ['_A','_B','_R','_AB']:checks.append(match(arms['base'],arms[arm],ids))
            for identifier in sorted(ids):
                grades={name:grade_turns(item[2][identifier]['case'],item[2][identifier]['steps']) for name,item in arms.items()}
                for metric in ['success','payload_correct']:
                    values={name:int(g[metric]) for name,g in grades.items()}
                    interaction=values['_AB']-values['_A']-values['_B']+values['base']
                    interactions.append({'model':root,'panel':panel,'case_id':identifier,'metric':metric,'values':values,'interaction':interaction})
    for root in ['qwen','lfm3','fable']:
        for precision in ['float','Q8_0','Q4_0']:
            key=root+'_'+precision
            checks.append(match(load('e3precision',key),load('e3precision',key+'_A')))
    summary=[]
    for root in ROOTS:
        for panel in ['engineering','joint_evidence']:
            for metric in ['success','payload_correct']:
                selected=[r for r in interactions if (r['model'],r['panel'],r['metric'])==(root,panel,metric)]
                summary.append({'model':root,'panel':panel,'metric':metric,'n':len(selected),
                  'nonzero_case_interactions':sum(r['interaction']!=0 for r in selected),
                  'positive_interactions':sum(r['interaction']>0 for r in selected),
                  'negative_interactions':sum(r['interaction']<0 for r in selected),
                  'mean_interaction_percentage_points':100*sum(r['interaction'] for r in selected)/len(selected)})
    write_json(out,{'utc':utc_now(),'git':git_state(),'matched_comparisons':checks,'case_interactions':interactions,'summary':summary,
              'interpretation':'Descriptive finite differences on reused development examples. Zero aggregate interaction can hide cancellation or floor effects; no independently useful patch pair or new compatibility method is established.'})
    print(json.dumps({'comparisons':len(checks),'interaction_summary':summary},indent=2))


if __name__=='__main__':main()
