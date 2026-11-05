"""Freeze deterministic development-only selections before confirmation."""
import argparse,json
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json

ROOTS=['qwen3_q8','lfm3_qad','fable_f16']


def records(label,key):
    root=Path(f'runs/p2-{label}-{key}-s17')
    if json.loads((root/'summary.json').read_text())['status']!='complete':raise ValueError('incomplete development')
    rows=[json.loads(x) for x in (root/'responses.jsonl').read_text().splitlines()]
    return {r['case']['id']:r for r in rows},str(root/'responses.jsonl')


def main():
    p=argparse.ArgumentParser();p.add_argument('study',choices=['e1','e2']);a=p.parse_args()
    output=Path(f'research/selections/{a.study}.json')
    if output.exists():raise FileExistsError(output)
    config=json.loads(Path('configs/portfolio.json').read_text())
    report={'utc':utc_now(),'git':git_state(),'study':a.study,'models':{},'sources':{}}
    for root in ROOTS:
        label='e1dev' if a.study=='e1' else 'e2dev-original'
        original,path=records(label,root);report['sources'][path]=file_sha256(path)
        suffixes=['_A','_B','_R'] if a.study=='e1' else ['_A','_B']
        keys=[root+s for s in suffixes]
        if root=='fable_f16' and a.study=='e1':keys+=['fable_v_only','fable_all_qkv_parent']
        candidates=[]
        primary_ids=[k for k,r in original.items() if r['case']['group'] in ('structured','literal_copy')] if a.study=='e1' else list(original)
        base_success=sum(original[k]['grading']['success'] for k in primary_ids)
        for order,key in enumerate(keys):
            comparison,path=records(label,key);report['sources'][path]=file_sha256(path)
            if comparison.keys()!=original.keys():raise ValueError('case mismatch')
            if a.study=='e1':
                preservation=[k for k,r in original.items() if r['case']['group'].startswith('functional_')]
                losses=[k for k in preservation if original[k]['grading']['payload_correct'] and not comparison[k]['grading']['payload_correct']]
                explanation_losses=[k for k,r in original.items() if r['case']['group']=='functional_explain' and r['grading']['success'] and not comparison[k]['grading']['success']]
            else:
                losses=[k for k,r in original.items() if r['case']['group']=='complete' and r['grading']['success'] and not comparison[k]['grading']['success']]
                explanation_losses=[]
            successes=sum(comparison[k]['grading']['success'] for k in primary_ids)
            manifest=config['models'][key].get('local_patch_manifest')
            changed=json.loads(Path(manifest).read_text())['changed_values'] if manifest else 10**12
            candidates.append({'key':key,'successes':successes,'n':len(primary_ids),'losses':losses,
                   'explanation_losses':explanation_losses,'changed_values':changed,'order':order,
                   'eligible':successes>base_success and not losses and not explanation_losses})
        eligible=sorted((c for c in candidates if c['eligible']),key=lambda c:(-c['successes'],c['changed_values'],c['order']))
        report['models'][root]={'original_successes':base_success,'n':len(primary_ids),'candidates':candidates,
              'eligible_repair_on_development':bool(eligible),'diagnostic_or_selected':eligible[0]['key'] if eligible else root+'_A',
              'selection_status':'development candidate; confirmation required' if eligible else 'no eligible improvement; A retained only as negative diagnostic'}
    write_json(output,report)
    print(json.dumps(report['models'],indent=2))


if __name__=='__main__':main()
