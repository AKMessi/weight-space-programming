"""Audit frozen inputs, full conversations, failures, reservations and artifacts."""
import argparse
import json
from pathlib import Path
from weightlab.budget import consumed
from weightlab.evidence import digest,file_sha256,git_state,utc_now,write_json


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def audit(verify_artifacts=False):
    started=utc_now()
    roots=sorted(p.parent for p in Path('runs').glob('*/manifest.json'))
    adjustments={r['request_id']:r for r in jsonl('research/budget-adjustments.jsonl')} if Path('research/budget-adjustments.jsonl').exists() else {}
    expected={};runs=[];models={}
    totals={'complete_runs':0,'failed_runs':0,'conversations_attempted':0,'inference_requests':0,
            'successful_responses':0,'failed_requests':0,'unattempted_cases_in_failed_runs':0,'generated_tokens':0}
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text());suite=json.loads((root/'suite.json').read_text())
        summary=json.loads((root/'summary.json').read_text());rows=jsonl(root/'responses.jsonl')
        assert digest(suite)==manifest['suite_sha256'],root
        cases=suite['cases'][:manifest['limit']] if manifest['limit'] else suite['cases']
        assert len(rows)==summary['attempt_count'],root
        assert sum('grading' in row for row in rows)==summary['response_count'],root
        assert [r['case'] for r in rows]==cases[:len(rows)],root
        complete=summary['status']=='complete'
        if complete:assert len(rows)==len(cases) and all('grading' in r for r in rows),root
        else:assert any('error' in r for r in rows),root
        generated=sum(r.get('usage',{}).get('completion_tokens',0) for r in rows)
        requests=responses=failures=0
        for row in rows:
            key=f'{root}/{row["case"]["id"]}/seed-{manifest["seed"]}'
            assert key not in expected,key
            if 'error' in row:
                correction=adjustments[key]
                assert file_sha256(correction['evidence'])==correction['evidence_sha256'],key
                assert row['error']==correction['original_error'],key
                # Only an explicit pre-decoding context rejection is settled as zero.
                assert 'exceed_context_size_error' in row['error'] and correction['known_generated_tokens']==0,key
                expected[key]=0;requests+=1;failures+=1
                continue
            expected[key]=row['usage']['completion_tokens'];steps=row.get('steps')
            if steps:
                assert len(steps)==1+len(row['case'].get('followups',[])),key
                for name in ('prompt_tokens','completion_tokens','total_tokens'):
                    assert row['usage'][name]==sum(s['usage'].get(name,0) for s in steps),key
                messages=list(row['case']['messages'])
                for index,step in enumerate(steps):
                    if index:messages += [{'role':'assistant','content':steps[index-1]['content']}]+row['case']['followups'][index-1]['messages']
                    assert step['request']['messages']==messages,key
                    assert step['usage']==step['raw_response']['usage'],key
                    assert step['content']==(step['raw_response']['choices'][0]['message'].get('content') or ''),key
                requests+=len(steps);responses+=len(steps)
            else:
                assert row['usage']==row['raw_response']['usage'],key
                requests+=1;responses+=1
        assert generated==summary['generated_tokens'],root
        assert requests==summary.get('inference_requests',summary['attempt_count']),root
        totals['complete_runs' if complete else 'failed_runs']+=1
        totals['conversations_attempted']+=len(rows);totals['inference_requests']+=requests
        totals['successful_responses']+=responses;totals['failed_requests']+=failures;totals['generated_tokens']+=generated
        if not complete:totals['unattempted_cases_in_failed_runs']+=len(cases)-len(rows)
        model_path=manifest['model']['path']
        assert model_path not in models or models[model_path]==manifest['verified_model_sha256'],model_path
        models[model_path]=manifest['verified_model_sha256']
        runs.append({'run':str(root),'status':summary['status'],'manifest_sha256':file_sha256(root/'manifest.json'),
                     'responses_sha256':file_sha256(root/'responses.jsonl'),'suite_sha256':manifest['suite_sha256']})
    ledger={};reservations={}
    for row in jsonl('research/budget.jsonl'):
        key=row['request_id']
        if row['event']=='reserve':
            assert key not in reservations,key
            reservations[key]=row['tokens']
        else:assert key in reservations,key
        ledger[key]=row
    assert set(ledger)==set(reservations)==set(expected),'Run/ledger identity mismatch'
    assert all(r['event']=='settle' and r['tokens']==expected[k] for k,r in ledger.items()),'Unsettled/mismatched charge'
    assert consumed()==totals['generated_tokens'],'Ledger/token mismatch'
    ledger_sha256=file_sha256('research/budget.jsonl')
    scored={}
    if Path('research/scored-budget.jsonl').exists():
        for row in jsonl('research/scored-budget.jsonl'):scored[row['request_id']]=row
        assert all(row['event']=='settle' for row in scored.values()),'Unsettled diagnostic scoring'
    checked=[]
    if verify_artifacts:
        for path,sha in sorted(models.items()):
            assert file_sha256(path)==sha,path
            checked.append({'path':path,'sha256':sha})
    return {'started_utc':started,'utc':utc_now(),'git':git_state(),'budget_sha256':ledger_sha256,**totals,'all_requests_settled':True,'ledger_matches':True,
            'generated_budget_ceiling':1800000,'known_zero_failed_request_adjustments':list(adjustments.values()),
            'separate_native_diagnostic_prefill_and_embedding_tokens':sum(r['tokens'] for r in scored.values()),
            'diagnostic_scoring_note':'The separate scored ledger covers the native-state diagnostic only. Ordinary chat prompt usage remains in every raw response; it is not additionally charged as generated tokens.',
            'verified_current_model_files':checked,'runs':runs,
            'scope':'Recorded evidence consistency and optional current artifact identity; not scientific validity or model improvement.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='research/evidence-integrity-final-v1.json')
    parser.add_argument('--verify-artifacts',action='store_true')
    args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError('Version audits instead of overwriting historical evidence')
    result=audit(args.verify_artifacts);write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','verified_current_model_files')},indent=2))
