"""Token-length preflight and a complete versioned retry; no evidence truncation."""
import argparse,json
from pathlib import Path
from weightlab.budget import settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json,runtime_libraries
from portfolio_batch import run_one

p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);a=p.parse_args()
if a.stage=='run':
    config=json.loads(Path('configs/diagnosis_editor_retry1.json').read_text())
    run_one(config,'local_editor','datasets/portfolio/diagnosis-blinded.json','e5editor-retry1',max_tokens=128)
else:
    output=Path('research/diagnosis-preflight-v2')
    output.mkdir(exist_ok=False)
    config=json.loads(Path('configs/diagnosis_editor.json').read_text())
    suite=json.loads(Path('datasets/portfolio/diagnosis-blinded.json').read_text())
    spec=config['models']['local_editor'];runtime={**config['runtime'],'context':8192}
    if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('editor artifact hash')
    write_json(output/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':spec,'runtime':runtime,
             'binary_sha256':file_sha256(runtime['binary']),'libraries':runtime_libraries(runtime['binary']),
             'purpose':'tokenization only; no task completions or hidden states requested'})
    lengths={}
    with server(runtime['binary'],spec['path'],output,runtime) as (url,command,proc):
        write_json(output/'runtime.json',{'command':command,'props':request_json(url+'/props')})
        for case in suite['cases']:
            payload={'model':'local','messages':case['messages'],'chat_template_kwargs':{'enable_thinking':False},
                     'max_tokens':128,'seed':17,**spec['sampling'],**case.get('request_overrides',{})}
            rendered=request_json(url+'/apply-template',payload)
            text=rendered['prompt'] if isinstance(rendered,dict) else rendered
            token_request={'content':text,'add_special':True,'parse_special':True}
            tokens=request_json(url+'/tokenize',token_request)
            lengths[case['id']]=len(tokens['tokens'])
            append_jsonl(output/'tokenization.jsonl',{'case_id':case['id'],'template_request':payload,
                 'rendered':rendered,'token_request':token_request,'token_response':tokens})
    required=max(lengths.values())+128+32
    chosen=next((n for n in [2048,4096,8192] if n>=required),None)
    if chosen is None:raise ValueError('evidence exceeds the declared editor-context envelope')
    config['runtime']['context']=chosen
    write_json('configs/diagnosis_editor_retry1.json',config)
    write_json('research/diagnosis-editor-run.json',{'utc':utc_now(),'git':git_state(),'label':'e5editor-retry1',
         'previous_failed_run':'runs/p2-e5editor-local_editor-s17','policy':'Full frozen batch replay at one context size; no omitted/modified cases.',
         'suite_sha256':digest(suite),'prompt_token_counts':lengths,'max_prompt_tokens':max(lengths.values()),
         'context':chosen,'max_generated_tokens':128})
    failed_path=Path('runs/p2-e5editor-local_editor-s17/responses.jsonl')
    failed=[json.loads(x) for x in failed_path.read_text().splitlines() if 'exceed_context_size_error' in x]
    if len(failed)!=1:raise ValueError('expected one explicit pre-decoding context rejection')
    row=failed[0]
    key=f'runs/p2-e5editor-local_editor-s17/{row["case"]["id"]}/seed-17'
    ledger=[json.loads(x) for x in Path('research/budget.jsonl').read_text().splitlines()]
    last=next(r for r in reversed(ledger) if r['request_id']==key)
    if last['event']!='reserve':raise ValueError('reservation already adjudicated; inspect before repeating')
    append_jsonl('research/budget-adjustments.jsonl',{'utc':utc_now(),'request_id':key,'known_generated_tokens':0,
                  'reason':'HTTP 400 exceed_context_size_error rejected before decoding; server states exact prompt/context counts.',
                  'evidence':str(failed_path),'evidence_sha256':file_sha256(failed_path),'original_error':row['error']})
    settle(key,0)
    print('max prompt:',max(lengths.values()),'chosen context:',chosen,flush=True)
