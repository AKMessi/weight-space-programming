"""Verify raw final states against native next-token logits before editing them."""
import copy,json,sys
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.budget import reserve,settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json
from portfolio_batch import run_one,publish

config=json.loads(Path('configs/portfolio_precision.json').read_text())
suite={'name':'native-final-state-parity-v1','status':'compatibility calibration only; no model quality or repair claim','cases':[]}
for i,prompt in enumerate(['Record: label=CY-17; count=4. Return only JSON with label as a string and count as an integer.',
                          'Return a Python function def add(a, b): that adds its two arguments, in a code fence, and explain why it works.']):
    suite['cases'].append({'id':f'raw-state-{i}','split':'compatibility_calibration','family':'raw_state','group':'raw_state',
        'grader':'measurement_only','expected':None,'messages':[{'role':'system','content':'Follow the requested response format.'},{'role':'user','content':prompt}],
        'request_overrides':{'logprobs':True,'top_logprobs':20,'temperature':1.0,'top_k':0,'top_p':1.0,'min_p':0.0,'repeat_penalty':1.0}})
dataset=Path('datasets/portfolio/native-state-parity.json')
if sys.argv[1:]==['freeze']:
    if dataset.exists():raise FileExistsError(dataset)
    write_json(dataset,suite)
    raise SystemExit(0)
if not dataset.exists():raise FileNotFoundError('Freeze this exact suite in a separate preparation step before running')
if json.loads(dataset.read_text())!=suite:raise ValueError('frozen suite mismatch')
results=[]
for root in ['qwen','lfm3','fable']:
    key=root+'_float';local=copy.deepcopy(config)
    if root=='lfm3':local['runtime']['gpu_layers']=18
    # Count model-prompt diagnostic work separately from generated tokens.
    score_id=f'native-state/{root}/generation-prefill'
    reserve(score_id,2*local['runtime']['context'],1000000,'research/scored-budget.jsonl')
    run_one(local,key,str(dataset),'native-state',max_tokens=1)
    native_dir=Path(f'runs/p2-native-state-{key}-s17')
    records=[json.loads(x) for x in (native_dir/'responses.jsonl').read_text().splitlines()]
    settle(score_id,sum(r['usage']['prompt_tokens'] for r in records),'research/scored-budget.jsonl')
    out=Path(f'research/native-state-probes/{root}');out.mkdir(parents=True,exist_ok=False)
    runtime={**local['runtime'],'embedding':True}
    weight_reader=GGUFReader(local['models'][key]['path'])
    tensors={t.name:t for t in weight_reader.tensors}
    head=tensors.get('output.weight',tensors['token_embd.weight']).data.astype(np.float32)
    write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'source_responses_sha256':file_sha256(native_dir/'responses.jsonl'),
               'model':local['models'][key],'runtime':runtime,'criteria':{'top1_equal':True,'max_centered_logit_error':.05,'relative_centered_error':.01}})
    with server(runtime['binary'],local['models'][key]['path'],out,runtime) as (url,command,proc):
        write_json(out/'runtime.json',{'command':command,'props':request_json(url+'/props')})
        for row in records:
            rendered=row['rendered_prompt'];text=rendered['prompt'] if isinstance(rendered,dict) else rendered
            tokenizations=[]
            for add_special in [True,False]:
                request={'content':text,'add_special':add_special,'parse_special':True}
                response=request_json(url+'/tokenize',request)
                tokenizations.append({'request':request,'response':response})
            options=[x['response']['tokens'] for x in tokenizations if len(x['response']['tokens'])==row['usage']['prompt_tokens']]
            if not options:raise ValueError('No tokenization matches the native chat prompt length')
            tokens=options[0]
            request={'content':tokens,'embd_normalize':-1}
            scored_id=f'native-state/{root}/{row["case"]["id"]}/embedding'
            reserve(scored_id,len(tokens),1000000,'research/scored-budget.jsonl')
            append_jsonl(out/'events.jsonl',{'event':'embedding_request','case_id':row['case']['id'],'tokenizations':tokenizations,'request':request})
            response=request_json(url+'/embedding',request)
            append_jsonl(out/'events.jsonl',{'event':'embedding_response','case_id':row['case']['id'],'response':response})
            settle(scored_id,len(tokens),'research/scored-budget.jsonl')
            item=response[0] if isinstance(response,list) else response
            vector=np.asarray(item['embedding'],dtype=np.float32).reshape(-1)
            if vector.shape!=(head.shape[1],) or not np.isfinite(vector).all():raise ValueError('Not one finite raw final state')
            native=row['raw_response']['choices'][0]['logprobs']['content'][0]['top_logprobs']
            ids=np.array([p['id'] for p in native],dtype=int);logp=np.array([p['logprob'] for p in native],dtype=float)
            logits=head@vector;native_center=logp-logp[0];manual=logits[ids]-logits[ids[0]]
            maximum=float(np.max(np.abs(native_center-manual)));relative=float(np.linalg.norm(native_center-manual)/max(np.linalg.norm(native_center),1e-9))
            same=int(np.argmax(logits))==int(ids[0]);passed=same and maximum<=.05 and relative<=.01
            result={'model':root,'case_id':row['case']['id'],'top1_equal':same,'max_centered_logit_error':maximum,
                    'relative_centered_error':relative,'passed':passed,'raw_state_l2_norm':float(np.linalg.norm(vector))}
            append_jsonl(out/'responses.jsonl',{'case':row['case'],'tokenizations':tokenizations,'request':request,'response':response,'result':result})
            results.append(result);print(result,flush=True)
    write_json(out/'summary.json',{'status':'complete','results':[x for x in results if x['model']==root]})
    publish([out,'research/scored-budget.jsonl'],f'Verify native final-state compatibility for {root}')
write_json('research/native-state-compatibility.json',{'utc':utc_now(),'git':git_state(),'results':results,
            'scope':'Two calibration contexts per architecture; not a repair or a guarantee on all contexts.'})
