"""Durable native forwards/generation using immutable exact-token request files."""
import argparse,fcntl,json,subprocess
from pathlib import Path
from weightlab.budget import reserve,settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.resources import monitor
from weightlab.runtime import runtime_libraries
from portfolio_batch import publish
from p3_batch import CAPS,stage_used

def run(model,requests,label,stage,config=None):
 out=Path(f'research/p3/native-{stage}-{label}');out.mkdir(parents=True,exist_ok=False)
 binary=str(Path((config or {}).get('client_binary','local/weightlab-native-v4')).resolve());library='/home/akmessi/llama-vulkan-build/bin'
 if file_sha256(model['path'])!=model['sha256']:raise ValueError('model hash')
 cfg={'model':str(Path(model['path']).resolve()),'library_dir':library,'context':2048,'gpu_layers':99,
      'callback':True,'capture':True,'filter':'result_norm|component_down-[0-9]+'}
 cfg.update(config or {})
 write_json(out/'config.json',cfg)
 write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':model,'requests_sha256':digest(requests),
  'binary':binary,'binary_sha256':file_sha256(binary),'native_source_sha256':file_sha256((config or {}).get('client_source','native/trace_v4.cpp')),
  'runtime_version':subprocess.check_output([library+'/llama-server','--version'],stderr=subprocess.STDOUT,text=True),
  'libraries':runtime_libraries(library+'/llama-server'),'stage':stage,'config':cfg})
 write_json(out/'requests.json',requests)
 records=[];error=None;proc=None
 with Path('local/execution.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   with (out/'native.log').open('w') as log:
    proc=subprocess.Popen([binary,str(out/'config.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
    with monitor(proc.pid) as metrics:
     ready=proc.stdout.readline()
     if not ready:raise RuntimeError('native startup failed')
     write_json(out/'ready.json',json.loads(ready))
     for case in requests:
      req={**case,'trace_dir':str((Path('local/p3-native')/stage/label/case['id']).resolve())}
      prompt_bound=len(case['tokens']) if 'tokens' in case else cfg['context']
      score_id=f'p3-native-scored/{stage}/{label}/{case["id"]}'
      score_charges={}
      for line in Path('research/scored-budget.jsonl').read_text().splitlines():
       item=json.loads(line)
       if item['request_id'].startswith(f'p3-native-scored/{stage}/'):score_charges[item['request_id']]=item['tokens']
      score_caps={'repair':100000,'adapter':25000,'measurement':50000,'reserve':25000,'precision':50000,'clarify':50000}
      if sum(score_charges.values())+prompt_bound>score_caps[stage]:raise RuntimeError('Native stage scored-token ceiling')
      reserve(score_id,prompt_bound,1000000,'research/scored-budget.jsonl')
      maximum=case.get('max_tokens',0);generation_id=f'p3-native/{stage}/{label}/{case["id"]}' if maximum else None
      if maximum:
       if stage_used(stage)+maximum>CAPS[stage]:raise RuntimeError('Native stage generated-token ceiling')
       reserve(generation_id,maximum,1800000)
      append_jsonl(out/'events.jsonl',{'event':'request','request':req,'scored_request_id':score_id,'generated_request_id':generation_id,'utc':utc_now()})
      proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush()
      line=proc.stdout.readline()
      if not line:raise RuntimeError('native request process exited')
      response=json.loads(line)
      append_jsonl(out/'events.jsonl',{'event':'response','response':response,'scored_request_id':score_id,'generated_request_id':generation_id})
      if response['status']!='complete':raise RuntimeError(response.get('error'))
      if response['prompt_tokens']>prompt_bound:raise ValueError('diagnostic reservation underbound')
      settle(score_id,response['prompt_tokens'],'research/scored-budget.jsonl')
      if maximum:settle(generation_id,response['generated_tokens'])
      hashes={k:file_sha256(v['path']) for k,v in response['captures'].items()}
      row={'case':case,'request':req,'raw_response':response,'array_hashes':hashes,
           'logits_sha256':file_sha256(response['logits_path']),'scored_request_id':score_id,'generated_request_id':generation_id}
      append_jsonl(out/'responses.jsonl',row);records.append(row)
      if maximum:append_jsonl('research/p3-native-generation.jsonl',{'request_id':generation_id,'tokens':response['generated_tokens'],
          'evidence':str(out/'responses.jsonl'),'case_id':case['id'],'row_sha256':digest(row),'stage':stage})
      print(label,case['id'],'scored',response['prompt_tokens'],'generated',response['generated_tokens'],flush=True)
    write_json(out/'resources.json',metrics)
   write_json(out/'summary.json',{'status':'complete','requests':len(records),'generated_tokens':sum(r['raw_response']['generated_tokens'] for r in records),
                 'scored_tokens':sum(r['raw_response']['prompt_tokens'] for r in records)})
  except BaseException as exc:
   error=exc;write_json(out/'summary.json',{'status':'failed','error':repr(exc),'completed_responses':len(records)})
  finally:
   if proc is not None and proc.poll() is None:
    proc.stdin.close()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
 paths=[out,'research/scored-budget.jsonl','research/budget.jsonl','research/p3-native-generation.jsonl']
 publish([p for p in paths if Path(p).exists()],f'Record native P3 {stage} {label}')
 if error:raise error
 return records

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('requests');p.add_argument('model');p.add_argument('label');p.add_argument('stage');p.add_argument('--config',default='configs/p3_repair.json');a=p.parse_args()
 config=json.loads(Path(a.config).read_text());run(config['models'][a.model],json.loads(Path(a.requests).read_text()),a.label,a.stage)
