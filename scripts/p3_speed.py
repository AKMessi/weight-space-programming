"""Exact-token, cache-disabled, bounded SSE timing and separate operator profiles."""
import argparse,fcntl,json,random,subprocess,time,urllib.request
from pathlib import Path
from weightlab.budget import reserve,settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json,runtime_libraries
from p3_batch import stage_used,CAPS
from portfolio_batch import publish
BASE=json.loads(Path('configs/fable_speed_followup.json').read_text())
KEYS=['fable_float','fable_Q8_0','fable_Q4_0','fable_Q4_K_M']
ROOT=Path('research/p3/speed-v1')

def freeze():
 out=ROOT/'tokenization';out.mkdir(parents=True,exist_ok=False);prompts={}
 for family,key in [('fable','fable_float'),('qwen','qwen_float')]:
  path=out/family;path.mkdir()
  with server(BASE['runtime']['binary'],BASE['models'][key]['path'],path,{**BASE['runtime'],'no_warmup':True}) as (url,cmd,proc):
   req={'content':'Continue this detailed archival story in ordinary prose. '+('The robot catalogued a garden and numbered every flower in a ledger. '*500),'add_special':True,'parse_special':True}
   raw=request_json(url+'/tokenize',req);tokens=raw['tokens']
   if len(tokens)<1536:raise ValueError('Insufficient tokens')
   write_json(path/'record.json',{'utc':utc_now(),'command':cmd,'model':BASE['models'][key],'request':req,'response':raw})
   prompts[family]={str(n):tokens[:n] for n in [128,512,1536]}
 write_json('datasets/p3/speed-prefixes.json',prompts)
 rng=random.Random(8917);schedule=[]
 for session in range(3):
  for backend in (['vulkan','cpu'] if session%2==0 else ['cpu','vulkan']):
   order=KEYS.copy();rng.shuffle(order)
   schedule.extend({'model':k,'session':session,'backend':backend} for k in order)
 schedule.extend({'model':k,'session':0,'backend':'vulkan'} for k in ['qwen_float','qwen_Q8_0'])
 write_json(ROOT/'schedule.json',schedule)
 publish([out,'datasets/p3/speed-prefixes.json',ROOT/'schedule.json'],'Freeze exact speed prompts and randomized three-session schedule')

def condition(key,backend,session,profile=False):
 label=f'{"profile" if profile else "timing"}-{backend}-{session}-{key}';out=ROOT/label
 if out.exists():
  if json.loads((out/'summary.json').read_text())['status']=='complete':return
  raise FileExistsError('Use a versioned retry after failure: '+str(out))
 out.mkdir(parents=True);spec=BASE['models'][key]
 if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Model checksum')
 cfg={**BASE['runtime'],'no_warmup':True,'cpu_only':backend=='cpu','gpu_layers':0 if backend=='cpu' else 99}
 if profile:cfg['environment']={'GGML_VK_PERF_LOGGER':'1','GGML_VK_PERF_LOGGER_FREQUENCY':'999999'}
 prompts=json.loads(Path('datasets/p3/speed-prefixes.json').read_text())['qwen' if key.startswith('qwen') else 'fable']
 write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':spec,'runtime':cfg,'binary_sha256':file_sha256(cfg['binary']),
  'version':subprocess.check_output([cfg['binary'],'--version'],stderr=subprocess.STDOUT,text=True),'libraries':runtime_libraries(cfg['binary']),
  'prefixes_sha256':file_sha256('datasets/p3/speed-prefixes.json'),'condition':label,'stage':'speed','profile_overhead':'separate from timings','warmups_included_in_budget':True})
 records=[];error=None
 try:
  with Path('local/execution.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with server(cfg['binary'],spec['path'],out,cfg) as (url,cmd,proc):
    write_json(out/'server.json',{'command':cmd,'props':request_json(url+'/props')})
    for length in ([128] if profile else [128,512,1536]):
     for rep in range(2 if profile else 8):
      rid=f'p3-custom/speed/{label}/{length}-{rep}';bound=16
      if stage_used('speed')+bound>CAPS['speed']:raise RuntimeError('Speed stage cap')
      reserve(rid,bound,1800000)
      req={'prompt':prompts[str(length)],'n_predict':bound,'temperature':0,'seed':17,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':False,'stream':True}
      append_jsonl(out/'events.jsonl',{'event':'request','utc':utc_now(),'request_id':rid,'request':req})
      chunks=[];first=None;start=time.perf_counter()
      http=urllib.request.Request(url+'/completion',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
      with urllib.request.urlopen(http,timeout=240) as response:
       for line in response:
        if not line.startswith(b'data: '):continue
        payload=line[6:].strip()
        if payload==b'[DONE]':continue
        raw=json.loads(payload);now=time.perf_counter()
        if first is None and raw.get('content'):first=now-start
        chunk={'elapsed_seconds':now-start,'raw':raw};chunks.append(chunk)
        append_jsonl(out/'events.jsonl',{'event':'chunk','request_id':rid,**chunk})
      elapsed=time.perf_counter()-start;final=chunks[-1]['raw'];usage=final.get('timings',{});actual=final.get('tokens_predicted',usage.get('predicted_n'))
      if not isinstance(actual,int):raise ValueError('No actual token count')
      settle(rid,actual)
      row={'request_id':rid,'request':req,'chunks':chunks,'seconds':elapsed,'ttft_seconds':first,'warmup':rep==0,'profile':profile,'model':key,'backend':backend,'session':session,'prompt_length':length,'generated_tokens':actual,'timings':usage,
           'valid_timing':actual==16 and usage.get('prompt_n')==length and usage.get('cache_n')==0}
      append_jsonl(out/'responses.jsonl',row);records.append(row)
      append_jsonl('research/p3-custom-generation.jsonl',{'request_id':rid,'tokens':actual,'evidence':str(out/'responses.jsonl'),'row_sha256':digest(row),'stage':'speed'})
      print(label,length,rep,'tokens',actual,'prefill',usage.get('prompt_n'),'cache',usage.get('cache_n'),'tps',usage.get('predicted_per_second'),flush=True)
  write_json(out/'summary.json',{'status':'complete','requests':len(records),'generated_tokens':sum(r['generated_tokens'] for r in records),'valid_timings':sum(r['valid_timing'] for r in records),'server_log_sha256':file_sha256(out/'server.log')})
 except BaseException as exc:
  error=exc;write_json(out/'summary.json',{'status':'failed','error':repr(exc),'requests':len(records)})
 publish([out,'research/budget.jsonl','research/p3-custom-generation.jsonl'],'Record P3 speed '+label)
 if error:raise error

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','profile','bench','hybrids']);a=p.parse_args()
 if a.phase=='freeze':freeze()
 elif a.phase=='profile':
  for key in KEYS:condition(key,'vulkan',0,True)
 elif a.phase=='bench':
  for c in json.loads((ROOT/'schedule.json').read_text()):condition(c['model'],c['backend'],c['session'])
 else:
  BASE['models'].update(json.loads(Path('configs/p3_hybrids.json').read_text())['models'])
  for session in range(3):
   for key in sorted(k for k in BASE['models'] if k.startswith('hybrid_')):condition(key,'vulkan',session)
if __name__=='__main__':main()
