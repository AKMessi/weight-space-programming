"""Small interleaved same-time format controls and separate frozen hybrid profiles."""
import fcntl,json,random,subprocess,time,urllib.request
from pathlib import Path
from weightlab.budget import reserve,settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json,runtime_libraries
from p3_batch import stage_used,CAPS
from portfolio_batch import publish
ROOT=Path('research/p3/speed-followup-v1');config=json.loads(Path('configs/fable_speed_followup.json').read_text());config['models'].update(json.loads(Path('configs/p3_hybrids.json').read_text())['models']);keys=['fable_float','fable_Q4_0','hybrid_head','hybrid_head_ffn','hybrid_head_shortconv']
rng=random.Random(1927);schedule=[]
for block in range(3):
 order=keys.copy();rng.shuffle(order);schedule.extend({'block':block,'model':k,'profile':False} for k in order)
schedule.extend({'block':0,'model':k,'profile':True} for k in keys if k.startswith('hybrid_'))
ROOT.mkdir(parents=True,exist_ok=True)
if not (ROOT/'schedule.json').exists():write_json(ROOT/'schedule.json',schedule);publish([ROOT/'schedule.json'],'Freeze interleaved timing order before follow-up outcomes')
for item in schedule:
 key=item['model'];profile=item['profile'];label=f'followup-{"profile" if profile else "timing"}-{item["block"]}-{key}';out=ROOT/label
 if out.exists():
  if json.loads((out/'summary.json').read_text())['status']=='complete':continue
  raise FileExistsError('Version failed follow-up instead of overwriting')
 out.mkdir();spec=config['models'][key]
 if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Model identity')
 cfg={**config['runtime'],'no_warmup':True};prefix=json.loads(Path('datasets/p3/speed-prefixes.json').read_text())['fable']['128']
 if profile:cfg['environment']={'GGML_VK_PERF_LOGGER':'1','GGML_VK_PERF_LOGGER_FREQUENCY':'999999'}
 write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':spec,'runtime':cfg,'binary_sha256':file_sha256(cfg['binary']),'libraries':runtime_libraries(cfg['binary']),'runtime_version':subprocess.check_output([cfg['binary'],'--version'],stderr=subprocess.STDOUT,text=True),'condition':item,'design':'2profile or1warm+3measured128-prefix16decode','prefixes_sha256':file_sha256('datasets/p3/speed-prefixes.json')})
 records=[];error=None
 try:
  with Path('local/execution.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with server(cfg['binary'],spec['path'],out,cfg) as (url,cmd,proc):
    write_json(out/'server.json',{'command':cmd,'props':request_json(url+'/props')})
    for rep in range(2 if profile else 4):
     rid=f'p3-custom/speed/{label}/{rep}'
     if stage_used('speed')+16>CAPS['speed']:raise RuntimeError('Speed cap')
     reserve(rid,16,1800000);req={'prompt':prefix,'n_predict':16,'temperature':0,'seed':17,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':False,'stream':True};append_jsonl(out/'events.jsonl',{'event':'request','utc':utc_now(),'request_id':rid,'request':req});chunks=[];first=None;start=time.perf_counter()
     http=urllib.request.Request(url+'/completion',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
     with urllib.request.urlopen(http,timeout=240) as response:
      for line in response:
       if not line.startswith(b'data: '):continue
       payload=line[6:].strip()
       if payload==b'[DONE]':continue
       raw=json.loads(payload);now=time.perf_counter()
       if first is None and raw.get('content'):first=now-start
       chunk={'elapsed_seconds':now-start,'raw':raw};chunks.append(chunk);append_jsonl(out/'events.jsonl',{'event':'chunk','request_id':rid,**chunk})
     elapsed=time.perf_counter()-start;final=chunks[-1]['raw'];timings=final.get('timings',{});actual=final.get('tokens_predicted',timings.get('predicted_n'))
     if not isinstance(actual,int):raise ValueError('No actual token count')
     settle(rid,actual);row={'request_id':rid,'request':req,'chunks':chunks,'generated_tokens':actual,'model':key,'block':item['block'],'profile':profile,'warmup':rep==0,'seconds':elapsed,'ttft_seconds':first,'timings':timings,'valid_timing':actual==16 and timings.get('prompt_n')==128 and timings.get('cache_n')==0};append_jsonl(out/'responses.jsonl',row);records.append(row);append_jsonl('research/p3-custom-generation.jsonl',{'request_id':rid,'tokens':actual,'evidence':str(out/'responses.jsonl'),'row_sha256':digest(row),'stage':'speed'});print(label,rep,timings.get('predicted_per_second'),flush=True)
  write_json(out/'summary.json',{'status':'complete','requests':len(records),'generated_tokens':sum(r['generated_tokens'] for r in records),'valid_timings':sum(r['valid_timing'] for r in records)})
 except BaseException as exc:error=exc;write_json(out/'summary.json',{'status':'failed','error':repr(exc),'requests':len(records)})
 publish([out,'research/budget.jsonl','research/p3-custom-generation.jsonl'],'Record '+label)
 if error:raise error
