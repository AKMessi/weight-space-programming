"""Verified architecture-aware development contrasts and bounded semantic interventions."""
import argparse,json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from weightlab.evidence import append_jsonl,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json
from weightlab.grading import first_json,typed_equal,grade
from p3_clarification import metrics
from p3_native_run import run
from portfolio_batch import publish
C=json.loads(Path('configs/p3_clarification.json').read_text());ROOT=Path('research/p3/clarify-edit-v1')
def data(key,arm):return list(map(json.loads,Path(f'runs/p3-clarify-evidence-dev-{arm}-{key}-s17/responses.jsonl').read_text().splitlines()))
def paths(key):return ROOT/key
SET={'client_binary':'local/weightlab-native-v6','client_source':'native/trace_v6.cpp','filter':'result_norm|component_down-[0-9]+|component_post-[0-9]+'}
def freeze(key):
 out=paths(key);out.mkdir(parents=True,exist_ok=False);scores={a:metrics(data(key,a)) for a in ['strong','combined']};eligible=[a for a,s in scores.items() if s['supported_correct']/s['supported']>=.8 and s['update_correct']/s['cases']>=.8]
 if not eligible:raise RuntimeError('No competent controller')
 arm=min(eligible,key=lambda a:(-scores[a]['joint_semantic'],scores[a]['unsupported_nonnull'],a!='combined'));rows=data(key,arm);byid={r['case']['id']:r for r in rows};bad=[];records=[]
 for row in rows:
  parsed=first_json(row['steps'][0]['content']);case=row['case']
  if case['evidence']!='complete' and isinstance(parsed,dict) and parsed.get('value') is not None and case['record_id'] not in records:bad.append(row);records.append(case['record_id'])
  if len(records)==4:break
 if len(records)<4:raise RuntimeError('Fewer than4natural failing records: requires explicit protocol amendment')
 selection={'utc':utc_now(),'git':git_state(),'model':key,'controller':arm,'controller_scores':scores,'record_ids':records,'failure_case_ids':[r['case']['id'] for r in bad],'scope':'Development only; no confirmation outcomes'};write_json(out/'selection.json',selection)
 requests=[];parity=[];token_dir=out/'tokenization';token_dir.mkdir()
 with server(C['runtime']['binary'],C['models'][key]['path'],token_dir,{**C['runtime'],'no_warmup':True}) as (url,command,proc):
  write_json(token_dir/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':C['models'][key],'command':command,'purpose':'tokenization only, no issued diagnostic/generation request; explicit no-warmup'})
  def tokenize(text,expected=None,policy=None):
   options=[]
   for add in ([policy] if policy is not None else [True,False]):
    req={'content':text,'add_special':add,'parse_special':True};append_jsonl(token_dir/'events.jsonl',{'event':'request','request':req,'utc':utc_now()});response=request_json(url+'/tokenize',req);append_jsonl(token_dir/'events.jsonl',{'event':'response','response':response});options.append((add,response['tokens']))
   match=[r for r in options if expected is None or len(r[1])==expected]
   if not match:raise ValueError('Exact tokenization mismatch')
   if len(match[0][1])>512:raise ValueError('Prefix bound512')
   return match[0]
  for record in records:
   for row in rows:
    if row['case']['record_id']!=record:continue
    case=row['case'];text=row['steps'][0]['rendered_prompt']['prompt'];policy,ids=tokenize(text,row['steps'][0]['usage']['prompt_tokens'])
    requests.append({'id':case['id']+'-key','role':'key','kind':case['evidence'],'record_id':record,'prompt':text,'tokens':ids,'max_tokens':0,'expected':case['expected'],'source_case_id':case['id'],'baseline_correct':typed_equal(first_json(row['steps'][0]['content']),case['expected'])})
    if case['evidence']=='complete':
     step=row['steps'][1];text=step['rendered_prompt']['prompt'];_,ids=tokenize(text,step['usage']['prompt_tokens']);requests.append({'id':case['id']+'-update','role':'update','kind':'update','record_id':record,'prompt':text,'tokens':ids,'max_tokens':0,'expected':case['followups'][0]['expected'],'source_case_id':case['id'],'baseline_correct':typed_equal(first_json(step['content']),case['followups'][0]['expected'])})
   row=next(r for r in bad if r['case']['record_id']==record);case=row['case'];text=row['steps'][0]['rendered_prompt']['prompt'];policy,_=tokenize(text,row['steps'][0]['usage']['prompt_tokens']);value=first_json(row['steps'][0]['content'])['value']
   for role,prefix in [('wanted','{"value":null,"ask":'),('unwanted','{"value":'+json.dumps(value,separators=(',',':'))+',"ask":')]:
    _,ids=tokenize(text+prefix,policy=policy);requests.append({'id':case['id']+'-'+role,'role':role,'kind':case['evidence'],'record_id':record,'prompt':text+prefix,'tokens':ids,'max_tokens':0,'source_case_id':case['id']})
  for row in data(key,'strong')[:4]:
   step=row['steps'][0];text=step['rendered_prompt']['prompt'];_,ids=tokenize(text,step['usage']['prompt_tokens']);parity.append({'id':row['case']['id'],'prompt':text,'tokens':ids,'max_tokens':48,'expected_server_content':step['content']})
 write_json(out/'trace-requests.json',requests);write_json(out/'parity-requests.json',parity);publish([out],'Freeze '+key+' evidence contrasts, protected keys and parity requests')

def native_rows(key,label):return list(map(json.loads,Path(f'research/p3/native-clarify-{key}-{label}/responses.jsonl').read_text().splitlines()))
def analyze_rows(key,rows,cpu=False):
 reader=GGUFReader(C['models'][key]['path']);weights={t.name:t for t in reader.tensors};layers=sorted(int(n.split('.')[1]) for n in weights if n.endswith('.ffn_down.weight'));eps_fields=[f.contents() for n,f in reader.fields.items() if n.endswith('layer_norm_rms_epsilon')];eps=float(eps_fields[0]) if eps_fields else 1e-6;checks=[]
 for l in layers:
  w=dequantize(weights[f'blk.{l}.ffn_down.weight'].data,weights[f'blk.{l}.ffn_down.weight'].tensor_type).astype(np.float32)
  def mat(kind):return np.stack([np.fromfile(r['raw_response']['captures'][f'component_{kind}-{l}']['path'],np.float32) for r in rows])
  h=mat('input');actual=mat('down');operand=h.astype(np.float16).astype(np.float32) if cpu else h;pred=operand@w.T;error=actual-pred;maximum=float(abs(error).max());relative=float(np.linalg.norm(error)/max(np.linalg.norm(actual),1e-6));passed=maximum<=.05 and relative<=.001;post=None
  normname=f'blk.{l}.post_ffw_norm.weight'
  if normname in weights:
   observed=mat('post');expected=actual/np.sqrt(np.mean(actual*actual,axis=1,keepdims=True)+eps)*weights[normname].data;err=observed-expected;pm=float(abs(err).max());pr=float(np.linalg.norm(err)/max(np.linalg.norm(observed),1e-6));post={'max_error':pm,'relative_error':pr,'passed':pm<=.05 and pr<=.001};passed=passed and post['passed']
  checks.append({'layer':l,'max_error':maximum,'relative_error':relative,'postnorm':post,'passed':bool(passed)})
 return checks

def gate(key):
 requests=json.loads((paths(key)/'trace-requests.json').read_text())[:4];rows=run(C['models'][key],requests,key+'-gpu-gate','clarify',SET);checks=analyze_rows(key,rows);backend='gpu'
 if key=='qwen_float' and not all(c['passed'] for c in checks):
  rows=run(C['models'][key],requests,key+'-cpu-gate','clarify',{**SET,'gpu_layers':0,'op_offload':False,'offload_kqv':False});cpu=analyze_rows(key,rows,True);backend='cpu';report={'gpu':checks,'cpu':cpu,'backend':backend,'eligible_layers':[c['layer'] for c in cpu if c['passed']]}
 else:report={'gpu':checks,'backend':backend,'eligible_layers':[c['layer'] for c in checks if c['passed']]}
 out=paths(key)/'numeric-gate.json';write_json(out,report);publish([out],'Gate '+key+' raw and postnorm component algebra');print(json.dumps(report),flush=True)

def trace(key):
 gate=json.loads((paths(key)/'numeric-gate.json').read_text())
 if not gate['eligible_layers']:return
 settings={**SET}
 if gate['backend']=='cpu':settings.update(gpu_layers=0,op_offload=False,offload_kqv=False)
 run(C['models'][key],json.loads((paths(key)/'trace-requests.json').read_text()),key+'-component-trace','clarify',settings)
 rows=native_rows(key,'component-trace');checks=analyze_rows(key,rows,gate['backend']=='cpu');eligible=[c['layer'] for c in checks if c['passed'] and c['layer'] in gate['eligible_layers']];ratios=[];all_arrays={}
 for l in eligible:
  def mat(role,kind):return np.stack([np.fromfile(r['raw_response']['captures'][f'component_{kind}-{l}']['path'],np.float32) for r in rows if r['case']['role']==role])
  raw=mat('wanted','down');bad=mat('unwanted','down');v=(raw-bad).mean(0);postkey=f'component_post-{l}'
  if postkey in rows[0]['raw_response']['captures']:pos=mat('wanted','post');neg=mat('unwanted','post')
  else:pos=raw;neg=bad
  ratio=float(np.linalg.norm((pos-neg).mean(0))/max(np.mean(np.linalg.norm(pos,axis=1)),1e-9));ratios.append({'layer':l,'post_or_raw_contrast_ratio':ratio,'raw_contrast_norm':float(np.linalg.norm(v))});all_arrays[l]={'v':v,'h_all':np.stack([np.fromfile(r['raw_response']['captures'][f'component_input-{l}']['path'],np.float32) for r in rows]),'row_roles':np.array([r['case']['role'] for r in rows])}
 selected=min(ratios,key=lambda r:(-r['post_or_raw_contrast_ratio'],r['layer'])) if ratios else None;out=paths(key)/'direction-selection.json'
 if selected:
  cache=Path('artifacts/p3/clarify')/(key+'-direction.npz');cache.parent.mkdir(parents=True,exist_ok=True)
  if cache.exists():raise FileExistsError(cache)
  np.savez(cache,**all_arrays[selected['layer']]);sha=file_sha256(cache)
 else:sha=None
 write_json(out,{'utc':utc_now(),'numeric_checks':checks,'eligible_layers':eligible,'ratios':ratios,'selected':selected,'arrays_sha256':sha});publish([out],'Freeze '+key+' verified evidence-sensitive direction');print(json.dumps(selected),flush=True)

def causal(key):
 root=paths(key);selection=json.loads((root/'selection.json').read_text());direction=json.loads((root/'direction-selection.json').read_text()) if (root/'direction-selection.json').exists() else {'selected':None}
 if not direction['selected']:
  write_json(root/'causal-gate.json',{'status':'closed','reason':'No verified component layer','eligible_for_permanent':False});publish([root/'causal-gate.json'],'Close unverified '+key+' component route');return
 parity=run(C['models'][key],json.loads((root/'parity-requests.json').read_text()),key+'-semantic-parity','clarify',{**SET,'capture':False,'callback':False})
 parity_checks=[{'id':r['case']['id'],'exact':r['case']['expected_server_content']==r['raw_response']['completion']} for r in parity]
 if not all(r['exact'] for r in parity_checks):
  write_json(root/'causal-gate.json',{'status':'closed','reason':'Native/server greedy parity failed','parity':parity_checks,'eligible_for_permanent':False});publish([root/'causal-gate.json'],'Preserve failed '+key+' native semantic parity');return
 layer=direction['selected']['layer'];v=np.load('artifacts/p3/clarify/'+key+'-direction.npz')['v'];rng=np.random.default_rng(811);random=rng.standard_normal(len(v));random*=float(np.linalg.norm(v)/np.linalg.norm(random));requests=json.loads((root/'trace-requests.json').read_text());panel=[r for r in requests if r['record_id'] in selection['record_ids'][:3] and r['role'] in ['key','update']];cases=[]
 for name,scale,vec in [('zero',0.,v),('half',.5,v),('one',1.,v),('two',2.,v),('random',1.,random)]:
  for q in panel:cases.append({**q,'id':q['id']+'-'+name,'max_tokens':48,'arm':name,'intervention':{'tensor':f'component_down-{layer}','scale':scale,'vector':vec.tolist()}})
 results=run(C['models'][key],cases,key+'-semantic-causal','clarify',{**SET,'capture':False});grades=[]
 for row in results:
  q=row['case'];g=grade({'grader':'json','expected':q['expected']},row['raw_response']['completion']);grades.append({'id':q['id'],'source_id':q['id'].rsplit('-',1)[0],'arm':q['arm'],'kind':q['kind'],'role':q['role'],'grading':g,'content':row['raw_response']['completion']})
 zero={r['source_id']:r for r in grades if r['arm']=='zero'};counts={}
 for arm in ['zero','half','one','two','random']:
  rs=[r for r in grades if r['arm']==arm];counts[arm]={'semantic':sum(r['grading']['payload_correct'] for r in rs),'rescued_unsupported':sum(r['kind'] in ['absent','conflicting'] and not zero[r['source_id']]['grading']['payload_correct'] and r['grading']['payload_correct'] for r in rs),'new_failures':sum(zero[r['source_id']]['grading']['payload_correct'] and not r['grading']['payload_correct'] for r in rs),'supported_success':sum(r['grading']['payload_correct'] for r in rs if r['kind'] in ['complete','update']),'supported_total':sum(r['kind'] in ['complete','update'] for r in rs)}
 good=[a for a in ['half','one','two'] if counts['zero']['supported_success']/counts['zero']['supported_total']>=.8 and counts[a]['rescued_unsupported']>counts['random']['rescued_unsupported'] and counts[a]['rescued_unsupported']>0 and counts[a]['new_failures']==0]
 write_json(root/'causal-gate.json',{'utc':utc_now(),'parity':parity_checks,'counts':counts,'records':grades,'useful_doses':good,'eligible_for_permanent':bool(good),'scope':'Development reversible feasibility with exact-site zero and random controls; native greedy and final schema-controlled decoding are distinct.'});publish([root/'causal-gate.json'],'Grade '+key+' evidence-sensitive causal feasibility');print(json.dumps(counts),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','gate','trace','causal']);p.add_argument('model',choices=['lfm3_qad','qwen_float']);a=p.parse_args();globals()[a.phase](a.model)
