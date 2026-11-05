"""Select an eligible subject, freeze exact-token development traces, verify algebra."""
import argparse,json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.evidence import append_jsonl,file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json
from p3_native_run import run
from portfolio_batch import publish

config=json.loads(Path('configs/p3_repair.json').read_text())
p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','analyze','parity']);a=p.parse_args()
selection_path=Path('research/p3/repair-subject-v1.json')
if a.phase=='freeze':
 if selection_path.exists():raise FileExistsError(selection_path)
 subjects=[]
 for key in ['qwen_float','fable_float']:
  root=Path(f'runs/p3-repair-switch-dev-original-{key}-s17')
  data=list(map(json.loads,(root/'responses.jsonl').read_text().splitlines()))
  plain=[r for r in data if r['case']['response_mode']=='plain']
  content=sum(r['grading']['payload_correct'] for r in data)/len(data)
  failure=1-sum(r['grading']['success'] for r in plain)/len(plain)
  subjects.append({'key':key,'content':content,'plain_failure':failure,'eligible':content>=.8 and failure>=.2})
 eligible=[s for s in subjects if s['eligible']]
 if not eligible:raise RuntimeError('No eligible switching subject; preserve this gate outcome')
 chosen=eligible[0]['key']
 write_json(selection_path,{'utc':utc_now(),'git':git_state(),'subjects':subjects,'chosen':chosen})
 publish([selection_path],'Freeze naturally eligible conditional-repair subject')
 root=Path(f'runs/p3-repair-switch-dev-original-{chosen}-s17')
 rows=list(map(json.loads,(root/'responses.jsonl').read_text().splitlines()))
 lookup={r['case']['id']:r for r in rows}
 out=Path('research/p3/repair-tokenization-v1');out.mkdir(parents=True,exist_ok=False)
 trace_requests=[];parity=[]
 with server(config['runtime']['binary'],config['models'][chosen]['path'],out,config['runtime']) as (url,command,proc):
  write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'command':command,'source_sha256':file_sha256(root/'responses.jsonl'),
            'status':'tokenization only; zero diagnostic forwards/generation'})
  def tokens(text,expected=None):
   options=[]
   for special in [True,False]:
    request={'content':text,'add_special':special,'parse_special':True}
    response=request_json(url+'/tokenize',request);options.append({'request':request,'response':response})
   matched=[x for x in options if expected is None or len(x['response']['tokens'])==expected]
   if not matched:raise ValueError('tokenization mismatch')
   append_jsonl(out/'events.jsonl',{'options':options,'chosen_add_special':matched[0]['request']['add_special']})
   return matched[0]['response']['tokens']
  for row in rows:
   if row['case']['response_mode']!='plain':continue
   partner=lookup[row['case']['id'].replace('-plain','-fenced')]
   prompt=row['steps'][0]['rendered_prompt']['prompt'];protected=partner['steps'][0]['rendered_prompt']['prompt']
   for branch,text in [('bare',prompt+'{'),('fence',prompt+'\x60\x60\x60json\n{'),('key_plain',prompt),('key_protect',protected)]:
    expected=row['usage']['prompt_tokens'] if branch=='key_plain' else partner['usage']['prompt_tokens'] if branch=='key_protect' else None
    ids=tokens(text,expected)
    if len(ids)>512:raise ValueError('trace prefix bound')
    trace_requests.append({'id':row['case']['id']+'-'+branch,'record_id':row['case']['id'],'branch':branch,'prompt':text,'tokens':ids,'max_tokens':0})
  for row in rows[:4]:
   prompt=row['steps'][0]['rendered_prompt']['prompt']
   parity.append({'id':row['case']['id'],'prompt':prompt,'tokens':tokens(prompt,row['usage']['prompt_tokens']),
                  'max_tokens':96,'expected_server_content':row['content'],'case':row['case']})
 write_json('datasets/p3/repair-trace-prefixes.json',trace_requests)
 write_json('datasets/p3/repair-native-parity.json',parity)
 publish([out,'datasets/p3/repair-trace-prefixes.json','datasets/p3/repair-native-parity.json'],'Freeze contrast, protected-input and native parity requests')
else:
 selection=json.loads(selection_path.read_text());key=selection['chosen']
 if a.phase=='run':
  run(config['models'][key],json.loads(Path('datasets/p3/repair-trace-prefixes.json').read_text()),'component-contrast','repair')
 elif a.phase=='parity':
  result=run(config['models'][key],json.loads(Path('datasets/p3/repair-native-parity.json').read_text()),'generation-parity','repair',{'capture':False,'callback':False})
  comparisons=[{'id':r['case']['id'],'content_equal':r['case']['expected_server_content']==r['raw_response']['completion']} for r in result]
  write_json('research/p3/repair-generation-parity-v1.json',{'utc':utc_now(),'comparisons':comparisons,'passed':all(r['content_equal'] for r in comparisons)})
  publish(['research/p3/repair-generation-parity-v1.json'],'Check native greedy generation against untouched server')
 else:
  path=Path('research/p3/native-repair-component-contrast/responses.jsonl')
  records=list(map(json.loads,path.read_text().splitlines()))
  branches={b:[r for r in records if r['case']['branch']==b] for b in ['bare','fence','key_plain','key_protect']}
  model=GGUFReader(config['models'][key]['path']);weights={t.name:t for t in model.tensors}
  layers=sorted(int(n.split('.')[1]) for n in weights if n.startswith('blk.') and n.endswith('.ffn_down.weight'))
  arrays={};checks=[];ratios=[]
  for layer in layers:
   def matrix(branch,kind):
    return np.stack([np.fromfile(r['raw_response']['captures'][f'component_{kind}-{layer}']['path'],np.float32) for r in branches[branch]])
   w=weights[f'blk.{layer}.ffn_down.weight'].data.astype(np.float32)
   hpos=matrix('bare','input');hneg=matrix('fence','input');hplain=matrix('key_plain','input');hprotect=matrix('key_protect','input')
   pos=matrix('bare','down');neg=matrix('fence','down');direction=(pos-neg).mean(axis=0)
   # Actual GPU projection uses the captured F32 operand; no guessed label.
   actual=pos[:4];predicted=hpos[:4]@w.T
   maximum=float(np.max(np.abs(actual-predicted)));relative=float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(actual),1e-6))
   checks.append({'layer':layer,'max_abs_error':maximum,'relative_error':relative,'passed':maximum<=.05 and relative<=.001})
   ratio=float(np.linalg.norm(direction)/max(np.mean(np.linalg.norm(pos,axis=1)),1e-9));ratios.append({'layer':layer,'ratio':ratio,'contrast_norm':float(np.linalg.norm(direction))})
   arrays[f'v{layer}']=direction;arrays[f'hplain{layer}']=hplain;arrays[f'hprotect{layer}']=hprotect
   arrays[f'hpos{layer}']=hpos;arrays[f'hneg{layer}']=hneg
  chosen_layer=min(ratios,key=lambda r:(-r['ratio'],r['layer']))
  cache=Path('artifacts/p3/repair-directions.npz');cache.parent.mkdir(parents=True,exist_ok=True)
  if cache.exists():raise FileExistsError(cache)
  np.savez(cache,**arrays)
  report={'utc':utc_now(),'git':git_state(),'source_sha256':file_sha256(path),'arrays_sha256':file_sha256(cache),'subject':key,
          'projection_checks':checks,'ratios':ratios,'chosen_layer':chosen_layer,'boundary_passed':all(c['passed'] for c in checks)}
  out=Path('research/p3/repair-component-analysis-v1.json')
  if out.exists():raise FileExistsError(out)
  write_json(out,report);publish([out],'Verify true FFN component algebra and freeze causal layer')
  print(json.dumps({'boundary_passed':report['boundary_passed'],'chosen_layer':chosen_layer}),flush=True)
