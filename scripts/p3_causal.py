"""Predeclared three-dose causal feasibility screen, CPU-derived directions on Vulkan."""
import json
from pathlib import Path
import numpy as np
from weightlab.evidence import file_sha256,utc_now,write_json
from weightlab.grading import grade
from p3_native_run import run
from portfolio_batch import publish
config=json.loads(Path('configs/p3_repair.json').read_text());analysis=json.loads(Path('research/p3/repair-cpu-analysis-v2.json').read_text())
if not analysis['boundary_passed']:raise RuntimeError('Component gate')
if not json.loads(Path('research/p3/repair-generation-parity-v1.json').read_text())['passed']:raise RuntimeError('Generation parity')
layer=analysis['chosen_layer']['layer'];v=np.load('artifacts/p3/repair-cpu-directions.npz')[f'v{layer}'].tolist()
prefixes=json.loads(Path('datasets/p3/repair-trace-prefixes.json').read_text());prefixes={r['id']:r for r in prefixes}
base=list(map(json.loads,Path('runs/p3-repair-switch-dev-original-fable_float-s17/responses.jsonl').read_text().splitlines()))[:16]
requests=[]
for dose in [.5,1.,2.]:
 for row in base:
  case=row['case'];plain=case['id'].replace('-fenced','-plain');branch='key_protect' if case['response_mode']=='fenced' else 'key_plain';prefix=prefixes[plain+'-'+branch]
  requests.append({'id':case['id']+'-dose'+str(dose),'prompt':prefix['prompt'],'tokens':prefix['tokens'],'max_tokens':96,'case':case,'baseline_grading':row['grading'],
   'intervention':{'tensor':f'component_down-{layer}','vector':v,'scale':dose}})
rows=run(config['models']['fable_float'],requests,'causal-cpu-direction-v1','repair',{'capture':False})
results=[]
for row in rows:
 request=row['case'];g=grade(request['case'],row['raw_response']['completion'])
 results.append({'id':request['id'],'dose':request['intervention']['scale'],'case':request['case'],'baseline':request['baseline_grading'],'grading':g,'rescued_plain':request['case']['response_mode']=='plain' and not request['baseline_grading']['success'] and g['success'] and g['payload_correct']})
out=Path('research/p3/repair-causal-feasibility-v1.json')
write_json(out,{'utc':utc_now(),'analysis_sha256':file_sha256('research/p3/repair-cpu-analysis-v2.json'),'layer':layer,'records':results,'open_permanent_search':any(r['rescued_plain'] for r in results),
 'dose_counts':{str(d):{'plain_success':sum(r['grading']['success'] for r in results if r['dose']==d and r['case']['response_mode']=='plain'),'fenced_success':sum(r['grading']['success'] for r in results if r['dose']==d and r['case']['response_mode']=='fenced'),'payload':sum(r['grading']['payload_correct'] for r in results if r['dose']==d),'rescues':sum(r['rescued_plain'] for r in results if r['dose']==d)} for d in [.5,1.,2.]}})
publish([out],'Grade predeclared component causal feasibility screen')
print(json.dumps(json.loads(out.read_text())['dose_counts']),flush=True)
