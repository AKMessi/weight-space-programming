"""Export fixed analytic candidates, screen development, freeze before confirmation."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.components import coefficients
from weightlab.edit import tensor_hash
from weightlab.evidence import digest,file_sha256,git_state,utc_now,write_json
from p3_batch import run_one
from portfolio_batch import publish
BASE=json.loads(Path('configs/p3_repair.json').read_text());root=Path('research/p3/repair-search-v1');configpath=Path('configs/p3_repair_candidates.json')

def prepare():
 if not json.loads(Path('research/p3/repair-causal-feasibility-v1.json').read_text())['open_permanent_search']:raise RuntimeError('No causal rescue')
 root.mkdir(exist_ok=False,parents=True);base=BASE['models']['fable_float'];reader=GGUFReader(base['path']);ts={t.name:t for t in reader.tensors}
 if file_sha256(base['path'])!=base['sha256']:raise ValueError('Original identity')
 weights={l:ts[f'blk.{l}.ffn_down.weight'].data.astype(np.float32) for l in range(14)}
 methods,directions,norm,selection=coefficients(weights,np.load('artifacts/p3/repair-cpu-directions.npz'))
 write_json(root/'sites.json',{'utc':utc_now(),'arrays_sha256':file_sha256('artifacts/p3/repair-cpu-directions.npz'),'all_down_norm':norm,**selection})
 config=copy.deepcopy(BASE)
 for method in methods:
  for dose in [.005,.01,.025,.05]:
   key=method+'_'+str(dose);parts=methods[method];cnorm=np.sqrt(sum(np.sum(c*c) for _,c in parts.values()));scale=norm*dose/cnorm
   output=Path('artifacts/p3/repair')/(key+'.gguf');output.parent.mkdir(parents=True,exist_ok=True)
   if output.exists():raise FileExistsError(output)
   shutil.copyfile(base['path'],output);target=GGUFReader(output,mode='r+');dest={t.name:t for t in target.tensors};changes=[];changed=0;actual_norm2=0.
   for layer,(cols,c) in parts.items():
    name=f'blk.{layer}.ffn_down.weight';old=ts[name].data;new=(old[:,cols].astype(np.float64)+directions[layer][:,None]*(c*scale)[None,:]).astype(np.float16)
    if not np.isfinite(new).all():raise ValueError('Nonfinite cast')
    dest[name].data[:,cols]=new;delta=new.astype(float)-old[:,cols].astype(float);changed+=int(np.count_nonzero(delta));actual_norm2+=float(np.sum(delta*delta))
    changes.append({'tensor':name,'columns':cols.tolist(),'old_sha256':tensor_hash(old),'edited_sha256':tensor_hash(dest[name].data)})
   target.data.flush();touched={r['tensor'] for r in changes}
   for name,t in ts.items():
    if name not in touched and tensor_hash(t.data)!=tensor_hash(dest[name].data):raise ValueError('Unexpected untouched change')
    if name in touched:
     layer=int(name.split('.')[1]);mask=np.ones(t.data.shape[1],bool);mask[parts[layer][0]]=False
     if not np.array_equal(t.data[:,mask],dest[name].data[:,mask]):raise ValueError('Unexpected unselected column change')
   manifest=root/(key+'.json');sha=file_sha256(output)
   write_json(manifest,{'utc':utc_now(),'git':git_state(),'base':base,'output':str(output),'sha256':sha,'method':method,'requested_dose':dose,'actual_relative_norm':np.sqrt(actual_norm2)/norm,'changed_values':changed,'tensors':changes,'untouched_tensor_count':len(ts)-len(touched),'status':'verified'})
   config['models'][key]={**base,'path':str(output),'sha256':sha,'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
 if file_sha256(base['path'])!=base['sha256']:raise ValueError('Original changed')
 write_json(configpath,config);publish([root,configpath],'Export twelve fixed sparse MLP weight candidates')

def rows(label,key):return list(map(json.loads,Path(f'runs/p3-repair-{label}-{key}-s17/responses.jsonl').read_text().splitlines()))
def counts(data):return {'plain':sum(r['grading']['success'] for r in data if r['case']['response_mode']=='plain'),'fenced':sum(r['grading']['success'] for r in data if r['case']['response_mode']=='fenced'),'payload':sum(r['grading']['payload_correct'] for r in data)}
def search():
 cfg=json.loads(configpath.read_text());keys=[k for k in cfg['models'] if k not in BASE['models']]
 for key in keys:run_one(cfg,key,'datasets/p3/switch-screen.json','candidate-screen','repair',96)
 baseline=counts([r for r in rows('switch-dev-original','fable_float') if r['case']['id'] in {c['id'] for c in json.loads(Path('datasets/p3/switch-screen.json').read_text())['cases']}])
 screened=[{'key':key,**counts(rows('candidate-screen',key))} for key in keys]
 ordered=sorted(screened,key=lambda r:(-(r['plain']+r['fenced']),-r['payload'],r['key']))[:3]
 shortlist=root/'shortlist.json';write_json(shortlist,{'utc':utc_now(),'baseline':baseline,'screened':screened,'advance':ordered});publish([shortlist],'Freeze three full-development repair candidates')
 for r in ordered:run_one(cfg,r['key'],'datasets/p3/switch-development.json','candidate-dev','repair',96)
 basecounts=counts(rows('switch-dev-original','fable_float'));eligible=[]
 for r in ordered:
  score=counts(rows('candidate-dev',r['key']));meta=json.loads((root/(r['key']+'.json')).read_text())
  if score['plain']>basecounts['plain'] and score['fenced']>=basecounts['fenced'] and score['payload']>=basecounts['payload']:eligible.append({'key':r['key'],'counts':score,'changed':meta['changed_values'],'dose':meta['actual_relative_norm']})
 eligible.sort(key=lambda r:(-r['counts']['plain'],r['changed'],r['dose'],r['key']))
 result={'utc':utc_now(),'baseline':basecounts,'eligible':eligible,'selected':eligible[0]['key'] if eligible else None,'reason':'Improve plain with no observed fenced/content decline, then changed-values/dose ties; development only'}
 write_json(root/'selection.json',result);publish([root/'selection.json'],'Freeze repair selection before independent confirmation')
 print(json.dumps(result),flush=True)

def confirm():
 cfg=json.loads(configpath.read_text());selection=json.loads((root/'selection.json').read_text());key=selection['selected']
 if not key:return
 for label,suite in [('confirmation-original','switch-confirmation.json'),('confirmation-strong','switch-confirmation-strong.json'),('confirmation-grammar','switch-confirmation-grammar.json')]:run_one(cfg,'fable_float','datasets/p3/'+suite,label,'repair',96)
 run_one(cfg,key,'datasets/p3/switch-confirmation.json','confirmation-patch','repair',96)
 for k in ['fable_float',key]:run_one(cfg,k,'datasets/p3/fable-protection.json','repair-protection','repair',96)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','search','confirm']);a=p.parse_args();globals()[a.phase]()
