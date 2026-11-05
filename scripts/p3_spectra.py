"""Analyze Fable attention deltas in one matrix at a time; no model forwards."""
import json,re,time
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.precision import floating_tensor_values
from portfolio_batch import publish

config=json.loads(Path('configs/portfolio.json').read_text());target=config['models']['fable_f16']
variants={'aligned':config['models']['parent_uniform_bf16'],'f16':config['models']['parent_f16']}
p=json.loads(Path('research/patches/parent-bf16-values-f16.json').read_text())
variants['same_source_rounded']={**p['base'],'path':p['output'],'sha256':p['edited_file_sha256']}
out=Path('research/p3/fable-spectra-v1');out.mkdir(parents=True,exist_ok=False)
for spec in [target,*variants.values()]:
 if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('source identity')
write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'target':target,'bases':variants,'ranks':[4,8,16,32,64]})
targets={t.name:t for t in GGUFReader(target['path']).tensors};results=[]
try:
 for label,base in variants.items():
  parent={t.name:t for t in GGUFReader(base['path']).tensors}
  cache=Path('artifacts/p3/svd-cache')/label;cache.mkdir(parents=True,exist_ok=False)
  for name,t in targets.items():
   if not re.fullmatch(r'blk\.\d+\.attn_[qkv]\.weight',name):continue
   begin=time.monotonic();delta=t.data.astype(np.float32)-floating_tensor_values(parent[name],np.float32)
   u,s,vh=np.linalg.svd(delta,full_matrices=False);energy=np.cumsum(s.astype(np.float64)**2);total=float(energy[-1])
   row={'source':label,'tensor':name,'shape':list(delta.shape),'relative_delta_norm':float(np.linalg.norm(delta)/np.linalg.norm(t.data.astype(np.float32))),
    'energy_at_rank':{str(k):float(energy[min(k,len(s))-1]/total) for k in [4,8,16,32,64]},
    'singular_values':s.tolist(),'seconds':time.monotonic()-begin}
   np.savez(cache/(name+'.npz'),u=u[:,:64],s=s,vh=vh[:64]);row['cache_sha256']=file_sha256(cache/(name+'.npz'))
   results.append(row);write_json(out/'partial-results.json',results)
   print(label,name,row['energy_at_rank'],flush=True)
  write_json(out/(label+'.json'),[r for r in results if r['source']==label]);publish([out],f'Record Fable delta spectra for {label}')
 write_json(out/'summary.json',{'status':'complete','matrices':len(results),'generated_tokens':0,'scored_tokens':0})
except BaseException as exc:
 write_json(out/'summary.json',{'status':'failed','error':repr(exc)});publish([out],'Preserve failed Fable spectrum analysis');raise
publish([out],'Complete precision-controlled Fable delta spectrum analysis')
