"""Matched-site random diagnostic, historical attenuation and fixed sampling check."""
import copy,json,shutil
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.edit import tensor_hash
from weightlab.evidence import file_sha256,write_json,utc_now
from p3_batch import run_one
from portfolio_batch import publish
cfg=json.loads(Path('configs/p3_repair_candidates.json').read_text());selection=json.loads(Path('research/p3/repair-search-v1/selection.json').read_text());key=selection['selected']
if key:
 meta=json.loads(Path(f'research/p3/repair-search-v1/{key}.json').read_text());base=cfg['models']['fable_float'];source=GGUFReader(base['path']);ts={t.name:t for t in source.tensors};rng=np.random.default_rng(811)
 noises={r['tensor']:(r['columns'],rng.standard_normal((ts[r['tensor']].data.shape[0],len(r['columns'])))) for r in meta['tensors']}
 oldnorm=float(json.loads(Path('research/p3/repair-search-v1/sites.json').read_text())['all_down_norm']);noise_norm=np.sqrt(sum(float(np.sum(n*n)) for _,n in noises.values()));scale=oldnorm*meta['requested_dose']/noise_norm
 out=Path('artifacts/p3/repair/random-control.gguf');manifest=Path('research/p3/repair-search-v1/random-control.json')
 if out.exists() or manifest.exists():raise FileExistsError(out)
 shutil.copyfile(base['path'],out);result=GGUFReader(out,mode='r+');dest={t.name:t for t in result.tensors};norm2=0.;changed=0
 for name,(cols,noise) in noises.items():
  new=(ts[name].data[:,cols].astype(float)+noise*scale).astype(np.float16);
  if not np.isfinite(new).all():raise ValueError('Nonfinite random control')
  dest[name].data[:,cols]=new;delta=new.astype(float)-ts[name].data[:,cols].astype(float);norm2+=float(np.sum(delta*delta));changed+=int(np.count_nonzero(delta))
 result.data.flush()
 for name,t in ts.items():
  if name not in noises and tensor_hash(t.data)!=tensor_hash(dest[name].data):raise ValueError('Unexpected tensor')
  if name in noises:
   mask=np.ones(t.data.shape[1],bool);mask[noises[name][0]]=False
   if not np.array_equal(t.data[:,mask],dest[name].data[:,mask]):raise ValueError('Unselected column changed')
 if file_sha256(base['path'])!=base['sha256']:raise ValueError('Original changed')
 write_json(manifest,{'utc':utc_now(),'base':base,'selected_patch':key,'seed':811,'selected_manifest_sha256':file_sha256(f'research/p3/repair-search-v1/{key}.json'),'sha256':file_sha256(out),'output':str(out),'actual_relative_norm':np.sqrt(norm2)/oldnorm,'changed_values':changed,'status':'verified'})
 cfg['models']['random_control']={**base,'path':str(out),'sha256':file_sha256(out),'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
 write_json('configs/p3_repair_controls.json',cfg);publish([manifest,'configs/p3_repair_controls.json'],'Export matched-site random repair diagnostic')
 run_one(cfg,'random_control','datasets/p3/switch-screen.json','random-control','repair',96)
 old=json.loads(Path('configs/fable_speed_followup.json').read_text());cfg['models']['attenuation_control']=old['models']['fable_float_A'];cfg['models']['attenuation_control']['sampling']=copy.deepcopy(base['sampling'])
 run_one(cfg,'attenuation_control','datasets/p3/switch-screen.json','attenuation-control','repair',96)
 suite=json.loads(Path('datasets/p3/switch-confirmation.json').read_text());suite['cases']=suite['cases'][:24];suite['status']='fixed confirmation subset decoding stability, not independent';path=Path('datasets/p3/switch-sampling-subset.json');write_json(path,suite);publish([path],'Materialize predeclared fixed decoding-stability subset')
 for k in ['fable_float',key]:
  sample=copy.deepcopy(cfg);sample['models'][k]['sampling'].update(temperature=.1,top_k=50,top_p=1,min_p=0,repeat_penalty=1)
  for seed in [17,29]:run_one(sample,k,path,'sampling-stability','repair',96,seed=seed)
