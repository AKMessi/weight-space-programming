"""Bounded CPU fallback after the documented Vulkan component gate failure."""
import argparse,json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from portfolio_batch import publish
from p3_native_run import run
p=argparse.ArgumentParser();p.add_argument('phase',choices=['gate','run','analyze']);a=p.parse_args()
config=json.loads(Path('configs/p3_repair.json').read_text());model=config['models']['fable_float']
requests=json.loads(Path('datasets/p3/repair-trace-prefixes.json').read_text())
if a.phase in ['gate','run']:
 if a.phase=='run' and not json.loads(Path('research/p3/repair-cpu-gate-v1.json').read_text())['boundary_passed']:raise RuntimeError('CPU component gate failed')
 run(model,requests[:4] if a.phase=='gate' else requests,'cpu-component-'+a.phase,'repair',{'gpu_layers':0})
 if a.phase=='run':raise SystemExit()
path=Path('research/p3/native-repair-cpu-component-'+('gate' if a.phase=='gate' else 'run')+'/responses.jsonl')
rows=list(map(json.loads,path.read_text().splitlines()));weights={t.name:t for t in GGUFReader(model['path']).tensors};checks=[];ratios=[];arrays={}
for layer in range(14):
 def mat(items,kind):return np.stack([np.fromfile(r['raw_response']['captures'][f'component_{kind}-{layer}']['path'],np.float32) for r in items])
 w=weights[f'blk.{layer}.ffn_down.weight'].data.astype(np.float32)
 h=mat(rows[:4],'input').astype(np.float16).astype(np.float32);y=mat(rows[:4],'down');pred=h@w.T
 err=y-pred;maximum=float(abs(err).max());rel=float(np.linalg.norm(err)/max(np.linalg.norm(y),1e-6))
 checks.append({'layer':layer,'max_abs_error':maximum,'relative_error':rel,'passed':maximum<=.05 and rel<=.001})
 if a.phase=='analyze':
  branches={b:[r for r in rows if r['case']['branch']==b] for b in ['bare','fence','key_plain','key_protect']}
  pos=mat(branches['bare'],'down');neg=mat(branches['fence'],'down');v=(pos-neg).mean(0)
  ratios.append({'layer':layer,'ratio':float(np.linalg.norm(v)/max(np.mean(np.linalg.norm(pos,axis=1)),1e-9)),'contrast_norm':float(np.linalg.norm(v))})
  arrays[f'v{layer}']=v
  for label,b in [('pos','bare'),('neg','fence'),('plain','key_plain'),('protect','key_protect')]:arrays[f'h{label}{layer}']=mat(branches[b],'input')
report={'utc':utc_now(),'git':git_state(),'source_sha256':file_sha256(path),'projection_checks':checks,'boundary_passed':all(c['passed'] for c in checks),'arithmetic':'CPU W_F16 * round_F16(h), accumulated float32'}
if arrays:
 cache=Path('artifacts/p3/repair-cpu-directions.npz')
 if cache.exists():raise FileExistsError(cache)
 np.savez(cache,**arrays);report.update(arrays_sha256=file_sha256(cache),ratios=ratios,chosen_layer=min(ratios,key=lambda r:(-r['ratio'],r['layer'])))
out=Path('research/p3/repair-cpu-'+('gate' if a.phase=='gate' else 'analysis')+'-v1.json')
if out.exists():raise FileExistsError(out)
write_json(out,report);publish([out],'Record CPU fallback component boundary checks');print(json.dumps(report),flush=True)
