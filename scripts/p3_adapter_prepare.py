"""Build every predeclared numerical Fable delta candidate before quality selection."""
import json
from pathlib import Path
from weightlab.adapters import pack,merge
from weightlab.evidence import file_sha256,write_json
from portfolio_batch import publish
config=json.loads(Path('configs/portfolio.json').read_text());base=config['models']['parent_uniform_bf16'];target=config['models']['fable_f16']
config['models']={key:config['models'][key] for key in ['fable_f16','parent_uniform_bf16']}
for spec in config['models'].values():spec['sampling'].update(temperature=0.,top_k=0,top_p=1.,min_p=0.,repeat_penalty=1.)
choices=[('svd',r,False) for r in [4,8,16,32,64]]+[('svd',32,True)]+[(kind,32,False) for kind in ['int8','int4','sign']]
for method,rank,rounded in choices:
 key=f'fable_{method}{rank if method=="svd" else ""}'+('_bf16' if rounded else '')
 manifest=Path(f'research/p3/adapters/{key}-pack.json');dest=Path(f'artifacts/p3/adapters/{key}.npz');merged=dest.with_suffix('.gguf');mergepath=manifest.with_name(key+'-merge.json')
 if manifest.exists():
  report=json.loads(manifest.read_text());mergedreport=json.loads(mergepath.read_text())
  if report['status']!='verified' or file_sha256(dest)!=report['package_sha256'] or file_sha256(merged)!=mergedreport['edited_file_sha256']:raise ValueError('artifact cache mismatch')
 else:
  report=pack(base,target,dest,manifest,method,rank,Path('artifacts/p3/svd-cache/aligned'),rounded)
  mergedreport=merge(base['path'],dest,merged,mergepath)
  publish([manifest,mergepath],f'Verify recovered Fable delta package {key}')
 config['models'][key]={**target,'sampling':dict(config['models']['fable_f16']['sampling']),'path':str(merged),
   'sha256':mergedreport['edited_file_sha256'],'package_path':str(dest),'package_sha256':report['package_sha256'],
   'package_bytes':report['bytes'],'base_sha256':base['sha256']}
 print(key,report['bytes'],flush=True)
write_json('configs/p3_adapters.json',config)
publish(['configs/p3_adapters.json'],'Freeze all nine recovered Fable candidates before quality inference')
