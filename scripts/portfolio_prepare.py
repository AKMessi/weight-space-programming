"""Prepare fixed normalization candidates; verified artifacts are never overwritten."""
import argparse,json,re
from pathlib import Path
from gguf import GGUFReader
from weightlab.transforms import intervene
from weightlab.evidence import file_sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/validation_runs.json');p.add_argument('--output',default='configs/portfolio.json')
a=p.parse_args()
config=json.loads(Path(a.config).read_text())
for key in ['qwen3_q8','lfm3_qad','fable_f16']:
    base=config['models'][key]
    r=GGUFReader(base['path'])
    layers=sorted({int(m[1]) for t in r.tensors if (m:=re.fullmatch(r'blk\.(\d+)\.ffn_norm\.weight',t.name))})
    last=layers[-1]
    ops={'A':[{'tensor':f'blk.{last}.ffn_norm.weight','kind':'scale','factor':0.9}],
         'B':[{'tensor':f'blk.{last}.attn_norm.weight','kind':'scale','factor':0.9}],
         'R':[{'tensor':f'blk.{last}.ffn_norm.weight','kind':'random_l2','relative_norm':0.1,'seed':617}]}
    ops['AB']=ops['A']+ops['B'];ops['BA']=ops['B']+ops['A']
    for name,operations in ops.items():
        dest=Path(f'artifacts/portfolio/{key}-{name}.gguf')
        manifest=Path(f'research/patches/portfolio-{key}-{name}.json')
        if manifest.exists():
            report=json.loads(manifest.read_text())
            if report['status']!='verified' or file_sha256(dest)!=report['edited_file_sha256']:raise ValueError('existing artifact invalid')
        else:report=intervene(base,operations,dest,manifest)
        config['models'][f'{key}_{name}']={**base,'path':str(dest),'sha256':report['edited_file_sha256'],
             'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
        print(key,name,report['edited_file_sha256'],flush=True)
write_json(a.output,config)
