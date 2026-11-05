"""Run the predeclared precision and diagnosis-target panels sequentially."""
import argparse,copy,json
from pathlib import Path
from portfolio_batch import run_one

p=argparse.ArgumentParser();p.add_argument('stage',choices=['precision','diagnosis']);a=p.parse_args()
config=json.loads(Path('configs/portfolio_precision.json').read_text())
if a.stage=='precision':
    for root in ['qwen','lfm3','fable']:
        local=copy.deepcopy(config)
        # The 1.7B floating reference leaves VRAM headroom; hold offload constant across its precision arms.
        if root=='lfm3':local['runtime']['gpu_layers']=18
        for precision in ['float','Q8_0','Q4_0']:
            for suffix in ['','_A']:
                run_one(local,f'{root}_{precision}{suffix}','datasets/portfolio/engineering.json','e3precision')
    for key in ['lfm3_qad','qad_rewrite']:
        run_one(config,key,'datasets/portfolio/diagnosis-probes.json','e3qadparity',limit=6)
else:
    for root in ['qwen','lfm3','fable']:
        local=copy.deepcopy(config)
        if root=='lfm3':local['runtime']['gpu_layers']=18
        for state in ['healthy','output_sign','ffn_sign']:
            for layout in ['','_permuted']:
                run_one(local,f'{root}_diagnosis_{state}{layout}','datasets/portfolio/diagnosis-probes.json','e5probes')
