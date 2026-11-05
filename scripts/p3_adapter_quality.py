"""Calibrate once, freeze a package, then expose fresh confirmation outcomes."""
import argparse,json,statistics
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from p3_batch import run_one
from portfolio_batch import publish
config=json.loads(Path('configs/p3_adapters.json').read_text())
keys=[k for k in config['models'] if k not in ['fable_f16','parent_uniform_bf16']]

def rows(label,key):
 root=Path(f'runs/p3-adapter-{label}-{key}-s17')
 if json.loads((root/'summary.json').read_text())['status']!='complete':raise ValueError('incomplete calibration')
 return list(map(json.loads,(root/'responses.jsonl').read_text().splitlines()))

p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibrate','select','confirm','protect']);a=p.parse_args()
if a.phase=='calibrate':
 for key in ['fable_f16','parent_uniform_bf16']:
  run_one(config,key,'datasets/p3/fable-repo-development.json','repo-dev','adapter',max_tokens=64)
 for key in ['fable_f16',*keys]:
  run_one(config,key,'datasets/p3/fable-repo-calibration.json','repo-cal','adapter',max_tokens=64)
elif a.phase=='select':
 out=Path('research/p3/adapter-selection-v1.json')
 if out.exists():raise FileExistsError(out)
 reference=rows('repo-cal','fable_f16');score=sum(r['grading']['payload_correct'] for r in reference);choices=[]
 for key in keys:
  data=rows('repo-cal',key)
  if [r['case'] for r in data]!=[r['case'] for r in reference]:raise ValueError('pair mismatch')
  correct=sum(r['grading']['payload_correct'] for r in data)
  agreement=sum(r['grading']['first_line']==b['grading']['first_line'] for r,b in zip(data,reference))/len(reference)
  pack=json.loads(Path(f'research/p3/adapters/{key}-pack.json').read_text())
  error=statistics.mean(t['relative_delta_error'] for t in pack['tensors'] if 'relative_delta_error' in t)
  choices.append({'key':key,'correct':correct,'n':len(data),'agreement':agreement,'package_bytes':config['models'][key]['package_bytes'],
                  'mean_relative_delta_error':error,'eligible':correct>=score and agreement>=.9})
 eligible=[r for r in choices if r['eligible']]
 chosen=min(eligible,key=lambda r:(r['package_bytes'],r['mean_relative_delta_error'],r['key'])) if eligible else min(choices,key=lambda r:(-r['agreement'],-r['correct'],r['package_bytes']))
 write_json(out,{'utc':utc_now(),'git':git_state(),'reference_score':score,'candidates':choices,'selected':chosen,
    'development_eligible':bool(eligible),'confirmation_suite_sha256':file_sha256('datasets/p3/fable-repo-confirmation.json'),
    'policy':'Frozen before confirmation; no confirmation rows or outcomes used to select.'})
 publish([out],'Freeze recovered Fable package selection before confirmation')
 print(json.dumps(chosen),flush=True)
else:
 selection=json.loads(Path('research/p3/adapter-selection-v1.json').read_text());chosen=selection['selected']['key']
 if a.phase=='confirm':
  for key in ['fable_f16','parent_uniform_bf16',chosen]:
   run_one(config,key,'datasets/p3/fable-repo-confirmation.json','repo-confirm','adapter',max_tokens=64)
 else:
  for key in ['fable_f16',chosen]:
   run_one(config,key,'datasets/p3/fable-protection.json','protection','adapter',max_tokens=64)
