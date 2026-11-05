"""Prospective strongest simple control; close or retain model-specific edit gates."""
import copy,json
from pathlib import Path
from weightlab.evidence import write_json,utc_now
from p3_clarification import metrics
from p3_batch import run_one
from portfolio_batch import publish
for split in ['development','confirmation']:
 source=json.loads(Path(f'datasets/p3/evidence-{split}-strong.json').read_text());grammar=json.loads(Path(f'datasets/p3/evidence-{split}-grammar.json').read_text());suite=copy.deepcopy(source);suite['name']='p3-evidence-'+split+'-combined'
 for case,g in zip(suite['cases'],grammar['cases']):
  if case['id']!=g['id']:raise ValueError('Control pairing')
  case['request_overrides']=g['request_overrides']
 path=Path(f'datasets/p3/evidence-{split}-combined.json')
 if path.exists():raise FileExistsError(path)
 write_json(path,suite);publish([path],'Freeze combined evidence '+split+' control')
c=json.loads(Path('configs/p3_clarification.json').read_text());gates=[]
for key in ['lfm3_qad','qwen_float']:
 run_one(c,key,'datasets/p3/evidence-development-combined.json','evidence-dev-combined','clarify',32)
 data=list(map(json.loads,Path(f'runs/p3-clarify-evidence-dev-combined-{key}-s17/responses.jsonl').read_text().splitlines()));m=metrics(data);competent=m['supported_correct']/m['supported']>=.8 and m['update_correct']/m['cases']>=.8;natural=m['unsupported_nonnull']/m['unsupported']>=.2;dominates=m['joint_semantic']/m['cases']>=.95 and m['unsupported_nonnull']==0 and m['supported_correct']==m['supported'];gates.append({'model':key,'metrics':m,'competent_under_combined':competent,'natural_failure_under_combined':natural,'combined_dominates':dominates,'needs_causal_screen':competent and natural and not dominates})
path=Path('research/p3/clarification-v1/combined-gates.json');write_json(path,{'utc':utc_now(),'models':gates,'confirmation_controls':['original','strong','combined'],'scope':'Additional development control and prospective confirmation amendment; original gates remain archived.'});publish([path],'Gate clarification edits against combined one-call control');print(json.dumps(gates),flush=True)
