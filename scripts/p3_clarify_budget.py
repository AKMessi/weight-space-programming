"""Choose a diagnostic panel size from exact prefix costs, before forward outcomes."""
import json
from pathlib import Path
from weightlab.evidence import write_json,utc_now
from p3_batch import stage_used
from portfolio_batch import publish
root=Path('research/p3/clarify-edit-v1');plan=root/'budget-plan.json'
if plan.exists():raise FileExistsError(plan)
charges={}
for line in Path('research/scored-budget.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['request_id'].startswith('p3-native-scored/clarify/'):charges[r['request_id']]=r['tokens']
choices=[]
for n in [3,2]:
 details=[]
 for key in ['lfm3_qad','qwen_float']:
  out=root/key;requests=json.loads((out/'trace-requests.json').read_text());parity=json.loads((out/'parity-requests.json').read_text());selection=json.loads((out/'selection.json').read_text());panel=[r for r in requests if r['record_id'] in selection['record_ids'][:n] and r['role'] in ['key','update']]
  tokens=sum(len(r['tokens']) for r in requests)+sum(len(r['tokens']) for r in requests[:4])*(2 if key=='qwen_float' else 1)+sum(len(r['tokens']) for r in parity)+5*sum(len(r['tokens']) for r in panel)
  details.append({'model':key,'scored_max':tokens,'panel_records':n,'panel_cases':len(panel),'generated_max':48*(len(parity)+5*len(panel))})
 projected=sum(charges.values())+sum(r['scored_max'] for r in details);choices.append({'panel_records':n,'details':details,'projected_scored_tokens':projected})
 if projected<=50000:chosen=choices[-1];break
else:raise RuntimeError('Exact diagnostics exceed scored cap; prospectively redesign before any new forwards')
write_json(plan,{'utc':utc_now(),'already_scored':sum(charges.values()),'choices_checked':choices,'selected':chosen,'clarify_generated_used':stage_used('clarify'),'planned_primary_confirmation_generated_max':61440,'new_permanent_route_generated_max':26368,'scope':'Prefix-count choice only, no native causal/component outcomes inspected.'});publish([plan],'Freeze exact-cost semantic diagnostic panel before forwards');print(json.dumps(chosen),flush=True)
