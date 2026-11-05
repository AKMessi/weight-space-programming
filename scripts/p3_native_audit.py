"""Audit P3 native request/response charges and optional local array integrity."""
import argparse,json
from pathlib import Path
from weightlab.evidence import file_sha256,utc_now,write_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--verify-arrays',action='store_true');a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
ledger={r['request_id']:r for r in map(json.loads,Path('research/scored-budget.jsonl').read_text().splitlines())};seen={};runs=[];arrays=0
for path in sorted(Path('research/p3').rglob('events.jsonl')):
 if not any(part.startswith('native-') for part in path.parts):continue
 requests={};responses={}
 for event in map(json.loads,path.read_text().splitlines()):
  if event['event'] not in ['request','response']:continue
  key=event.get('scored_request_id',event.get('request_id'))
  if key is None:raise ValueError('Missing scored event identity')
  destination=requests if event['event']=='request' else responses
  if key in destination:raise ValueError('Duplicate native event')
  destination[key]=event
 if requests.keys()!=responses.keys():raise ValueError('Unresolved native request in '+str(path))
 for key,event in responses.items():
  raw=event['response']
  if raw['status']!='complete':raise ValueError('Native response failed, audit conservative charge explicitly')
  if key in seen:raise ValueError('Duplicate native scored identity')
  if ledger[key]['event']!='settle' or ledger[key]['tokens']!=raw['prompt_tokens']:raise ValueError('Native scoring mismatch')
  seen[key]=raw['prompt_tokens']
 root=path.parent;summary=json.loads((root/'summary.json').read_text());runs.append({'run':str(root),'status':summary['status'],'native_requests':len(requests),'scored_tokens':sum(ledger[k]['tokens'] for k in requests),'events_sha256':file_sha256(path)})
 records=root/'responses.jsonl'
 if a.verify_arrays and records.exists():
  for row in map(json.loads,records.read_text().splitlines()):
   raw=row['raw_response'];hashes=row.get('array_hashes',raw.get('array_hashes',{}))
   for name,expected in hashes.items():
    if file_sha256(raw['captures'][name]['path'])!=expected:raise ValueError('Trace hash mismatch')
    arrays+=1
   expected=row.get('logits_sha256',raw.get('logits_sha256'))
   if expected:
    if file_sha256(raw['logits_path'])!=expected:raise ValueError('Logit hash mismatch')
    arrays+=1
p3={k:v for k,v in ledger.items() if k.startswith('p3/') or k.startswith('p3-native-scored/')}
if p3.keys()!=seen.keys():raise ValueError('P3 scored ledger / native raw evidence mismatch: '+str(set(p3)^set(seen)))
write_json(a.output,{'utc':utc_now(),'scored_ledger_sha256':file_sha256('research/scored-budget.jsonl'),'native_runs':len(runs),'native_forward_requests':len(seen),'p3_scored_tokens':sum(seen.values()),'arrays_and_logits_verified':arrays,'all_requests_settled':True,'runs':runs,'scope':'P3 native prefill/state forwards, including failed analysis runs with successfully recorded responses; generated native tokens are audited separately.'})
print(json.dumps({'runs':len(runs),'forwards':len(seen),'scored_tokens':sum(seen.values()),'arrays':arrays}),flush=True)
