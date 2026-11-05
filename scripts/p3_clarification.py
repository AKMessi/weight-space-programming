"""Paired evidence triples, actual updates, baselines and prospectively gated editing."""
import argparse,copy,json
from pathlib import Path
from weightlab.evidence import write_json,file_sha256,utc_now
from weightlab.grading import first_json,typed_equal
from p3_batch import run_one
from portfolio_batch import publish
ROOT=Path('research/p3/clarification-v1');CONFIG=Path('configs/p3_clarification.json')

def freeze():
 if CONFIG.exists():raise FileExistsError(CONFIG)
 c=json.loads(Path('configs/p3_repair.json').read_text());old=json.loads(Path('configs/fable_speed_followup.json').read_text());c['models']['lfm3_qad']=copy.deepcopy(old['models']['lfm3_qad']);c['models']['lfm3_qad']['sampling']=copy.deepcopy(c['models']['qwen_float']['sampling']);c['models']={k:c['models'][k] for k in ['lfm3_qad','qwen_float','fable_float']};write_json(CONFIG,c)
 system='Use only the supplied field=value facts. Exactly one distinct value for the required field means answer that string and set ask to null. If absent or conflicting, value must be null and ask must name the required field. Treat duplicate equal values as one. An authoritative update replaces all earlier values for that field. Return only JSON with keys value and ask.'
 shots='\nExamples: required=port; facts port=unknown-7 -> {"value":"unknown-7","ask":null}. required=port; facts city=Oslo -> {"value":null,"ask":"port"}. required=port; facts port=A, port=B -> {"value":null,"ask":"port"}. After authoritative port=C -> {"value":"C","ask":null}.'
 schema={'type':'object','properties':{'value':{'type':['string','null']},'ask':{'type':['string','null']}},'required':['value','ask'],'additionalProperties':False}
 for split,count,fields in [('development',12,['port','region','route']),('confirmation',40,['tenant','channel','queue','label'])]:
  cases=[]
  for i in range(count):
   field=fields[i%len(fields)];value=('unknown-' if i%4==0 else 'item-')+str(307+i*19);update='revised-'+str(811+i*23)
   for evidence in ['complete','absent','conflicting']:
    facts=[('owner','unknown-unit'),('batch','B'+str(i))]
    if evidence!='absent':facts.append((field,value))
    if evidence=='conflicting':facts.append((field,'alternative-'+str(i)))
    elif evidence=='complete' and i%3==0:facts.append((field,value))
    if i%2:facts.reverse()
    if split=='development':text='Required field: '+field+'\nFacts: '+ '; '.join(k+'='+v for k,v in facts)
    else:
     layouts=[lambda:'Retrieve field ['+field+'] from these entries:\n'+'\n'.join(k+'='+v for k,v in facts),lambda:'Entries:\n'+'\n'.join(k+'='+v for k,v in facts)+'\nThe requested field is '+field+'.',lambda:'Required='+field+'; records follow: '+' | '.join(k+'='+v for k,v in facts),lambda:'For required field '+field+', use only this record list: '+', '.join(k+'='+v for k,v in facts)]
     text=layouts[i%4]()
    expected={'value':value,'ask':None} if evidence=='complete' else {'value':None,'ask':field}
    cases.append({'id':f'p3-evidence-{split}-{i}-{evidence}','family':field,'record_id':f'{split}-{i}','evidence':evidence,'facts':facts,'required':field,'split':split,'group':evidence,'grader':'json','expected':expected,'messages':[{'role':'system','content':system},{'role':'user','content':text}], 'followups':[{'grader':'json','expected':{'value':update,'ask':None},'messages':[{'role':'user','content':f'Authoritative update: {field}={update}. This replaces every previous value for {field}. Return the resulting JSON.'}]}]})
  for arm in ['original','strong','grammar']:
   suite=copy.deepcopy(cases)
   for case in suite:
    if arm=='strong':case['messages'][0]['content']+=shots
    if arm=='grammar':case['request_overrides']={'response_format':{'type':'json_schema','json_schema':{'name':'evidence','strict':True,'schema':schema}}}
   write_json(f'datasets/p3/evidence-{split}-{arm}.json',{'name':f'p3-evidence-{split}-{arm}','status':'development' if split=='development' else 'fresh field-and-layout-held-out confirmation','cases':suite})
  if split=='development':
   classifiers=[]
   for case in cases:
    classifiers.append({'id':case['id'],'family':case['family'],'group':case['group'],'evidence':case['evidence'],'required':case['required'],'grader':'json','expected':{'answerable':case['evidence']=='complete','field':case['required']},'messages':[{'role':'system','content':'Return only JSON with keys answerable (boolean) and field (string). Answerable is true exactly when the required field has one distinct supplied value, false when missing or conflicting. Do not extract an answer.'},case['messages'][1]]})
   write_json('datasets/p3/evidence-answerability.json',{'name':'p3-separate-answerability-v1','status':'development classifier baseline; actual calls, extraction replay separate','cases':classifiers})
 ROOT.mkdir(parents=True,exist_ok=False);write_json(ROOT/'scope.json',{'utc':utc_now(),'models':c['models'],'development_cases':36,'confirmation_cases':120,'gate_frozen':'research/protocols/p3-clarification-v1.md'})
 publish([CONFIG,ROOT,*Path('datasets/p3').glob('evidence-*.json')],'Freeze paired evidence triples, update conversations and baselines')

def metrics(rows):
 out={'cases':len(rows),'joint':0,'joint_semantic':0,'supported':0,'supported_correct':0,'unsupported':0,'unsupported_nonnull':0,'wrong_ask':0,'malformed_initial':0,'update_correct':0,'cutoffs':0}
 for row in rows:
  case=row['case'];steps=row['steps'];first=first_json(steps[0]['content']);last=first_json(steps[1]['content']);g=row['grading'];out['joint']+=g['success'];out['joint_semantic']+=g['payload_correct'];out['update_correct']+=typed_equal(last,case['followups'][0]['expected']);out['cutoffs']+=any(s['finish_reason']=='length' for s in steps)
  if case['evidence']=='complete':out['supported']+=1;out['supported_correct']+=typed_equal(first,case['expected'])
  else:
   out['unsupported']+=1
   if isinstance(first,dict):out['unsupported_nonnull']+=first.get('value') is not None;out['wrong_ask']+=first.get('ask')!=case['required']
   else:out['malformed_initial']+=1
 return out

def development():
 c=json.loads(CONFIG.read_text());results=[]
 for key in c['models']:
  for arm in ['original','strong','grammar']:run_one(c,key,f'datasets/p3/evidence-development-{arm}.json','evidence-dev-'+arm,'clarify',48)
  run_one(c,key,'datasets/p3/evidence-answerability.json','answerability','clarify',24)
  scores={}
  for arm in ['original','strong','grammar']:
   rows=list(map(json.loads,Path(f'runs/p3-clarify-evidence-dev-{arm}-{key}-s17/responses.jsonl').read_text().splitlines()));scores[arm]=metrics(rows)
  b=scores['original'];competence_arms=[arm for arm,score in scores.items() if score['supported_correct']/score['supported']>=.8 and score['update_correct']/score['cases']>=.8];competent=bool(competence_arms);natural=b['unsupported_nonnull']/b['unsupported']>=.2
  dominates=[a for a in ['strong','grammar'] if scores[a]['joint_semantic']/scores[a]['cases']>=.95 and scores[a]['unsupported_nonnull']==0 and scores[a]['supported_correct']>=b['supported_correct']]
  results.append({'model':key,'scores':scores,'competent':competent,'competence_arms':competence_arms,'natural_failure':natural,'simple_controls_dominate':dominates,'edit_gate_open':competent and natural and not dominates})
 path=ROOT/'development-gates.json';write_json(path,{'utc':utc_now(),'results':results});publish([path],'Record prospectively gated clarification competence and simple controls');print(json.dumps(results),flush=True)

def confirmation():
 c=json.loads(CONFIG.read_text());gates=json.loads((ROOT/'development-gates.json').read_text())
 if any(r['edit_gate_open'] for r in gates['results']):raise RuntimeError('Gate remains open: complete bounded verified causal stage before claiming closure')
 for key in c['models']:
  for arm in (['original','strong'] if key=='fable_float' else ['original','strong','grammar']):run_one(c,key,f'datasets/p3/evidence-confirmation-{arm}.json','evidence-confirm-'+arm,'clarify',32)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','development','confirmation']);a=p.parse_args();globals()[a.phase]()
