"""Offline summaries; never use these final outcomes to select model edits."""
import argparse,json,re
from pathlib import Path
from collections import defaultdict
import numpy as np
from weightlab.evidence import file_sha256,utc_now,write_json
from p3_clarification import metrics
from portfolio_batch import publish
ROOT=Path('research/p3/analysis-v1')
def rows(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def runrows(stage,label,key):return rows(f'runs/p3-{stage}-{label}-{key}-s17/responses.jsonl')
def quality(data):
 groups=defaultdict(list)
 for r in data:groups[r['case']['group']].append(r)
 return {g:{'n':len(rs),'strict':sum(r['grading']['success'] for r in rs),'payload':sum(r['grading']['payload_correct'] for r in rs),'cutoff':sum(r['finish_reason']=='length' for r in rs),'generated_tokens':sum(r['usage']['completion_tokens'] for r in rs)} for g,rs in groups.items()}
def paired(a,b,field,mode=None):
 lookup={r['case']['id']:r for r in a};pairs=[(lookup[r['case']['id']],r) for r in b if mode is None or r['case']['response_mode']==mode]
 rescues=sum(not x['grading'][field] and y['grading'][field] for x,y in pairs);losses=sum(x['grading'][field] and not y['grading'][field] for x,y in pairs)
 groups=defaultdict(list)
 for x,y in pairs:groups[x['case']['family']].append(int(y['grading'][field])-int(x['grading'][field]))
 rng=np.random.default_rng(811);names=sorted(groups);means=[]
 for _ in range(2000):
  chosen=rng.integers(0,len(names),len(names));values=[v for i in chosen for v in groups[names[i]]];means.append(float(np.mean(values)))
 return {'n':len(pairs),'rescues':rescues,'new_failures':losses,'net_percentage_points':100*(rescues-losses)/len(pairs),'family_cluster_bootstrap_95pp':(100*np.quantile(means,[.025,.975])).tolist(),'family_clusters':len(names),'scope':'descriptive small-cluster bootstrap, not broad method generalization'}
def repair():
 key=json.loads(Path('research/p3/repair-search-v1/selection.json').read_text())['selected'];base=runrows('repair','confirmation-original','fable_float');patch=runrows('repair','confirmation-patch',key)
 arms={'original':quality(base),'patch':quality(patch),'strong':quality(runrows('repair','confirmation-strong','fable_float')),'grammar':quality(runrows('repair','confirmation-grammar','fable_float'))}
 comparisons={m:paired(base,patch,'success',m) for m in ['plain','fenced']};comparisons['payload']=paired(base,patch,'payload_correct')
 protection={k:quality(runrows('repair','repair-protection',k)) for k in ['fable_float',key]}
 controls={k:quality(runrows('repair',k,'random_control' if k=='random-control' else 'attenuation_control')) for k in ['random-control','attenuation-control']}
 sampling={k:{str(seed):quality(rows(f'runs/p3-repair-sampling-stability-{k}-s{seed}/responses.jsonl')) for seed in [17,29]} for k in ['fable_float',key]}
 return {'selected':key,'confirmation':arms,'paired':comparisons,'protection':protection,'diagnostics':controls,'sampling_fixed_confirmation_subset':sampling,'repair_deployment_gate':False,'gate_reason':'Fails requested-fence preservation on independent families; protected explanations also have cutoffs. Do not deploy as a validated repair. New-family task competence was weaker than development.'}
def speed():
 groups=defaultdict(list);profiles={}
 for root in sorted(Path('research/p3/speed-v1').glob('*')):
  if not (root/'responses.jsonl').exists():continue
  data=rows(root/'responses.jsonl')
  if root.name.startswith('profile-'):
   text=(root/'server.log').read_text();ops=[]
   for m in re.finditer(r'^(.*?): (\d+) x ([\d.e+\-]+) us = ([\d.e+\-]+) us',text,re.M):ops.append({'op':m[1],'count':int(m[2]),'mean_us':float(m[3]),'total_us':float(m[4])})
   profiles[data[0]['model']]={'operations':ops,'source_sha256':file_sha256(root/'server.log'),'note':'Profiling includes initialization/scheduler work; overhead not mixed with normal timing.'};continue
  for r in data:
   if not r['warmup'] and r['valid_timing']:groups[(r['model'],r['backend'],r['prompt_length'])].append(r)
 summaries=[]
 for (model,backend,length),data in groups.items():
  sessions=sorted({r['session'] for r in data});per_session={str(s):float(np.median([r['timings']['predicted_per_second'] for r in data if r['session']==s])) for s in sessions}
  summaries.append({'model':model,'backend':backend,'prompt_tokens':length,'n':len(data),'sessions':sessions,'decode_tps_median':float(np.median([r['timings']['predicted_per_second'] for r in data])),'session_decode_medians':per_session,'ttft_ms_median':1000*float(np.median([r['ttft_seconds'] for r in data])),'prefill_ms_median':float(np.median([r['timings']['prompt_ms'] for r in data])),'total_ms_median':1000*float(np.median([r['seconds'] for r in data]))})
 return {'conditions':summaries,'profiles':profiles,'caveat':'Shared desktop, not controlled idle machine; session variation is material, especially CPU. Runtime-reported decode tokens/s uses its own first-token accounting. Quality evaluated separately.'}
def precision():
 c=json.loads(Path('configs/p3_precision_final.json').read_text());target=rows('research/p3/native-precision-final-target_float/responses.jsonl');out={}
 for key,spec in c['models'].items():
  data=rows(f'research/p3/native-precision-final-{key}/responses.jsonl');kls=[];top=0
  for ref,row in zip(target,data):
   a=np.fromfile(ref['raw_response']['logits_path'],np.float32).astype(float);b=np.fromfile(row['raw_response']['logits_path'],np.float32).astype(float);lp=a-a.max();lp-=np.log(np.exp(lp).sum());lq=b-b.max();lq-=np.log(np.exp(lq).sum());kls.append(float(np.sum(np.exp(lp)*(lp-lq))));top+=int(np.argmax(a)==np.argmax(b))
  data=runrows('precision','adapter-precision-quality',key);ref=runrows('precision','adapter-precision-quality','target_float');out[key]={'quality':quality(data),'bytes':Path(spec['path']).stat().st_size,'mean_heldout_kl_to_original_fable':float(np.mean(kls)),'kl_cases':kls,'same_next_top1':top,'next_prefixes':len(kls),'full_response_agreement':sum(x['content']==y['content'] for x,y in zip(data,ref)),'responses':len(data)}
 return {'arms':out,'allocations':json.loads(Path('research/p3/precision-adapter-v1/allocation-selection.json').read_text()),'scope':'Recovered-adapter native distribution/functional fidelity; no proven specialist advantage, new repair, native LoRA loading or general allocation-method novelty.'}
def clarify():
 c=json.loads(Path('configs/p3_clarification.json').read_text());result={}
 for key in c['models']:
  result[key]={}
  for arm in (['original','strong'] if key=='fable_float' else ['original','strong','grammar']):result[key][arm]=metrics(runrows('clarify','evidence-confirm-'+arm,key))
 return {'confirmation':result,'development_gates':json.loads(Path('research/p3/clarification-v1/development-gates.json').read_text()),'scope':'Controlled supplied-record extraction; field and layout held out together; no claim of open-world epistemic calibration.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['repair','speed','precision','clarify']);a=p.parse_args();path=ROOT/(a.phase+'.json')
 if path.exists():raise FileExistsError(path)
 result={'utc':utc_now(),**globals()[a.phase]()};write_json(path,result);publish([path],'Summarize P3 '+a.phase+' without reselection');print(json.dumps(result),flush=True)
