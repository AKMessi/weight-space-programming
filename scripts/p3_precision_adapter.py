"""Paired tensor precision allocation for a frozen recovered adapter; no fitting weights."""
import argparse,copy,itertools,json,random,subprocess
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.hybrid import replace_tensors
from weightlab.runtime import server,request_json,runtime_libraries
from p3_native_run import run
from p3_batch import run_one
from portfolio_batch import publish
ROOT=Path('research/p3/precision-adapter-v1');ART=Path('artifacts/p3/precision');CFG=Path('configs/p3_precision_cal.json');FINAL=Path('configs/p3_precision_final.json')

def quantize(spec,key,kind):
 out=ART/(key+'.gguf');report=ROOT/(key+'.json');binary='/home/akmessi/llama-vulkan-build/bin/llama-quantize'
 if out.exists() or report.exists():raise FileExistsError(out)
 if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Source identity')
 cmd=[binary,spec['path'],str(out),kind,'4'];record={'utc':utc_now(),'git':git_state(),'base':spec,'command':cmd,'binary_sha256':file_sha256(binary),'libraries':runtime_libraries(binary),'status':'started'};write_json(report,record)
 try:
  with (ROOT/(key+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Source changed')
  sha=file_sha256(out);record.update(status='verified',sha256=sha,bytes=out.stat().st_size)
 except BaseException as exc:record.update(status='failed',error=repr(exc));raise
 finally:write_json(report,record)
 return {**spec,'path':str(out),'sha256':sha,'local_patch_manifest':str(report),'local_patch_manifest_sha256':file_sha256(report)}

def hybrid(base,donor,names,key):
 manifest=ROOT/(key+'.json');r=replace_tensors(base,donor,names,ART/(key+'.gguf'),manifest)
 return {**base,'path':r['output'],'sha256':r['sha256'],'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}

def prepare():
 ROOT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=True)
 c=json.loads(Path('configs/p3_adapters.json').read_text());models={k:c['models'][old] for k,old in [('base_float','parent_uniform_bf16'),('edited_float','fable_svd32'),('target_float','fable_f16')]}
 for spec in models.values():spec['sampling'].update(temperature=0.,top_k=0,top_p=1.,min_p=0.,repeat_penalty=1.)
 for source in ['base','edited','target']:
  for kind in ['Q8_0','Q4_0']:models[source+'_'+kind]=quantize(models[source+'_float'],source+'_'+kind,kind)
 old=json.loads(Path('configs/fable_speed_followup.json').read_text())
 matches={kind:models['target_'+kind]['sha256']==old['models']['fable_'+kind]['sha256'] for kind in ['Q8_0','Q4_0']}
 if not all(matches.values()):raise ValueError('Target ordinary quantization no-op hash drift')
 donor={t.name:t for t in GGUFReader(models['edited_float']['path']).tensors};quant={t.name:t for t in GGUFReader(models['edited_Q4_0']['path']).tensors}
 names=sorted(n for n in donor if any(n.endswith('.attn_'+k+'.weight') for k in ['q','k','v']))
 if len(names)!=18:raise ValueError('Expected18quantizable adapted tensors')
 costs={n:int(donor[n].n_bytes-quant[n].n_bytes) for n in names};budget=4*min(costs.values())
 errors={n:float(np.sum((donor[n].data.astype(float)-dequantize(quant[n].data,quant[n].tensor_type).astype(float))**2)) for n in names}
 for i,name in enumerate(names):models['single_'+str(i)]=hybrid(models['edited_Q4_0'],models['edited_float'],[name],'single_'+str(i))
 write_json(ROOT/'tensor-policy.json',{'utc':utc_now(),'names':names,'costs':costs,'budget':budget,'reconstruction_errors':errors,'original_quantization_hash_matches':matches,'scope':'18quantizable QKV matrices; tiny F32 scalar corrections retained source-wise'})
 cal={'runtime':c['runtime'],'budgets':c['budgets'],'models':models};write_json(CFG,cal)
 publish([ROOT,CFG],'Prepare fixed same-lineage parent/recovered/target precision arms')

def prefixes():
 c=json.loads(CFG.read_text());out=ROOT/'tokenization';out.mkdir(exist_ok=False)
 texts=['Copy exactly the bracketed text: [Blue Cedar 291].','Return JSON with tag="PB-619" and count=4.','Complete Python: def plus_two(x):\n    return','Explain in one sentence why a list can be empty.','Return only the number of letters in the word cat.','Continue the line: result = sorted(','Record: token=LM-71. Copy only the token.','Explain what return x*2 does in a Python function.']
 quality=json.loads(Path('datasets/p3/format-quality.json').read_text());code=[r for r in quality['cases'] if r['response_mode']=='code'];heldout=code[:4]+code[-4:];requests={'cal':[],'final':[]}
 with server(c['runtime']['binary'],c['models']['target_float']['path'],out,c['runtime']) as (url,command,proc):
  events=[]
  for split,items in [('cal',[{'id':'precision-cal-'+str(i),'messages':[{'role':'user','content':s}]} for i,s in enumerate(texts)]),('final',heldout)]:
   for case in items:
    form={'messages':case['messages'],'chat_template_kwargs':{'enable_thinking':False}}
    rendered=request_json(url+'/apply-template',form);req={'content':rendered['prompt'],'add_special':True,'parse_special':True};raw=request_json(url+'/tokenize',req);ids=raw['tokens']
    if len(ids)>128:raise ValueError('Frozen prefix longer than128')
    events.append({'split':split,'case':case,'template_request':form,'template_response':rendered,'token_request':req,'token_response':raw})
    requests[split].append({'id':case['id'],'prompt':rendered['prompt'],'tokens':ids,'max_tokens':0})
  write_json(out/'record.json',{'utc':utc_now(),'command':command,'events':events,'model':c['models']['target_float']})
 write_json('datasets/p3/precision-prefixes.json',requests);publish([out,'datasets/p3/precision-prefixes.json'],'Freeze exact calibration and held-out precision prefixes')

def kl(x,y):
 x=x.astype(float);y=y.astype(float);p=x-x.max();p-=np.log(np.exp(p).sum());q=y-y.max();q-=np.log(np.exp(q).sum());return float(np.sum(np.exp(p)*(p-q)))
def native_rows(label):return list(map(json.loads,Path(f'research/p3/native-precision-{label}/responses.jsonl').read_text().splitlines()))
def logits(row):return np.fromfile(row['raw_response']['logits_path'],np.float32)

def calibrate():
 c=json.loads(CFG.read_text());requests=json.loads(Path('datasets/p3/precision-prefixes.json').read_text())['cal']
 for key in ['target_float','edited_Q4_0']+['single_'+str(i) for i in range(18)]:run(c['models'][key],requests,'cal-'+key,'precision',{'filter':'result_norm'})
 target=native_rows('cal-target_float');base=native_rows('cal-edited_Q4_0');baseline=float(np.mean([kl(logits(a),logits(b)) for a,b in zip(target,base)]));policy=json.loads((ROOT/'tensor-policy.json').read_text());benefits={}
 for i,name in enumerate(policy['names']):benefits[name]=baseline-float(np.mean([kl(logits(a),logits(b)) for a,b in zip(target,native_rows('cal-single_'+str(i)))]))
 feasible=[];names=policy['names'];costs=policy['costs'];budget=policy['budget']
 for size in range(1,5):
  feasible.extend(combo for combo in itertools.combinations(names,size) if sum(costs[n] for n in combo)==budget)
 if not feasible:raise ValueError('No feasible byte-exact subset')
 random_subset=random.Random(811).choice(feasible)
 reconstruction=min(feasible,key=lambda s:(-sum(policy['reconstruction_errors'][n] for n in s),s));behavior=min(feasible,key=lambda s:(-sum(benefits[n] for n in s),s))
 allocations={'all':names,'random':list(random_subset),'reconstruction':list(reconstruction),'distribution':list(behavior)}
 selection={'utc':utc_now(),'baseline_cal_kl':baseline,'single_tensor_kl_reductions':benefits,'feasible_subsets':len(feasible),'seed':811,'allocations':allocations,'budget':budget,'allocation_method':'additive native full-vocabulary KL improvement; calibration only'}
 write_json(ROOT/'allocation-selection.json',selection);publish([ROOT/'allocation-selection.json'],'Freeze byte-matched precision allocations before final measurement')
 final=copy.deepcopy(c);final['models']={k:v for k,v in final['models'].items() if not k.startswith('single_')}
 for arm,names in allocations.items():
  for source in ['base','edited']:final['models'][source+'_'+arm]=hybrid(c['models'][source+'_Q4_0'],c['models'][source+'_float'],names,source+'_'+arm)
 sizes={arm:{s:Path(final['models'][s+'_'+arm]['path']).stat().st_size for s in ['base','edited']} for arm in allocations}
 if len({sizes[a][s] for a in ['random','reconstruction','distribution'] for s in ['base','edited']})!=1:raise ValueError('Matched extra-byte allocations have unequal files')
 write_json(ROOT/'allocation-bytes.json',sizes);write_json(FINAL,final);publish([ROOT,FINAL],'Export paired precision allocations with exact byte budgets')

def final():
 c=json.loads(FINAL.read_text());requests=json.loads(Path('datasets/p3/precision-prefixes.json').read_text())['final']
 for key in c['models']:run(c['models'][key],requests,'final-'+key,'precision',{'filter':'result_norm'})
 for key in c['models']:run_one(c,key,'datasets/p3/format-quality.json','adapter-precision-quality','precision',128)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prefixes','calibrate','final']);a=p.parse_args();globals()[a.phase]()
