"""Native distribution fidelity on fixed protection prefixes; no adapter selection."""
import argparse,json
from pathlib import Path
import numpy as np
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.runtime import server,request_json
from p3_native_run import run
from portfolio_batch import publish
config=json.loads(Path('configs/p3_adapters.json').read_text())
p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','analyze']);a=p.parse_args()
path=Path('datasets/p3/fidelity-prefixes.json')
if a.phase=='freeze':
 if path.exists():raise FileExistsError(path)
 original=Path('runs/p3-adapter-protection-fable_f16-s17/responses.jsonl')
 rows=list(map(json.loads,original.read_text().splitlines()));chosen=rows[:8]+rows[-8:]
 out=Path('research/p3/fidelity-tokenization-v1');out.mkdir(exist_ok=False,parents=True)
 requests=[]
 with server(config['runtime']['binary'],config['models']['fable_f16']['path'],out,config['runtime']) as (url,command,proc):
  write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'command':command,'source':str(original),'source_sha256':file_sha256(original),
    'selection':'fixed first8 plus last8 protection rows; no quality/outcome filtering; tokenizer only, zero forward calls'})
  for row in chosen:
   prompt=row['steps'][0]['rendered_prompt']['prompt'];options=[]
   for add in [True,False]:
    req={'content':prompt,'add_special':add,'parse_special':True};response=request_json(url+'/tokenize',req);options.append({'request':req,'response':response})
   matches=[v['response']['tokens'] for v in options if len(v['response']['tokens'])==row['usage']['prompt_tokens']]
   if not matches:raise ValueError('native tokenization does not match recorded prompt')
   if len(matches[0])>512:raise ValueError('diagnostic prefix length')
   requests.append({'id':row['case']['id'],'prompt':prompt,'tokens':matches[0],'max_tokens':0,'tokenizer_options':options})
 write_json(path,requests);publish([path,out],'Freeze exact-token held-out Fable fidelity prefixes')
elif a.phase=='run':
 requests=json.loads(path.read_text())
 for key in ['fable_f16','fable_svd16','fable_svd32']:
  run(config['models'][key],requests,'fidelity-'+key,'adapter',{'filter':'result_norm'})
else:
 paths={key:Path(f'research/p3/native-adapter-fidelity-{key}/responses.jsonl') for key in ['fable_f16','fable_svd16','fable_svd32']}
 rows={k:list(map(json.loads,p.read_text().splitlines())) for k,p in paths.items()};results=[]
 for key in ['fable_svd16','fable_svd32']:
  for ref,item in zip(rows['fable_f16'],rows[key]):
   x=np.fromfile(ref['raw_response']['logits_path'],np.float32).astype(float);y=np.fromfile(item['raw_response']['logits_path'],np.float32).astype(float)
   lp=x-np.max(x);lp-=np.log(np.exp(lp).sum());lq=y-np.max(y);lq-=np.log(np.exp(lq).sum())
   prob=np.exp(lp);kl=float(np.sum(prob*(lp-lq)))
   results.append({'model':key,'id':item['case']['id'],'top1_equal':int(np.argmax(x))==int(np.argmax(y)),
                   'kl_reference_to_candidate':kl,'max_centered_logit_difference':float(np.max(np.abs((x-x.max())-(y-y.max()))))})
 out=Path('research/p3/fidelity-analysis-v1.json')
 if out.exists():raise FileExistsError(out)
 write_json(out,{'utc':utc_now(),'git':git_state(),'source_sha256':{k:file_sha256(p) for k,p in paths.items()},'cases':results,
   'summary':{k:{'n':sum(r['model']==k for r in results),'same_top1':sum(r['top1_equal'] for r in results if r['model']==k),
                 'mean_kl':float(np.mean([r['kl_reference_to_candidate'] for r in results if r['model']==k]))} for k in ['fable_svd16','fable_svd32']},
   'scope':'Fixed protection-prefix distribution fidelity, not coding superiority or a new independent adapter selection.'})
 publish([out],'Measure recovered Fable native distribution fidelity')
 print(json.dumps(json.loads(out.read_text())['summary']),flush=True)
