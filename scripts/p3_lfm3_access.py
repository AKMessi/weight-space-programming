"""Same-forward QAD readout, no-op and known-vector access gate."""
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from weightlab.evidence import file_sha256,utc_now,write_json
from weightlab.runtime import server,request_json
from p3_native_run import run
from portfolio_batch import publish
c=json.loads(Path('configs/p3_clarification.json').read_text());model=c['models']['lfm3_qad'];out=Path('research/p3/lfm3-access-tokenization');out.mkdir(parents=True,exist_ok=False)
cases=json.loads(Path('datasets/p3/evidence-development-original.json').read_text())['cases'][:4];requests=[];events=[]
with server(c['runtime']['binary'],model['path'],out,c['runtime']) as (url,command,proc):
 for case in cases:
  form={'messages':case['messages'],'chat_template_kwargs':{'enable_thinking':False}};rendered=request_json(url+'/apply-template',form);req={'content':rendered['prompt'],'add_special':True,'parse_special':True};raw=request_json(url+'/tokenize',req)
  if len(raw['tokens'])>512:raise ValueError('Prefix bound')
  events.append({'case':case,'template_request':form,'template_response':rendered,'token_request':req,'token_response':raw});requests.append({'id':case['id'],'prompt':rendered['prompt'],'tokens':raw['tokens'],'max_tokens':0})
 write_json(out/'record.json',{'utc':utc_now(),'command':command,'model':model,'events':events})
publish([out],'Freeze LFM3 QAD same-forward access requests')
settings={'filter':'result_norm','client_binary':'local/weightlab-native-v5','client_source':'native/trace_v5.cpp'}
rows=run(model,requests,'lfm3-readout','clarify',settings)
reader=GGUFReader(model['path']);ts={t.name:t for t in reader.tensors};tensor=ts.get('output.weight',ts['token_embd.weight']);w=dequantize(tensor.data,tensor.tensor_type).astype(np.float32);checks=[]
for row in rows:
 r=row['raw_response'];h=np.fromfile(r['captures']['result_norm']['path'],np.float32);manual=w@h;native=np.fromfile(r['logits_path'],np.float32);ids=[x['id'] for x in r['top64']];ref=int(native.argmax());a=native[ids]-native[ref];b=manual[ids]-manual[ref];maximum=float(abs(a-b).max());rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-6));checks.append({'id':r['id'],'max_centered_error':maximum,'relative_centered_error':rel,'top1_equal':int(manual.argmax())==ref,'passed':maximum<=.05 and rel<=.01 and int(manual.argmax())==ref})
noops=run(model,requests[:2],'lfm3-no-callback','clarify',{**settings,'capture':False,'callback':False})
vector=(.01*np.sin(np.arange(w.shape[1]))).astype(np.float32)
causal=run(model,[{**q,'intervention':{'tensor':'result_norm','vector':vector.tolist(),'scale':1.}} for q in requests[:2]],'lfm3-causal-readout','clarify',settings)
controls=[]
for mode,data in [('no_callback',noops),('known_vector',causal)]:
 for original,new in zip(rows,data):
  r=original['raw_response'];s=new['raw_response'];a=np.fromfile(r['logits_path'],np.float32);b=np.fromfile(s['logits_path'],np.float32);ref=int(a.argmax())
  if mode=='no_callback':maximum=float(abs(a-b).max());passed=maximum<=.05 and int(a.argmax())==int(b.argmax());record={'max_logit_difference':maximum,'exact_equal':bool(np.array_equal(a,b))}
  else:
   predicted=w@vector;shift=b-a;error=(shift-shift[ref])-(predicted-predicted[ref]);maximum=float(abs(error).max());h0=np.fromfile(r['captures']['result_norm']['path'],np.float32);h1=np.fromfile(s['captures']['result_norm']['path'],np.float32);state_error=float(abs(h1-h0-vector).max());passed=maximum<=.1 and state_error<=1e-5;record={'max_centered_shift_error':maximum,'max_state_delta_error':state_error}
  controls.append({'mode':mode,'id':s['id'],**record,'passed':bool(passed)})
result={'utc':utc_now(),'model':model,'readout_tensor':tensor.name,'readout_encoding':str(tensor.tensor_type),'checks':checks,'controls':controls,'passed':all(r['passed'] for r in checks+controls),'scope':'Same-forward final readout and interventions, not unverified intermediate/post-FFN-normalization algebra.'}
write_json('research/p3/lfm3-access-gate-v1.json',result);publish(['research/p3/lfm3-access-gate-v1.json'],'Validate LFM3 QAD final-state access and known interventions');print(json.dumps(result),flush=True)
