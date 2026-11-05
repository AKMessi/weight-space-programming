"""Post-hoc exact-site no-op audit; cannot select another patch."""
import json
from pathlib import Path
import numpy as np
from weightlab.evidence import file_sha256,utc_now,write_json
from weightlab.grading import grade
from p3_native_run import run
from portfolio_batch import publish
c=json.loads(Path('configs/p3_repair.json').read_text());old=json.loads(Path('research/p3/native-repair-causal-cpu-direction-v1/requests.json').read_text())
requests=[{**q,'id':q['id']+'-zero','intervention':{**q['intervention'],'scale':0.}} for q in old[:16]]
results=run(c['models']['fable_float'],requests,'component-exact-site-zero','repair',{'capture':False})
baseline=list(map(json.loads,Path('runs/p3-repair-switch-dev-original-fable_float-s17/responses.jsonl').read_text().splitlines()))[:16];checks=[]
for original,row in zip(baseline,results):
 g=grade(original['case'],row['raw_response']['completion']);checks.append({'id':original['case']['id'],'content_equal':original['content']==row['raw_response']['completion'],'baseline_grading':original['grading'],'zero_grading':g,'zero_rescued_plain':original['case']['response_mode']=='plain' and not original['grading']['success'] and g['success']})
prefixes=json.loads(Path('datasets/p3/fidelity-prefixes.json').read_text())[:2]
rows=run(c['models']['fable_float'],prefixes,'fable-fidelity-no-callback','measurement',{'capture':False,'callback':False})
references=list(map(json.loads,Path('research/p3/native-adapter-fidelity-fable_f16/responses.jsonl').read_text().splitlines()))[:2];readout=[]
for ref,row in zip(references,rows):
 a=np.fromfile(ref['raw_response']['logits_path'],np.float32);b=np.fromfile(row['raw_response']['logits_path'],np.float32);maximum=float(abs(a-b).max());readout.append({'id':row['case']['id'],'max_abs_logit_difference':maximum,'exact_equal':bool(np.array_equal(a,b)),'passed':maximum<=.05 and int(a.argmax())==int(b.argmax())})
report={'utc':utc_now(),'status':'post-hoc diagnostic','component_zero':checks,'component_all_exact':all(r['content_equal'] for r in checks),'zero_plain_rescues':sum(r['zero_rescued_plain'] for r in checks),'readout_no_callback':readout,'readout_passed':all(r['passed'] for r in readout),'reversible_attribution_unconfounded_in_tested_cases':all(r['content_equal'] for r in checks)}
out=Path('research/p3/instrumentation-noops-v1.json');write_json(out,report);publish([out],'Audit exact-site zero intervention and Fable readout no-op');print(json.dumps(report),flush=True)
