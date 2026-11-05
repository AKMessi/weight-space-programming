"""Same-forward state/logit parity with exact-version native libraries."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.budget import reserve,settle
from weightlab.evidence import append_jsonl,digest,file_sha256,git_state,utc_now,write_json
from weightlab.resources import monitor
from weightlab.runtime import runtime_libraries
from portfolio_batch import publish

ROOT=Path('research/p3/native-gate-v2')
CONFIG=json.loads(Path('configs/portfolio_precision.json').read_text())

def freeze():
    path=Path('datasets/p3/native-gate-v1.json')
    if path.exists():raise FileExistsError(path)
    forms=['Return only JSON: tag=NX-47, count=9. Assistant:',
           'Explain why def add(a,b): return a+b works. Assistant:',
           'Copy only the payload inside brackets: [Blue Raven 407]. Assistant:',
           'Continue Python: def last(xs):\n    ',
           'Record: city=Oslo. What is the city? Assistant:',
           'Required field: port. Facts: host=local. Ask for any absent required value. Assistant:']
    cases=[]
    for i,p in enumerate(forms):
        for j in range(4):
            prefix=('Background note: the archive lists ordinary numbered objects. '*j)
            suffix=['',' {',' The answer','\nIt follows that'][j]
            cases.append({'id':f'gate-{i}-{j}','prompt':prefix+p+suffix,'diagnostic_only':True})
    write_json(path,{'status':'visible compatibility calibration, never quality confirmation','cases':cases})

def run(backend,attempt=None):
    out=ROOT/(attempt or backend)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    suite=json.loads(Path('datasets/p3/native-gate-v1.json').read_text());model=CONFIG['models']['qwen_float']
    if file_sha256(model['path'])!=model['sha256']:raise ValueError('Source model hash')
    binary=Path('local/weightlab-native').resolve();libs='/home/akmessi/llama-vulkan-build/bin'
    base={'model':str(Path(model['path']).resolve()),'library_dir':libs,'context':2048,
          'gpu_layers':0 if backend=='cpu' else 99,'capture':True,'callback':True,'filter':'result_norm'}
    write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'model':model,'suite_sha256':digest(suite),
       'binary_sha256':file_sha256(binary),'native_source_revision':'338fce6511c744893cc0d3d958b275f40552c7a4',
       'source_sha256':file_sha256('native/trace.cpp'),'libraries':runtime_libraries(libs+'/llama-server'),'config':base})
    reader=GGUFReader(model['path']);ts={t.name:t for t in reader.tensors}
    head=ts.get('output.weight',ts['token_embd.weight']).data.astype(np.float32)
    results=[];error=None
    with Path('local/execution.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for mode in ['trace','no_callback','causal']:
                cfg={**base,'callback':mode!='no_callback'};configpath=out/(mode+'-config.json');write_json(configpath,cfg)
                cases=suite['cases'] if mode=='trace' else suite['cases'][:2]
                with (out/(mode+'-native.log')).open('w') as log:
                    proc=subprocess.Popen([str(binary),str(configpath)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
                    try:
                        with monitor(proc.pid) as metrics:
                            ready=proc.stdout.readline()
                            if not ready:raise RuntimeError('Native startup failed; inspect log')
                            write_json(out/(mode+'-ready.json'),json.loads(ready))
                            for case in cases:
                                req={**case,'trace_dir':str((Path('local/p3-traces')/out.name/mode/case['id']).resolve())}
                                if mode=='causal':req['intervention']={'tensor':'result_norm','scale':1.,'vector':(.01*np.sin(np.arange(head.shape[1]))).astype(np.float32).tolist()}
                                rid=f'p3/native-v2/{out.name}/{mode}/{case["id"]}'
                                reserve(rid,512,1000000,'research/scored-budget.jsonl')
                                append_jsonl(out/'events.jsonl',{'event':'request','utc':utc_now(),'request_id':rid,'request':req})
                                proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush()
                                raw=proc.stdout.readline()
                                if not raw:raise RuntimeError('Native process terminated during request')
                                response=json.loads(raw)
                                append_jsonl(out/'events.jsonl',{'event':'response','request_id':rid,'response':response})
                                if response['status']!='complete':raise RuntimeError(response['error'])
                                settle(rid,response['prompt_tokens'],'research/scored-budget.jsonl')
                                response['array_hashes']={k:file_sha256(v['path']) for k,v in response['captures'].items()}
                                response['logits_sha256']=file_sha256(response['logits_path'])
                                logits=np.fromfile(response['logits_path'],dtype=np.float32)
                                row={'mode':mode,'case':case,'request':req,'raw_response':response}
                                if mode!='no_callback':
                                    h=np.fromfile(response['captures']['result_norm']['path'],dtype=np.float32)
                                    readout=h.astype(np.float16).astype(np.float32) if backend=='cpu' else h
                                    manual=head@readout;ids=np.array([x['id'] for x in response['top64']]);ref=int(np.argmax(logits))
                                    a=logits[ids]-logits[ref];b=manual[ids]-manual[ref]
                                    maximum=float(np.max(np.abs(a-b)));relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-6))
                                    same=int(np.argmax(manual))==ref
                                    threshold=.001 if backend=='cpu' else .05;reltol=.001 if backend=='cpu' else .01
                                    row['comparison']={'top1_equal':same,'max_centered_error':maximum,'relative_centered_error':relative,
                                                       'passed':same and maximum<=threshold and relative<=reltol}
                                append_jsonl(out/'responses.jsonl',row);results.append(row)
                                print(backend,mode,case['id'],row.get('comparison',{}),flush=True)
                        write_json(out/(mode+'-resources.json'),metrics)
                    finally:
                        if proc.poll() is None:
                            proc.stdin.close()
                            try:proc.wait(timeout=20)
                            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            traces={r['case']['id']:r for r in results if r['mode']=='trace'};controls=[]
            for r in results:
                if r['mode']=='trace':continue
                original=traces[r['case']['id']]['raw_response'];new=r['raw_response']
                a=np.fromfile(original['logits_path'],dtype=np.float32);b=np.fromfile(new['logits_path'],dtype=np.float32)
                if r['mode']=='no_callback':
                    maximum=float(np.max(np.abs(a-b)));limit=.001 if backend=='cpu' else .05
                    controls.append({'mode':r['mode'],'id':r['case']['id'],'max_abs_logit_difference':maximum,
                                     'exact_equal':bool(np.array_equal(a,b)),'passed':bool(maximum<=limit and np.argmax(a)==np.argmax(b))})
                else:
                    h0=np.fromfile(original['captures']['result_norm']['path'],dtype=np.float32)
                    h1=np.fromfile(new['captures']['result_norm']['path'],dtype=np.float32)
                    vector=np.array(r['request']['intervention']['vector'],dtype=np.float32)
                    effective=(h1.astype(np.float16).astype(np.float32)-h0.astype(np.float16).astype(np.float32)) if backend=='cpu' else vector
                    predicted=head@effective;ref=int(np.argmax(a));difference=(b-a)-(b-a)[ref];prediction=predicted-predicted[ref]
                    maximum=float(np.max(np.abs(difference-prediction)));limit=.002 if backend=='cpu' else .1
                    controls.append({'mode':r['mode'],'id':r['case']['id'],'max_state_delta_error':float(np.max(np.abs(h1-h0-vector))),
                         'max_centered_shift_error':maximum,'passed':bool(maximum<=limit and np.max(np.abs(h1-h0-vector))<=1e-5)})
            write_json(out/'controls.json',controls)
            checks=[r['comparison']['passed'] for r in results if 'comparison' in r]
            write_json(out/'summary.json',{'status':'complete','boundary_passes':sum(checks),'boundary_checks':len(checks),
                         'controls':controls,'gate_passed':all(checks) and all(c['passed'] for c in controls)})
        except BaseException as exc:
            error=exc;write_json(out/'summary.json',{'status':'failed','error':repr(exc)})
    publish([out,'research/scored-budget.jsonl'],f'Record P3 same-forward gate {backend}')
    if error:raise error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','cpu','vulkan']);p.add_argument('--attempt');a=p.parse_args()
    freeze() if a.action=='freeze' else run(a.action,a.attempt)
