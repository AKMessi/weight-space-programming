"""Freeze anonymized diagnosis presentations only after phenotype/parity screening."""
import argparse,json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from portfolio_batch import run_one


def read(root,state,permuted=False):
    key=f'{root}_diagnosis_{state}'+('_permuted' if permuted else '')
    path=Path(f'runs/p2-e5probes-{key}-s17/responses.jsonl')
    summary=json.loads((path.parent/'summary.json').read_text())
    if summary['status']!='complete':raise ValueError('incomplete phenotype screen')
    return [json.loads(x) for x in path.read_text().splitlines()],path,key


def features(path):
    tensors={t.name:t for t in GGUFReader(path).tensors}
    norm=tensors.get('output_norm.weight',tensors.get('token_embd_norm.weight')).data.astype(np.float64)
    def stats(x):
        return {'mean':round(float(x.mean()),7),'std':round(float(x.std()),7),
                'negative_fraction':round(float(np.mean(x<0)),5)}
    output={'output_normalization':stats(norm),'output_normalization_sample':np.round(norm[:8],4).tolist(),
            'ffn_down_statistics':{},'first_layer_samples':{}}
    for i in range(3):output['ffn_down_statistics'][str(i)]=stats(tensors[f'blk.{i}.ffn_down.weight'].data.astype(np.float32))
    for kind in ['gate','up','down']:
        output['first_layer_samples'][kind]=np.round(tensors[f'blk.0.ffn_{kind}.weight'].data[:3,:8].astype(np.float64),4).tolist()
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args()
    if a.stage=='run':
        config=json.loads(Path('configs/diagnosis_editor.json').read_text())
        run_one(config,'local_editor','datasets/portfolio/diagnosis-blinded.json','e5editor',max_tokens=128)
        return
    if Path('datasets/portfolio/diagnosis-blinded.json').exists():raise FileExistsError('already frozen')
    config=json.loads(Path('configs/portfolio_precision.json').read_text())
    numerical=json.loads(Path('research/symmetry-numerical.json').read_text())
    numeric={(r['model'],r['state']):r['within_tolerance'] for r in numerical['checks']}
    report={'utc':utc_now(),'git':git_state(),'screens':[],'sources':{},'editor':'pinned local Qwen3 Q8; no API or interactive tools',
            'limitations':['Synthetic conspicuous faults; no natural-failure claim.','Bounded summaries/samples, not full-weight interactive inspection.','Only one small local editor and one decoding seed.']}
    cases=[];rng=np.random.default_rng(711)
    schema={'type':'object','properties':{'diagnosis':{'type':'string','enum':['healthy','degraded']},
            'confidence':{'type':'number','minimum':0,'maximum':1}},'required':['diagnosis','confidence'],'additionalProperties':False}
    instruction=('Diagnose this anonymous language-model variant as healthy (intact relative to its original) or degraded (at least a 25 percentage-point loss on simple supplied-record extraction tasks). '
                 'You have no healthy reference checkpoint. Return only JSON with diagnosis and confidence, a number from 0 to 1 expressing confidence in your chosen diagnosis. ')
    for root in ['qwen','lfm3','fable']:
        baseline,base_path,_=read(root,'healthy')
        baseline_score=sum(r['grading']['payload_correct'] for r in baseline)
        for state in ['healthy','output_sign','ffn_sign']:
            original,path,key=read(root,state);permuted,perm_path,perm_key=read(root,state,True)
            for source in [path,perm_path,base_path]:report['sources'][str(source)]=file_sha256(source)
            score=sum(r['grading']['payload_correct'] for r in original)
            decline=(baseline_score-score)/len(original)
            phenotype=state=='healthy' or decline>=.25
            parity=all(x['content']==y['content'] and x['finish_reason']==y['finish_reason'] and
                       x['usage']['completion_tokens']==y['usage']['completion_tokens'] for x,y in zip(original,permuted))
            screen={'model':root,'state':state,'healthy_payload_successes':baseline_score,'variant_payload_successes':score,
                    'n':len(original),'decline':decline,'clear_phenotype':phenotype,'numerical_parity':numeric[(root,state)],
                    'runtime_content_parity':parity,'presentations':[]}
            for layout,rows,target_key in [('original',original,key),('permuted',permuted,perm_key)]:
                valid=phenotype and (layout=='original' or parity and numeric[(root,state)])
                entry={'layout':layout,'included':valid,'reason':None if valid else ('no 25pp behavioral decline' if not phenotype else 'failed symmetry parity')}
                if valid:
                    anonymous=f'case-{int(rng.integers(10**8,10**9)):09d}'
                    f=features(config['models'][target_key]['path'])
                    weight_prediction='degraded' if f['output_normalization']['mean']<0 else 'healthy'
                    behavior_prediction='degraded' if not all(r['grading']['payload_correct'] for r in rows[:2]) else 'healthy'
                    label='healthy' if state=='healthy' else 'degraded'
                    entry.update(anonymous_id=anonymous,label=label,weight_anomaly_baseline=weight_prediction,
                                 behavior_baseline=behavior_prediction)
                    for arm in ['weights','behavior']:
                        evidence={'architecture':root,'weight_observations':f} if arm=='weights' else {
                            'architecture':root,'probes':[{'messages':r['case']['messages'],'output':r['content'],
                                                          'schema_constrained':bool(r['case'].get('request_overrides',{}).get('response_format'))} for r in rows[:2]]}
                        prompt=instruction+f'Presentation ID: {anonymous}. Evidence arm: {arm}.\n'+json.dumps(evidence,separators=(',',':'))
                        cases.append({'id':anonymous+'-'+arm,'split':'blinded_diagnosis','family':arm,'group':arm,
                             'grader':'diagnosis','expected':label,'messages':[{'role':'system','content':'Use only the provided anonymous evidence. No tools or target inference are available.'},
                             {'role':'user','content':prompt}], 'request_overrides':{'temperature':0.0,
                             'response_format':{'type':'json_schema','json_schema':{'name':'diagnosis','strict':True,'schema':schema}}}})
                screen['presentations'].append(entry)
            report['screens'].append(screen)
    rng.shuffle(cases)
    write_json('research/diagnosis-screening.json',report)
    write_json('datasets/portfolio/diagnosis-blinded.json',{'name':'anonymous-diagnosis-v1','status':'fixed editor inputs after disclosed fault screening; expected labels never sent to editor','cases':cases})
    original=json.loads(Path('configs/portfolio.json').read_text())
    editor={'runtime':original['runtime'],'budgets':original['budgets'],'models':{'local_editor':original['models']['qwen3_q8']}}
    write_json('configs/diagnosis_editor.json',editor)
    print('Frozen presentations:',len(cases),'editor calls; screens:',len(report['screens']))


if __name__=='__main__':main()
