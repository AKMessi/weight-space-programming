"""Matched prompt/schema baselines, frozen before confirmation outcomes."""
import argparse,json
from pathlib import Path
from weightlab.evidence import write_json
from portfolio_batch import run_one,ROOTS

p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
if a.action=='freeze':
    for split in ['development','confirmation']:
        for control in ['strong','grammar']:
            suite=json.loads(Path(f'datasets/portfolio/e1-{split}.json').read_text())
            suite['name']+=f'-{control}'
            for case in suite['cases']:
                case['control']=control
                case['messages'][0]['content']+=' For copying, exclude the opening/closing delimiter markers themselves. For code-only requests output Python directly, without backticks. For JSON output just one JSON object and stop. When an explanation is explicitly requested include it.'
                if control=='grammar' and case['grader']=='json':
                    case['request_overrides']={'response_format':{'type':'json_schema','schema':{'type':'object',
                        'properties':{'tag':{'type':'string'},'count':{'type':'integer'}},'required':['tag','count'],'additionalProperties':False}}}
            write_json(f'datasets/portfolio/e1-{split}-{control}.json',suite)
else:
    config=json.loads(Path('configs/portfolio.json').read_text())
    for split in ['development','confirmation']:
        for root in ROOTS:
            for control in ['strong','grammar']:
                run_one(config,root,f'datasets/portfolio/e1-{split}-{control}.json',f'e1-{split}-{control}',max_tokens=192)
