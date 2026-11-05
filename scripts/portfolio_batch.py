"""Sequential resumable pilot batches; commit/push each run, including failures."""
import argparse,json,subprocess,fcntl
from pathlib import Path
from weightlab.runner import execute
from weightlab.evidence import append_jsonl,utc_now,digest

ROOTS=['qwen3_q8','lfm3_qad','fable_f16']


def publish(paths,message):
    Path('local').mkdir(exist_ok=True)
    with Path('local/publication.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _publish(paths,message)


def _publish(paths,message):
    subprocess.run(['git','add',*map(str,paths)],check=True)
    staged=subprocess.run(['git','diff','--cached','--quiet']).returncode
    if staged:subprocess.run(['git','commit','-m',message],check=True)
    subprocess.run(['git','push','origin','main'],check=True)


def run_one(config,key,suite_path,label,seed=17,max_tokens=128,limit=None):
    out=Path(f'runs/p2-{label}-{key}-s{seed}')
    suite=json.loads(Path(suite_path).read_text())
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('status')=='complete':
            manifest=json.loads((out/'manifest.json').read_text())
            expected={'config_sha256':digest(config),'suite_sha256':digest(suite),
                      'model_key':key,'seed':seed,'max_tokens':max_tokens,'limit':limit}
            if any(manifest.get(k)!=v for k,v in expected.items()):
                raise ValueError(f'Resume inputs differ from completed immutable run: {out}')
            print('Already complete:',out,flush=True);return previous
        raise RuntimeError(f'Previous failed run requires an explicitly versioned retry: {out}')
    error=None
    try:summary=execute(config,key,suite,out,max_tokens,seed,limit)
    except BaseException as exc:
        error=exc
        summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {'status':'failed','error':repr(exc)}
    append_jsonl('research/portfolio-events.jsonl',{'utc':utc_now(),'label':label,'run':str(out),'summary':summary})
    with Path('research/JOURNAL.md').open('a') as f:
        f.write(f'\n### {utc_now()} — {out.name}\n\nRun status: {summary.get("status")}; generated tokens: {summary.get("generated_tokens",0)}. Group counts: `{json.dumps(summary.get("groups",{}),sort_keys=True)}`. Full immutable requests, raw responses, and settings are in `{out}`. These counts are observations; interpretation is deferred to the study report.\n')
    publish([out,'research/budget.jsonl','research/portfolio-events.jsonl','research/JOURNAL.md'],f'Record {label} {key} seed {seed} {summary.get("status")}')
    if error:raise error
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['e1dev','e1confirm','e2dev','e2schema','e2confirm','e4'])
    p.add_argument('--config',default='configs/portfolio.json');a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    if a.phase=='e1dev':
        for root in ROOTS:
            for suffix in ['', '_A','_B','_R']:
                run_one(config,root+suffix,'datasets/portfolio/e1-development.json','e1dev',max_tokens=192)
        for key in ['parent_uniform_bf16','fable_v_only','fable_all_qkv_parent']:
            run_one(config,key,'datasets/portfolio/e1-development.json','e1dev',max_tokens=192)
    elif a.phase=='e1confirm':
        selected=json.loads(Path('research/selections/e1.json').read_text())
        for root in ROOTS:
            for key in [root,selected['models'][root]['diagnostic_or_selected']]:
                for seed in [17,29]:run_one(config,key,'datasets/portfolio/e1-confirmation.json','e1confirm',seed,max_tokens=192)
        for seed in [17,29]:run_one(config,'parent_uniform_bf16','datasets/portfolio/e1-confirmation.json','e1confirm',seed,max_tokens=192)
    elif a.phase=='e2dev':
        for root in ROOTS:
            for suffix,control in [('', 'original'),('_A','original'),('_B','original'),('','strong'),('','grammar')]:
                run_one(config,root+suffix,f'datasets/portfolio/e2-development-{control}.json',f'e2dev-{control}')
    elif a.phase=='e2schema':
        for root in ROOTS:run_one(config,root,'datasets/portfolio/e2-development-grammar-v2.json','e2dev-grammar-v2')
    elif a.phase=='e2confirm':
        selected=json.loads(Path('research/selections/e2.json').read_text())
        for root in ROOTS:
            for key,control in [(root,'original'),(selected['models'][root]['diagnostic_or_selected'],'original'),(root,'strong'),(root,'grammar')]:
                version='-v2' if control=='grammar' else ''
                run_one(config,key,f'datasets/portfolio/e2-confirmation-{control}{version}.json',f'e2confirm-{control}{version}')
    elif a.phase=='e4':
        for root in ROOTS:
            run_one(config,root+'_AB','datasets/portfolio/engineering.json','e4engineering')
            run_one(config,root+'_BA','datasets/portfolio/engineering.json','e4ordercheck',limit=6)
            for suffix in ['_R','_AB']:
                run_one(config,root+suffix,'datasets/portfolio/e2-development-original.json','e4joint')


if __name__=='__main__':main()
