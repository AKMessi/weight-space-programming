"""Four-format follow-up after an unexpected deployment timing difference."""
import argparse,copy,json,subprocess
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.runtime import runtime_libraries,server,request_json
from portfolio_batch import run_one

p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
if a.stage=='prepare':
    config=json.loads(Path('configs/portfolio_precision.json').read_text());base=config['models']['fable_float']
    binary=str(Path(config['runtime']['binary']).with_name('llama-quantize'))
    output=Path('artifacts/portfolio/fable_Q4_K_M.gguf');manifest=Path('research/patches/portfolio-fable_Q4_K_M.json')
    if output.exists() or manifest.exists():raise FileExistsError('version preparation instead of overwriting')
    command=[binary,base['path'],str(output),'Q4_K_M','4']
    log=Path('research/quantization/fable_Q4_K_M.log')
    with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    reference='0c40569a1736081c7f33d33ec18208cf089e8ae0e762806c515281740e60c2f3'
    report={'utc':utc_now(),'git':git_state(),'base':base,'output':str(output),'command':command,'status':'verified',
       'edited_file_sha256':file_sha256(output),'bytes':output.stat().st_size,'binary_sha256':file_sha256(binary),
       'libraries':runtime_libraries(binary),'published_reference_sha256':reference,
       'matches_published_reference':file_sha256(output)==reference,'log_sha256':file_sha256(log)}
    if file_sha256(base['path'])!=base['sha256']:raise ValueError('original changed')
    write_json(manifest,report)
    config['models']['fable_Q4_K_M']={**base,'path':str(output),'sha256':report['edited_file_sha256'],
         'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
    for key in ['fable_float','fable_Q8_0','fable_Q4_0','fable_Q4_K_M']:
        config['models'][key]['sampling'].update(temperature=.1,repeat_penalty=1.0,ignore_eos=True)
    write_json('configs/fable_speed_followup.json',config)
    cases=[{'id':f'speed-{i}','group':'warmup' if i==0 else 'decode_measurement','family':'timing','split':'timing_diagnostic',
       'grader':'measurement_only','expected':None,'messages':[{'role':'user','content':'Write a detailed story about a robot tending a garden, using at least 200 words.'}]} for i in range(4)]
    write_json('datasets/portfolio/fable-speed-followup.json',{'name':'fable-four-format-timing-v1',
       'status':'post-hoc timing follow-up; first case warms model; three repeated fixed-prompt measurements; no quality score', 'cases':cases})
    print('Q4_K_M matches public release:',report['matches_published_reference'],flush=True)
else:
    config=json.loads(Path('configs/fable_speed_followup.json').read_text())
    # Separate startup trace; benchmark runs retain the original default verbosity.
    trace=Path('research/fable-backend-startup-trace');trace.mkdir(exist_ok=False)
    runtime={**config['runtime'],'log_verbosity':5}
    with server(runtime['binary'],config['models']['fable_Q8_0']['path'],trace,runtime) as (url,command,proc):
        write_json(trace/'runtime.json',{'utc':utc_now(),'git':git_state(),'command':command,'props':request_json(url+'/props'),
                   'model_sha256':file_sha256(config['models']['fable_Q8_0']['path']),'binary_sha256':file_sha256(runtime['binary'])})
    for key in ['fable_float','fable_Q8_0','fable_Q4_0','fable_Q4_K_M']:
        run_one(config,key,'datasets/portfolio/fable-speed-followup.json','speed-followup',max_tokens=64)
