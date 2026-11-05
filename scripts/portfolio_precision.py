"""Prepare pinned floating references, controlled quantizations, and symmetry cases."""
import argparse,json,re,subprocess,time
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.transforms import intervene,rewrite
from weightlab.runtime import runtime_libraries


def register(config,key,report,manifest):
    spec={**report['base'],'path':report['output'],'sha256':report['edited_file_sha256'],
          'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
    config['models'][key]=spec
    return spec


def existing(manifest):
    if not Path(manifest).exists():return None
    report=json.loads(Path(manifest).read_text())
    if report['status']!='verified' or file_sha256(report['output'])!=report['edited_file_sha256']:
        raise ValueError(f'previous artifact invalid: {manifest}')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['floating','quantize','symmetry']);a=p.parse_args()
    config=json.loads(Path('configs/floating_sources.json' if a.stage=='floating' else 'configs/portfolio_precision.json').read_text())
    if a.stage=='floating':
        manifest='research/patches/portfolio-qwen-float.json'
        r=existing(manifest) or rewrite(config['models']['qwen3_bf16'],'artifacts/portfolio/qwen-float.gguf',manifest,mode='f16')
        register(config,'qwen_float',r,manifest)
        config['models']['fable_float']=config['models']['fable_f16']
        for root in ['qwen','lfm3','fable']:
            base=config['models'][root+'_float']
            reader=GGUFReader(base['path'])
            layer=max(int(m[1]) for t in reader.tensors if (m:=re.fullmatch(r'blk\.(\d+)\.ffn_norm\.weight',t.name)))
            op=[{'tensor':f'blk.{layer}.ffn_norm.weight','kind':'scale','factor':.9}]
            manifest=f'research/patches/portfolio-{root}-float-A.json'
            r=existing(manifest) or intervene(base,op,f'artifacts/portfolio/{root}-float-A.gguf',manifest)
            register(config,root+'_float_A',r,manifest)
        # The QAD no-op preserves every quantized tensor byte and all original metadata.
        base=json.loads(Path('configs/portfolio.json').read_text())['models']['lfm3_qad']
        manifest='research/patches/portfolio-qad-rewrite.json'
        r=existing(manifest) or rewrite(base,'artifacts/portfolio/qad-rewrite.gguf',manifest)
        config['models']['lfm3_qad']=base
        register(config,'qad_rewrite',r,manifest)
    elif a.stage=='quantize':
        binary=str(Path(config['runtime']['binary']).with_name('llama-quantize'))
        version=subprocess.run([binary,'--version'],capture_output=True,text=True)
        provenance={'binary_sha256':file_sha256(binary),'version':version.stdout+version.stderr,
                    'libraries':runtime_libraries(binary)}
        for root in ['qwen','lfm3','fable']:
            for suffix in ['','_A']:
                base=config['models'][root+'_float'+suffix]
                for precision in ['Q8_0','Q4_0']:
                    key=f'{root}_{precision}{suffix}'
                    manifest=f'research/patches/portfolio-{key}.json'
                    r=existing(manifest)
                    if r is None:
                        output=f'artifacts/portfolio/{key}.gguf'
                        if Path(output).exists():raise FileExistsError(output)
                        log=Path(f'research/quantization/{key}.log');log.parent.mkdir(exist_ok=True)
                        command=[binary,base['path'],output,precision,'4']
                        r={'utc':utc_now(),'git':git_state(),'base':base,'output':output,'quantizer':provenance,
                           'command':command,'status':'started','precision':precision}
                        write_json(manifest,r);start=time.monotonic()
                        try:
                            with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                            if file_sha256(base['path'])!=base['sha256']:raise ValueError('original hash')
                            source={t.name:t for t in GGUFReader(base['path']).tensors}
                            norms=[t for t in GGUFReader(output).tensors if t.name.endswith('norm.weight')]
                            checks={t.name:bool(np.array_equal(t.data,source[t.name].data)) for t in norms}
                            if not all(checks.values()):raise ValueError('normalization values changed')
                            r.update(status='verified',edited_file_sha256=file_sha256(output),bytes=Path(output).stat().st_size,
                                     seconds=time.monotonic()-start,normalization_bytes_preserved=checks,log_sha256=file_sha256(log))
                        except BaseException as exc:r.update(status='failed',error=repr(exc));raise
                        finally:write_json(manifest,r)
                    register(config,key,r,manifest)
                    print('verified',key,flush=True)
    elif a.stage=='symmetry':
        numerical=[]
        for root in ['qwen','lfm3','fable']:
            base=config['models'][root+'_float']
            reader=GGUFReader(base['path']);tensors={t.name:t for t in reader.tensors}
            norm='output_norm.weight' if 'output_norm.weight' in tensors else 'token_embd_norm.weight'
            faults={'healthy':[], 'output_sign':[{'tensor':norm,'kind':'scale','factor':-1.0}],
                    'ffn_sign':[{'tensor':f'blk.{i}.ffn_down.weight','kind':'scale','factor':-1.0} for i in range(3)]}
            for state,ops in faults.items():
                key=f'{root}_diagnosis_{state}'
                if not ops:config['models'][key]=base
                else:
                    manifest=f'research/patches/portfolio-{key}.json'
                    r=existing(manifest) or intervene(base,ops,f'artifacts/portfolio/{key}.gguf',manifest)
                    register(config,key,r,manifest)
                manifest=f'research/patches/portfolio-{key}-permuted.json'
                r=existing(manifest) or rewrite(config['models'][key],f'artifacts/portfolio/{key}-permuted.gguf',manifest,mode='permute')
                permuted=register(config,key+'_permuted',r,manifest)
                left={t.name:t.data.astype(np.float32) for t in GGUFReader(config['models'][key]['path']).tensors if t.name.startswith('blk.0.ffn_') and t.name.endswith('.weight') and t.data.ndim==2}
                right={t.name:t.data.astype(np.float32) for t in GGUFReader(permuted['path']).tensors if t.name.startswith('blk.0.ffn_') and t.name.endswith('.weight') and t.data.ndim==2}
                x=np.random.default_rng(919).normal(0,.02,(4,left['blk.0.ffn_gate.weight'].shape[1])).astype(np.float32)
                def mlp(t):
                    gate=x@t['blk.0.ffn_gate.weight'].T;up=x@t['blk.0.ffn_up.weight'].T
                    return ((gate/(1+np.exp(-gate)))*up)@t['blk.0.ffn_down.weight'].T
                u,v=mlp(left),mlp(right)
                numerical.append({'model':root,'state':state,'synthetic_inputs':4,'seed':919,
                    'max_abs_difference':float(np.max(np.abs(u-v))),
                    'relative_l2_difference':float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-30)),
                    'within_tolerance':bool(np.allclose(u,v,rtol=1e-4,atol=1e-4))})
                print('symmetry prepared',key,numerical[-1],flush=True)
        write_json('research/symmetry-numerical.json',{'utc':utc_now(),'git':git_state(),'checks':numerical,
                   'note':'Gated-MLP FP32 numerical checks; actual runtime output parity is tested separately. No editor received these forward calculations.'})
    write_json('configs/portfolio_precision.json',config)


if __name__=='__main__':main()
