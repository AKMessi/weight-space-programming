"""Rebuild a low-rank-plus-exact-residual capsule in a new ignored directory."""
import argparse,json,platform,sys
from pathlib import Path
import numpy as np
from weightlab.adapters import pack,merge
from weightlab.patchcodec import create_patch,apply_patch_file
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/p3_adapters.json');p.add_argument('--base',default='parent_uniform_bf16');p.add_argument('--target',default='fable_f16');p.add_argument('--rank',type=int,default=32);p.add_argument('--word-method',choices=['xor','signed','zigzag_shuffle'],default='zigzag_shuffle');p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if not any(out.resolve().is_relative_to(Path(d).resolve()) for d in ['local','artifacts']):raise ValueError('Use an ignored local/ or artifacts/ output directory')
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);c=json.loads(Path(a.config).read_text());base=c['models'][a.base];target=c['models'][a.target]
write_json(out/'invocation.json',{'utc':utc_now(),'git':git_state(),'argv':sys.argv,'config_sha256':file_sha256(a.config),'python':platform.python_version(),'numpy':np.__version__,'base':base,'target':target,'no_model_forwards':True,'scope':'New reproduction; archive bytes/factor signs can vary with numeric libraries, restored target identity must not.'})
try:
 package=out/'factors.npz';packed=pack(base,target,package,out/'factors.json',method='svd',rank=a.rank)
 prediction=merge(base['path'],package,out/'predictor.gguf',out/'predictor.json',deterministic=True)
 residual=create_patch({'path':prediction['output'],'sha256':prediction['edited_file_sha256']},target,out/'residual.wpatch',out/'residual.json',word_method=None if a.word_method=='xor' else a.word_method)
 restored=apply_patch_file(prediction['output'],residual['output'],out/'restored.gguf',out/'restore.json')
 result={'utc':utc_now(),'status':'verified','base_required_sha256':base['sha256'],'target_sha256':restored['target_sha256'],'byte_identical':restored['byte_identical'],'factors_bytes':package.stat().st_size,'residual_bytes':residual['patch_bytes'],'total_delivery_bytes':package.stat().st_size+residual['patch_bytes'],'rank':a.rank,'word_method':a.word_method,'note':'Delivery size excludes the required base and merged runtime model. Established coding, not a new algorithm.'}
 write_json(out/'summary.json',result);print(json.dumps(result,indent=2))
except BaseException as exc:write_json(out/'failure.json',{'utc':utc_now(),'error':repr(exc)});raise
