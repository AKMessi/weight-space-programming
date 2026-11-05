"""Lossless residual after a portable dyadic low-rank predictor; zero forwards."""
import json,time
from pathlib import Path
from weightlab.adapters import merge
from weightlab.patchcodec import create_patch,apply_patch_file
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from portfolio_batch import publish
config=json.loads(Path('configs/p3_adapters.json').read_text());base=config['models']['parent_uniform_bf16'];target=config['models']['fable_f16']
out=Path('research/p3/lossless-capsules-v1');out.mkdir(parents=True,exist_ok=False);results=[]
write_json(out/'manifest.json',{'utc':utc_now(),'git':git_state(),'base':base,'target':target,'ranks':[16,32,64],
                               'predictor_arithmetic':'exact-f16-dyadic-int64-v1','generated_tokens':0,'scored_tokens':0})
try:
 for rank in [16,32,64]:
  key=f'fable_svd{rank}';factor=config['models'][key]['package_path'];prefix=Path(f'artifacts/p3/capsules/rank{rank}')
  start=time.monotonic();prediction=merge(base['path'],factor,str(prefix)+'-predictor.gguf',out/f'rank{rank}-predictor.json',deterministic=True)
  intermediate={'path':prediction['output'],'sha256':prediction['edited_file_sha256']}
  patch=create_patch(intermediate,target,str(prefix)+'-residual.wpatch',out/f'rank{rank}-residual.json')
  restore=apply_patch_file(intermediate['path'],patch['output'],str(prefix)+'-restored.gguf',out/f'rank{rank}-restore.json')
  row={'rank':rank,'factor_package_bytes':Path(factor).stat().st_size,'factor_package_sha256':file_sha256(factor),
       'residual_patch_bytes':patch['patch_bytes'],'total_delivery_bytes':Path(factor).stat().st_size+patch['patch_bytes'],
       'base_sha256':base['sha256'],'predictor_sha256':intermediate['sha256'],'target_sha256':restore['target_sha256'],
       'exact_target_identity':restore['byte_identical'],'seconds':time.monotonic()-start}
  results.append(row);write_json(out/f'rank{rank}-summary.json',row)
  publish([out],f'Verify exact Fable low-rank-plus-residual capsule rank {rank}')
  print(json.dumps(row),flush=True)
 write_json(out/'results.json',{'utc':utc_now(),'git':git_state(),'status':'complete','candidates':results,
   'baseline_exact_patch_bytes':20308656,'base_excluded':True,'algorithm_claim':'Established low-rank prediction and lossless residual coding; one precision-controlled delivery case.'})
except BaseException as exc:
 write_json(out/'failure.json',{'utc':utc_now(),'error':repr(exc),'finished':results});publish([out],'Preserve failed exact capsule experiment');raise
publish([out],'Complete all three predeclared exact Fable delivery capsules')
