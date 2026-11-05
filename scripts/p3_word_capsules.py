"""Two frozen reversible residual transforms for the existing rank32 predictor."""
import json
from pathlib import Path
from weightlab.evidence import file_sha256,utc_now,write_json
from weightlab.patchcodec import create_patch,apply_patch_file
from portfolio_batch import publish
ROOT=Path('research/p3/lossless-word-residuals-v1');ROOT.mkdir(parents=True,exist_ok=False)
old=json.loads(Path('research/p3/lossless-capsules-v1/rank32-summary.json').read_text());base={'path':'artifacts/p3/capsules/rank32-predictor.gguf','sha256':old['predictor_sha256']};target={'path':'local/models/lfm2.5-230m-fable-5-f16.gguf','sha256':old['target_sha256']}
factor=Path(json.loads(Path('configs/p3_adapters.json').read_text())['models']['fable_svd32']['package_path'])
if file_sha256(factor)!=old['factor_package_sha256'] or factor.stat().st_size!=old['factor_package_bytes']:raise ValueError('Frozen factor package identity')
write_json(ROOT/'manifest.json',{'utc':utc_now(),'base':base,'target':target,'old_summary_sha256':file_sha256('research/p3/lossless-capsules-v1/rank32-summary.json'),'methods':['signed','zigzag_shuffle'],'scope':'post-hoc exact coding control, zero model calls, established transforms, not a new compression algorithm'})
for mode in ['signed','zigzag_shuffle']:
 archive=Path('artifacts/p3/capsules')/('rank32-'+mode+'.wpatch');out=Path('artifacts/p3/capsules')/('rank32-'+mode+'-restored.gguf')
 packed=create_patch(base,target,archive,ROOT/(mode+'-pack.json'),word_method=mode)
 restored=apply_patch_file(base['path'],archive,out,ROOT/(mode+'-restore.json'))
 summary={'utc':utc_now(),'method':mode,'factor_package_bytes':old['factor_package_bytes'],'factor_package_sha256':old['factor_package_sha256'],'residual_bytes':packed['patch_bytes'],'total_delivery_bytes':old['factor_package_bytes']+packed['patch_bytes'],'base_required_sha256':old['base_sha256'],'predictor_sha256':base['sha256'],'target_sha256':restored['target_sha256'],'byte_identical':restored['byte_identical'],'pack_seconds':packed['seconds'],'restore_seconds':restored['seconds']}
 write_json(ROOT/(mode+'-summary.json'),summary);publish([ROOT],'Record fixed exact residual representation '+mode);print(json.dumps(summary),flush=True)
