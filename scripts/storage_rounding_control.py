"""Post-hoc control: round the SAME F16 parent before exact delta encoding."""
import json
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.patchcodec import create_patch,apply_patch_file

config=json.loads(Path('configs/portfolio.json').read_text())
prior=json.loads(Path('research/patches/parent-bf16-values-f16.json').read_text())
if prior['status']!='verified':raise ValueError('unverified rounding control')
base={**prior['base'],'path':prior['output'],'sha256':prior['edited_file_sha256']}
target=config['models']['fable_f16']
name='fable-same-parent-rounded'
packed=create_patch(base,target,f'artifacts/patches/{name}.wpatch',f'research/patches/storage-{name}.json')
applied=apply_patch_file(base['path'],packed['output'],f'artifacts/patches/{name}-restored.gguf',f'research/patches/storage-{name}-roundtrip.json')
write_json('research/exact-patch-rounding-control.json',{'utc':utc_now(),'git':git_state(),
 'status':'complete','reason':'Post-hoc control after the source-choice result: same original F16 parent with deterministic BF16 value rounding, not another source checkpoint.',
 'prior_results_sha256':file_sha256('research/exact-patch-storage.json'),
 'rounding_manifest_sha256':file_sha256('research/patches/parent-bf16-values-f16.json'),
 'patch_bytes':packed['patch_bytes'],'changed_tensors':len(packed['changed_tensors']),
 'exact_roundtrip':applied['byte_identical'],'target_sha256':applied['target_sha256']})
print(packed['patch_bytes'],applied['byte_identical'],flush=True)
