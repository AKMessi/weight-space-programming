"""Four fixed exact-patch round trips and precision-alignment storage comparison."""
import json
from pathlib import Path
from weightlab.patchcodec import create_patch,apply_patch_file,compressed_stream_size
from weightlab.evidence import file_sha256,git_state,utc_now,write_json

config=json.loads(Path('configs/portfolio.json').read_text())
pairs=[('qwen-norm','qwen3_q8','qwen3_q8_A'),('qad-norm','lfm3_qad','lfm3_qad_A'),
       ('fable-unaligned','parent_f16','fable_f16'),('fable-aligned','parent_uniform_bf16','fable_f16')]
result={'utc':utc_now(),'git':git_state(),'pairs':{},'scope':'Exact file-delivery utility; established XOR/deflate coding, not model improvement.'}
compressed={}
for name,left,right in pairs:
    base,target=config['models'][left],config['models'][right]
    patch=f'artifacts/patches/{name}.wpatch';manifest=f'research/patches/storage-{name}.json'
    if Path(manifest).exists():
        packed=json.loads(Path(manifest).read_text())
        if packed['status']!='verified' or file_sha256(patch)!=packed['patch_sha256']:raise ValueError('existing patch invalid')
    else:packed=create_patch(base,target,patch,manifest)
    applied_manifest=f'research/patches/storage-{name}-roundtrip.json'
    if Path(applied_manifest).exists():
        applied=json.loads(Path(applied_manifest).read_text())
        if applied['status']!='verified' or file_sha256(applied['output'])!=target['sha256']:raise ValueError('existing reconstruction invalid')
    else:applied=apply_patch_file(base['path'],patch,f'artifacts/patches/{name}-restored.gguf',applied_manifest)
    if target['sha256'] not in compressed:compressed[target['sha256']]=compressed_stream_size(target['path'])
    result['pairs'][name]={'base_key':left,'target_key':right,'patch_bytes':packed['patch_bytes'],
       'target_bytes':packed['target_bytes'],'target_zlib_bytes':compressed[target['sha256']],
       'changed_tensor_count':len(packed['changed_tensors']),'exact_roundtrip':applied['byte_identical'],
       'patch_sha256':packed['patch_sha256'],'target_sha256':target['sha256'],
       'pack_seconds':packed['seconds'],'apply_seconds':applied['seconds']}
    write_json('research/exact-patch-storage.json',result)
    print(name,result['pairs'][name],flush=True)
result['status']='complete';write_json('research/exact-patch-storage.json',result)
