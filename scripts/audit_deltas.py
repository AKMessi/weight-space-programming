"""Compare pinned F16 checkpoint tensors, including a BF16-rounding diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader

from weightlab.evidence import file_sha256, git_state, utc_now, write_json

p = argparse.ArgumentParser()
p.add_argument('output')
p.add_argument('--config', default='configs/local_models.json')
a = p.parse_args()
if Path(a.output).exists():
    raise FileExistsError(a.output)
config = json.loads(Path(a.config).read_text())
sources = [config['models'][key] for key in ['fable_f16','parent_f16']]
for spec in sources:
    if file_sha256(spec['path']) != spec['sha256']:
        raise ValueError('Source checksum mismatch')
fine,parent = [GGUFReader(spec['path']) for spec in sources]
pt = {t.name:t for t in parent.tensors}
report = {'utc':utc_now(),'git':git_state(),'sources':sources,
          'interpretation':'Checkpoint differences; actual original training revision not yet established. BF16 rounding equality is a diagnostic, not proof of training provenance.',
          'tensors':[]}
for tensor in fine.tensors:
    other = pt[tensor.name]
    if tensor.data.dtype.kind != 'f' or other.data.dtype != tensor.data.dtype or tensor.data.shape != other.data.shape:
        raise ValueError('Expected aligned floating tensor')
    x,y = tensor.data.reshape(-1),other.data.reshape(-1)
    norm_sq=delta_sq=0.0
    changed=rounded_equal=0
    max_delta=0.0
    for start in range(0,len(x),1048576):
        u=x[start:start+1048576].astype(np.float32)
        v=y[start:start+1048576].astype(np.float32)
        delta=u-v
        changed += int(np.count_nonzero(delta))
        norm_sq += float(np.square(v).sum(dtype=np.float64))
        delta_sq += float(np.square(delta).sum(dtype=np.float64))
        max_delta=max(max_delta,float(np.max(np.abs(delta))))
        bits=u.view(np.uint32)
        rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
        rounded_equal += int(np.count_nonzero(rounded==v))
    row={'name':tensor.name,'shape':list(tensor.data.shape),'dtype':str(tensor.data.dtype),
         'values':len(x),'changed_values':changed,'relative_frobenius_delta':float(np.sqrt(delta_sq/norm_sq)) if norm_sq else None,
         'max_absolute_delta':max_delta,'fine_rounded_bf16_equals_parent_fraction':rounded_equal/len(x)}
    report['tensors'].append(row)
    write_json(a.output,report)
report['status']='complete'
write_json(a.output,report)
for row in sorted(report['tensors'],key=lambda r:r['relative_frobenius_delta'] or 0,reverse=True)[:18]:
    print(row['name'],row['relative_frobenius_delta'],row['fine_rounded_bf16_equals_parent_fraction'])
