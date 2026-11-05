"""Stream lossless tensor replacements into a mixed-format GGUF."""
from pathlib import Path
import numpy as np
from gguf import GGUFReader,GGUFWriter,GGUFValueType
from .edit import tensor_hash
from .evidence import file_sha256,git_state,utc_now,write_json
from .transforms import check_output

def replace_tensors(base,donor,names,output,manifest):
 output=check_output(output,manifest);names=set(names)
 for spec in [base,donor]:
  if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Source identity')
 reader=GGUFReader(base['path']);replacement=GGUFReader(donor['path']);ts={t.name:t for t in reader.tensors};ds={t.name:t for t in replacement.tensors}
 if not names<=ts.keys() or not names<=ds.keys():raise ValueError('Missing replacement')
 for name in names:
  if not np.array_equal(ts[name].shape,ds[name].shape):raise ValueError('Logical shape differs')
 for key in ['general.architecture','tokenizer.ggml.tokens','tokenizer.chat_template']:
  a,b=reader.get_field(key),replacement.get_field(key)
  if (a is None)!=(b is None) or a is not None and a.contents()!=b.contents():raise ValueError('Incompatible sources')
 report={'utc':utc_now(),'git':git_state(),'base':base,'donor':donor,'selected':sorted(names),'output':str(output),'status':'started'};write_json(manifest,report)
 partial=output.with_suffix(output.suffix+'.partial')
 if partial.exists():raise FileExistsError(partial)
 try:
  writer=GGUFWriter(str(partial),reader.get_field('general.architecture').contents())
  for key,f in reader.fields.items():
   if key.startswith('GGUF.') or key=='general.architecture':continue
   writer.add_key_value(key,f.contents(),f.types[0],f.types[-1] if f.types[0]==GGUFValueType.ARRAY else None)
  for t in reader.tensors:
   x=ds[t.name] if t.name in names else t;writer.add_tensor_info(t.name,x.data.shape,x.data.dtype,x.data.nbytes,raw_dtype=x.tensor_type)
  writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
  for t in reader.tensors:writer.write_tensor_data((ds[t.name] if t.name in names else t).data)
  writer.close();result=GGUFReader(partial);checks=[]
  for t in result.tensors:
   expected=ds[t.name] if t.name in names else ts[t.name]
   if tensor_hash(t.data)!=tensor_hash(expected.data) or t.tensor_type!=expected.tensor_type:raise ValueError('Tensor byte/encoding drift')
   checks.append({'tensor':t.name,'encoding':str(t.tensor_type),'sha256':tensor_hash(t.data),'replaced':t.name in names})
  for key,f in reader.fields.items():
   if not key.startswith('GGUF.') and result.get_field(key).contents()!=f.contents():raise ValueError('Metadata drift')
  for spec in [base,donor]:
   if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('Original changed')
  partial.replace(output);report.update(status='verified',sha256=file_sha256(output),bytes=output.stat().st_size,tensors=checks)
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:write_json(manifest,report)
 return report
