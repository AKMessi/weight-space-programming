"""Verified floating-vector interventions and streaming GGUF rewrites."""
import json
from pathlib import Path
import shutil
import numpy as np

from .edit import tensor_hash
from .evidence import digest, file_sha256, git_state, utc_now, write_json
from .precision import floating_tensor_values


def transformed(values,operation):
    if values.dtype.kind!='f' or not np.isfinite(values).all():raise ValueError('finite floating input required')
    if operation['kind']=='scale':result=values.astype(np.float32)*operation['factor']
    elif operation['kind']=='random_l2':
        rng=np.random.default_rng(operation['seed'])
        noise=rng.standard_normal(values.shape).astype(np.float32)
        noise*=float(np.linalg.norm(values.astype(np.float64)))*operation['relative_norm']/float(np.linalg.norm(noise.astype(np.float64)))
        result=values.astype(np.float32)+noise
    else:raise ValueError('unknown transform')
    result=result.astype(values.dtype)
    if not np.isfinite(result).all():raise ValueError('nonfinite result')
    return result


def check_output(output,manifest):
    output=Path(output)
    if not any(output.resolve().is_relative_to(Path(d).resolve()) for d in ('artifacts','local')):
        raise ValueError('binary output must be ignored local artifact')
    if output.exists() or Path(manifest).exists():raise FileExistsError('unique output/manifest required')
    dirty=git_state()['status']
    if any(l[3:].startswith(('src/','scripts/','configs/','tests/')) for l in dirty.splitlines()):
        raise RuntimeError('commit source/config/scripts before artifact construction')
    output.parent.mkdir(parents=True,exist_ok=True)
    return output


def intervene(base,operations,output,manifest):
    from gguf import GGUFReader
    output=check_output(output,manifest)
    if file_sha256(base['path'])!=base['sha256']:raise ValueError('source hash')
    reader=GGUFReader(base['path'])
    sources={t.name:t for t in reader.tensors}
    if len({op['tensor'] for op in operations})!=len(operations):raise ValueError('one operation per tensor')
    report={'schema_version':2,'utc':utc_now(),'git':git_state(),'base':base,'operations':operations,
            'output':str(output),'status':'started','tensors':[]}
    write_json(manifest,report)
    partial=output.with_suffix(output.suffix+'.partial')
    try:
        if partial.exists():raise FileExistsError(partial)
        shutil.copyfile(base['path'],partial)
        changed=GGUFReader(partial,mode='r+')
        target={t.name:t for t in changed.tensors}
        for op in operations:
            tensor=sources[op['tensor']]
            new=transformed(tensor.data,op)
            target[tensor.name].data[:]=new
            delta=new.astype(np.float64)-tensor.data.astype(np.float64)
            report['tensors'].append({'name':tensor.name,'base_sha256':tensor_hash(tensor.data),
                 'edited_sha256':tensor_hash(new),'changed_values':int(np.count_nonzero(delta)),
                 'relative_frobenius_change':float(np.linalg.norm(delta)/np.linalg.norm(tensor.data.astype(np.float64)))})
        changed.data.flush()
        touched={op['tensor'] for op in operations}
        for name,t in sources.items():
            if name not in touched and tensor_hash(t.data)!=tensor_hash(target[name].data):raise ValueError('untouched tensor changed')
        if file_sha256(base['path'])!=base['sha256']:raise ValueError('original changed')
        report.update(untouched_tensors_verified=len(sources)-len(touched),edited_file_sha256=file_sha256(partial),
                      changed_values=sum(t['changed_values'] for t in report['tensors']),bytes=partial.stat().st_size,status='verified')
        if hasattr(changed.data,'_mmap') and changed.data._mmap is not None:
            changed.data._mmap.close()
        del target,changed
        partial.replace(output)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:write_json(manifest,report)
    return report


def rewrite(base,output,manifest,mode='identity',permutation_seed=811,layer=0,fault=None):
    from gguf import GGUFReader,GGUFWriter,GGUFValueType,GGMLQuantizationType
    output=check_output(output,manifest)
    if file_sha256(base['path'])!=base['sha256']:raise ValueError('source hash')
    reader=GGUFReader(base['path'])
    source={t.name:t for t in reader.tensors}
    selected=[f'blk.{layer}.ffn_{kind}.weight' for kind in ('gate','up','down')]
    values={}
    if mode=='permute':
        for name in selected:values[name]=floating_tensor_values(source[name]).copy()
        width=values[selected[0]].shape[0]
        permutation=np.random.default_rng(permutation_seed).permutation(width)
        values[selected[0]]=np.ascontiguousarray(values[selected[0]][permutation])
        values[selected[1]]=np.ascontiguousarray(values[selected[1]][permutation])
        values[selected[2]]=np.ascontiguousarray(values[selected[2]][:,permutation])
    elif mode not in ('identity','f16'):raise ValueError('rewrite mode')
    report={'schema_version':2,'utc':utc_now(),'git':git_state(),'base':base,'mode':mode,'fault':fault,
            'permutation_seed':permutation_seed if mode=='permute' else None,'layer':layer,
            'output':str(output),'status':'started','tensors':[]}
    write_json(manifest,report)
    partial=output.with_suffix(output.suffix+'.partial')
    try:
        if partial.exists():raise FileExistsError(partial)
        writer=GGUFWriter(str(partial),reader.get_field('general.architecture').contents())
        for key,field in reader.fields.items():
            if key.startswith('GGUF.') or key=='general.architecture':continue
            value=1 if mode=='f16' and key=='general.file_type' else field.contents()
            writer.add_key_value(key,value,field.types[0],field.types[-1] if field.types[0]==GGUFValueType.ARRAY else None)
        # Stream one tensor at a time: never materialize a whole dequantized model.
        def data_for(t):
            if t.name in values:return values[t.name]
            if mode=='f16' and t.tensor_type==GGMLQuantizationType.BF16:
                result=floating_tensor_values(t,dtype=np.float16)
                if not np.isfinite(result).all():raise ValueError('nonfinite F16 conversion')
                return result
            return t.data
        for t in reader.tensors:
            data=data_for(t)
            raw=None if t.name in values or mode=='f16' and t.tensor_type==GGMLQuantizationType.BF16 else t.tensor_type
            writer.add_tensor_info(t.name,data.shape,data.dtype,data.nbytes,raw_dtype=raw)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
        for t in reader.tensors:writer.write_tensor_data(data_for(t))
        writer.close()
        result=GGUFReader(partial)
        for key,field in reader.fields.items():
            if key.startswith('GGUF.'):continue
            expected=1 if mode=='f16' and key=='general.file_type' else field.contents()
            if result.get_field(key).contents()!=expected:raise ValueError(f'metadata drift: {key}')
        for t in result.tensors:
            old=source[t.name]
            if not np.array_equal(t.shape,old.shape):raise ValueError('logical shape drift')
            expected=data_for(old)
            if tensor_hash(t.data)!=tensor_hash(expected):raise ValueError('tensor rewrite mismatch')
            row={'name':t.name,'encoding':str(t.tensor_type),
                 'unchanged_bytes':tensor_hash(t.data)==tensor_hash(old.data),'sha256':tensor_hash(t.data)}
            if mode=='f16' and old.tensor_type==GGMLQuantizationType.BF16:
                delta=t.data.astype(np.float32)-floating_tensor_values(old)
                row.update(converted_changed_values=int(np.count_nonzero(delta)),max_abs_conversion_error=float(np.max(np.abs(delta))))
            report['tensors'].append(row)
        if file_sha256(base['path'])!=base['sha256']:raise ValueError('original changed')
        report.update(status='verified',edited_file_sha256=file_sha256(partial),bytes=partial.stat().st_size,
                      metadata_digest=digest({k:v.contents() for k,v in reader.fields.items() if not k.startswith('GGUF.')}))
        if hasattr(result.data,'_mmap') and result.data._mmap is not None:
            result.data._mmap.close()
        del result
        partial.replace(output)
    except BaseException as exc:report.update(status='failed',error=repr(exc));raise
    finally:write_json(manifest,report)
    return report
