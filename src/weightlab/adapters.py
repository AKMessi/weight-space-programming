"""Training-free matrix-delta packages with exact base identity and verified merge."""
import json,re,shutil,zipfile
from pathlib import Path
import numpy as np
from .evidence import file_sha256,git_state,utc_now,write_json
from .edit import tensor_hash
from .transforms import check_output

PATTERN=r'blk\.\d+\.attn_[qkv]\.weight'


def factorize(delta,rank):
    u,s,vh=np.linalg.svd(np.asarray(delta,dtype=np.float32),full_matrices=False)
    rank=min(rank,len(s));root=np.sqrt(s[:rank])
    a=(u[:,:rank]*root).astype(np.float16);b=(root[:,None]*vh[:rank]).astype(np.float16)
    return a,b,s


def encode(delta,method,rank=32,factors=None):
    x=np.asarray(delta,dtype=np.float32)
    if method=='svd':
        a,b,_=factorize(x,rank) if factors is None else factors
        return {'a':a,'b':b}
    if method=='sign':
        return {'scale':np.array([np.mean(np.abs(x),dtype=np.float64)],dtype=np.float32),
                'bits':np.packbits((x>=0).reshape(-1))}
    if method in ['int8','int4']:
        limit=127 if method=='int8' else 7
        scales=np.max(np.abs(x),axis=1)/limit;scales=np.where(scales==0,1.,scales).astype(np.float32)
        q=np.clip(np.rint(x/scales[:,None]),-limit,limit).astype(np.int8)
        if method=='int8':return {'scale':scales,'q':q}
        flat=(q.reshape(-1).astype(np.int16)+8).astype(np.uint8)
        if len(flat)%2:flat=np.append(flat,np.uint8(8))
        return {'scale':scales,'packed':flat[::2]|(flat[1::2]<<4)}
    raise ValueError('Unsupported adapter encoding')


def decode(data,method,shape):
    if method=='svd':return data['a'].astype(np.float32)@data['b'].astype(np.float32)
    if method=='sign':
        bits=np.unpackbits(data['bits'])[:int(np.prod(shape))].reshape(shape)
        return (bits.astype(np.float32)*2-1)*data['scale'][0]
    if method=='int8':return data['q'].astype(np.float32)*data['scale'][:,None]
    if method=='int4':
        packed=data['packed'];q=np.empty(len(packed)*2,dtype=np.int8)
        q[::2]=(packed&15).astype(np.int8)-8;q[1::2]=(packed>>4).astype(np.int8)-8
        return q[:int(np.prod(shape))].reshape(shape).astype(np.float32)*data['scale'][:,None]
    raise ValueError(method)


def pack(base,target,output,manifest,method='svd',rank=32,cache=None,round_bf16=False):
    from gguf import GGUFReader
    output=check_output(output,manifest)
    for spec in [base,target]:
        if file_sha256(spec['path'])!=spec['sha256']:raise ValueError('source checksum')
    left=GGUFReader(base['path']);right=GGUFReader(target['path'])
    a={t.name:t for t in left.tensors};b={t.name:t for t in right.tensors}
    if a.keys()!=b.keys():raise ValueError('tensor sets')
    meta={'base_sha256':base['sha256'],'reference_sha256':target['sha256'],'method':method,'rank':rank,
          'round_bf16':round_bf16,'tensors':[],'format':'weightlab-approx-delta-v1'}
    arrays={};report={'utc':utc_now(),'git':git_state(),'base':base,'target':target,'output':str(output),
                    'method':method,'rank':rank,'round_bf16':round_bf16,'status':'started','tensors':[]}
    write_json(manifest,report)
    try:
        for index,(name,t) in enumerate(a.items()):
            old=t.data;new=b[name].data
            if old.shape!=new.shape or old.dtype!=new.dtype or old.dtype.kind!='f':raise ValueError('Floating aligned arrays required')
            if np.array_equal(old,new):continue
            prefix=f't{index}';changed=int(np.count_nonzero(old!=new))
            if re.fullmatch(PATTERN,name):
                delta=new.astype(np.float32)-old.astype(np.float32)
                factors=None
                if cache is not None and method=='svd':
                    z=np.load(cache/(name+'.npz'),allow_pickle=False)
                    factors=(z['u'][:,:rank].astype(np.float32)*np.sqrt(z['s'][:rank]),
                             np.sqrt(z['s'][:rank])[:,None]*z['vh'][:rank].astype(np.float32))
                    factors=(factors[0].astype(np.float16),factors[1].astype(np.float16),z['s'])
                encoded=encode(delta,method,rank,factors)
                for key,value in encoded.items():arrays[prefix+'_'+key]=value
                row={'name':name,'prefix':prefix,'shape':list(old.shape),'kind':method}
                approximation=decode(encoded,method,old.shape)
                row['relative_delta_error']=float(np.linalg.norm(approximation-delta)/max(np.linalg.norm(delta),1e-30))
            else:
                # Preserve only genuinely tiny residuals, not a hidden dense model payload.
                if changed>64:raise ValueError('Unexpected substantial residual outside declared Q/K/V: '+name)
                ids=np.flatnonzero(old.reshape(-1)!=new.reshape(-1)).astype(np.uint64)
                arrays[prefix+'_indices']=ids;arrays[prefix+'_values']=new.reshape(-1)[ids]
                row={'name':name,'prefix':prefix,'shape':list(old.shape),'kind':'exact_sparse','changed_values':changed}
            meta['tensors'].append(row);report['tensors'].append(row)
        arrays['metadata']=np.array(json.dumps(meta))
        with output.open('wb') as stream:np.savez_compressed(stream,**arrays)
        report.update(status='verified',bytes=output.stat().st_size,package_sha256=file_sha256(output),
                      changed_tensor_count=len(meta['tensors']))
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:write_json(manifest,report)
    return report


def deterministic_add(base,a,b):
    """Exact dyadic F16 predictor: integer products/sums, then one IEEE F16 cast."""
    if base.dtype!=np.float16 or a.dtype!=np.float16 or b.dtype!=np.float16:raise ValueError('F16-only deterministic predictor')
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or base.shape!=(a.shape[0],b.shape[1]):raise ValueError('predictor shape')
    if not all(np.isfinite(x).all() for x in [base,a,b]):raise ValueError('finite predictor')
    # Every finite F16 value is an integer multiple of 2^-24.
    ai=(a.astype(np.float64)*(2.**24)).astype(np.int64)
    bi=(b.astype(np.float64)*(2.**24)).astype(np.int64)
    product_bound=int(np.max(np.abs(ai)))*int(np.max(np.abs(bi)))*a.shape[1]
    base_bound=float(np.max(np.abs(base)))*(2.**48)
    # Keep both int64 accumulation and subsequent float64 conversion exact.
    if product_bound+base_bound>=2**53:raise ValueError('Predictor exact-integer bound exceeded')
    numerators=ai@bi
    base_int=(base.astype(np.float64)*(2.**48)).astype(np.int64)
    total=numerators+base_int
    return (total.astype(np.float64)*(2.**-48)).astype(np.float16)


def merge(base,package,output,manifest,deterministic=False):
    from gguf import GGUFReader
    from .precision import round_bf16_values
    output=check_output(output,manifest)
    z=np.load(package,allow_pickle=False);meta=json.loads(str(z['metadata']))
    if meta['format']!='weightlab-approx-delta-v1' or file_sha256(base)!=meta['base_sha256']:raise ValueError('Wrong base')
    report={'utc':utc_now(),'git':git_state(),'base_path':str(base),'base_sha256':meta['base_sha256'],
            'package':str(package),'package_sha256':file_sha256(package),'output':str(output),'status':'started',
            'predictor_arithmetic':'exact-f16-dyadic-int64-v1' if deterministic else 'numpy-fp32-v1'}
    write_json(manifest,report)
    partial=output.with_suffix('.gguf.partial')
    try:
        if partial.exists():raise FileExistsError(partial)
        shutil.copyfile(base,partial)
        original={t.name:t for t in GGUFReader(base).tensors};reader=GGUFReader(partial,mode='r+');edited={t.name:t for t in reader.tensors}
        expected={}
        for row in meta['tensors']:
            name,prefix=row['name'],row['prefix'];old=original[name].data
            if list(old.shape)!=row['shape']:raise ValueError('shape')
            if row['kind']=='exact_sparse':
                new=old.copy();new.reshape(-1)[z[prefix+'_indices']]=z[prefix+'_values']
            else:
                data={key[len(prefix)+1:]:z[key] for key in z.files if key.startswith(prefix+'_')}
                if deterministic:
                    if row['kind']!='svd':raise ValueError('Deterministic predictor supports SVD factors only')
                    new=deterministic_add(old,data['a'],data['b'])
                else:new=(old.astype(np.float32)+decode(data,row['kind'],old.shape)).astype(old.dtype)
                if meta['round_bf16']:new=round_bf16_values(new)
            if not np.isfinite(new).all():raise ValueError('nonfinite merge')
            edited[name].data[:]=new;expected[name]=tensor_hash(new)
        reader.data.flush()
        for name,t in edited.items():
            if tensor_hash(t.data)!=expected.get(name,tensor_hash(original[name].data)):raise ValueError('Merge verification: '+name)
        if file_sha256(base)!=meta['base_sha256']:raise ValueError('Original mutated')
        report.update(status='verified',edited_file_sha256=file_sha256(partial),bytes=partial.stat().st_size,
                      touched_tensors=len(expected),untouched_tensors_verified=len(edited)-len(expected))
        partial.replace(output)
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:write_json(manifest,report)
    return report
