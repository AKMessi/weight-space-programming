"""Exact named-tensor GGUF delta archives. Originals are opened read-only."""
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import zlib
import numpy as np
from .evidence import file_sha256,git_state,utc_now,write_json

CHUNK=1024*1024


def output_path(path):
    path=Path(path)
    if not any(path.resolve().is_relative_to(Path(d).resolve()) for d in ('artifacts','local')):
        raise ValueError('new binary must be under ignored artifacts/ or local/')
    if path.exists() or path.with_name(path.name+'.partial').exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def range_hash(stream,offset,n):
    stream.seek(offset);h=hashlib.sha256()
    while n:
        block=stream.read(min(n,CHUNK))
        if not block:raise ValueError('truncated source range')
        h.update(block);n-=len(block)
    return h.hexdigest()


def create_patch(base,target,output,manifest):
    from gguf import GGUFReader
    output=output_path(output)
    if Path(manifest).exists():raise FileExistsError(manifest)
    start=time.monotonic()
    report={'utc':utc_now(),'git':git_state(),'base':base,'target':target,'output':str(output),
            'format':'weightlab-gguf-xor-v1','status':'started','zlib_version':zlib.ZLIB_RUNTIME_VERSION}
    write_json(manifest,report);partial=output.with_name(output.name+'.partial')
    try:
        if file_sha256(base['path'])!=base['sha256'] or file_sha256(target['path'])!=target['sha256']:
            raise ValueError('source checksum mismatch')
        left,right=GGUFReader(base['path']),GGUFReader(target['path'])
        lt={t.name:t for t in left.tensors};rt={t.name:t for t in right.tensors}
        if lt.keys()!=rt.keys():raise ValueError('tensor sets must match')
        for name,t in rt.items():
            u=lt[name]
            if t.n_bytes!=u.n_bytes or t.tensor_type!=u.tensor_type or not np.array_equal(t.shape,u.shape):
                raise ValueError('tensor shape/encoding mismatch')
        index={'format':'weightlab-gguf-xor-v1','base_sha256':base['sha256'],'target_sha256':target['sha256'],
               'target_bytes':Path(target['path']).stat().st_size,'segments':[]}
        changed=[]
        with open(base['path'],'rb') as a,open(target['path'],'rb') as b,zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            def add_segment(offset,n,base_offset=None,tensor_name=None):
                offset,n=int(offset),int(n)
                base_offset=int(base_offset) if base_offset is not None else None
                if not n:return
                target_hash=range_hash(b,offset,n)
                same=base_offset is not None and range_hash(a,base_offset,n)==target_hash
                item={'offset':offset,'bytes':n,'target_sha256':target_hash,'tensor':tensor_name}
                if same:item.update(kind='copy',base_offset=base_offset)
                else:
                    item.update(kind='xor' if tensor_name else 'literal',payload=f'blocks/{len(index["segments"]):05d}')
                    if tensor_name:item['base_offset']=base_offset;changed.append(tensor_name)
                    b.seek(offset)
                    if tensor_name:a.seek(base_offset)
                    with archive.open(item['payload'],'w',force_zip64=True) as dst:
                        remaining=n
                        while remaining:
                            data=b.read(min(CHUNK,remaining))
                            if not data:raise ValueError('truncated target')
                            if tensor_name:
                                old=a.read(len(data))
                                if len(old)!=len(data):raise ValueError('truncated base')
                                data=np.bitwise_xor(np.frombuffer(data,np.uint8),np.frombuffer(old,np.uint8)).tobytes()
                            dst.write(data);remaining-=len(data)
                index['segments'].append(item)
            cursor=0
            for t in sorted(right.tensors,key=lambda t:t.data_offset):
                if t.data_offset<cursor:raise ValueError('overlapping tensors')
                # Prefix/gap bytes are compressed literally unless an exact base range matches.
                if t.data_offset>cursor:add_segment(cursor,t.data_offset-cursor,0 if cursor==0 and t.data_offset<=Path(base['path']).stat().st_size else None)
                add_segment(t.data_offset,t.n_bytes,lt[t.name].data_offset,t.name)
                cursor=t.data_offset+t.n_bytes
            if cursor<index['target_bytes']:add_segment(cursor,index['target_bytes']-cursor)
            archive.writestr('index.json',json.dumps(index,sort_keys=True,separators=(',',':')))
        with partial.open('r+b') as stream:os.fsync(stream.fileno())
        if file_sha256(base['path'])!=base['sha256'] or file_sha256(target['path'])!=target['sha256']:raise ValueError('original changed')
        report.update(status='verified',changed_tensors=changed,unchanged_tensors=len(rt)-len(changed),
                      patch_sha256=file_sha256(partial),patch_bytes=partial.stat().st_size,
                      target_bytes=index['target_bytes'],seconds=time.monotonic()-start,
                      note='Exact delta encoding, not a learned repair or new compression algorithm.')
        partial.replace(output)
    except BaseException as exc:report.update(status='failed',error=repr(exc));raise
    finally:write_json(manifest,report)
    return report


def apply_patch_file(base_path,patch_path,output,manifest):
    output=output_path(output)
    if Path(manifest).exists():raise FileExistsError(manifest)
    start=time.monotonic();partial=output.with_name(output.name+'.partial')
    report={'utc':utc_now(),'git':git_state(),'base_path':str(base_path),'patch_path':str(patch_path),
            'output':str(output),'status':'started','patch_sha256':file_sha256(patch_path)}
    write_json(manifest,report)
    try:
        with zipfile.ZipFile(patch_path) as archive:
            if archive.getinfo('index.json').file_size>16*1024*1024:raise ValueError('index too large')
            index=json.loads(archive.read('index.json'))
            if index['format']!='weightlab-gguf-xor-v1':raise ValueError('unknown patch format')
            if file_sha256(base_path)!=index['base_sha256']:raise ValueError('wrong base checkpoint')
            size=index['target_bytes']
            if type(size) is not int or not 0<size<=16*1024**3:raise ValueError('target size bound')
            cursor=0;base_size=Path(base_path).stat().st_size
            with open(base_path,'rb') as source,partial.open('xb+') as target:
                target.truncate(size)
                for item in index['segments']:
                    n=item['bytes']
                    if type(n) is not int or n<=0 or item['offset']!=cursor or cursor+n>size:raise ValueError('invalid/overlapping segment')
                    kind=item['kind']
                    if kind not in ('copy','literal','xor'):raise ValueError('unknown segment')
                    if kind in ('copy','xor'):
                        start_at=item['base_offset']
                        if type(start_at) is not int or start_at<0 or start_at+n>base_size:raise ValueError('source range')
                        source.seek(start_at)
                    payload=None
                    if kind!='copy':
                        if archive.getinfo(item['payload']).file_size!=n:raise ValueError('payload size')
                        payload=archive.open(item['payload'])
                    h=hashlib.sha256();remaining=n;target.seek(cursor)
                    try:
                        while remaining:
                            count=min(CHUNK,remaining)
                            data=source.read(count) if kind=='copy' else payload.read(count)
                            if len(data)!=count:raise ValueError('truncated segment')
                            if kind=='xor':
                                old=source.read(count)
                                if len(old)!=count:raise ValueError('truncated source')
                                data=np.bitwise_xor(np.frombuffer(data,np.uint8),np.frombuffer(old,np.uint8)).tobytes()
                            target.write(data);h.update(data);remaining-=count
                        if payload is not None and payload.read(1):raise ValueError('extra payload data')
                    finally:
                        if payload is not None:payload.close()
                    if h.hexdigest()!=item['target_sha256']:raise ValueError('reconstructed segment checksum')
                    cursor+=n
                if cursor!=size:raise ValueError('uncovered target bytes')
                target.flush();os.fsync(target.fileno())
            actual=file_sha256(partial)
            if actual!=index['target_sha256']:raise ValueError('complete target checksum')
            if file_sha256(base_path)!=index['base_sha256']:raise ValueError('base changed')
            report.update(status='verified',target_sha256=actual,byte_identical=True,bytes=size,
                          seconds=time.monotonic()-start)
            partial.replace(output)
    except BaseException as exc:report.update(status='failed',error=repr(exc));raise
    finally:write_json(manifest,report)
    return report


def compressed_stream_size(path):
    compressor=zlib.compressobj(6);size=0
    with open(path,'rb') as source:
        while data:=source.read(CHUNK):size+=len(compressor.compress(data))
    return size+len(compressor.flush())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for kind in ['pack','apply']:
        p=sub.add_parser(kind);p.add_argument('base');p.add_argument('other');p.add_argument('output');p.add_argument('--manifest',required=True)
    args=parser.parse_args()
    if args.action=='pack':
        base={'path':str(Path(args.base).expanduser()),'sha256':file_sha256(Path(args.base).expanduser())}
        target={'path':str(Path(args.other).expanduser()),'sha256':file_sha256(Path(args.other).expanduser())}
        result=create_patch(base,target,args.output,args.manifest)
    else:result=apply_patch_file(Path(args.base).expanduser(),Path(args.other).expanduser(),args.output,args.manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
