"""Rebuild ignored GGUFs from committed recipes without rewriting their manifests."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import time
from .evidence import file_sha256,utc_now,write_json


class Rebuilder:
    def __init__(self,manifest_dir='research/patches',download=False,quantizer=None):
        self.recipes={}
        for path in Path(manifest_dir).glob('*.json'):
            data=json.loads(path.read_text())
            if data.get('status')=='verified' and 'edited_file_sha256' in data and 'output' in data:
                key=str(Path(data['output']))
                if key in self.recipes and self.recipes[key][1]['edited_file_sha256']!=data['edited_file_sha256']:
                    raise ValueError('Ambiguous artifact recipe: '+key)
                self.recipes[key]=(path,data)
        self.download=download;self.quantizer=quantizer
        self.logdir=Path('local/rebuild')/str(time.time_ns());self.active=set();self.cache={};self.number=0

    def ensure(self,spec):
        result=copy.deepcopy(spec);path=Path(spec['path']).expanduser();key=str(path)
        if key in self.cache:return {**result,'path':self.cache[key]}
        if path.exists():
            if file_sha256(path)!=spec['sha256']:raise ValueError('Existing file has wrong hash: '+key)
            self.cache[key]=key;return result
        if key in self.active:raise ValueError('Cyclic recipe: '+key)
        recipe=self.recipes.get(key)
        if recipe is None:
            # An old machine's absolute source path is portable via the pinned basename.
            if path.is_absolute():path=Path('local/models')/spec['file']
            result['path']=str(path)
            if path.exists():
                if file_sha256(path)!=spec['sha256']:raise ValueError('Relocated source has wrong hash')
            elif self.download:
                if not all(spec.get(k) for k in ('repo','revision','file')):raise ValueError('Missing pinned download provenance')
                from .artifacts import fetch_artifacts
                fetch_artifacts({'models':{'rebuild-source':result}})
            else:raise FileNotFoundError(f'{key}; fetch the pinned source first or use --download')
            self.cache[key]=str(path);return result
        historical,report=recipe
        if report['edited_file_sha256']!=spec['sha256']:raise ValueError('Recipe identity does not match requested model')
        self.active.add(key)
        try:
            base=self.ensure(report['base']);self.logdir.mkdir(parents=True,exist_ok=True)
            self.number+=1;manifest=self.logdir/f'{self.number:03d}-{historical.name}'
            if 'operations' in report:
                from .transforms import intervene
                fresh=intervene(base,report['operations'],path,manifest)
            elif 'mode' in report:
                from .transforms import rewrite
                fresh=rewrite(base,path,manifest,mode=report['mode'],permutation_seed=report.get('permutation_seed') or 811,
                              layer=report.get('layer',0),fault=report.get('fault'))
            elif report.get('operation')=='partial_parent_rollback_of_selected_floating_tensors':
                from .edit import create_rollback
                parent=self.ensure(report['parent']);names=report['selected']
                matches=[re.match(r'blk\.(\d+)\.',name) for name in names]
                layers=sorted({int(m[1]) for m in matches}) if all(matches) else None
                fresh=create_rollback({'models':{'base':base,'parent':parent}},'base','parent',path,manifest,report['alpha'],layers,
                      pattern=report['selection_pattern'],parent_round_bf16=report.get('parent_transformation')=='round_bf16_values_then_cast_back')
                if set(fresh['selected'])!=set(names):raise ValueError('Rebuilt selection differs')
            elif 'command' in report and report['command'][3].startswith('Q'):
                binary=self.quantizer or report['command'][0]
                expected=report.get('quantizer',{}).get('binary_sha256',report.get('binary_sha256'))
                if file_sha256(binary)!=expected:raise ValueError('Quantizer hash differs; use the recorded build')
                path.parent.mkdir(parents=True,exist_ok=True)
                command=[binary,base['path'],str(path),*report['command'][3:]]
                fresh={'utc':utc_now(),'historical_manifest':str(historical),'command':command,'status':'started'}
                write_json(manifest,fresh)
                try:
                    with manifest.with_suffix('.log').open('w') as stream:
                        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    fresh.update(status='verified',edited_file_sha256=file_sha256(path))
                except BaseException as exc:
                    fresh.update(status='failed',error=repr(exc));raise
                finally:write_json(manifest,fresh)
            else:raise ValueError('Unsupported rebuild recipe: '+str(historical))
            if fresh['edited_file_sha256']!=spec['sha256']:
                write_json(manifest.with_suffix('.identity-error.json'),{'expected':spec['sha256'],'actual':fresh['edited_file_sha256']})
                raise ValueError('Rebuilt bytes differ from historical artifact: '+key)
            self.cache[key]=str(path);result['path']=str(path)
            return result
        finally:self.active.remove(key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/portfolio_precision.json');p.add_argument('--keys',nargs='+')
    p.add_argument('--download',action='store_true');p.add_argument('--quantizer')
    p.add_argument('--output-config',required=True);a=p.parse_args()
    if Path(a.output_config).exists():raise FileExistsError(a.output_config)
    config=json.loads(Path(a.config).read_text());worker=Rebuilder(download=a.download,quantizer=a.quantizer)
    selected=a.keys or list(config['models'])
    for key in selected:
        config['models'][key]=worker.ensure(config['models'][key]);print('Verified/rebuilt',key,flush=True)
    write_json(a.output_config,config)


if __name__=='__main__':main()
