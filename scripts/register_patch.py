"""Derive a run configuration from a verified local patch manifest."""
import argparse
import json
from pathlib import Path

from weightlab.evidence import file_sha256, write_json

p = argparse.ArgumentParser()
p.add_argument('manifest')
p.add_argument('key')
p.add_argument('output_config')
p.add_argument('--config', default='configs/local_models.json')
a = p.parse_args()
manifest = json.loads(Path(a.manifest).read_text())
if manifest['status'] != 'verified' or file_sha256(manifest['output']) != manifest['edited_file_sha256']:
    raise ValueError('Patch must be verified and unchanged')
config = json.loads(Path(a.config).read_text())
row = dict(manifest['base'])
row.update(path=manifest['output'], sha256=manifest['edited_file_sha256'],
           local_patch_manifest=a.manifest, local_patch_manifest_sha256=file_sha256(a.manifest))
config['models'][a.key] = row
write_json(a.output_config, config)
