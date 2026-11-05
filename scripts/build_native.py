"""Compile the isolated trace client; never change installed native libraries."""
import subprocess
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
out=Path('research/p3/native-build-v2');out.mkdir(parents=True,exist_ok=False)
source=Path('local/runtime-source')
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
if revision!='338fce6511c744893cc0d3d958b275f40552c7a4':raise ValueError('source revision')
command=['c++','-O2','-std=c++17','native/trace.cpp','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),
 '-I'+str(source/'vendor/nlohmann'),'-L/home/akmessi/llama-vulkan-build/bin',
 '-Wl,-rpath,/home/akmessi/llama-vulkan-build/bin','-lllama','-lggml','-lggml-base','-o','local/weightlab-native']
report={'utc':utc_now(),'git':git_state(),'command':command,'source_revision':revision,'source_sha256':file_sha256('native/trace.cpp'),'status':'started'}
write_json(out/'manifest.json',report)
try:
 with (out/'build.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
 report.update(status='complete',binary_sha256=file_sha256('local/weightlab-native'))
except BaseException as e:report.update(status='failed',error=repr(e));raise
finally:write_json(out/'manifest.json',report)
