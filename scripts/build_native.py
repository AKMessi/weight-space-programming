"""Compile the isolated trace client; never change installed native libraries."""
import argparse,subprocess
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
parser=argparse.ArgumentParser();parser.add_argument('--output',default='research/p3/native-build-v2');parser.add_argument('--source',default='native/trace.cpp');parser.add_argument('--binary',default='local/weightlab-native');args=parser.parse_args()
if Path(args.binary).exists():raise FileExistsError(args.binary)
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
source=Path('local/runtime-source')
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
if revision!='338fce6511c744893cc0d3d958b275f40552c7a4':raise ValueError('source revision')
command=['c++','-O2','-std=c++17',args.source,'-I'+str(source/'include'),'-I'+str(source/'ggml/include'),
 '-I'+str(source/'vendor/nlohmann'),'-L/home/akmessi/llama-vulkan-build/bin',
 '-Wl,-rpath,/home/akmessi/llama-vulkan-build/bin','-lllama','-lggml','-lggml-base','-o',args.binary]
report={'utc':utc_now(),'git':git_state(),'command':command,'source_revision':revision,'source_sha256':file_sha256(args.source),'status':'started'}
write_json(out/'manifest.json',report)
try:
 with (out/'build.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
 report.update(status='complete',binary_sha256=file_sha256(args.binary))
except BaseException as e:report.update(status='failed',error=repr(e));raise
finally:write_json(out/'manifest.json',report)
