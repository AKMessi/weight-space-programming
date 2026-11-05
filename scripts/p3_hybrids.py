"""Profile-informed mixed-format export, with exact source-byte checks."""
import json
from pathlib import Path
from gguf import GGUFReader
from weightlab.hybrid import replace_tensors
from weightlab.evidence import file_sha256,write_json
from portfolio_batch import publish
baseconfig=json.loads(Path('configs/fable_speed_followup.json').read_text());base=baseconfig['models']['fable_Q4_0'];donor=baseconfig['models']['fable_float']
names=[t.name for t in GGUFReader(donor['path']).tensors];head=[n for n in names if n in ['token_embd.weight','output.weight']]
short=[n for n in names if 'shortconv' in n and ('in_proj' in n or 'out_proj' in n) and n.endswith('.weight')]
if not short:raise ValueError('No short-conv projection sites')
sets={'hybrid_head':head,'hybrid_head_ffn':head+[n for n in names if any('.ffn_'+k+'.weight' in n for k in ['gate','up','down'])],'hybrid_head_shortconv':head+short,'hybrid_noop':[]}
config={'runtime':baseconfig['runtime'],'budgets':baseconfig['budgets'],'models':{}}
for key,selected in sets.items():
 manifest=Path('research/p3/hybrids')/(key+'.json');manifest.parent.mkdir(parents=True,exist_ok=True)
 report=replace_tensors(base,donor,selected,Path('artifacts/p3/hybrids')/(key+'.gguf'),manifest)
 config['models'][key]={**base,'path':report['output'],'sha256':report['sha256'],'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
write_json('configs/p3_hybrids.json',config);publish(['research/p3/hybrids','configs/p3_hybrids.json'],'Export profile-informed verified mixed-format Fable models')
