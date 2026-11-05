"""Original and practical controls on new response-switching development cases."""
import argparse,copy,json
from pathlib import Path
from p3_batch import run_one
config=json.loads(Path('configs/p3_repair.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['qwen_float']);args=parser.parse_args()
for key in args.models:
 local=copy.deepcopy(config)
 if key=='lfm3_float':local['runtime']['gpu_layers']=18
 for control in ['original','strong','grammar']:
  suffix='' if control=='original' else '-'+control
  run_one(local,key,f'datasets/p3/switch-development{suffix}.json','switch-dev-'+control,'repair',max_tokens=96)
