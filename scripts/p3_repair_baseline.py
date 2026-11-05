"""Original and practical controls on new response-switching development cases."""
import json
from pathlib import Path
from p3_batch import run_one
config=json.loads(Path('configs/p3_repair.json').read_text())
for control in ['original','strong','grammar']:
 suffix='' if control=='original' else '-'+control
 run_one(config,'qwen_float',f'datasets/p3/switch-development{suffix}.json','switch-dev-'+control,'repair',max_tokens=96)
