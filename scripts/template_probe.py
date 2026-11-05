"""Generate explicit discovery conditions to diagnose template/parser confounds."""
import copy
import json
from pathlib import Path

from weightlab.evidence import write_json

suite = json.loads(Path('datasets/discovery-v1.json').read_text())
families = {'requested_repeat', 'evidence_missing', 'evidence_conflict', 'code_only'}
selected = [c for c in suite['cases'] if c['family'] in families and c['id'].endswith('-01')]
conditions = {
    'false_raw': {'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'none'},
    'false_parsed': {'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'deepseek'},
    'false_effort_none': {'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_effort': 'none', 'reasoning_format': 'none'},
    'true_parsed': {'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_format': 'deepseek'},
}
cases = []
for label, overrides in conditions.items():
    for original in selected:
        case = copy.deepcopy(original)
        case['id'] = f'probe-v1-{label}-{original["family"]}'
        case['group'] = label
        case['request_overrides'] = overrides
        cases.append(case)
write_json('datasets/template-probe-v1.json', {'name': 'template-probe-v1', 'status': 'diagnostic discovery; not held-out', 'cases': cases})
config = json.loads(Path('configs/local_models.json').read_text())
config['runtime'].update(reasoning='auto', chat_template_kwargs={})
config['models']['lfm3_q8'] = dict(config['models']['lfm3_qad'],
    path='/home/akmessi/models/lfm3/LFM3-1.7B-Q8_0.gguf', file='LFM3-1.7B-Q8_0.gguf',
    sha256='8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6')
write_json('configs/template_probe.json', config)
