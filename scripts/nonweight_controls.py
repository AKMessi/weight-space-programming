"""Strong development-only prompt and grammar baselines; never encode gold values."""
import copy
import json
from pathlib import Path

from weightlab.evidence import write_json


def schema_for(value):
    if isinstance(value, bool):
        return {'type': 'boolean'}
    if isinstance(value, str):
        return {'type': 'string'}
    if isinstance(value, int):
        return {'type': 'integer'}
    if isinstance(value, dict):
        return {'type': 'object', 'properties': {k: schema_for(v) for k,v in value.items()},
                'required': list(value), 'additionalProperties': False}
    raise TypeError(type(value))


suite = json.loads(Path('datasets/discovery-v1.json').read_text())
cases = []
for original in suite['cases']:
    case = copy.deepcopy(original)
    case.update(id='strong-prompt-'+original['id'], parent_case_id=original['id'], control='strong_prompt')
    case['messages'][0]['content'] = (
        'You are a precise text assistant. Treat identifiers in tasks as arbitrary strings. '
        'Do not invent or call tools. If asked for a mock tool call, return its requested JSON as text. '
        'When asked for exact text, a code line, or JSON, return only that content without markdown or explanation. '
        'When an explanation is requested, explain. Use only supplied facts for extraction and tool arguments.')
    cases.append(case)
    if original['grader'] == 'json':
        grammar = copy.deepcopy(case)
        grammar.update(id='grammar-'+original['id'], control='strong_prompt_plus_schema')
        # Only key names/types already specified by the task. No enum/const/gold values.
        grammar['request_overrides'] = {'response_format': {'type': 'json_object', 'schema': schema_for(original['expected'])}}
        cases.append(grammar)
write_json('datasets/nonweight-controls-v1.json', {'name': 'nonweight-controls-v1',
    'status': 'development controls designed after discovery; not held-out', 'cases': cases})
