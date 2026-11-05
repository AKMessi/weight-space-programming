"""Freeze fresh fixtures before post-discovery ablation evaluation."""
import random

from weightlab.evidence import write_json

rng=random.Random(20261105517)
cases=[]
def add(group,i,grader,expected,prompt):
    cases.append({'id':f'validation-v1-{group}-{i:02d}','split':'validation_frozen',
                  'family':group,'group':group,'grader':grader,'expected':expected,
                  'messages':[{'role':'system','content':"Follow the user's instructions. Use only supplied facts for extraction and tool arguments."},
                              {'role':'user','content':prompt}]})
for i in range(12):
    tag=f'obj_{rng.randint(1000,9999)}'
    name=rng.choice(['Asha','Felix','Leila','Omar'])
    count=rng.randint(2,37)
    literal=rng.choice([f'pkg::{tag}::{tag}',f'/src/{tag}.py | {name}',f'[{name}]::{tag}#v2'])
    add('literal_copy',i,'copy',literal,
        f'Reproduce the following payload exactly once, with no introduction or tool call.\nPAYLOAD START\n{literal}\nPAYLOAD END')
    active=bool(i%2)
    expected={'owner':name,'active':active,'count':count}
    add('new_json_fields',i,'json',expected,
        f'Data card:\nOwner: {name}\nActive: {str(active).lower()}\nCount: {count}\n'
        'Return only JSON with owner (string), active (boolean), count (integer). Use those exact keys, no markdown.')
    field=rng.choice(['score','label','amount','state'])
    expression=f'return record["{field}"]'
    add('dictionary_code',i,'code_line',expression,
        f'Write only the missing return statement for this function; no fences or function declaration.\n'
        f'def read_{field}(record):\n    """Return the value stored under the key {field!r}."""\n    # missing return statement')
    mode=i%3
    instruction=('Return one line using these reply forms: ANSWER: <value>, ASK: <field>, CONFLICT: <field>. '
                 'Use ANSWER when exactly one distinct value for the requested field is supplied, ASK when none is supplied, '
                 'and CONFLICT when multiple distinct values are supplied. Do not add anything else.\n')
    if mode==0:
        facts=f'owner: {name}\ntag: {tag}\ncount: {count}'
        expected=f'ANSWER: {tag}'
    elif mode==1:
        facts=f'owner: {name}\nreference: {tag}\ncount: {count}'
        expected='ASK: tag'
    else:
        facts=f'owner: {name}\ntag: {tag}\ntag: other_0000\ncount: {count}'
        expected='CONFLICT: tag'
    add('evidence_decision',i,'evidence_action',expected,instruction+f'Source record:\n{facts}\nRequested field: tag.')
rng.shuffle(cases)
write_json('datasets/validation-v1-frozen.json',{'name':'validation-v1-frozen',
    'seed':20261105517,'status':'Frozen after discovery, before validation inference. Not for choosing sites/doses. Synthetic fixtures, not an external benchmark.',
    'cases':cases})
