"""Freeze bounded follow-up pilots before feedback; all fixtures are synthetic."""
import json
from pathlib import Path
from weightlab.evidence import write_json

SYSTEM="Follow the requested response contract. Use only supplied facts. Do not call real tools."
TASKS=[
 ('add',['a','b'],'Return the sum of a and b.',[([0,0],0),([-3,8],5),([11,-4],7)]),
 ('positive_sum',['xs'],'Return the sum of the strictly positive numbers in xs.',[([[]],0),([[2,-3,5,0]],7),([[-1,-2]],0)]),
 ('last_or_zero',['xs'],'Return the last item of xs, or integer zero if xs is empty.',[([[]],0),([[3,1]],1),([[-9]],-9)]),
 ('double',['xs'],'Return a list with every number in xs multiplied by two.',[([[]],[]),([[2,-1,0]],[4,-2,0]),([[7]],[14])]),
 ('lower_text',['text'],'Return text with surrounding whitespace removed and letters made lowercase.',[([' A B '],'a b'),([' HELLO'],'hello'),([''],'')]),
 ('lookup',['record','key'],'Return record[key] using dictionary indexing. The key always exists.',[([{'a':3},'a'],3),([{'b':'word'},'b'],'word'),([{'x':False},'x'],False)]),
 ('difference',['a','b'],'Return a minus b.',[([0,2],-2),([-3,-7],4),([10,4],6)]),
 ('count_positive',['xs'],'Return the number of strictly positive entries in xs.',[([[]],0),([[2,-1,0,9]],2),([[-3]],0)]),
 ('first_or_zero',['xs'],'Return the first entry of xs, or integer zero when empty.',[([[]],0),([[4,8]],4),([[-2]],-2)]),
 ('reverse_list',['xs'],'Return a new list with xs in reverse order.',[([[]],[]),([[1,4,2]],[2,4,1]),([[9]],[9])]),
 ('absolute',['value'],'Return the absolute value of value.',[([-4],4),([0],0),([7],7)]),
 ('maximum',['xs'],'Return the maximum value in the nonempty list xs.',[([[1,5,2]],5),([[-7,-2]],-2),([[0]],0)])]


def case(key,group,grader,expected,prompt,split,**extra):
    return {'id':key,'split':split,'family':group,'group':group,'grader':grader,'expected':expected,
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],**extra}


def e1(split):
    dev=split=='development'
    tasks=TASKS[:6] if dev else TASKS[6:]
    cases=[]
    for name,params,instruction,tests in tasks:
        expected={'parameters':params,'tests':[{'args':args,'result':value} for args,value in tests]}
        signature=f'def {name}({", ".join(params)}):'
        for mode in ['code','explain']:
            contract=('Return one complete Python function only, without Markdown or commentary.' if mode=='code'
                      else 'Return a complete Python function in a Python code fence and explain its behavior in at least eight words.')
            cases.append(case(f'e1-{split}-{name}-{mode}',f'functional_{mode}','functional_code',expected,
                              f'{contract}\nRequired signature: {signature}\nSpecification: {instruction}',split,response_mode=mode))
    for i in range(8):
        value=f'{"DV" if dev else "CF"}-{130+i*19}'
        n=3+i
        data={'tag':value,'count':n}
        prompt=(f'Entry: tag={value}; count={n}. Emit just the JSON object using tag and count.' if dev
                else f'From this note extract count as an integer and tag as text: [{n} items / tag {value}]. Your entire reply must be JSON.')
        cases.append(case(f'e1-{split}-json-{i}','structured','json',data,prompt,split))
        literal=f'{value} / x={n} / {value}' if dev else f'{value}\n    [{n}]  {value.lower()}!'
        prompt=(f'Return exactly the payload in <payload>{literal}</payload>, with no surrounding text.' if dev
                else f'Reproduce the following payload verbatim, including its spaces and newline. Output the payload alone.\nBEGIN\n{literal}\nEND')
        cases.append(case(f'e1-{split}-copy-{i}','literal_copy','copy',literal,prompt,split))
    return {'name':f'portfolio-e1-{split}-v1','status':f'frozen synthetic {split}; no broad benchmark claim','cases':cases}


SCHEMA={'type':'object','properties':{'action':{'type':'string','enum':['answer','ask','conflict']},
        'field':{'type':'string'},'value':{'type':['string','null']}},'required':['action','field','value'],'additionalProperties':False}
RULE='Answer only JSON with action, field, value. If the required field has one unambiguous value, action is answer and value is that string. If absent, action is ask and value is null. If contradictory with no precedence rule, action is conflict and value is null. field names the requested field.'


def e2(split,control='original'):
    dev=split=='development'
    cases=[]
    fields=['address','port','branch','destination'] if dev else ['hostname','language','directory','recipient']
    for i,field in enumerate(fields):
        value=f'{"blue" if dev else "amber"}-{13+i*7}'
        replacement=f'{"green" if dev else "violet"}-{81+i}'
        for state in ['complete','missing','conflict']:
            facts=[('project',f'work-{i}'),('note','ordinary')]
            facts+= [(field,value)] if state=='complete' else [('reference',value)]
            if state=='conflict':facts[-1:]=[(field,value),(field,'other-99')]
            else:facts.append(('code','other-99'))
            context='; '.join(f'{k}={v}' for k,v in facts) if dev else '\n'.join(f'[{k}] {v}' for k,v in reversed(facts))
            expected={'action':{'complete':'answer','missing':'ask','conflict':'conflict'}[state],'field':field,'value':value if state=='complete' else None}
            prompt=f'{RULE}\nRequired field: {field}\nSupplied records:\n{context}\nNo precedence rule is provided.'
            follow={'grader':'json','expected':{'action':'answer','field':field,'value':replacement},
                    'messages':[{'role':'user','content':f'Authoritative update: {field}={replacement}. This replaces any earlier value of {field}. Complete the same task now.'}]}
            entry=case(f'e2-{split}-{field}-{state}',state,'json',expected,prompt,split,followups=[follow],control=control)
            if control in ['strong','grammar']:
                entry['messages'][0]['content']+=' Check every supplied record and follow the decision rule carefully. Emit a single JSON object and stop.'
            if control=='grammar':entry['request_overrides']={'response_format':{'type':'json_schema','schema':SCHEMA}}
            cases.append(entry)
    return {'name':f'portfolio-e2-{split}-{control}-v1','status':f'frozen synthetic {split}; actual assistant response passed into next turn','cases':cases}


def main():
    target=Path('datasets/portfolio')
    target.mkdir(exist_ok=True)
    for split in ['development','confirmation']:
        write_json(target/f'e1-{split}.json',e1(split))
        for control in ['original','strong','grammar']:
            write_json(target/f'e2-{split}-{control}.json',e2(split,control))
    # Shared engineering panel; already declared development fixtures, never labelled a new confirmation set.
    engineering=e1('development')
    engineering['cases']=[c for c in engineering['cases'] if c['group']!='functional_explain'][:18]
    engineering.update(name='portfolio-engineering-v1',status='development stress panel shared by precision/composition; not independent validation')
    write_json(target/'engineering.json',engineering)
    # A separate symmetry/diagnosis panel; schema never contains a gold value.
    probes=[]
    for i in range(12):
        expected={'tag':f'ID-{71+i*13}','count':i+2}
        c=case(f'diagnosis-probe-{i}','diagnosis_probe','json',expected,
               f'Item tag={expected["tag"]}; count={i+2}. Return only JSON with tag as text and count as integer.',
               'diagnosis_screening')
        if i<8:c['request_overrides']={'response_format':{'type':'json_schema','schema':{'type':'object','properties':{'tag':{'type':'string'},'count':{'type':'integer'}},'required':['tag','count'],'additionalProperties':False}}}
        c.setdefault('request_overrides',{}).update(temperature=0.0)
        probes.append(c)
    write_json(target/'diagnosis-probes.json',{'name':'diagnosis-probes-v1','status':'fault screening and symmetry probes; first two responses exposed only in behavior arm','cases':probes})
    print('Frozen suite sizes:',{p.name:len(json.loads(p.read_text())['cases']) for p in target.glob('*.json')})


if __name__=='__main__':main()
