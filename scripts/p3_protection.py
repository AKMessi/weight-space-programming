"""Fresh bounded protection fixtures for recovered Fable; no model feedback."""
from pathlib import Path
from weightlab.evidence import write_json
cases=[]
for i in range(24):
 cases.append({'id':f'p3-protect-json-{i}','split':'protection_confirmation','family':'fresh_extract','group':'structured','grader':'json',
  'expected':{'code':f'VX-{709+i*13}','units':i+3},'messages':[{'role':'system','content':'Follow the requested response format. Use only supplied facts.'},
   {'role':'user','content':f'Record code VX-{709+i*13} has {i+3} units. Return only JSON with code (string) and units (integer).'}]})
functions=[('advance','x','Return x plus seven.',[([-4],3),([0],7),([11],18)]),
 ('repeat_twice','s','Return the input string concatenated with itself.',[(['ab'],'abab'),([''],''),(['Z'],'ZZ')]),
 ('squared','x','Return x multiplied by itself.',[([-3],9),([0],0),([7],49)]),
 ('count_items','xs','Return the length of the input list.',[([[]],0),([[1,2]],2),([[9]],1)]),
 ('tail_value','xs','Return the last item of xs, or -7 if xs is empty.',[([[]],-7),([[4,9]],9),([[-1]],-1)]),
 ('upper_label','s','Return s converted to uppercase.',[(['one'],'ONE'),(['Ab'],'AB'),([''],'')])]
for name,param,spec,tests in functions:
 for version in [1,2]:
  for mode in ['code','explain']:
   request='Return one complete Python function only, without Markdown or commentary.' if mode=='code' else 'Return a complete Python function in a Python code fence and explain it in at least eight words.'
   cases.append({'id':f'p3-protect-{name}{version}-{mode}','split':'protection_confirmation','family':name,'group':'functional_'+mode,
    'grader':'functional_code','response_mode':mode,'expected':{'parameters':[param],'tests':[{'args':a,'result':r} for a,r in tests]},
    'messages':[{'role':'system','content':'Follow the requested format. Do not call tools.'},{'role':'user','content':request+f'\nRequired signature: def {name}{version}({param}):\nSpecification: '+spec}]})
p=Path('datasets/p3/fable-protection.json')
if p.exists():raise FileExistsError(p)
write_json(p,{'name':'p3-fable-protection-v1','status':'fresh protection confirmation; no candidate selection','cases':cases})
assert len(cases)==48
