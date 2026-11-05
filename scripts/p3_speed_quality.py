"""Frozen fresh useful-output checks for format choice; never fit on these."""
import argparse,copy,json
from pathlib import Path
from weightlab.evidence import write_json
from weightlab.functional import evaluate_code
from p3_batch import run_one
from portfolio_batch import publish
p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
if a.phase=='freeze':
 specs=[('triple','x','Return x multiplied by three.','x*3',[([-2],-6),([0],0),([9],27)]),
 ('offset','x','Return x minus two.','x-2',[([-1],-3),([2],0),([10],8)]),
 ('double_items','xs','Return a new list with each input number doubled.','[x*2 for x in xs]',[([[]],[]),([[1,-2]],[2,-4])]),
 ('smallest','xs','Return the minimum of the nonempty input list.','min(xs)',[([[3,1]],1),([[-5,-1]],-5)]),
 ('largest','xs','Return the maximum of the nonempty input list.','max(xs)',[([[3,1]],3),([[-5,-1]],-1)]),
 ('mirror','s','Return the string in reversed order.','s[::-1]',[(['ab'],'ba'),([''],''),(['AbC'],'CbA')]),
 ('total','xs','Return the sum of the numbers in the list.','sum(xs)',[([[]],0),([[2,-3,5]],4)]),
 ('head_or_zero','xs','Return first list item, or zero if empty.','xs[0] if xs else 0',[([[]],0),([[3,9]],3)]),
 ('negatives','xs','Return a list of only the negative input numbers, preserving order.','[x for x in xs if x<0]',[([[0,-2,4,-1]],[-2,-1]),([[]],[])]),
 ('absolute','x','Return the absolute value of x.','abs(x)',[([-7],7),([0],0),([3],3)]),
 ('lower_label','s','Return s converted to lowercase.','s.lower()', [(['AbC'],'abc'),([''],''),(['X-7'],'x-7')]),
 ('join_dash','xs','Return the list of strings joined using a dash between items.','"-".join(xs)',[([[]],''),([['a','b']],'a-b')]),
 ('get_label','d','Return the dictionary value at key label, or empty string if absent.','d.get("label", "")',[([{}],''),([{'label':'A'}],'A')]),
 ('clamp_zero','x','Return zero if x is negative, otherwise x.','max(x,0)',[([-2],0),([0],0),([7],7)]),
 ('is_even','x','Return a boolean that is true exactly when x is even.','x%2==0',[([-2],True),([3],False),([0],True)]),
 ('sort_numbers','xs','Return the numbers in ascending order as a new list.','sorted(xs)',[([[3,1,2]],[1,2,3]),([[]],[])])]
 cases=[]
 for name,param,spec,expr,tests in specs:
  for mode in ['code','explain']:
   contract='Return one complete Python function only, without Markdown or commentary.' if mode=='code' else 'Return a complete Python function in a Python code fence and explain it in at least eight words.'
   case={'id':'p3-format-'+name+'-'+mode,'family':name,'group':'functional_'+mode,'split':'fresh_format_quality','grader':'functional_code','response_mode':mode,'expected':{'parameters':[param],'tests':[{'args':args,'result':result} for args,result in tests]},'messages':[{'role':'system','content':'Follow the requested response format.'},{'role':'user','content':contract+f'\nRequired signature: def {name}({param}):\nSpecification: '+spec}]}
   reference=f'def {name}({param}):\n    return {expr}'
   if not evaluate_code({**case,'response_mode':'code'},reference)['payload_correct']:raise ValueError('Fixture oracle failed')
   cases.append(case)
 path=Path('datasets/p3/format-quality.json')
 if path.exists():raise FileExistsError(path)
 write_json(path,{'name':'p3-format-quality-v1','status':'fresh protected functional fixtures; related elementary function families, not a broad coding benchmark','cases':cases})
 write_json('datasets/p3/format-noop-quality.json',{'name':'p3-format-noop-v1','cases':[c for c in cases if c['response_mode']=='code']})
 publish([path,'datasets/p3/format-noop-quality.json'],'Freeze useful-output quality fixtures before hybrid feedback')
else:
 config=json.loads(Path('configs/fable_speed_followup.json').read_text());hybrids=json.loads(Path('configs/p3_hybrids.json').read_text());config['models'].update(hybrids['models'])
 for spec in config['models'].values():spec['sampling'].update(temperature=0,repeat_penalty=1.,ignore_eos=False)
 for key in ['fable_float','fable_Q8_0','fable_Q4_0','fable_Q4_K_M','hybrid_head','hybrid_head_ffn','hybrid_head_shortconv']:
  run_one(config,key,'datasets/p3/format-quality.json','format-quality','reserve',128)
 run_one(config,'hybrid_noop','datasets/p3/format-noop-quality.json','format-noop','reserve',96)
