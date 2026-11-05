"""One post-hoc deployment hypothesis; independently frozen function panel."""
import argparse,copy,json
from pathlib import Path
from gguf import GGUFReader
from weightlab.hybrid import replace_tensors
from weightlab.functional import evaluate_code
from weightlab.evidence import file_sha256,write_json
from p3_batch import run_one
from portfolio_batch import publish
CONFIG=Path('configs/p3_format_final.json')
def prepare():
 c=json.loads(Path('configs/fable_speed_followup.json').read_text());floatbase=c['models']['fable_float'];q4=c['models']['fable_Q4_0'];names=[t.name for t in GGUFReader(floatbase['path']).tensors];keep=[n for n in names if not any(n.endswith('.ffn_'+k+'.weight') for k in ['gate','up','down'])]
 manifest=Path('research/p3/final-format-v1/artifact.json');manifest.parent.mkdir(parents=True,exist_ok=False)
 report=replace_tensors(q4,floatbase,keep,'artifacts/p3/hybrids/hybrid_ffn_only.gguf',manifest)
 cfg={'runtime':c['runtime'],'budgets':c['budgets'],'models':{k:copy.deepcopy(c['models'][k]) for k in ['fable_float','fable_Q4_0']}};cfg['models']['hybrid_head_shortconv']=json.loads(Path('configs/p3_hybrids.json').read_text())['models']['hybrid_head_shortconv'];cfg['models']['hybrid_ffn_only']={**floatbase,'path':report['output'],'sha256':report['sha256'],'local_patch_manifest':str(manifest),'local_patch_manifest_sha256':file_sha256(manifest)}
 for spec in cfg['models'].values():spec['sampling'].update(temperature=0,top_k=0,top_p=1,min_p=0,repeat_penalty=1,ignore_eos=False)
 write_json(CONFIG,cfg);publish([manifest,CONFIG],'Export one FFN-only quantized format before fresh quality outcomes')
 specs=[('reciprocal',['x'],'Return one divided by the nonzero number x.','return 1/x',[([2],.5),([-2],-.5),([4],.25)]),
 ('ceil_thirds',['x'],'Return the smallest integer greater than or equal to x divided by three.','return (x+2)//3',[([-4],-1),([0],0),([7],3)]),
 ('product_values',['xs'],'Return the product of all numbers in xs; the empty product is one.','p = 1\nfor x in xs:\n    p *= x\nreturn p',[([[]],1),([[2,-3,4]],-24),([[0,3]],0)]),
 ('running_totals',['xs'],'Return a list of cumulative sums of the input numbers, preserving input order.','out = []\nacc = 0\nfor x in xs:\n    acc += x\n    out.append(acc)\nreturn out',[([[]],[]),([[1,2,-1]],[1,3,2])]),
 ('count_positive',['xs'],'Return the number of input numbers strictly greater than zero.','return sum(1 for x in xs if x>0)',[([[]],0),([[-1,0,3,2]],2)]),
 ('sum_of_squares',['xs'],'Return the sum of squares of the input numbers.','return sum(x*x for x in xs)',[([[]],0),([[-2,3]],13)]),
 ('second_largest',['xs'],'For a list of at least two numbers, return the second largest item, counting duplicates separately.','return sorted(xs)[-2]',[([[1,3,2]],2),([[5,5]],5),([[-1,-3]],-3)]),
 ('drop_first',['xs'],'Return a new list containing every input item except the first; return empty list for empty input.','return xs[1:]',[([[]],[]),([[2,5,1]],[5,1])]),
 ('inside_letters',['s'],'Return the string with its first and last characters removed. For length less than two return empty string.','return s[1:-1]',[(['abcd'],'bc'),(['ab'],''),(['a'],'')]),
 ('rotate_left',['xs'],'Return a new list with the first item moved to the end; empty input remains empty.','return xs[1:]+xs[:1]',[([[]],[]),([[1,2,3]],[2,3,1]),([[9]],[9])]),
 ('divisible_three',['x'],'Return a boolean true exactly when integer x is divisible by three.','return x%3==0',[([-3],True),([4],False),([0],True)]),
 ('clamp_between',['x','low','high'],'Given low <= high, clamp x into the inclusive interval from low to high.','return min(max(x,low),high)',[([5,1,4],4),([-1,0,9],0),([3,0,9],3)]),
 ('prefix_chars',['s','n'],'Return the first n characters of s, where n is nonnegative.','return s[:n]',[(['abc',2],'ab'),(['abc',0],''),(['',4],'')]),
 ('name_or_default',['d','fallback'],'Return the dictionary value for string key name, or fallback if that key is absent.','return d.get("name",fallback)',[([{},'X'],'X'),([{'name':'A'},'X'],'A')]),
 ('reverse_words',['s'],'Split the string on whitespace and return the words in reversed order joined with single spaces.','return " ".join(reversed(s.split()))',[(['one two three'],'three two one'),([''],''),(['  a  b '],'b a')]),
 ('lengths_of_words',['s'],'Split the string on whitespace and return the length of each word as a list, in order.','return [len(w) for w in s.split()]',[(['one a'],[3,1]),([''],[]),(['ab cd'],[2,2])])]
 for mode in ['code','explain']:
  cases=[]
  for name,params,spec,body,tests in specs:
   request='Return one complete Python function only, without Markdown or commentary.' if mode=='code' else 'Return a complete Python function in a Python code fence and explain it in at least eight words.'
   case={'id':'p3-final-format-'+name+'-'+mode,'family':name,'group':'functional_'+mode,'split':'fresh_deployment_confirmation','grader':'functional_code','response_mode':mode,'expected':{'parameters':params,'tests':[{'args':args,'result':value} for args,value in tests]},'messages':[{'role':'system','content':'Follow the requested response format. Use only built-in Python operations; do not import libraries or call tools.'},{'role':'user','content':request+'\nRequired signature: def '+name+'('+', '.join(params)+'):\nSpecification: '+spec}]}
   reference='def '+name+'('+', '.join(params)+'):\n'+'\n'.join('    '+line for line in body.splitlines())
   if not evaluate_code({**case,'response_mode':'code'},reference)['payload_correct']:raise ValueError('Fixture oracle failed: '+name)
   cases.append(case)
  path=Path('datasets/p3/final-format-'+mode+'.json')
  if path.exists():raise FileExistsError(path)
  write_json(path,{'name':'p3-final-format-'+mode,'status':'fresh final deployment function families; not used for format selection','cases':cases});publish([path],'Freeze fresh '+mode+' deployment confirmation fixtures')
def run():
 c=json.loads(CONFIG.read_text())
 for key in c['models']:
  for mode,cap in [('code',128),('explain',256)]:run_one(c,key,'datasets/p3/final-format-'+mode+'.json','format-confirm-'+mode,'precision',cap)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);a=p.parse_args();globals()[a.phase]()
