"""New matched JSON response-switching families; no model feedback used."""
import copy,json
from pathlib import Path
from weightlab.evidence import write_json
config=json.loads(Path('configs/portfolio_precision.json').read_text())
config['models']={k:config['models'][k] for k in ['qwen_float','fable_float','lfm3_float']}
for spec in config['models'].values():spec['sampling'].update(temperature=0.,top_k=0,top_p=1.,min_p=0.,repeat_penalty=1.)
write_json('configs/p3_repair.json',config)
for split,n in [('development',32),('confirmation',120)]:
 cases=[]
 for i in range(n):
  if split=='development':
   first,second='tag','count';value=f'AC-{411+17*i}';number=3+i
   forms=[f'Record: tag={value}; count={number}.',f'The entry has tag {value} and count {number}.']
  else:
   pairs=[('name','size'),('label','amount'),('item','qty'),('identifier','total'),('mark','units'),('symbol','number')]
   first,second=pairs[(i//20)%6];value=f'NX-{823+11*i}';number=6+2*i
   forms=[f'Supplied fields: {first}: {value} | {second}: {number}.',f'{first.upper()} [{value}]\n{second.upper()} [{number}]',f'For this entry the {second} is {number}; its {first} is {value}.']
  facts=forms[i%len(forms)];expected={first:value,second:number}
  for mode in ['plain','fenced']:
   contract='Return only the JSON object, without Markdown or commentary.' if mode=='plain' else 'Return only the JSON object inside one json code fence, with no other text.'
   if split=='confirmation':
    contract=('Your entire reply must be a bare JSON object. Do not include a code fence or other text.' if mode=='plain' else 'Your entire reply must be one Markdown code block labeled json, containing the JSON object.')
   prompt=f'{facts}\nExtract {first} as a string and {second} as an integer. '+contract
   cases.append({'id':f'p3-switch-{split}-{i}-{mode}','split':split,'family':f'{first}/{second}',
    'group':'json_'+mode,'grader':'json_switch','response_mode':mode,'expected':expected,
    'messages':[{'role':'system','content':'Follow the response format requested by the user. Use only supplied facts.'},{'role':'user','content':prompt}]})
 path=Path(f'datasets/p3/switch-{split}.json')
 if path.exists():raise FileExistsError(path)
 suite={'name':'p3-switch-'+split,'status':split+'; paired content with different requested formats','cases':cases}
 write_json(path,suite)
 if split=='development':
  screen={**suite,'name':'p3-switch-screen','cases':cases[:16]+cases[32:40]}
  write_json('datasets/p3/switch-screen.json',screen)
 for control in ['strong','grammar']:
  altered=copy.deepcopy(suite);altered['name']+='-'+control
  for c in altered['cases']:
   c['messages'][0]['content']+=' Decide whether the user requests plain JSON or a json code fence. Follow that choice exactly; do not add an explanation.'
   if control=='grammar' and c['response_mode']=='plain':
    properties={k:{'type':'string' if type(v)==str else 'integer'} for k,v in c['expected'].items()}
    c['request_overrides']={'response_format':{'type':'json_schema','json_schema':{'schema':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}}
  write_json(f'datasets/p3/switch-{split}-{control}.json',altered)
