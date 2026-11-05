"""Pin a bounded repository-disjoint next-line evaluation; never print held-out code."""
import hashlib,json,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
revision='8a7cf0c8942cc1aa066bf261839650ac55a2ff79'
repo='tianyang/repobench_python_v1.1';filename='data/cross_file_first-00000-of-00002-baebda7f3a6e980a.parquet'
url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
source=Path('local/datasets/repobench-first.parquet');source.parent.mkdir(parents=True,exist_ok=True)
if not source.exists():
 with urllib.request.urlopen(url,timeout=60) as response,source.with_suffix('.partial').open('wb') as out:
  while block:=response.read(1024*1024):out.write(block)
 source.with_suffix('.partial').replace(source)
rows=pq.read_table(source).to_pylist()
def h(value):return hashlib.sha256(('p3-repo-617/'+value).encode()).hexdigest()
by_split={'development':[],'calibration':[],'confirmation':[]};counts={}
for i,row in enumerate(rows):
 name=row['repo_name'];group=int(h(name)[:8],16)%10
 split='development' if group<3 else 'calibration' if group<5 else 'confirmation'
 # A deterministic short-context adaptation: no gold-snippet selection and no generated actions.
 contexts='\n'.join(f"# {c['path']}\n{c['snippet']}" for c in row['context'][:2])[:850]
 code=row['cropped_code'][-650:];expected=row['next_line'].rstrip('\r\n')
 prompt=f"Repository: {name}\nReference snippets:\n{contexts}\nCurrent file prefix:\n{code}\nReturn the next code line only."
 if not prompt.isascii() or len(prompt)>1800 or not expected or '\n' in expected:continue
 case={'id':f'repobench-first-{i}','split':split,'family':name,'group':'repo_next_line','grader':'next_line',
       'expected':expected,'source':{'dataset':repo,'revision':revision,'file':filename,'row_index':i,'repo_name':name,'file_path':row['file_path']},
       'messages':[{'role':'system','content':'Continue the supplied Python file. Return only the next code line, preserving indentation. Do not run tools.'},{'role':'user','content':prompt}]}
 by_split[split].append(case)
limits={'development':80,'calibration':24,'confirmation':240};selected={}
for split,cases in by_split.items():
 chosen=[];seen={}
 for case in sorted(cases,key=lambda c:h(c['id'])):
  repo_name=case['family']
  if seen.get(repo_name,0)>=4:continue
  chosen.append(case);seen[repo_name]=seen.get(repo_name,0)+1
  if len(chosen)==limits[split]:break
 if len(chosen)!=limits[split]:raise ValueError(f'Insufficient {split}: {len(chosen)}')
 selected[split]=chosen
 path=Path(f'datasets/p3/fable-repo-{split}.json')
 if path.exists():raise FileExistsError(path)
 write_json(path,{'name':'p3-fable-repo-'+split,'status':split+'; repository-disjoint short-context adaptation, not full official RepoBench', 'cases':chosen})
 sets={c['family'] for c in chosen};counts[split]={'cases':len(chosen),'repositories':len(sets)}
assert not ({c['family'] for c in selected['development']} & {c['family'] for c in selected['confirmation']})
write_json('research/p3/repobench-source.json',{'utc':utc_now(),'git':git_state(),'dataset':repo,'revision':revision,'url':url,
 'file_sha256':file_sha256(source),'rows_in_source':len(rows),'counts':counts,
 'selection':'Repository hash partitions 0-2 development,3-4 calibration,5-9 confirmation; deterministic row ordering, max4/repo, ASCII and predeclared length filter before outcomes.',
 'context_policy':'First two reference snippets truncated together to850chars, last650chars current prefix; no gold-snippet-index use; total user prompt <=1800ASCIIchars.',
 'primary_metric':'first extracted line exact match with indentation; strict whole-response match separately; text metric, no generated code execution.'})
print(json.dumps(counts),flush=True)
