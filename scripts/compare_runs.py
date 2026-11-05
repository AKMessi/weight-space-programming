"""Paired discovery comparison; raw outputs and original grades stay untouched."""
import argparse
import json
from pathlib import Path

from weightlab.evidence import file_sha256, git_state, utc_now, write_json
from weightlab.grading import GRADER_VERSION, grade

p=argparse.ArgumentParser()
p.add_argument('original')
p.add_argument('comparison')
p.add_argument('output')
a=p.parse_args()
if Path(a.output).exists():raise FileExistsError(a.output)
def load(path):
    return {r['case']['id']:r for r in (json.loads(l) for l in Path(path).read_text().splitlines())}
left,right=load(a.original),load(a.comparison)
rows=[]
for key,r in right.items():
    l=left[key]
    if l['case']!=r['case']:raise ValueError('Case content changed')
    g0,g1=grade(l['case'],l['content']),grade(r['case'],r['content'])
    rows.append({'case_id':key,'original_grade':g0,'comparison_grade':g1,'content_identical':l['content']==r['content'],
                 'tokens_identical':l['usage'].get('completion_tokens')==r['usage'].get('completion_tokens')})
write_json(a.output,{'utc':utc_now(),'git':git_state(),'grader_version':GRADER_VERSION,
                    'sources':[{'path':p,'sha256':file_sha256(p)} for p in [a.original,a.comparison]],
                    'paired_cases':len(rows),'identical_contents':sum(r['content_identical'] for r in rows),
                    'rescues':sum(not r['original_grade']['success'] and r['comparison_grade']['success'] for r in rows),
                    'losses':sum(r['original_grade']['success'] and not r['comparison_grade']['success'] for r in rows),
                    'rows':rows})
