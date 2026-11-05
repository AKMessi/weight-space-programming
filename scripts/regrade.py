"""Write a derived grading artifact; never overwrite original recorded grades."""
import argparse
import json
from pathlib import Path

from weightlab.evidence import file_sha256, git_state, utc_now, write_json
from weightlab.grading import GRADER_VERSION, grade
from weightlab.runner import summarize

p = argparse.ArgumentParser()
p.add_argument('responses')
p.add_argument('output')
a = p.parse_args()
if Path(a.output).exists():
    raise FileExistsError(a.output)
rows = [json.loads(line) for line in Path(a.responses).read_text().splitlines()]
changes = []
for row in rows:
    if 'content' not in row:
        continue
    updated = grade(row['case'], row['content'])
    if updated != row['grading']:
        changes.append({'case_id': row['case']['id'], 'original': row['grading'], 'derived': updated})
    row['grading'] = updated
write_json(a.output, {'utc': utc_now(), 'git': git_state(), 'source': a.responses,
                     'source_sha256': file_sha256(a.responses), 'grader_version': GRADER_VERSION,
                     'changes': changes, 'summary': summarize(rows)})
