"""Verify run completeness, exact frozen inputs, request IDs, and conservative ledger."""
import json
from pathlib import Path

from weightlab.budget import consumed
from weightlab.evidence import digest, file_sha256, git_state, utc_now, write_json

roots=sorted(p.parent for p in Path('runs').glob('*/manifest.json'))
total_tokens=total_requests=0
ids=set()
for root in roots:
    manifest=json.loads((root/'manifest.json').read_text())
    suite=json.loads((root/'suite.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert digest(suite)==manifest['suite_sha256'],root
    assert summary['status']=='complete',root
    rows=[json.loads(l) for l in (root/'responses.jsonl').read_text().splitlines()]
    assert len(rows)==summary['response_count']==summary['attempt_count'],root
    cases=suite['cases'][:manifest['limit']] if manifest['limit'] else suite['cases']
    assert [r['case'] for r in rows]==cases,root
    tokens=sum(r['usage']['completion_tokens'] for r in rows)
    assert tokens==summary['generated_tokens'],root
    for row in rows:
        key=f'{root}/{row["case"]["id"]}/seed-{manifest["seed"]}'
        assert key not in ids,key
        ids.add(key)
    total_tokens+=tokens
    total_requests+=len(rows)
ledger={}
for line in Path('research/budget.jsonl').read_text().splitlines():
    row=json.loads(line);ledger[row['request_id']]=row
assert set(ledger)==ids,'Run/ledger request identity mismatch'
assert all(row['event']=='settle' for row in ledger.values()),'Pending/failed reservation remains'
assert consumed()==total_tokens,'Ledger/token mismatch'
result={'utc':utc_now(),'git':git_state(),'complete_runs':len(roots),'requests':total_requests,
        'generated_tokens':total_tokens,'ledger_matches':True,'all_requests_settled':True,
        'scope':'Verifies recorded evidence consistency, not scientific validity or model improvement.'}
write_json('research/evidence-integrity-latest.json',result)
print(json.dumps(result,indent=2))
