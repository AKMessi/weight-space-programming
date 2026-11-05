"""Oracle-free record parser baseline and explicit initial-call answerability replay."""
import json,platform,time
from pathlib import Path
from weightlab.evidence import append_jsonl,file_sha256,git_state,utc_now,write_json
from weightlab.evidence_checker import check_record,check_update
from weightlab.grading import grade_turns,first_json,typed_equal
from portfolio_batch import publish
ROOT=Path('research/p3/checker-replay-v1');ROOT.mkdir(parents=True,exist_ok=False)
write_json(ROOT/'manifest.json',{'utc':utc_now(),'git':git_state(),'python':platform.python_version(),'checker_source_sha256':file_sha256('src/weightlab/evidence_checker.py'),'model':None,'model_requests':0,'datasets':{s:file_sha256(f'datasets/p3/evidence-{s}-original.json') for s in ['development','confirmation']},'scope':'Text parser supports the five explicitly declared field=value layouts. No expected labels or hidden facts are used in decisions. Not a general natural-language checker.'})
for split in ['development','confirmation']:
 cases=json.loads(Path(f'datasets/p3/evidence-{split}-original.json').read_text())['cases'];good=0
 for case in cases:
  start=time.perf_counter_ns();field,first=check_record(case['messages'][-1]['content']);second=check_update(field,case['followups'][0]['messages'][0]['content']);elapsed=time.perf_counter_ns()-start
  steps=[{'content':json.dumps(r)} for r in [first,second]];grading=grade_turns(case,steps);good+=grading['success'];append_jsonl(ROOT/'checker-responses.jsonl',{'case':case,'input_texts':[case['messages'][-1]['content'],case['followups'][0]['messages'][0]['content']],'raw_outputs':[first,second],'grading':grading,'parser_nanoseconds':elapsed,'split':split})
 write_json(ROOT/('checker-'+split+'.json'),{'n':len(cases),'joint_success':good,'model_calls':0,'scope':'Programmatic parser output, not learned clarification; formatting emission and logging excluded from parser timer.'})
c=json.loads(Path('configs/p3_clarification.json').read_text());summaries={}
for key in c['models']:
 baseline=list(map(json.loads,Path(f'runs/p3-clarify-evidence-dev-original-{key}-s17/responses.jsonl').read_text().splitlines()));classifiers=list(map(json.loads,Path(f'runs/p3-clarify-answerability-{key}-s17/responses.jsonl').read_text().splitlines()));counts={'n':len(baseline),'correct_initial':0,'supported':0,'supported_coverage':0,'supported_correct':0,'unsupported_nonnull':0,'invalid_classifiers':0,'actual_research_classifier_calls':len(classifiers),'actual_reused_extraction_calls':len(baseline),'sum_measured_initial_call_tokens':0,'sum_measured_initial_prompt_tokens':0,'invalid_policy_outputs':0}
 for b,cl in zip(baseline,classifiers):
  if b['case']['id']!=cl['case']['id']:raise ValueError('Replay pairing mismatch')
  classification=first_json(cl['content']);valid=isinstance(classification,dict) and set(classification)=={'answerable','field'} and type(classification['answerable']) is bool and classification['field']==b['case']['required'];counts['invalid_classifiers']+=not valid
  policy=(first_json(b['steps'][0]['content']) if classification['answerable'] else {'value':None,'ask':classification['field']}) if valid else None
  counts['invalid_policy_outputs']+=not isinstance(policy,dict);correct=typed_equal(policy,b['case']['expected']);counts['correct_initial']+=correct;nonnull=isinstance(policy,dict) and policy.get('value') is not None
  if b['case']['evidence']=='complete':counts['supported']+=1;counts['supported_coverage']+=nonnull;counts['supported_correct']+=correct
  else:counts['unsupported_nonnull']+=nonnull
  cost=b['steps'][0]['usage']['completion_tokens']+cl['usage']['completion_tokens'];counts['sum_measured_initial_call_tokens']+=cost;prompt_cost=b['steps'][0]['usage']['prompt_tokens']+cl['usage']['prompt_tokens'];counts['sum_measured_initial_prompt_tokens']+=prompt_cost
  append_jsonl(ROOT/'replay-responses.jsonl',{'model':key,'case_id':b['case']['id'],'classification':classification,'classifier_valid':valid,'replay_initial_output':policy,'semantic_correct':correct,'extraction_response_sha256':file_sha256(f'runs/p3-clarify-evidence-dev-original-{key}-s17/responses.jsonl'),'classification_responses_sha256':file_sha256(f'runs/p3-clarify-answerability-{key}-s17/responses.jsonl'),'measured_completion_tokens_sum':cost,'measured_prompt_tokens_sum':prompt_cost,'scope':'Two independent actually measured initial calls reused offline. No adaptive latency or replayed second-turn claim; no new model call or duplicated token charge.'})
 summaries[key]=counts
write_json(ROOT/'replay-summary.json',{'utc':utc_now(),'models':summaries,'scope':'Initial-call policy replay only; original second-turn histories differ if policy changes the first answer. Both actual requests remain charged even if deployment could skip extraction.'});publish([ROOT],'Record deterministic evidence checker and answerability replay baselines')
