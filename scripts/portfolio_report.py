"""Build all five reports from recorded evidence; missing required runs are errors."""
import argparse,json,statistics,random
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.grading import first_json,typed_equal,grade_turns,grade,GRADER_VERSION
from weightlab.runner import summarize
from validation_report import upper_event_probability
from portfolio_batch import ROOTS
import re

SOURCES={}


def rows(label,key,seeds=(17,)):
    result=[]
    for seed in seeds:
        root=Path(f'runs/p2-{label}-{key}-s{seed}')
        summary=json.loads((root/'summary.json').read_text())
        if summary['status']!='complete':raise ValueError(f'incomplete {root}')
        source=root/'responses.jsonl';SOURCES[str(source)]=file_sha256(source)
        for row in map(json.loads,source.read_text().splitlines()):
            result.append({**row,'recorded_grading':row['grading'],
                           'grading':grade_turns(row['case'],row['steps']), 'analysis_seed':seed})
    return result


def counts(data):
    return {'n':len(data),'strict':sum(r['grading']['success'] for r in data),
            'payload':sum(r['grading']['payload_correct'] for r in data),
            'cutoffs':sum(r['finish_reason']=='length' for r in data),
            'grader_rejections':sum('grading_error' in r['grading'] for r in data)}


def by_group(data):
    return {g:counts([r for r in data if r['case']['group']==g]) for g in sorted({r['case']['group'] for r in data})}


def table(items):
    out=['| Arm | Group | Strict | Payload / function | Cutoffs |','|---|---|---:|---:|---:|']
    for key,data in items.items():
        for group,n in by_group(data).items():out.append(f'| {key} | {group} | {n["strict"]}/{n["n"]} | {n["payload"]}/{n["n"]} | {n["cutoffs"]} |')
    return out


def pairing(original,changed,ids=None):
    a={(r['case']['id'],r['analysis_seed']):r for r in original}
    b={(r['case']['id'],r['analysis_seed']):r for r in changed}
    if a.keys()!=b.keys():raise ValueError('pair mismatch')
    selected=[k for k in a if ids is None or k[0] in ids]
    clusters={}
    for key in selected:
        case=a[key]['case'];identifier=key[0]
        if case.get('followups'):
            cluster=re.search(r'Required field:\s*(\w+)',case['messages'][-1]['content'])[1]
        elif identifier.endswith(('-code','-explain')):cluster=identifier.rsplit('-',1)[0]
        else:cluster=identifier.rsplit('-',1)[-1]
        clusters.setdefault(cluster,[]).append((a[key]['grading']['success'],b[key]['grading']['success']))
    effects=[statistics.mean(int(y)-int(x) for x,y in group) for group in clusters.values()]
    loss_events=sum(any(x and not y for x,y in group) for group in clusters.values())
    rng=random.Random(20261105)
    bootstrap=sorted(statistics.mean(rng.choices(effects,k=len(effects))) for _ in range(10000))
    return {'n':len(selected),'rescues':sum(not a[k]['grading']['success'] and b[k]['grading']['success'] for k in selected),
            'losses':sum(a[k]['grading']['success'] and not b[k]['grading']['success'] for k in selected),
            'clusters':len(clusters),'mean_cluster_gain':statistics.mean(effects),
            'conditional_cluster_bootstrap_95':[bootstrap[249],bootstrap[9749]],
            'one_sided_95_upper_cluster_loss_event_probability':upper_event_probability(loss_events,len(clusters)),
            'assumptions':'Seeds, code/explanation pairs and evidence triples stay grouped. Conditional authored-item resampling only; not population certainty or independent domain sampling.'}


def e1(out):
    selection=json.loads(Path('research/selections/e1.json').read_text());data={};result={}
    for root in ROOTS:
        candidate=selection['models'][root]['diagnostic_or_selected']
        original=rows('e1confirm',root,(17,29));changed=rows('e1confirm',candidate,(17,29))
        data[root]=original;data[candidate]=changed
        primary=[r['case']['id'] for r in original if r['case']['group'] in ('structured','literal_copy')]
        baseline=[r for r in original if r['case']['id'] in primary];compare=[r for r in changed if r['case']['id'] in primary]
        failures=len(baseline)-counts(baseline)['strict'];reduction=(counts(compare)['strict']-counts(baseline)['strict'])/failures if failures else None
        before={(r['case']['id'],r['analysis_seed']):r for r in original}
        functional_losses=sum(before[(r['case']['id'],r['analysis_seed'])]['grading']['payload_correct'] and not r['grading']['payload_correct']
                              for r in changed if r['case']['group'].startswith('functional_'))
        explanation_losses=sum(before[(r['case']['id'],r['analysis_seed'])]['grading']['success'] and not r['grading']['success']
                               for r in changed if r['case']['group']=='functional_explain')
        eligible=selection['models'][root]['eligible_repair_on_development']
        result[root]={'candidate':candidate,'development_eligible':eligible,'relative_primary_failure_reduction':reduction,
                      'functional_losses':functional_losses,'explanation_losses':explanation_losses,
                      'confirmed_pilot_repair':bool(eligible and reduction is not None and reduction>=.3 and functional_losses==explanation_losses==0),
                      'paired_primary':pairing(original,changed,primary)}
    data['aligned Fable parent']=rows('e1confirm','parent_uniform_bf16',(17,29))
    controls={}
    for root in ROOTS:
        for kind in ['strong','grammar-v2']:controls[root+'/'+kind]=rows('e1-confirmation-'+kind,root)
    lines=['# Experiment 1 — Repair habits while preserving utility','',
           'Completed bounded pilot. Development selection and confirmation were separated. Twelve functional fixtures per seed include code and explanation modes; the six underlying functions are correlated pairs. These tasks do not establish repository-level coding specialization.','',
           '## Frozen confirmation',*table(data),'','## Matched prompt / real-schema controls',*table(controls),'','## Decisions','']
    for root,r in result.items():lines.append(f'- {root}: development eligible={r["development_eligible"]}; confirmation paired rescues/losses={r["paired_primary"]["rescues"]}/{r["paired_primary"]["losses"]}; functional losses={r["functional_losses"]}; explanation losses={r["explanation_losses"]}; confirmed pilot repair={r["confirmed_pilot_repair"]}.')
    lines+=['','A retained diagnostic candidate cannot become an independently selected success merely because its confirmation looks attractive. Fable superiority over the parent is not presumed. The interpreter covers a bounded Python subset; syntax/unsupported-language rejections remain in the raw grading records and need manual audit before broad capability claims. Explanation length is a proxy.','',
            'All discovery, prior Q/K/V restorations, rejected candidates and exact selection rules are preserved in the journal, run directories and `research/selections/e1.json`. The new schema v1 controls were malformed and superseded by separately recorded v2 controls. No new editing-method claim or victory over unimplemented activation methods is made.']
    (out/'experiment-1.md').write_text('\n'.join(lines)+'\n')
    return result


def conversation_metrics(data):
    result={'conversations':len(data),'strict_joint':0,'semantic_joint':0,'first_semantic':0,'second_semantic':0,
            'complete_cases':0,'correct_supported_answers':0,'incomplete_cases':0,'unsupported_answer_actions':0,'invented_nonnull_values':0}
    for row in data:
        grades=row['grading']['turn_grades'];parsed=first_json(row['steps'][0]['content'])
        result['strict_joint']+=row['grading']['success'];result['semantic_joint']+=row['grading']['payload_correct']
        result['first_semantic']+=grades[0]['payload_correct'];result['second_semantic']+=grades[1]['payload_correct']
        if row['case']['group']=='complete':
            result['complete_cases']+=1;result['correct_supported_answers']+=grades[0]['payload_correct']
        else:
            result['incomplete_cases']+=1
            if isinstance(parsed,dict) and parsed.get('action')=='answer':
                result['unsupported_answer_actions']+=1;result['invented_nonnull_values']+=parsed.get('value') is not None
    return result


def checker(case):
    prompt=case['messages'][-1]['content'];field=re.search(r'Required field: ([^\n]+)',prompt)[1]
    facts=prompt.split('Supplied records:\n',1)[1].split('\nNo precedence',1)[0]
    pairs=re.findall(r'([a-z_]+)=([^;\n]+)',facts) or re.findall(r'^\[([a-z_]+)\]\s*(.*)$',facts,re.M)
    values={v.strip() for k,v in pairs if k==field}
    first={'action':'ask' if not values else 'conflict' if len(values)>1 else 'answer','field':field,
           'value':next(iter(values)) if len(values)==1 else None}
    update=case['followups'][0]['messages'][0]['content']
    value=re.search(re.escape(field)+r'=([^\s.]+)',update)[1]
    second={'action':'answer','field':field,'value':value}
    return typed_equal(first,case['expected']) and typed_equal(second,case['followups'][0]['expected'])


def e2(out):
    selection=json.loads(Path('research/selections/e2.json').read_text());items={};result={}
    for root in ROOTS:
        candidate=selection['models'][root]['diagnostic_or_selected']
        for key in [root,candidate]:items[key]=rows('e2confirm-original',key)
        for kind in ['strong','grammar-v2']:items[root+'/'+kind]=rows('e2confirm-'+kind,root)
        base=items[root];changed=items[candidate]
        losses=pairing([r for r in base if r['case']['group']=='complete'],[r for r in changed if r['case']['group']=='complete'])['losses']
        before=counts(base);after=counts(changed);failures=before['n']-before['strict']
        reduction=(after['strict']-before['strict'])/failures if failures else None
        result[root]={'candidate':candidate,'paired':pairing(base,changed),'complete_case_losses':losses,
                 'relative_joint_failure_reduction':reduction,'confirmed_pilot_repair':bool(selection['models'][root]['eligible_repair_on_development'] and reduction is not None and reduction>=.3 and not losses)}
    metrics={key:conversation_metrics(data) for key,data in items.items()}
    lines=['# Experiment 2 — Ask, then recover after evidence arrives','',
           'Completed two-turn pilot with actual prior model replies in the second-turn context. Twelve conversations = four requested-field families × three evidence conditions. These are correlated synthetic tasks, not 12 independently sampled domains.','',
           '| Arm | Strict joint / 12 | Semantic joint / 12 | First correct | Second correct | Supported answers / 4 | Unsupported answer actions / 8 | Invented non-null / 8 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,m in metrics.items():lines.append(f'| {key} | {m["strict_joint"]} | {m["semantic_joint"]} | {m["first_semantic"]} | {m["second_semantic"]} | {m["correct_supported_answers"]} | {m["unsupported_answer_actions"]} | {m["invented_nonnull_values"]} |')
    suite=json.loads(Path('datasets/portfolio/e2-confirmation-original.json').read_text());rule_success=sum(checker(c) for c in suite['cases'])
    result['deterministic_checker']={'joint_successes':rule_success,'n':len(suite['cases']),'scope':'Explicit structured records and authoritative updates only; parses supplied pairs without reading expected labels.'}
    lines+=['',f'Deterministic structured-record checker: {rule_success}/{len(suite["cases"])} complete conversations. This is a useful task-specific baseline, not evidence of general understanding.','',
            '## Interpretation','',
            'Schema correctness is insufficient for evidence-sensitive decisions. An answer action with null is a wrong decision but is separated from inventing a non-null value. A correct second turn cannot erase a bad first decision. Coverage and unsupported answering must be read together. The fixed A/B edits are black-box baselines; an activation-derived evidence direction was not implemented. No extra inference calls are hidden in the one-call arms.','']
    for root,r in result.items():
        if root!='deterministic_checker':lines.append(f'- {root}: confirmed pilot repair={r["confirmed_pilot_repair"]}; complete-case losses={r["complete_case_losses"]}; paired rescues/losses={r["paired"]["rescues"]}/{r["paired"]["losses"]}.')
    (out/'experiment-2.md').write_text('\n'.join(lines)+'\n')
    return {'decisions':result,'metrics':metrics}


def e3(out):
    config=json.loads(Path('configs/portfolio_precision.json').read_text());items={};metrics={}
    lines=['# Experiment 3 — Quantization and patch deployment','',
           'Completed same-lineage numerical/deployment stress pilot. Natural useful-repair retention requires an independently useful repair on this exact source artifact; that prerequisite is not established by these diagnostic A edits. No precision allocator or repair-retention success is claimed.','',
           '| Model | Precision | Original strict / 18 | A strict / 18 | Gain | Bytes original | A bytes | Median original decode tok/s |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for root in ['qwen','lfm3','fable']:
        metrics[root]={}
        for precision in ['float','Q8_0','Q4_0']:
            basekey=root+'_'+precision;patchkey=basekey+'_A'
            base=rows('e3precision',basekey);changed=rows('e3precision',patchkey)
            before=counts(base);after=counts(changed)
            speeds=[r['timings']['predicted_per_second'] for r in base if r.get('timings',{}).get('predicted_per_second')]
            source=config['models'][basekey];patch=config['models'][patchkey]
            gain=(after['strict']-before['strict'])/len(base)
            metrics[root][precision]={'base':by_group(base),'A':by_group(changed),'absolute_gain':gain,'paired':pairing(base,changed),
                'bytes':Path(source['path']).stat().st_size,'A_bytes':Path(patch['path']).stat().st_size,
                'median_decode_tps':statistics.median(speeds) if speeds else None,'natural_repair_retention_gate':'not established'}
            lines.append(f'| {root} | {precision} | {before["strict"]} | {after["strict"]} | {gain:+.1%} | {metrics[root][precision]["bytes"]} | {metrics[root][precision]["A_bytes"]} | {statistics.median(speeds):.2f} |')
    original=rows('e3qadparity','lfm3_qad');rewritten=rows('e3qadparity','qad_rewrite')
    parity=all(a['content']==b['content'] and a['usage']['completion_tokens']==b['usage']['completion_tokens'] for a,b in zip(original,rewritten))
    metrics['qad_identity']={'requests':len(original),'content_and_token_parity':parity,'paired':pairing(original,rewritten)}
    lines+=['',f'QAD no-op rewrite: exact content/token parity={parity} on {len(original)} greedy schema probes. Every quantized tensor byte and original metadata item was separately verified. This does not equate QAD to ordinary Q4.','',
            'The A intervention edits a normalization vector retained in F32 by these quantizers. Quantization manifests verify those bytes. The comparison tests propagation of an identical small parameter edit through a quantized network; it cannot show a sparse floating patch being rounded away, or demonstrate a novel mixed-precision allocation method.','',
            'Qwen uses a pinned community floating export as its own source, not an assumed reconstruction of the official Q8 file. LFM3 uses official ordinary F16; all its precision arms use 18 GPU layers for memory headroom. Actual templates, backend/settings, resource samples, quantizer logs and source hashes are retained. Speeds are short-task observations with background preparation/download activity, not isolated benchmarks. Ratios of tiny or zero behavioral gains are intentionally omitted.']
    (out/'experiment-3.md').write_text('\n'.join(lines)+'\n')
    return metrics


def e4(out):
    engineering=json.loads(Path('datasets/portfolio/engineering.json').read_text());ids={c['id'] for c in engineering['cases']}
    config=json.loads(Path('configs/portfolio.json').read_text());result={}
    lines=['# Experiment 4 — Patch interactions','',
           'Completed diagnostic composition pilot. A/B were not both independently established useful repairs; the useful-repair composition claim fails its prerequisite. These measurements test numerical commutation and behavioral interaction of fixed interventions.','',
           '| Model | Panel | Base | A | B | R | AB | Interaction (pp) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for root in ROOTS:
        result[root]={}
        for panel in ['engineering','joint_evidence']:
            data={}
            for suffix in ['', '_A','_B','_R','_AB']:
                if panel=='engineering':
                    candidate=rows('e4engineering-v2',root+suffix) if suffix=='_AB' else [r for r in rows('e1dev',root+suffix) if r['case']['id'] in ids]
                else:candidate=rows('e4joint',root+suffix) if suffix in ['_R','_AB'] else rows('e2dev-original',root+suffix)
                data[suffix or 'base']=counts(candidate)
            n=data['base']['n']
            interaction=100*(data['_AB']['strict']-data['_A']['strict']-data['_B']['strict']+data['base']['strict'])/n
            result[root][panel]={'arms':data,'interaction_percentage_points':interaction,'n':n}
            values=[data[k]['strict'] for k in ['base','_A','_B','_R','_AB']]
            lines.append(f'| {root} | {panel} (n={n}) | '+ ' | '.join(map(str,values))+f' | {interaction:+.1f} |')
        ab=rows('e4engineering-v2',root+'_AB')[:6];ba=rows('e4ordercheck-v2',root+'_BA')
        identity=config['models'][root+'_AB']['sha256']==config['models'][root+'_BA']['sha256']
        outputs=all(a['content']==b['content'] and a['usage']['completion_tokens']==b['usage']['completion_tokens'] for a,b in zip(ab,ba))
        result[root]['order']={'file_identity':identity,'six_request_content_token_identity':outputs}
    lines+=['','AB/BA file identity and six-request behavioral identity are reported in the JSON. Parameter commutation follows from disjoint edited tensors; additive behavior does not follow. Nonzero interactions on a small shared panel are exploratory, not a learned interference predictor. R has a matched L2 delta but is not a semantic repair.','',
            'The engineering subset reuses the declared development cases from Experiment 1; the joint panel reuses Experiment 2 development conversations. This reuse reduces compute and is explicit. Neither panel is presented as fresh confirmation for composition. A larger independently useful patch library is required for the original paper-level hypothesis.',
            '', 'The initial AB/BA engineering runs used 128 output tokens while the reused individual arms allowed 192. Those runs are preserved as a mismatched-cap diagnostic. The primary v2 engineering/order runs repeat the same frozen cases at 192, matching all individual arms; the joint-evidence arms already matched at 128.']
    (out/'experiment-4.md').write_text('\n'.join(lines)+'\n')
    return result


def e5(out):
    screen=json.loads(Path('research/diagnosis-screening.json').read_text())
    execution=json.loads(Path('research/diagnosis-editor-run.json').read_text())
    editor=rows(execution['label'],'local_editor')
    predictions={r['case']['id']:r for r in editor};metrics={};baseline={'weight_anomaly':[],'behavior_probe':[]}
    labels={}
    for s in screen['screens']:
        for p in s['presentations']:
            if not p['included']:continue
            labels[p['anonymous_id']]=p
            baseline['weight_anomaly'].append(p['weight_anomaly_baseline']==p['label'])
            baseline['behavior_probe'].append(p['behavior_baseline']==p['label'])
    for arm in ['weights','behavior']:
        selected=[r for r in editor if r['case']['group']==arm]
        confidence=[r for r in selected if r['grading'].get('stated_confidence') is not None]
        healthy=[r for r in selected if r['case']['expected']=='healthy']
        agreements=[]
        for s in screen['screens']:
            included=[p for p in s['presentations'] if p['included']]
            if len(included)==2:
                a,b=[predictions[p['anonymous_id']+'-'+arm]['grading']['diagnosis'] for p in included]
                agreements.append(a is not None and b is not None and a==b)
        metrics[arm]={'n':len(selected),'diagnosis_correct':sum(r['grading']['payload_correct'] for r in selected),
             'raw_label_correct':sum(r['grading']['diagnosis']==r['case']['expected'] for r in selected),
             'invalid_confidence_or_format':sum(not r['grading']['format_valid'] for r in selected),
             'healthy_presentations':len(healthy),'false_alarm_degraded':sum(r['grading']['diagnosis']=='degraded' for r in healthy),
             'layout_agreements':sum(agreements),'layout_pairs':len(agreements),
             'confidence_brier':statistics.mean((r['grading']['stated_confidence']-int(r['grading']['payload_correct']))**2 for r in confidence) if confidence else None,
             'editor_prompt_tokens':sum(r['usage']['prompt_tokens'] for r in selected)}
    metrics['baselines']={k:{'correct':sum(v),'n':len(v)} for k,v in baseline.items()}
    label_counts={label:sum(p['label']==label for p in labels.values()) for label in ['healthy','degraded']}
    metrics['majority_class_baseline']={'correct':max(label_counts.values()),'n':len(labels),'label_counts':label_counts}
    lines=['# Experiment 5 — Blinded diagnosis and equivalent layouts','',
           'Completed a bounded local-editor pilot. The editor receives anonymized summaries/small raw samples or two fixed target outputs in a fresh request per case. It never receives filenames, target hashes, a healthy reference, screening scores or expected labels. This is not frontier-LLM full-weight tool use and does not replicate the motivating report.',
           f'The full batch uses context {execution["context"]} after a tokenization preflight (maximum prompt {execution["max_prompt_tokens"]} tokens). The original 2K attempt completed three cases and rejected a fourth before decoding; its raw evidence and known-zero failed-request settlement remain in the ledger. The entire unchanged batch was repeated, rather than combining contexts or dropping the long case.','',
           '| Model | State | Healthy payload / 12 | Variant payload / 12 | Clear fault? | Numerical parity | Runtime parity |',
           '|---|---|---:|---:|---|---|---|']
    for s in screen['screens']:lines.append(f'| {s["model"]} | {s["state"]} | {s["healthy_payload_successes"]} | {s["variant_payload_successes"]} | {s["clear_phenotype"]} | {s["numerical_parity"]} | {s["runtime_content_parity"]} |')
    lines+=['','## Diagnosis results','','| Evidence | Correct | Healthy false alarms | Layout agreement | Confidence Brier |','|---|---:|---:|---:|---:|']
    for arm,m in metrics.items():
        if arm not in ['weights','behavior']:continue
        lines.append(f'| {arm} | {m["diagnosis_correct"]}/{m["n"]} | {m["false_alarm_degraded"]}/{m["healthy_presentations"]} | {m["layout_agreements"]}/{m["layout_pairs"]} | {m["confidence_brier"]} |')
    lines+=['',f'Deterministic baselines: {json.dumps(metrics["baselines"],sort_keys=True)}. The weight baseline flags a negative output-normalization mean; the behavior baseline uses exact supplied-fact answer checks on the two exposed probes.',
            f'Majority-class baseline: {metrics["majority_class_baseline"]["correct"]}/{metrics["majority_class_baseline"]["n"]}; class counts {label_counts}. Accuracy above 50% is not automatically better than this baseline.','',
            'Faults were screened before editor inference using a declared 25pp payload drop. Failed numerical/runtime layouts and perturbations without that decline were explicitly excluded from binary accuracy. Original/permuted presentations are correlated, and there are at most nine independent model-state combinations. High agreement can come from a constant diagnosis; read it with accuracy and false alarms.','',
            'Output-normalization sign reversal is conspicuous; FFN sign reversal is a different stress case. These injected changes are not natural failures or unique behavioral circuits. Summaries partly discard layout information by construction, while raw samples change under permutation; this measures only the limited evidence given. No diagnostic repair was executed, no external API editor was used, and confidence statistics are descriptive on this tiny set. Full case mapping, exclusions, frozen prompts and all editor outputs are retained.']
    (out/'experiment-5.md').write_text('\n'.join(lines)+'\n')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='research/portfolio-v3')
    output=Path(parser.parse_args().output)
    # Build only after every required study has evidence; a partial output is not completion.
    if output.exists():raise FileExistsError('version reports rather than overwrite')
    output.mkdir()
    result={'utc':utc_now(),'git':git_state(),'completion_scope':'five bounded pilots with negative prerequisite closures; not five paper-ready studies',
            'analysis_grader_version':GRADER_VERSION,'grading_note':'Post-hoc v6 functional extraction/signature sensitivity; original recorded v4/v5 and frozen selections preserved. Primary copying/JSON/evidence/diagnosis semantics unchanged.',
            'experiment_1':e1(output),'experiment_2':e2(output),'experiment_3':e3(output),'experiment_4':e4(output),'experiment_5':e5(output)}
    # Derived artifacts preserve both the raw-source checksum and every changed grade.
    regrades=output/'regrades';regrades.mkdir()
    grading_index=[]
    for raw in sorted(Path('runs').glob('*/responses.jsonl')):
        records=list(map(json.loads,raw.read_text().splitlines()));changes=[]
        manifest=json.loads((raw.parent/'manifest.json').read_text())
        for record in records:
            if 'content' not in record:continue
            old=record['grading']
            new=grade_turns(record['case'],record['steps']) if record.get('steps') else grade(record['case'],record['content'])
            if old!=new:changes.append({'case_id':record['case']['id'],'recorded':old,'derived':new})
            record['grading']=new
        recorded_version=manifest.get('grader_version','unversioned; identify implementation by source commit')
        sidecar={'source':str(raw),'source_sha256':file_sha256(raw),'recorded_grader_version':recorded_version,
                 'analysis_grader_version':GRADER_VERSION,'changes':changes,'derived_summary':summarize(records)}
        write_json(regrades/(raw.parent.name+'.json'),sidecar)
        grading_index.append({'run':str(raw.parent),'changes':len(changes),'recorded_grader_version':recorded_version})
    write_json(regrades/'index.json',grading_index)
    for report in output.glob('experiment-*.md'):
        with report.open('a') as f:
            f.write('\nCode scoring uses post-hoc derived grader v7 (v6 functional rules); original recorded grades, frozen selections and raw outputs are preserved. The [derived sidecars](regrades/) list every change with its source checksum. This correction is sensitivity analysis, not an independent prespecified confirmation.\n')
    runs=[]
    for p in sorted(Path('runs').glob('p2-*/summary.json')):
        summary=json.loads(p.read_text());runs.append({'run':str(p.parent),'summary':summary,'response_sha256':file_sha256(p.parent/'responses.jsonl')})
    result.update(sources=SOURCES,runs=runs,phase_generated_tokens=sum(r['summary']['generated_tokens'] for r in runs),
                  phase_inference_requests=sum(r['summary'].get('inference_requests',r['summary']['attempt_count']) for r in runs),
                  complete_runs=sum(r['summary']['status']=='complete' for r in runs),
                  failed_runs=sum(r['summary']['status']!='complete' for r in runs))
    if len(runs)>150 or result['phase_generated_tokens']>250000:raise ValueError('declared phase envelope exceeded')
    write_json(output/'results.json',result)
    lines=['# Five-pilot portfolio results','',result['completion_scope'], '',
           f'Phase totals: {len(runs)} runs, {result["phase_inference_requests"]} inference requests, {result["phase_generated_tokens"]} generated tokens. Raw evidence, schema corrections and rejected candidates are retained.','']
    lines += [f'Run statuses: {result["complete_runs"]} complete, {result["failed_runs"]} failed attempts. The intended editor trial completed through a separately frozen full-batch retry.','',
              'Code tables use current versioned sensitivity grading: v6 separates a leading function from trailing prose and checks requested API signatures/fences; v7 adds measurement-only fixtures. Original v4/v5 records and frozen selections remain intact. Copying, JSON, evidence decisions and diagnosis semantics are unchanged. Conditional grouped uncertainty is in results.json; it is not population certainty.','']
    for i in range(1,6):lines.append(f'- [Experiment {i}](experiment-{i}.md)')
    lines+=['','Useful-repair and composition prerequisites must be read separately from completed stress measurements. Unsupported activation methods, a precision allocator, a large independently useful patch library, interactive frontier diagnosis, and paper-scale independent confirmation have not been performed. See machine-readable results for every source hash, actual run and decision.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['phase_generated_tokens','phase_inference_requests']},indent=2))


if __name__=='__main__':main()
