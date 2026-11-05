"""Versioned storage, timing, symmetry and native-readout supplement with figures."""
import argparse,json,statistics
from pathlib import Path
from weightlab.evidence import file_sha256,git_state,utc_now,write_json
from weightlab.grading import grade_turns


def read(path):return json.loads(Path(path).read_text())


def rows(root):return list(map(json.loads,(Path(root)/'responses.jsonl').read_text().splitlines()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='research/portfolio-v2/supplement');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sources={};timing={}
    config=read('configs/fable_speed_followup.json')
    for key in ['fable_float','fable_Q8_0','fable_Q4_0','fable_Q4_K_M']:
        root=Path(f'runs/p2-speed-followup-{key}-s17');records=rows(root)
        assert len(records)==4 and all(r['usage']['completion_tokens']==64 for r in records)
        rates=[r['timings']['predicted_per_second'] for r in records[1:]]
        source=config['models'][key];resource=read(root/'resources.json')
        timing[key]={'bytes':Path(source['path']).stat().st_size,'sha256':source['sha256'],
             'first_warmup_decode_tps':records[0]['timings']['predicted_per_second'],'post_warmup_decode_tps':rates,
             'median_decode_tps':statistics.median(rates),'min_decode_tps':min(rates),'max_decode_tps':max(rates),
             'resources':resource}
        sources[str(root/'responses.jsonl')]=file_sha256(root/'responses.jsonl')
    symmetry=[]
    for model in ['qwen','lfm3','fable']:
        for state in ['healthy','output_sign','ffn_sign']:
            item={'model':model,'state':state}
            pair=[]
            for layout in ['original','permuted']:
                suffix='_permuted' if layout=='permuted' else ''
                root=Path(f'runs/p2-e5probes-{model}_diagnosis_{state}{suffix}-s17')
                records=rows(root);pair.append(records)
                item[layout]={'n':len(records),'payload':sum(grade_turns(r['case'],r['steps'])['payload_correct'] for r in records),
                              'strict':sum(grade_turns(r['case'],r['steps'])['success'] for r in records)}
                sources[str(root/'responses.jsonl')]=file_sha256(root/'responses.jsonl')
            item['exact_content_pairs']=sum(x['content']==y['content'] for x,y in zip(*pair))
            item['coarse_score_equal']=item['original']==item['permuted']
            symmetry.append(item)
    native=read('research/native-state-compatibility.json');storage=read('research/exact-patch-storage.json')
    rounding=read('research/exact-patch-rounding-control.json')
    for source in ['research/native-state-compatibility.json','research/exact-patch-storage.json','research/exact-patch-rounding-control.json',
                   'research/diagnosis-screening.json','research/symmetry-numerical.json']:
        sources[source]=file_sha256(source)
    result={'utc':utc_now(),'git':git_state(),'timing':timing,'symmetry':symmetry,'native_state':native,
            'storage':storage,'within_source_rounding':rounding,'sources':sources}
    write_json(out/'results.json',result)
    report=['# Deployment and integrity follow-ups','',
      'These are disclosed follow-ups to the bounded portfolio; quality decisions and primary diagnosis exclusions stay frozen.','',
      '## Fable: the larger file is faster on this device','',
      '| Format | File MB (decimal) | Median decode tokens/s | Range across three post-warmup calls |',
      '|---|---:|---:|---:|']
    for key,r in timing.items():report.append(f'| {key} | {r["bytes"]/1e6:.1f} | {r["median_decode_tps"]:.2f} | {r["min_decode_tps"]:.2f}–{r["max_decode_tps"]:.2f} |')
    report += ['',
      'Each format received the same 28-token rendered story prompt, seed 17, context 2048, four calls and 64 actual generated tokens per call; the first call was warmup. Three calls in one session are descriptive repeats, not independent machines or robust uncertainty bounds. Ignore-EOS was requested, so this is a decode microbenchmark with no quality score. Generation content differs between formats. F16 is about 14.5× faster than Q8 here while requiring a larger file and more VRAM. The locally exported Q8 and Q4_K_M hashes match the pinned published Fable artifacts.',
      '',
      'The separate Q8 verbosity-5 startup trace reports 15/15 layers offloaded. This does not identify which execution path causes the difference. No CPU-fallback or quantization-causes-slowdown mechanism is established. The matched repeated timing result is specific to this runtime/device and short-context workload; cache policy is disabled as recorded in each request.',
      '', '## Exact checkpoint delivery','',
      'A named-tensor XOR/deflate patch restores the complete original Fable GGUF from its exact matching aligned base in 20,308,656 bytes. The same-source BF16-rounded F16 parent needs 44,653,069 bytes; the original unrounded F16 parent needs 167,671,933 bytes. All reconstruct the same complete 461,883,712-byte target hash. The same-source rounding comparison reduces patch cost by 3.75×; changing to the separately published aligned BF16 base reduces it further. Source provenance and double-rounding effects prevent assigning the full 8.26× source-choice difference to rounding alone.',
      '',
      'These sizes exclude the matching base. This is an exact delivery tool and one measured fine-tune case, using established delta encoding; no new compression algorithm, standalone 20 MB model or validated behavioral repair is claimed. See [utility documentation](../../EXACT_PATCH_DELIVERY.md) and the raw roundtrip manifests.',
      '', '## Permutation audit','',
      '| Model | State | Original payload / 12 | Permuted payload / 12 | Exact text pairs / 12 |',
      '|---|---|---:|---:|---:|']
    for r in symmetry:report.append(f'| {r["model"]} | {r["state"]} | {r["original"]["payload"]} | {r["permuted"]["payload"]} | {r["exact_content_pairs"]} |')
    report += ['',
      'All nine gated-MLP numerical checks passed at the declared tolerance; three of nine full-runtime layouts passed strict text parity. Equal coarse task scores cannot establish functional equivalence. The six excluded layouts remain excluded from primary blinded-editor accuracy. This exposes a practical difference between mathematical symmetry and identical native generation; it does not prove a new capability loss or explain the numerical cause.',
      '', '## Native final-state gate','',
      '| Model | Context | Same top-1? | Max centered-logit error | Relative error | Passed? |',
      '|---|---|---|---:|---:|---|']
    for r in native['results']:report.append(f'| {r["model"]} | {r["case_id"]} | {r["top1_equal"]} | {r["max_centered_logit_error"]:.4f} | {r["relative_centered_error"]:.2%} | {r["passed"]} |')
    report += ['',
      'Six contexts retained top-1 but none met both predeclared error tolerances. No edit was derived from these unverified readouts. The two-context gate is a compatibility test at this boundary/backend, not proof that hidden-state access is impossible. A separately validated CPU/native or instrumented path is future method work, not a completed comparison in this portfolio.',
      '',
      'All code scores shown here use the disclosed derived v7 sensitivity policy; every raw grade remains unchanged. Data and source checksums are in [results.json](results.json).']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    figures(out,result)


def figures(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    labels=['F16','Q8_0','Q4_0','Q4_K_M'];values=list(data['timing'].values());colors=['#246B9A','#528A75','#DA9441','#8A69A3']
    bars=axes[0].bar(labels,[v['median_decode_tps'] for v in values],color=colors)
    axes[0].bar_label(bars,fmt='%.1f',padding=4);axes[0].set_ylim(0,225)
    axes[0].set_ylabel('Decode tokens / second');axes[0].set_title('Fable on RX 6400: matched timing follow-up')
    axes[0].text(.02,.96,'3 repeats after warmup; 64 tokens/call',transform=axes[0].transAxes,va='top',fontsize=9)
    sizes=[167671933,44653069,20308656]
    bars=axes[1].bar(['F16 parent','Same parent\nBF16-rounded','Aligned BF16\nsource values'],np.array(sizes)/1e6,color=['#777777','#DA9441','#246B9A'])
    axes[1].bar_label(bars,fmt='%.1f MB',padding=4);axes[1].set_ylim(0,200)
    axes[1].set_ylabel('Exact patch MB (base excluded)');axes[1].set_title('Same Fable target; byte-identical restoration')
    fig.suptitle('Practical deployment findings — one device and one fine-tune',fontsize=13)
    for extension in ['png','svg']:fig.savefig(out/f'deployment.{extension}',dpi=170)
    plt.close(fig)
    primary=read(out.parent/'results.json');metrics=primary['experiment_5']
    correct=[metrics['weights']['diagnosis_correct'],metrics['behavior']['diagnosis_correct'],
             metrics['baselines']['weight_anomaly']['correct'],metrics['baselines']['behavior_probe']['correct'],metrics['majority_class_baseline']['correct']]
    fig,ax=plt.subplots(figsize=(9,4.6))
    fig.subplots_adjust(left=.30,bottom=.21,top=.90,right=.94)
    bars=ax.barh(['0.6B editor: weight summaries','0.6B editor: two outputs','Output-norm sign detector','Exact two-output checker','Majority label'],correct,color=['#246B9A','#528A75','#8A69A3','#DA9441','#999999'])
    ax.invert_yaxis();ax.set_xlim(0,13.5);ax.set_xticks(range(0,13,2));ax.set_xlabel('Correct diagnoses / 12 retained presentations')
    ax.bar_label(bars,labels=[f'{x}/12' for x in correct],padding=4)
    ax.set_title('Blinded diagnosis pilot: limited evidence, injected faults')
    fig.text(.5,.035,'9 correlated model-state combinations; injected faults.\nTask-specific baselines; no frontier-editor claim.',
             ha='center',va='bottom',fontsize=9)
    for extension in ['png','svg']:fig.savefig(out/f'diagnosis.{extension}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
