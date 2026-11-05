"""Run discovery panels with immutable inputs and fsynced raw evidence."""

import collections
import fcntl
import json
from pathlib import Path
import platform
import subprocess
import time

from .evidence import append_jsonl, digest, file_sha256, git_state, utc_now, write_json
from .budget import reserve, settle
from .grading import GRADER_VERSION, grade_turns
from .runtime import request_json, runtime_libraries, server,validate_response_format


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["case"]["group"]].append(row)
    result = {}
    for group, data in sorted(groups.items()):
        ok = [r for r in data if "grading" in r]
        result[group] = {"attempted": len(data), "responses": len(ok),
                         "success": sum(r["grading"]["success"] for r in ok),
                         "payload_correct": sum(r["grading"]["payload_correct"] for r in ok),
                         "cutoff": sum(r.get("finish_reason") == "length" for r in ok)}
    return {"groups": result, "generated_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows),
            "inference_requests":sum(len(r.get('steps',[])) or 1 for r in rows),
            "response_count": sum("grading" in r for r in rows), "attempt_count": len(rows)}


def execute(config, model_key, suite, output, max_tokens=128, seed=17, limit=None):
    Path("local").mkdir(exist_ok=True)
    with open("local/execution.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(config, model_key, suite, output, max_tokens, seed, limit)


def _execute(config, model_key, suite, output, max_tokens, seed, limit):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    spec = config["models"][model_key]
    model = Path(spec["path"]).expanduser()
    rows = []
    state = git_state()
    # A run may coexist with uncommitted evidence, but never uncommitted source/config changes.
    dirty_source = [l for l in state["status"].splitlines()
                    if l[3:].startswith(("src/", "scripts/", "configs/", "tests/", "pyproject.toml"))]
    if dirty_source:
        raise RuntimeError(f"Commit experimental source/config before running: {dirty_source}")
    actual = file_sha256(model)
    if actual != spec["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {model_key}")
    binary = config["runtime"]["binary"]
    manifest = {"schema_version": 2, "started_utc": utc_now(), "purpose": suite.get('status','exploratory discovery'),
                "git": state, "python": platform.python_version(), "model_key": model_key,
                "model": spec, "verified_model_sha256": actual, "suite_sha256": digest(suite),
                "max_tokens": max_tokens, "seed": seed, "limit": limit, "grader_version": GRADER_VERSION,
                "runtime": config["runtime"], "config_sha256":digest(config), "binary_sha256": file_sha256(binary),
                "colocated_runtime_libraries": runtime_libraries(binary),
                "runtime_version": subprocess.check_output([binary, "--version"], stderr=subprocess.STDOUT, text=True)}
    write_json(out / "manifest.json", manifest)
    write_json(out / "suite.json", suite)
    max_run_tokens = config.get("budgets", {}).get("per_run_generated_tokens", 20000)
    cases = suite["cases"][:limit] if limit else suite["cases"]
    try:
        with server(binary, model, out, config["runtime"]) as (url, cmd, proc):
            write_json(out / "runtime.json", {"command": cmd, "props": request_json(url + "/props")})
            for case in cases:
                consumed = sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows)
                reservation=max_tokens*(1+len(case.get('followups',[])))
                if consumed + reservation > max_run_tokens:
                    raise RuntimeError("Per-run budget would be exceeded")
                sampling = spec["sampling"]
                payload = {"model": "local", "messages": case["messages"], "seed": seed,
                           "max_tokens": max_tokens, "stream": False, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}, **sampling}
                overrides = case.get("request_overrides", {})
                allowed = {"reasoning_format", "reasoning_effort", "chat_template_kwargs", "response_format",
                           "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"}
                if set(overrides) - allowed:
                    raise ValueError(f"Unsupported request overrides: {set(overrides) - allowed}")
                payload.update(overrides)
                validate_response_format(payload.get('response_format'))
                row = {"case": case, "request": payload, "started_utc": utc_now()}
                try:
                    row["rendered_prompt"] = request_json(url + "/apply-template", payload)
                except Exception as exc:
                    row["rendered_prompt_error"] = repr(exc)
                request_id = f"{out}/{case['id']}/seed-{seed}"
                reserve(request_id, reservation, config.get("budgets", {}).get("overall_generated_tokens", 1800000))
                append_jsonl(out / "events.jsonl", {"event": "request", **row})
                start = time.monotonic()
                try:
                    steps=[]
                    messages=list(payload['messages'])
                    for turn,specification in enumerate([case]+case.get('followups',[])):
                        if turn:
                            messages += [{'role':'assistant','content':steps[-1]['content']}]+specification['messages']
                        turn_payload={**payload,'messages':list(messages)}
                        prompt=request_json(url+'/apply-template',turn_payload)
                        append_jsonl(out/'events.jsonl',{'event':'turn_request','case_id':case['id'],'turn':turn+1,
                                                        'request':turn_payload,'rendered_prompt':prompt,'utc':utc_now()})
                        response = request_json(url + "/v1/chat/completions", turn_payload)
                        choice=response['choices'][0]
                        step={'request':turn_payload,'rendered_prompt':prompt,'raw_response':response,
                              'content':choice['message'].get('content') or '', 'finish_reason':choice.get('finish_reason'),
                              'usage':response.get('usage',{}),'timings':response.get('timings',{})}
                        steps.append(step)
                        append_jsonl(out/'events.jsonl',{'event':'turn_response','case_id':case['id'],'turn':turn+1,**step})
                    row['steps']=steps
                    row["raw_response"] = steps[-1]['raw_response']
                    row["elapsed_seconds"] = time.monotonic() - start
                    usage={key:sum(step['usage'].get(key,0) for step in steps)
                           for key in ('prompt_tokens','completion_tokens','total_tokens')}
                    row.update(content=steps[-1]['content'],
                               finish_reason='length' if any(s['finish_reason']=='length' for s in steps) else steps[-1]['finish_reason'],
                               usage=usage,timings=steps[-1]['timings'], grading=grade_turns(case,steps))
                except Exception as exc:
                    row.update(error=repr(exc), elapsed_seconds=time.monotonic() - start)
                    rows.append(row)
                    append_jsonl(out / "responses.jsonl", row)
                    raise
                rows.append(row)
                append_jsonl(out / "responses.jsonl", row)
                settle(request_id, row["usage"].get("completion_tokens", max_tokens))
                append_jsonl(out / "events.jsonl", {"event": "response_saved", "case_id": case["id"], "utc": utc_now()})
                write_json(out / "summary.json", summarize(rows))
                print(f"{model_key} {case['id']}: success={row['grading']['success']} tokens={row['usage'].get('completion_tokens')} time={row['elapsed_seconds']:.2f}s", flush=True)
        status = "complete"
    except Exception as exc:
        status = "failed"
        append_jsonl(out / "events.jsonl", {"event": "run_failed", "utc": utc_now(), "error": repr(exc)})
        raise
    finally:
        summary = summarize(rows)
        summary.update(status=locals().get("status", "interrupted"), ended_utc=utc_now())
        write_json(out / "summary.json", summary)
    return summary
