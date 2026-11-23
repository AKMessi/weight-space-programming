"""Concurrent RX 6400 Vulkan llama-server evaluator + llama-perplexity + statistical tests."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from .gguf_patch import file_sha256
from .math_grader import grade_arc_response, grade_copy_response, grade_math_response


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """Paired exact McNemar test comparing baseline A vs edited B on identical items."""
    assert len(a_correct) == len(b_correct)
    n = len(a_correct)
    b_wins = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    a_wins = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    both_ok = sum(1 for a, b in zip(a_correct, b_correct) if a and b)
    both_bad = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and (not b))
    disc = b_wins + a_wins
    pval = float(binomtest(b_wins, disc, 0.5).pvalue) if disc > 0 else 1.0
    delta = (b_wins - a_wins) / n if n > 0 else 0.0
    return {
        "n": n,
        "a_acc": (both_ok + a_wins) / n if n else 0.0,
        "b_acc": (both_ok + b_wins) / n if n else 0.0,
        "delta": delta,
        "rescued_by_b": b_wins,
        "broken_by_b": a_wins,
        "both_correct": both_ok,
        "both_wrong": both_bad,
        "mcnemar_p_two_sided": pval,
    }


def format_messages(item: dict[str, Any], benchmark: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if benchmark in ("math", "gsm8k"):
        user = (
            f"Solve the following math problem step by step and put your final answer "
            f"inside \\boxed{{}}.\n\nProblem:\n{item['problem']}"
        )
    elif benchmark == "arc":
        user = (
            f"Answer the following multiple-choice science question. "
            f"Put the single letter of the correct option inside \\boxed{{}}.\n\n"
            f"Question: {item['question']}\n\nOptions:\n{item['options_text']}"
        )
    elif benchmark == "copy":
        user = item["prompt"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    msgs.append({"role": "user", "content": user})
    return msgs


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LlamaServer:
    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "Vulkan0",
        ngl: int = 99,
        ctx_size: int = 8192,
        parallel: int = 4,
        log_path: Path | str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.ngl = ngl
        self.ctx_size = ctx_size
        self.parallel = parallel
        self.port = free_port()
        self.log_path = Path(log_path) if log_path else None
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, wait_timeout: float = 120.0) -> dict[str, Any]:
        cmd = [
            "llama-server",
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", str(self.ngl),
            "-dev", self.device,
            "-c", str(self.ctx_size),
            "-np", str(self.parallel),
            "--cache-ram", "0",
            "--reasoning-format", "none",
            "--no-webui",
        ] + self.extra_args
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("w", encoding="utf-8")
            stdout_dst = self._log_file
            stderr_dst = subprocess.STDOUT
        else:
            stdout_dst = subprocess.DEVNULL
            stderr_dst = subprocess.DEVNULL

        self.proc = subprocess.Popen(cmd, stdout=stdout_dst, stderr=stderr_dst)
        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            if self.proc.poll() is not None:
                tail = ""
                if self.log_path and self.log_path.exists():
                    tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"llama-server exited early (code={self.proc.returncode}):\n{tail}")
            try:
                st = _http_json(f"{self.base_url}/health", timeout=2.0)
                if st.get("status") == "ok":
                    return {"cmd": cmd, "port": self.port, "startup_sec": round(time.time() - t0, 2)}
            except Exception:
                time.sleep(0.35)
        self.stop()
        raise TimeoutError(f"llama-server failed to start within {wait_timeout}s")

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
            self.proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def run_benchmark(
    *,
    model_path: Path | str,
    dataset_path: Path | str,
    benchmark: str,
    run_dir: Path | str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
    enable_thinking: bool = True,
    system_prompt: str | None = None,
    parallel: int = 4,
    ctx_per_slot: int = 2560,
    device: str = "Vulkan0",
    ngl: int = 99,
    limit: int | None = None,
    sampling_extra: dict[str, Any] | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """Run a full benchmark against a GGUF checkpoint with immutable per-request logging."""
    model_p = Path(model_path).resolve()
    ds_p = Path(dataset_path).resolve()
    out_dir = Path(run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in ds_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        items = items[:limit]

    # Resume support: if responses.jsonl already has completed IDs, skip them cleanly
    resp_path = out_dir / "responses.jsonl"
    done_by_id: dict[str, dict[str, Any]] = {}
    if resp_path.exists():
        for line in resp_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "id" in rec and not rec.get("error"):
                    done_by_id[rec["id"]] = rec

    ver_proc = subprocess.run(["llama-server", "--version"], capture_output=True, text=True)
    llama_ver = (ver_proc.stdout + "\n" + ver_proc.stderr).strip().splitlines()[-2:]

    manifest = {
        "schema": "wsp_eval_v1",
        "tag": tag,
        "started_utc": utc_now(),
        "model_path": str(model_p),
        "model_sha256": file_sha256(model_p),
        "model_bytes": model_p.stat().st_size,
        "dataset_path": str(ds_p),
        "dataset_sha256": file_sha256(ds_p),
        "benchmark": benchmark,
        "num_items": len(items),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "system_prompt": system_prompt,
        "sampling_extra": sampling_extra or {},
        "parallel": parallel,
        "ctx_size": ctx_per_slot * parallel,
        "device": device,
        "ngl": ngl,
        "llama_version": llama_ver,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    pending = [it for it in items if it["id"] not in done_by_id]
    t_start = time.time()
    write_lock = threading.Lock()

    if pending:
        srv = LlamaServer(
            model_p,
            device=device,
            ngl=ngl,
            ctx_size=ctx_per_slot * parallel,
            parallel=parallel,
            log_path=out_dir / "server.log",
        )
        srv_info = srv.start()
        manifest["server_start"] = srv_info
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        def eval_one(item: dict[str, Any]) -> dict[str, Any]:
            msgs = format_messages(item, benchmark, system_prompt)
            body: dict[str, Any] = {
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            if top_k > 0:
                body["top_k"] = top_k
            if sampling_extra:
                body.update(sampling_extra)
            t0 = time.time()
            try:
                res = _http_json(f"{srv.base_url}/v1/chat/completions", body, timeout=1200.0)
                elapsed = round(time.time() - t0, 3)
                choice = res["choices"][0]
                msg = choice.get("message", {})
                raw = msg.get("content") or ""
                if msg.get("reasoning_content"):
                    raw = f"<think>\n{msg['reasoning_content']}\n</think>\n{raw}"
                finish_reason = choice.get("finish_reason", "")
                usage = res.get("usage", {})
                timings = res.get("timings", {})
                if benchmark in ("math", "gsm8k"):
                    grade = grade_math_response(raw, item["answer"], finish_reason)
                elif benchmark == "arc":
                    grade = grade_arc_response(raw, item["answer"], item["valid_labels"], finish_reason)
                elif benchmark == "copy":
                    grade = grade_copy_response(raw, item["answer"], finish_reason)
                else:
                    raise ValueError(benchmark)
                rec = {
                    "id": item["id"],
                    "utc": utc_now(),
                    "elapsed_sec": elapsed,
                    "finish_reason": finish_reason,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "predicted_per_second": timings.get("predicted_per_second"),
                    "prompt_per_second": timings.get("prompt_per_second"),
                    "raw_response": raw,
                    "grade": grade,
                    "meta": {k: item[k] for k in ("level", "subject", "task") if k in item},
                    "error": None,
                }
            except Exception as exc:
                rec = {
                    "id": item["id"],
                    "utc": utc_now(),
                    "elapsed_sec": round(time.time() - t0, 3),
                    "finish_reason": "error",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "raw_response": "",
                    "grade": {"correct": False, "hit_limit": False, "loop_detected": False},
                    "error": repr(exc),
                }
            with write_lock:
                with resp_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            return rec

        try:
            completed = len(done_by_id)
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                futs = [ex.submit(eval_one, it) for it in pending]
                for fut in concurrent.futures.as_completed(futs):
                    rec = fut.result()
                    done_by_id[rec["id"]] = rec
                    completed += 1
                    if completed % 25 == 0 or completed == len(items):
                        c_ok = sum(1 for r in done_by_id.values() if r["grade"].get("correct"))
                        c_lim = sum(1 for r in done_by_id.values() if r["grade"].get("hit_limit"))
                        c_lp = sum(1 for r in done_by_id.values() if r["grade"].get("loop_detected"))
                        print(
                            f"  [{tag or out_dir.name}] {completed}/{len(items)} | "
                            f"acc={c_ok}/{completed} ({100*c_ok/completed:.1f}%) | "
                            f"limit={c_lim} | loops={c_lp} | {time.time()-t_start:.1f}s",
                            flush=True,
                        )
        finally:
            srv.stop()

    # Deterministic ordered summary
    ordered = [done_by_id[it["id"]] for it in items if it["id"] in done_by_id]
    n = len(ordered)
    n_ok = sum(1 for r in ordered if r["grade"].get("correct"))
    n_lim = sum(1 for r in ordered if r["grade"].get("hit_limit"))
    n_loop = sum(1 for r in ordered if r["grade"].get("loop_detected"))
    n_close = sum(1 for r in ordered if r["grade"].get("has_think_close"))
    n_latent = sum(1 for r in ordered if r["grade"].get("latent_in_think"))
    n_err = sum(1 for r in ordered if r.get("error"))
    comp_toks = [r["completion_tokens"] for r in ordered]
    mean_toks = sum(comp_toks) / n if n else 0.0
    sorted_toks = sorted(comp_toks)
    med_toks = sorted_toks[n // 2] if n else 0
    total_gen_toks = sum(comp_toks)
    total_prompt_toks = sum(r["prompt_tokens"] for r in ordered)
    tps_vals = [r["predicted_per_second"] for r in ordered if r.get("predicted_per_second")]
    med_tps = sorted(tps_vals)[len(tps_vals) // 2] if tps_vals else None
    ci_lo, ci_hi = wilson_ci(n_ok, n)

    # Breakdown by level or task if present
    by_group: dict[str, dict[str, Any]] = {}
    for r in ordered:
        meta = r.get("meta", {})
        gkey = None
        if "level" in meta:
            gkey = f"level_{meta['level']}"
        elif "task" in meta:
            gkey = str(meta["task"])
        if gkey is not None:
            g = by_group.setdefault(gkey, {"n": 0, "correct": 0, "hit_limit": 0, "loop": 0, "tokens": 0})
            g["n"] += 1
            g["correct"] += int(bool(r["grade"].get("correct")))
            g["hit_limit"] += int(bool(r["grade"].get("hit_limit")))
            g["loop"] += int(bool(r["grade"].get("loop_detected")))
            g["tokens"] += r["completion_tokens"]
    for g in by_group.values():
        g["acc"] = g["correct"] / g["n"] if g["n"] else 0.0
        g["mean_tokens"] = g["tokens"] / g["n"] if g["n"] else 0.0

    summary = {
        "tag": tag,
        "ended_utc": utc_now(),
        "wall_sec": round(time.time() - t_start, 2),
        "model_path": str(model_p),
        "model_sha256": manifest["model_sha256"],
        "benchmark": benchmark,
        "n": n,
        "correct": n_ok,
        "accuracy": n_ok / n if n else 0.0,
        "wilson_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "hit_limit_count": n_lim,
        "hit_limit_rate": n_lim / n if n else 0.0,
        "loop_count": n_loop,
        "loop_rate": n_loop / n if n else 0.0,
        "think_close_rate": n_close / n if n else 0.0,
        "latent_in_think_count": n_latent,
        "error_count": n_err,
        "total_prompt_tokens": total_prompt_toks,
        "total_generated_tokens": total_gen_toks,
        "mean_completion_tokens": round(mean_toks, 1),
        "median_completion_tokens": med_toks,
        "median_slot_tps": round(med_tps, 2) if med_tps else None,
        "by_group": by_group,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def measure_perplexity(
    model_path: Path | str,
    text_path: Path | str = "datasets/wsp/test_wikitext2_raw.txt",
    *,
    chunks: int = 32,
    ctx_size: int = 512,
    device: str = "Vulkan0",
    ngl: int = 99,
) -> dict[str, Any]:
    """Run llama-perplexity on RX 6400 Vulkan for fast, exact WikiText-2 perplexity."""
    model_p = Path(model_path).resolve()
    text_p = Path(text_path).resolve()
    cmd = [
        "llama-perplexity",
        "-m", str(model_p),
        "-f", str(text_p),
        "-c", str(ctx_size),
        "--chunks", str(chunks),
        "-ngl", str(ngl),
        "-dev", device,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"Failed to parse llama-perplexity output (rc={proc.returncode}):\n{out[-1500:]}")
    ppl, se = float(m[-1][0]), float(m[-1][1])
    return {
        "model_path": str(model_p),
        "model_sha256": file_sha256(model_p),
        "text_path": str(text_p),
        "chunks": chunks,
        "ctx_size": ctx_size,
        "ppl": ppl,
        "ppl_se": se,
        "elapsed_sec": round(time.time() - t0, 2),
    }
