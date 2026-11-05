"""Lifecycle control for one local llama.cpp server; no inherited remote credentials."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .evidence import file_sha256
from .evidence import write_json
from .resources import monitor


def runtime_libraries(binary):
    """Fingerprint colocated libraries, including dlopen-loaded server backends."""
    paths = {p.resolve() for p in Path(binary).resolve().parent.glob('*.so*') if p.is_file()}
    return [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': file_sha256(p)} for p in sorted(paths)]


def validate_response_format(form):
    if form is None:return
    kind=form.get('type')
    if kind=='json_schema':
        wrapper=form.get('json_schema',{})
        if 'schema' in form or not isinstance(wrapper.get('schema'),dict) or not wrapper['schema']:
            raise ValueError('json_schema requires a nonempty response_format.json_schema.schema; top-level schema is ignored by this runtime')
    elif kind=='json_object':
        if 'schema' in form and not isinstance(form['schema'],dict):raise ValueError('schema must be an object')
    elif kind!='text':raise ValueError('unsupported response format')


def request_json(url, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


@contextmanager
def server(binary, model, run_dir, config):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [binary, "--model", str(Path(model).expanduser()), "--host", "127.0.0.1",
           "--port", str(port), "--ctx-size", str(config.get("context", 2048)),
           "--parallel", "1", "--threads", str(config.get("threads", 4)),
           "--n-gpu-layers", str(config.get("gpu_layers", 99)), "--jinja",
           "--reasoning", config.get("reasoning", "off"), "--chat-template-kwargs",
           json.dumps(config.get("chat_template_kwargs", {"enable_thinking": False}))]
    if config.get('embedding'):
        cmd += ['--embedding','--pooling','last','--embd-normalize','-1']
    if 'log_verbosity' in config:cmd += ['-lv',str(int(config['log_verbosity']))]
    if config.get('cpu_only'):cmd += ['--device','none','--no-op-offload','--no-kv-offload']
    if config.get('no_warmup'):cmd += ['--no-warmup']
    explicit_env=config.get('environment',{})
    if set(explicit_env)-{'GGML_VK_PERF_LOGGER','GGML_VK_PERF_LOGGER_FREQUENCY'}:
        raise ValueError('Unapproved runtime environment override')
    url = f"http://127.0.0.1:{port}"
    with open(Path(run_dir) / "server.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,env={**os.environ,**explicit_env})
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}; inspect server.log")
                try:
                    if request_json(url + "/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, RuntimeError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server did not become healthy within 120 seconds")
                time.sleep(0.3)
            with monitor(proc.pid) as metrics:
                try:yield url, cmd, proc
                finally:write_json(Path(run_dir)/'resources.json',metrics)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
