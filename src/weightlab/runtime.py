"""Lifecycle control for one local llama.cpp server; no inherited remote credentials."""

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


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
           "--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}']
    url = f"http://127.0.0.1:{port}"
    with open(Path(run_dir) / "server.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
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
            yield url, cmd, proc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
