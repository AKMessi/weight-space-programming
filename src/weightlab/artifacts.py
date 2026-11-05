"""Pinned artifact retrieval. Originals are immutable and never edited here."""

import json
from pathlib import Path
import time
import urllib.request

from .evidence import append_jsonl, file_sha256, utc_now


def fetch_artifacts(config, names=None):
    records = []
    for name, spec in config["models"].items():
        if names and name not in names:
            continue
        path = Path(spec["path"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"model_key": name, "utc": utc_now(), "source": spec, "path": str(path)}
        if not path.exists():
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}?download=true"
            partial = path.with_name(path.name + ".partial")
            print(f"Downloading {name}: {spec['file']}", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                actual = file_sha256(partial)
                if actual != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch for {name}: {actual}")
                partial.replace(path)
            except Exception as exc:
                row.update(status="failed", error=repr(exc))
                append_jsonl("research/artifact_events.jsonl", row)
                raise
        row.update(sha256=file_sha256(path), bytes=path.stat().st_size)
        if row["sha256"] != spec["sha256"]:
            row["status"] = "checksum_mismatch"
            append_jsonl("research/artifact_events.jsonl", row)
            raise ValueError(f"Existing artifact {name} does not match pinned source")
        row["status"] = "verified"
        append_jsonl("research/artifact_events.jsonl", row)
        records.append(row)
        print(f"Verified {name}: {row['bytes']:,} bytes {row['sha256']}", flush=True)
    return records
