"""Conservative append-only token reservations, including failed/pending requests."""

import json
from pathlib import Path

from .evidence import append_jsonl, utc_now


def consumed(path="research/budget.jsonl"):
    charges = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            charges[row["request_id"]] = row["tokens"]
    return sum(charges.values())


def reserve(request_id, tokens, ceiling, path="research/budget.jsonl"):
    if consumed(path) + tokens > ceiling:
        raise RuntimeError("Global generated-token budget would be exceeded")
    append_jsonl(path, {"event": "reserve", "utc": utc_now(), "request_id": request_id, "tokens": tokens})


def settle(request_id, tokens, path="research/budget.jsonl"):
    append_jsonl(path, {"event": "settle", "utc": utc_now(), "request_id": request_id, "tokens": tokens})
