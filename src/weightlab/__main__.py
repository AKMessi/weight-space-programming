import argparse
import json

from .artifacts import fetch_artifacts
from .evidence import write_json
from .runner import execute
from .suites import discovery_suite


def main():
    p = argparse.ArgumentParser(description="Auditable local weight-space discovery")
    p.add_argument("--config", default="configs/local_models.json")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("models", nargs="*")
    s = sub.add_parser("suite")
    s.add_argument("output")
    s.add_argument("--repetitions", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("suite")
    r.add_argument("output")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.command == "suite":
        write_json(a.output, discovery_suite(repetitions=a.repetitions))
        return
    with open(a.config) as f:
        config = json.load(f)
    if a.command == "fetch":
        fetch_artifacts(config, a.models)
    elif a.command == "run":
        with open(a.suite) as f:
            suite = json.load(f)
        print(json.dumps(execute(config, a.model, suite, a.output, a.max_tokens, a.seed, a.limit), indent=2))


if __name__ == "__main__":
    main()
