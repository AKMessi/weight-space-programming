"""Versioned synthetic DISCOVERY fixtures. No confirmatory data generated here."""

import json
import random


def discovery_suite(seed=20261105, repetitions=4):
    rng = random.Random(seed)
    cases = []
    def add(family, grader, expected, prompt, group, i):
        cases.append({"id": f"discovery-v1-{family}-{i:02d}", "split": "discovery",
                      "family": family, "group": group, "grader": grader,
                      "expected": expected, "messages": [
                          {"role": "system", "content": "Follow the user's instructions. Use only supplied facts for extraction and tool arguments."},
                          {"role": "user", "content": prompt}]})
    for i in range(repetitions):
        name = rng.choice(["Mira", "Nolan", "Tariq", "Sofia"])
        units = rng.randint(3, 31)
        code = f"ZX-{rng.randint(100, 999)}"
        data = {"name": name, "units": units}
        add("json_extract", "json", data,
            f"Record: name={name}; units={units}. Return only one JSON object with keys name (string) and units (integer). No explanation or code fences.", "format", i)
        add("json_reordered", "json", {"code": code, "available": True},
            f"Item {code} is available. Output only JSON with keys code (string), available (boolean).", "format", i)
        n = rng.randint(2, 9)
        line = f"    return value + {n}"
        context = f"def add_{n}(value):\n    \"\"\"Return value plus {n}.\"\"\"\n"
        add("code_only", "code_line", line,
            f"Complete the following function. Return only the next line of Python, without backticks or explanation.\n{context}", "coding", i)
        add("code_explain", "explanation_proxy", line,
            f"Give the next line of Python and explain why it implements this function.\n{context}", "explanation", i)
        literal = f"{code}|{name}|{units}|{code}"
        add("exact_copy", "copy", literal,
            f"Copy the text between <data> and </data> exactly once. Return only the text, preserving case, symbols and repeated parts.\n<data>{literal}</data>", "copy", i)
        repeated = " ".join([code] * 3)
        add("requested_repeat", "copy", repeated,
            f"Return {code} exactly three times separated by single spaces. No other text.", "copy", i)
        target = {"name": "reserve", "arguments": {"item": code, "quantity": units}}
        add("tool_complete", "json", target,
            f"Mock tool reserve requires item (string) and quantity (integer). User wants {units} units of {code}. Return only a JSON object with name='reserve' and arguments containing those two values. Do not execute anything.", "tool", i)
        evidence_instruction = "Use only these facts. Reply with exactly ANSWER: value if the requested fact exists; ASK: field if it is absent; CONFLICT: field if two stated values disagree."
        add("evidence_complete", "evidence_action", f"ANSWER: {code}",
            f"{evidence_instruction}\nFacts: recipient={name}; address={code}; quantity={units}.\nRequested field: address.", "evidence_complete", i)
        add("evidence_missing", "evidence_action", "ASK: address",
            f"{evidence_instruction}\nFacts: recipient={name}; reference={code}; quantity={units}.\nRequested field: address.", "evidence_missing", i)
        add("evidence_conflict", "evidence_action", "CONFLICT: address",
            f"{evidence_instruction}\nFacts: recipient={name}; address={code}; address=YY-000. No precedence rule exists.\nRequested field: address.", "evidence_conflict", i)
    return {"name": "discovery-v1", "seed": seed, "repetitions": repetitions,
            "status": "exploratory; templates visible to investigator; not a held-out benchmark",
            "cases": cases}
