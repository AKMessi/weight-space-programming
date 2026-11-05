"""Exact contract graders with separately labeled semantic/proxy scores."""

import ast
import json
import re


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-JSON constant: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
GRADER_VERSION = "5"


def code_payload_present(text, expected):
    """Static return-expression match; never execute generated code."""
    try:
        wanted = ast.parse(expected.strip()).body[0]
    except SyntaxError:
        return False
    snippets = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL) or [text]
    # Presence is a limited proxy, not a final-answer/function-correctness judgment.
    snippets += re.findall(r"`([^`\n]+)`", text)
    snippets += text.replace('<think>', '\n').replace('</think>', '\n').splitlines()
    for snippet in snippets:
        try:
            tree = ast.parse(snippet.strip())
            if any(isinstance(n, ast.Return) and ast.dump(n) == ast.dump(wanted) for n in ast.walk(tree)):
                return True
        except SyntaxError:
            pass
    return expected.strip() in [line.strip() for line in text.splitlines()]


def one_return_statement_matches(text, expected):
    """Ignore Python quote/spacing choices, but reject prose/functions/fences."""
    try:
        actual = ast.parse(text.strip())
        wanted = ast.parse(expected.strip())
        return (len(actual.body) == 1 and isinstance(actual.body[0], ast.Return)
                and ast.dump(actual) == ast.dump(wanted))
    except SyntaxError:
        return False


def typed_equal(left, right):
    # Python normally equates True with 1; that would corrupt exact-value grading.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def first_json(text):
    """Only the first decodable object/array, not the most favorable nested answer."""
    match = re.search(r"[\[{]", text)
    if match:
        try:
            return DECODER.raw_decode(text, match.start())[0]
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def grade(case, text):
    kind, expected = case["grader"], case["expected"]
    if kind == 'functional_code':
        from .functional import evaluate_code
        return evaluate_code(case,text)
    if kind=='diagnosis':
        parsed=first_json(text)
        confidence=parsed.get('confidence') if isinstance(parsed,dict) else None
        valid=(isinstance(parsed,dict) and set(parsed)=={'diagnosis','confidence'}
               and type(confidence) in (int,float) and 0<=confidence<=1
               and parsed['diagnosis'] in ('healthy','degraded'))
        payload=bool(valid and parsed['diagnosis']==expected)
        try:
            whole,end=DECODER.raw_decode(text.lstrip())
            plain=not text.lstrip()[end:].strip() and typed_equal(whole,parsed)
        except (ValueError,json.JSONDecodeError):plain=False
        return {'success':payload and plain,'payload_correct':payload,'format_valid':bool(valid and plain),
                'diagnosis':parsed.get('diagnosis') if isinstance(parsed,dict) else None,
                'stated_confidence':confidence if valid else None}
    if kind == "json":
        valid = False
        try:
            parsed, end = DECODER.raw_decode(text.lstrip())
            valid = not text.lstrip()[end:].strip()
        except (ValueError, json.JSONDecodeError):
            parsed = None
        return {"success": valid and typed_equal(parsed, expected),
                "format_valid": valid, "payload_correct": typed_equal(first_json(text), expected)}
    if kind in {"copy", "code_line"}:
        strict = text.rstrip("\r\n") == expected
        if kind == "code_line":
            strict = one_return_statement_matches(text, expected)
        payload = expected in text if kind == "copy" else code_payload_present(text, expected)
        return {"success": strict, "payload_correct": payload}
    if kind == "explanation_proxy":
        payload = expected.strip() in text
        rest = text.replace(expected.strip(), "")
        enough_words = len(re.findall(r"[A-Za-z]+", rest)) >= 8
        return {"success": payload and enough_words, "payload_correct": payload,
                "note": "contract proxy only; not a reasoning-quality judgment"}
    if kind == "evidence_action":
        found = re.search(r"(?m)^\s*(ANSWER|ASK|CONFLICT):\s*([^\n]+)", text)
        payload = found is not None and f"{found[1]}: {found[2].strip()}" == expected
        return {"success": text.strip() == expected, "payload_correct": payload}
    raise ValueError(f"Unknown grader: {kind}")


def grade_turns(case, steps):
    specifications=[case]+case.get('followups',[])
    if len(steps)!=len(specifications):raise ValueError('Incomplete conversation')
    results=[grade(spec,step['content']) for spec,step in zip(specifications,steps)]
    if len(results)==1:return results[0]
    return {'success':all(r['success'] for r in results),
            'payload_correct':all(r['payload_correct'] for r in results),
            'turn_grades':results,'note':'joint success requires every actual conversation turn'}
