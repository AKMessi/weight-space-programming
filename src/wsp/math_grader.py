"""Exact deterministic graders for MATH-500, GSM8K, ARC-Challenge, exact-copy, and loop diagnostics."""
from __future__ import annotations

import math
import re
from typing import Any

import sympy as sp


def extract_all_boxed(text: str) -> list[str]:
    results: list[str] = []
    pos = 0
    while True:
        idx = text.find("\\boxed{", pos)
        if idx == -1:
            break
        i = idx + len("\\boxed{")
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        if depth == 0:
            results.append("".join(buf).strip())
        pos = i + 1
    return results


def split_think_and_answer(raw: str) -> tuple[str, str, bool]:
    """Return (think_text, final_text, has_think_close)."""
    if "</think>" in raw:
        parts = raw.split("</think>")
        think = parts[0]
        if "<think>" in think:
            think = think.split("<think>", 1)[1]
        final = parts[-1].strip()
        return think.strip(), final, True
    if "<think>" in raw:
        return raw.split("<think>", 1)[1].strip(), "", False
    return "", raw.strip(), False


def _strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\ ", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\%", "%").replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.strip().rstrip(".")
    # Remove trailing units if numeric prefix like "42 meters" -> keep exact first, handle in sym
    return s


def _latex_to_sympy_str(s: str) -> str:
    s = _strip_latex_wrappers(s)
    # Remove thousands commas inside numbers: 1,234 -> 1234
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    # Handle \frac{a}{b} iteratively (handles nested fracs)
    for _ in range(5):
        m = re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
        if not m:
            break
        s = s[: m.start()] + f"(({m.group(1)})/({m.group(2)}))" + s[m.end() :]
    # Handle \frac12 shorthand
    s = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s)
    # Handle \sqrt{a} and \sqrt[n]{a}
    s = re.sub(r"\\sqrt\[([^{}]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    for _ in range(5):
        m = re.search(r"\\sqrt\{([^{}]+)\}", s)
        if not m:
            break
        s = s[: m.start()] + f"sqrt({m.group(1)})" + s[m.end() :]
    s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    # Handle trailing % as /100
    if s.endswith("%"):
        s = f"(({s[:-1]})/100)"
    # Implicit multiplication: 2pi -> 2*pi, 2( -> 2*(, )( -> )*(, 2sqrt -> 2*sqrt, )pi -> )*pi
    s = re.sub(r"(\d)(pi|sqrt|[a-zA-Z(])", r"\1*\2", s)
    s = re.sub(r"(\))(pi|sqrt|\d|[a-zA-Z(])", r"\1*\2", s)
    return s


def _try_sympy(expr_str: str):
    try:
        s = _latex_to_sympy_str(expr_str)
        if not s or len(s) > 200:
            return None
        return sp.sympify(s, locals={"pi": sp.pi, "sqrt": sp.sqrt, "oo": sp.oo, "i": sp.I})
    except Exception:
        return None


def math_answers_equal(pred: str, gold: str) -> bool:
    if not pred:
        return False
    p_norm = _strip_latex_wrappers(pred)
    g_norm = _strip_latex_wrappers(gold)
    # Direct normalized string match (ignoring spaces & thousands commas)
    p_compact = re.sub(r"\s+", "", p_norm).lower()
    g_compact = re.sub(r"\s+", "", g_norm).lower()
    if p_compact == g_compact:
        return True
    p_nocomma = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", p_compact)
    g_nocomma = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", g_compact)
    if p_nocomma == g_nocomma:
        return True
    # Tuple / list comparison (e.g. "(3, 4)" vs "(3,4)")
    if (p_norm.startswith("(") and p_norm.endswith(")") and
            g_norm.startswith("(") and g_norm.endswith(")")):
        p_items = [x.strip() for x in p_norm[1:-1].split(",")]
        g_items = [x.strip() for x in g_norm[1:-1].split(",")]
        if len(p_items) == len(g_items) and all(math_answers_equal(a, b) for a, b in zip(p_items, g_items)):
            return True
    # Comma-separated unordered or ordered small lists if gold has commas and no parens
    if "," in g_norm and "," in p_norm and not g_norm.startswith("(") and not re.search(r"\d,\d{3}", g_norm):
        p_items = [x.strip() for x in p_norm.split(",")]
        g_items = [x.strip() for x in g_norm.split(",")]
        if len(p_items) == len(g_items):
            if all(math_answers_equal(a, b) for a, b in zip(p_items, g_items)):
                return True
    # Strip leading "x = " if gold doesn't have "="
    if "=" in p_norm and "=" not in g_norm:
        if math_answers_equal(p_norm.split("=")[-1].strip(), g_norm):
            return True
    # Sympy exact & numerical equivalence
    sp_p = _try_sympy(p_norm)
    sp_g = _try_sympy(g_norm)
    if sp_p is not None and sp_g is not None:
        try:
            diff = sp.simplify(sp_p - sp_g)
            if diff == 0:
                return True
        except Exception:
            pass
        try:
            fp = complex(sp_p.evalf())
            fg = complex(sp_g.evalf())
            if abs(fp.imag) < 1e-7 and abs(fg.imag) < 1e-7:
                if math.isclose(fp.real, fg.real, rel_tol=1e-5, abs_tol=1e-6):
                    return True
            elif abs(fp - fg) <= 1e-5 * max(1.0, abs(fg)):
                return True
        except Exception:
            pass
    return False


def extract_final_math_answer(raw: str) -> tuple[str, str]:
    """Return (extracted_answer, extraction_method) from the visible final response (or fallback)."""
    _, final, has_close = split_think_and_answer(raw)
    search_text = final if (has_close and final) else raw
    boxed = extract_all_boxed(search_text)
    if boxed:
        return boxed[-1], "boxed"
    # Check "#### <num>" format
    m = re.findall(r"####\s*([^\n]+)", search_text)
    if m:
        return m[-1].strip(), "hash"
    # Only fall back to explicit final answer phrases if think actually closed (or non-thinking)
    if has_close or "<think>" not in raw:
        # Strip markdown bold/italic markers so "**Answer:** 34" extracts cleanly as "34"
        clean_search = re.sub(r"\*\*|__", "", search_text)
        m2 = re.findall(
            r"(?:final answer is|the answer is|answer:|therefore,?\s*(?:the answer is)?)\s*[:$]?\s*([^\n\.]+)",
            clean_search,
            flags=re.IGNORECASE,
        )
        if m2:
            cand = m2[-1].strip().rstrip(".")
            mu = re.match(r"^([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)\s*(?:[a-zA-Z]+)$", cand)
            if mu:
                cand = mu.group(1)
            return cand, "phrase"
        # Last numeric token in final section if short
        if final:
            nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", final)
            if nums:
                return nums[-1], "last_num"
    return "", "none"


def detect_repetition_loop(text: str) -> dict[str, Any]:
    """Detect token/word n-gram repetition loops and hesitation markers."""
    words = text.split()
    n_words = len(words)
    loop_detected = False
    loop_onset_word = -1
    max_ngram_repeats = 0

    # Check 12-word and 24-word sliding n-grams
    for ng in (12, 24):
        if n_words < ng * 3:
            continue
        seen: dict[tuple[str, ...], list[int]] = {}
        for i in range(0, n_words - ng + 1, 2):
            key = tuple(words[i : i + ng])
            lst = seen.setdefault(key, [])
            lst.append(i)
            if len(lst) > max_ngram_repeats:
                max_ngram_repeats = len(lst)
            if len(lst) >= 3 and not loop_detected:
                loop_detected = True
                loop_onset_word = lst[1]

    # Also check high-frequency short sentence/line repetition
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 25]
    line_counts: dict[str, int] = {}
    for ln in lines:
        line_counts[ln] = line_counts.get(ln, 0) + 1
    max_line_rep = max(line_counts.values(), default=0)
    if max_line_rep >= 4:
        loop_detected = True

    hesitation_patterns = [
        r"\bwait\b", r"\bhold on\b", r"\blet me (?:double[- ]?check|re-?check|verify|re-?read|re-?calculate)\b",
        r"\bactually,\b", r"\bhmm\b", r"\balternatively\b", r"\bbut wait\b",
    ]
    hesitation_count = sum(
        len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in hesitation_patterns
    )

    return {
        "loop_detected": loop_detected,
        "loop_onset_word": loop_onset_word,
        "max_ngram_repeats": max_ngram_repeats,
        "max_line_repeats": max_line_rep,
        "hesitation_count": hesitation_count,
        "word_count": n_words,
    }


def grade_math_response(raw: str, gold: str, finish_reason: str = "") -> dict[str, Any]:
    think, final, has_close = split_think_and_answer(raw)
    pred, method = extract_final_math_answer(raw)
    correct = math_answers_equal(pred, gold)
    loop_info = detect_repetition_loop(raw)
    hit_limit = finish_reason == "length"

    # Check "correct-then-corrupt" / latent answer inside think trace
    latent_in_think = False
    if not correct and think:
        # Check boxed answers inside think or exact standalone gold number/expression
        think_boxed = extract_all_boxed(think)
        if any(math_answers_equal(b, gold) for b in think_boxed):
            latent_in_think = True
        else:
            # Check clean numeric gold in think (avoid matching single-digit noise like "1" or "2")
            g_clean = _strip_latex_wrappers(gold)
            if re.fullmatch(r"-?\d{2,}(?:\.\d+)?", g_clean):
                if re.search(rf"(?<!\d){re.escape(g_clean)}(?!\d)", think):
                    latent_in_think = True

    return {
        "correct": bool(correct),
        "pred": pred,
        "gold": gold,
        "extraction_method": method,
        "has_think_close": has_close,
        "hit_limit": hit_limit,
        "latent_in_think": latent_in_think,
        "think_chars": len(think),
        "final_chars": len(final),
        **loop_info,
    }


def grade_arc_response(raw: str, gold: str, valid_labels: list[str], finish_reason: str = "") -> dict[str, Any]:
    _, final, has_close = split_think_and_answer(raw)
    text = final if (has_close and final) else raw
    pred = ""
    boxed = extract_all_boxed(text)
    if boxed:
        cand = _strip_latex_wrappers(boxed[-1]).upper()
        m = re.search(r"\b([A-E])\b", cand)
        if m and m.group(1) in valid_labels:
            pred = m.group(1)
    if not pred:
        m = re.findall(r"(?:answer(?:\s+is)?|option|choice)\s*[:\*\(\[]*\s*([A-E])\b", text, flags=re.IGNORECASE)
        for c in reversed(m):
            if c.upper() in valid_labels:
                pred = c.upper()
                break
    if not pred and final:
        m = re.match(r"^\s*([A-E])(?:\b|\.|\))", final.strip(), flags=re.IGNORECASE)
        if m and m.group(1).upper() in valid_labels:
            pred = m.group(1).upper()
    loop_info = detect_repetition_loop(raw)
    return {
        "correct": pred == gold.upper(),
        "pred": pred,
        "gold": gold.upper(),
        "has_think_close": has_close,
        "hit_limit": finish_reason == "length",
        **loop_info,
    }


def grade_copy_response(raw: str, gold: str, finish_reason: str = "") -> dict[str, Any]:
    _, final, has_close = split_think_and_answer(raw)
    text = final if (has_close and final) else raw
    clean = text.strip().strip("`").strip('"').strip("'").strip()
    exact = clean == gold
    contains = gold in text
    return {
        "correct": exact,
        "contains_gold": contains,
        "pred": clean[:120],
        "gold": gold,
        "has_think_close": has_close,
        "hit_limit": finish_reason == "length",
    }
