"""Prepare leak-free, SHA-256-pinned calibration, dev, and held-out test benchmarks."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# Avoid CWD's local `datasets/` directory shadowing HuggingFace `datasets`
_cwd = str(Path.cwd().resolve())
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
from datasets import load_dataset


OUT_DIR = Path("datasets/wsp")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_gsm8k_answer(ans_text: str) -> str:
    if "####" in ans_text:
        return ans_text.split("####")[-1].strip().replace(",", "")
    return ans_text.strip()


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


def build_copy_recall_suite(seed: int, n: int, split_tag: str) -> list[dict[str, Any]]:
    """Deterministic exact-copy & key-value retrieval tasks to test dual-function entanglement."""
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    words = [
        "cobalt", "velvet", "quartz", "nebula", "cipher", "harbor", "zenith", "prism",
        "matrix", "falcon", "orchid", "tundra", "anchor", "beacon", "canopy", "delta",
        "ember", "fjord", "glacier", "horizon", "ivory", "jasper", "kernel", "lantern",
        "marble", "nexus", "opal", "pixel", "quasar", "ripple", "saffron", "timber",
        "umbra", "vortex", "willow", "xenon", "yarrow", "zephyr", "basalt", "cedar",
    ]
    rows = []
    for idx in range(n):
        kind = idx % 3
        if kind == 0:
            # Exact structured token/code string repetition (tests pure induction/copy circuit)
            toks = ["".join(rng.choices(alphabet, k=4)) for _ in range(4)]
            target = "-".join(toks)
            prompt = (
                f"Repeat the following identifier string verbatim and nothing else:\n"
                f"{target}"
            )
            rows.append({
                "id": f"copy_{split_tag}_{idx:03d}",
                "task": "exact_copy",
                "prompt": prompt,
                "answer": target,
            })
        elif kind == 1:
            # Needle-in-a-haystack key-value lookup among 16 distractors
            keys = rng.sample(words, 16)
            pairs = []
            vals = {}
            for k in keys:
                v = f"{''.join(rng.choices(alphabet, k=3))}-{rng.randint(100, 999)}"
                vals[k] = v
                pairs.append(f"  {k}: {v}")
            query_k = keys[rng.randrange(len(keys))]
            table = "\n".join(pairs)
            prompt = (
                f"Look up the exact value for key '{query_k}' in the table below.\n"
                f"{table}\n\n"
                f"Return ONLY the exact value for '{query_k}'."
            )
            rows.append({
                "id": f"copy_{split_tag}_{idx:03d}",
                "task": "kv_retrieval",
                "prompt": prompt,
                "answer": vals[query_k],
            })
        else:
            # Repeated-pattern sequence completion (tests induction head health!)
            seq = rng.sample(words, 6)
            # Show sequence twice, ask for element after seq[k]
            k_pos = rng.randrange(5)
            stream = " ".join(seq * 2 + seq[: k_pos + 1])
            prompt = (
                f"The following list repeats a fixed 6-word cycle:\n"
                f"{stream}\n\n"
                f"What is the exact single next word in the cycle? Reply with ONLY that single word."
            )
            rows.append({
                "id": f"copy_{split_tag}_{idx:03d}",
                "task": "cycle_induction",
                "prompt": prompt,
                "answer": seq[k_pos + 1],
            })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"datasets": {}}

    # 1. MATH-500 (100% HELD-OUT TEST SET)
    print("Loading HuggingFaceH4/MATH-500...")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m500_rows = []
    m500_problems = set()
    for i, ex in enumerate(m500):
        prob = ex["problem"].strip()
        m500_problems.add(prob)
        m500_rows.append({
            "id": f"math500_{i:03d}",
            "problem": prob,
            "answer": str(ex["answer"]).strip(),
            "subject": ex.get("subject", ""),
            "level": int(ex.get("level", 0)),
        })
    manifest["datasets"]["test_math500"] = {
        "source": "HuggingFaceH4/MATH-500[test]",
        "count": len(m500_rows),
        "sha256": write_jsonl(OUT_DIR / "test_math500.jsonl", m500_rows),
    }

    # Stratified 150-problem fast evaluation subset of MATH-500 (for sweeps across many models/quants,
    # PLUS full 500 for final confirmatory runs!)
    rng_m500 = random.Random(20261121)
    by_lvl: dict[int, list[dict[str, Any]]] = {}
    for r in m500_rows:
        by_lvl.setdefault(r["level"], []).append(r)
    fast_m500 = []
    for lvl in sorted(by_lvl):
        grp = list(by_lvl[lvl])
        rng_m500.shuffle(grp)
        k = max(1, round(len(grp) * 150 / len(m500_rows)))
        fast_m500.extend(grp[:k])
    fast_m500 = fast_m500[:150]
    manifest["datasets"]["test_math500_strat150"] = {
        "source": "HuggingFaceH4/MATH-500[test] stratified-150",
        "count": len(fast_m500),
        "sha256": write_jsonl(OUT_DIR / "test_math500_strat150.jsonl", fast_m500),
    }

    # 2. MATH train (STRICTLY DISJOINT FROM MATH-500 -> calibration & dev!)
    print("Loading DigitalLearningGmbH/MATH-lighteval train...")
    mtrain = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    rng = random.Random(42)
    pool = []
    for i, ex in enumerate(mtrain):
        prob = ex["problem"].strip()
        if prob in m500_problems:
            continue
        ans = extract_boxed(ex["solution"])
        if not ans:
            continue
        lvl_str = str(ex.get("level", "Level 3"))
        lvl = int("".join(c for c in lvl_str if c.isdigit()) or "3")
        pool.append({
            "id": f"mathtrain_{i:05d}",
            "problem": prob,
            "answer": ans,
            "subject": ex.get("type", ""),
            "level": lvl,
        })
    # Stratify by level 1..5 for calibration (50) and dev (60)
    by_l: dict[int, list[dict[str, Any]]] = {}
    for r in pool:
        by_l.setdefault(r["level"], []).append(r)
    cal_math, dev_math = [], []
    for lvl in sorted(by_l):
        grp = by_l[lvl]
        rng.shuffle(grp)
        cal_math.extend(grp[:10])      # 50 total (10 per level 1..5)
        dev_math.extend(grp[10:22])    # 60 total (12 per level 1..5)
    manifest["datasets"]["cal_math_50"] = {
        "source": "DigitalLearningGmbH/MATH-lighteval[train] disjoint from MATH-500",
        "count": len(cal_math),
        "sha256": write_jsonl(OUT_DIR / "cal_math_50.jsonl", cal_math),
    }
    manifest["datasets"]["dev_math_60"] = {
        "source": "DigitalLearningGmbH/MATH-lighteval[train] disjoint from cal & MATH-500",
        "count": len(dev_math),
        "sha256": write_jsonl(OUT_DIR / "dev_math_60.jsonl", dev_math),
    }

    # 3. GSM8K (train -> cal/dev, test -> held-out!)
    print("Loading openai/gsm8k...")
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    rng_g = random.Random(101)
    tr_idx = list(range(len(gsm_train)))
    rng_g.shuffle(tr_idx)
    cal_gsm = [{
        "id": f"gsm_cal_{i:03d}",
        "problem": gsm_train[idx]["question"].strip(),
        "answer": extract_gsm8k_answer(gsm_train[idx]["answer"]),
    } for i, idx in enumerate(tr_idx[:48])]
    dev_gsm = [{
        "id": f"gsm_dev_{i:03d}",
        "problem": gsm_train[idx]["question"].strip(),
        "answer": extract_gsm8k_answer(gsm_train[idx]["answer"]),
    } for i, idx in enumerate(tr_idx[48:108])]
    test_gsm_full = [{
        "id": f"gsm_test_{i:04d}",
        "problem": ex["question"].strip(),
        "answer": extract_gsm8k_answer(ex["answer"]),
    } for i, ex in enumerate(gsm_test)]
    rng_gt = random.Random(2026)
    test_gsm_250 = list(test_gsm_full)
    rng_gt.shuffle(test_gsm_250)
    test_gsm_250 = test_gsm_250[:250]

    manifest["datasets"]["cal_gsm8k_48"] = {
        "source": "openai/gsm8k[train]",
        "count": len(cal_gsm),
        "sha256": write_jsonl(OUT_DIR / "cal_gsm8k_48.jsonl", cal_gsm),
    }
    manifest["datasets"]["dev_gsm8k_60"] = {
        "source": "openai/gsm8k[train] disjoint from cal",
        "count": len(dev_gsm),
        "sha256": write_jsonl(OUT_DIR / "dev_gsm8k_60.jsonl", dev_gsm),
    }
    manifest["datasets"]["test_gsm8k_250"] = {
        "source": "openai/gsm8k[test] seeded-250",
        "count": len(test_gsm_250),
        "sha256": write_jsonl(OUT_DIR / "test_gsm8k_250.jsonl", test_gsm_250),
    }
    manifest["datasets"]["test_gsm8k_full"] = {
        "source": "openai/gsm8k[test] full",
        "count": len(test_gsm_full),
        "sha256": write_jsonl(OUT_DIR / "test_gsm8k_full.jsonl", test_gsm_full),
    }

    # 4. ARC-Challenge test (OOD non-math reasoning preservation & transfer!)
    print("Loading allenai/ai2_arc ARC-Challenge...")
    arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_rows = []
    for i, ex in enumerate(arc_test):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        # Normalize numeric labels 1,2,3,4 -> A,B,C,D
        norm_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        norm_labels = [norm_map.get(l, l) for l in labels]
        ans_key = norm_map.get(ex["answerKey"], ex["answerKey"])
        opts = "\n".join(f"{l}. {t}" for l, t in zip(norm_labels, texts))
        arc_rows.append({
            "id": f"arc_c_{i:04d}",
            "question": ex["question"].strip(),
            "options_text": opts,
            "valid_labels": norm_labels,
            "answer": ans_key,
        })
    rng_a = random.Random(303)
    arc_250 = list(arc_rows)
    rng_a.shuffle(arc_250)
    arc_250 = arc_250[:250]
    manifest["datasets"]["test_arc_challenge_250"] = {
        "source": "allenai/ai2_arc[ARC-Challenge:test] seeded-250",
        "count": len(arc_250),
        "sha256": write_jsonl(OUT_DIR / "test_arc_challenge_250.jsonl", arc_250),
    }

    # 5. Copy / KV-Retrieval / Cycle-Induction suite (Dual-Function Entanglement probe!)
    cal_copy = build_copy_recall_suite(seed=701, n=36, split_tag="cal")
    test_copy = build_copy_recall_suite(seed=702, n=90, split_tag="test")
    manifest["datasets"]["cal_copy_36"] = {
        "source": "synthetic deterministic exact-copy/kv/cycle calibration",
        "count": len(cal_copy),
        "sha256": write_jsonl(OUT_DIR / "cal_copy_36.jsonl", cal_copy),
    }
    manifest["datasets"]["test_copy_90"] = {
        "source": "synthetic deterministic exact-copy/kv/cycle held-out test",
        "count": len(test_copy),
        "sha256": write_jsonl(OUT_DIR / "test_copy_90.jsonl", test_copy),
    }

    # 6. WikiText-2 validation (calibration) & test (perplexity preservation!)
    print("Loading Salesforce/wikitext wikitext-2-raw-v1...")
    wt_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    wt_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    val_paras = [x["text"].strip() for x in wt_val if len(x["text"].strip()) > 120]
    cal_wt = [{"id": f"wt_val_{i:03d}", "text": p} for i, p in enumerate(val_paras[:64])]
    manifest["datasets"]["cal_wikitext_64"] = {
        "source": "Salesforce/wikitext[wikitext-2-raw-v1:validation]",
        "count": len(cal_wt),
        "sha256": write_jsonl(OUT_DIR / "cal_wikitext_64.jsonl", cal_wt),
    }
    test_text = "\n".join(x["text"] for x in wt_test)
    wt_path = OUT_DIR / "test_wikitext2_raw.txt"
    wt_path.write_text(test_text, encoding="utf-8", newline="\n")
    manifest["datasets"]["test_wikitext2_raw"] = {
        "source": "Salesforce/wikitext[wikitext-2-raw-v1:test]",
        "bytes": wt_path.stat().st_size,
        "sha256": hashlib.sha256(wt_path.read_bytes()).hexdigest(),
    }

    man_path = OUT_DIR / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Saved manifest to", man_path)
    for k, v in manifest["datasets"].items():
        print(f"  {k:28s}: {v.get('count', v.get('bytes'))} | sha256={v['sha256'][:12]}")


if __name__ == "__main__":
    main()
