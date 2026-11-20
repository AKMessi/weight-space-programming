"""Unit tests for core WSP grading, BF16 codec, and exact MLP permutation invariance."""
from __future__ import annotations

import numpy as np
import pytest

from src.wsp.gguf_patch import (
    bf16_bytes_to_f32,
    build_block_aligned_permutation,
    f32_to_bf16_u16,
)
from src.wsp.llama_eval import mcnemar_exact, wilson_ci
from src.wsp.math_grader import (
    detect_repetition_loop,
    grade_arc_response,
    grade_copy_response,
    grade_math_response,
    math_answers_equal,
)


def test_math_answers_equal():
    assert math_answers_equal("42", "42")
    assert math_answers_equal("1,234", "1234")
    assert math_answers_equal(r"\frac{1}{2}", "0.5")
    assert math_answers_equal(r"\dfrac{3\sqrt{2}}{2}", r"\frac{3\sqrt{2}}{2}")
    assert math_answers_equal(r"2\pi", r"2*\pi")
    assert math_answers_equal(r"x = 7", "7")
    assert math_answers_equal("(3, 4)", "(3,4)")
    assert not math_answers_equal("41", "42")
    assert not math_answers_equal("", "42")


def test_grade_math_and_loop_detection():
    good = "<think>First 2+2=4. Done.</think>\nThe answer is \\boxed{4}."
    g1 = grade_math_response(good, "4", finish_reason="stop")
    assert g1["correct"] is True
    assert g1["has_think_close"] is True
    assert g1["loop_detected"] is False
    assert g1["hit_limit"] is False

    # Repetition loop hitting max tokens with latent gold inside think
    chunk = "Wait, let me double check whether the sum is actually 144 or something else entirely. "
    loop_raw = "<think>" + chunk * 10
    g2 = grade_math_response(loop_raw, "144", finish_reason="length")
    assert g2["correct"] is False
    assert g2["hit_limit"] is True
    assert g2["loop_detected"] is True
    assert g2["latent_in_think"] is True
    assert g2["hesitation_count"] >= 10


def test_grade_arc_and_copy():
    g_arc = grade_arc_response("<think>Gravity pulls down.</think>\n\\boxed{C}", "C", ["A", "B", "C", "D"])
    assert g_arc["correct"] is True
    g_copy = grade_copy_response("<think>Just repeat it.</think>\nAB12-CD34", "AB12-CD34")
    assert g_copy["correct"] is True


def test_bf16_roundtrip_exact():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 32)).astype(np.float32)
    u16 = f32_to_bf16_u16(x)
    raw_u8 = u16.view(np.uint8)
    back = bf16_bytes_to_f32(raw_u8, (16, 32))
    # Re-encoding back must be 100% bit-identical
    assert np.array_equal(f32_to_bf16_u16(back), u16)
    # Relative error within BF16 7-bit mantissa precision (~0.8%)
    assert np.max(np.abs(back - x) / (np.abs(x) + 1e-6)) < 0.01


def test_block_aligned_permutation_is_exact_swiglu_identity():
    rng = np.random.default_rng(123)
    d_model, d_ff = 64, 256
    W_gate = rng.standard_normal((d_ff, d_model)).astype(np.float32)
    W_up = rng.standard_normal((d_ff, d_model)).astype(np.float32)
    W_down = rng.standard_normal((d_model, d_ff)).astype(np.float32)
    x = rng.standard_normal((8, d_model)).astype(np.float32)

    def swiglu(wg, wu, wd, inp):
        g = inp @ wg.T
        silu_g = g / (1.0 + np.exp(-g))
        u = inp @ wu.T
        return (silu_g * u) @ wd.T

    y0 = swiglu(W_gate.astype(np.float64), W_up.astype(np.float64), W_down.astype(np.float64), x.astype(np.float64))
    dormant = [5, 19, 44, 77, 102, 131, 188, 240]
    perm = build_block_aligned_permutation(d_ff, dormant, block_size=32, dest_block=0)
    assert list(perm[: len(dormant)]) == dormant

    y1 = swiglu(W_gate[perm].astype(np.float64), W_up[perm].astype(np.float64), W_down[:, perm].astype(np.float64), x.astype(np.float64))
    assert np.allclose(y0, y1, atol=1e-10, rtol=1e-10)

    # Also verify exact gauge scaling invariance between W_up and W_down
    eta = rng.uniform(0.2, 5.0, size=(d_ff,)).astype(np.float64)
    y2 = swiglu(
        W_gate[perm].astype(np.float64),
        W_up[perm].astype(np.float64) * eta[:, None],
        W_down[:, perm].astype(np.float64) / eta[None, :],
        x.astype(np.float64),
    )
    assert np.allclose(y0, y2, atol=1e-10, rtol=1e-10)


def test_mcnemar_and_wilson():
    a = [True, False, False, False, False, False, False, False, False, False]
    b = [True, True, True, True, True, True, True, True, True, False]
    res = mcnemar_exact(a, b)
    assert res["rescued_by_b"] == 8
    assert res["broken_by_b"] == 0
    assert res["mcnemar_p_two_sided"] < 0.01
    lo, hi = wilson_ci(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
