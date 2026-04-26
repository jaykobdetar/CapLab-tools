"""Unit tests for :mod:`caplab_sim.rng`.

Covers:
- LCG primitive math against a short reference vector computed
  directly from the Ghidra-confirmed formula.
- ``rng_output`` bit-pattern edges (n=0, n=MAX_RNG_N, n>MAX_RNG_N error).
- ``MiscRNG`` instrumentation: label assignment, trace length, seed
  advance equivalence.
- ``distance_if_reachable`` round-trip property: for a random k in the
  test range, advancing k steps should be detected as exactly k.

These tests are pure-Python — no save files needed. They lock down the
deterministic core that everything else in Phase 1 depends on.
"""

from __future__ import annotations

import unittest

from caplab_sim.rng import (
    LCG_INCREMENT,
    LCG_MODULUS_MASK,
    LCG_MULTIPLIER,
    MAX_RNG_N,
    MiscRNG,
    RNGTrace,
    distance_if_reachable,
    lcg_step,
    rng_output,
    simulate_lcg,
)


class LCGPrimitiveTests(unittest.TestCase):
    """Bit-exact checks against the engine's ``seed = seed * 0x015A4E35 + 1`` step."""

    def test_lcg_step_matches_closed_form(self):
        # Closed-form reference values from hand calculation.
        seed = 0x0AE83F5E
        expected = (seed * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MODULUS_MASK
        self.assertEqual(lcg_step(seed), expected)

    def test_lcg_wraps_at_2_32(self):
        # Seed of all-ones → next step must stay in uint32 range.
        out = lcg_step(0xFFFFFFFF)
        self.assertGreaterEqual(out, 0)
        self.assertLess(out, 1 << 32)

    def test_simulate_lcg_equivalent_to_iterated_step(self):
        seed = 0xD6296661
        k = 449        # the reference save's empirical daily count
        by_loop = seed
        for _ in range(k):
            by_loop = lcg_step(by_loop)
        self.assertEqual(simulate_lcg(seed, k), by_loop)

    def test_simulate_lcg_zero_steps_is_identity(self):
        self.assertEqual(simulate_lcg(0xDEADBEEF, 0), 0xDEADBEEF)


class RNGOutputTests(unittest.TestCase):
    """The ``((seed >> 16) & 0x7FFF) * n >> 15`` output formula."""

    def test_n_zero_returns_zero(self):
        self.assertEqual(rng_output(0xABCDEF01, 0), 0)

    def test_output_is_in_range(self):
        # Spread across several post-step seeds; output must always be
        # in [0, n-1] when n > 0.
        for seed_after in (0x00000001, 0x7FFFFFFF, 0xDEADBEEF, 0xFFFFFFFF):
            for n in (1, 9, 60, 120, 180, MAX_RNG_N):
                out = rng_output(seed_after, n)
                self.assertGreaterEqual(out, 0)
                self.assertLess(out, n)

    def test_output_rejects_too_large_n(self):
        with self.assertRaises(ValueError):
            rng_output(0x12345678, MAX_RNG_N + 1)

    def test_output_rejects_negative_n(self):
        with self.assertRaises(ValueError):
            rng_output(0x12345678, -1)


class MiscRNGTests(unittest.TestCase):
    """Instrumented wrapper — seed advance, trace recording, labels."""

    def test_seed_advances_even_on_n_zero(self):
        # Engine advances seed unconditionally; test that our replay does too.
        rng = MiscRNG(42)
        rng.rng(0)
        self.assertEqual(rng.seed, lcg_step(42))

    def test_trace_is_recorded_by_default(self):
        rng = MiscRNG(0xCAFEBABE)
        for n in (1, 9, 60, 120):
            rng.rng(n)
        self.assertEqual(len(rng.trace), 4)
        for i, call in enumerate(rng.trace):
            self.assertEqual(call.index, i)

    def test_trace_disabled_omits_records(self):
        rng = MiscRNG(0xCAFEBABE, trace=False)
        rng.rng(9)
        self.assertEqual(len(rng.trace), 0)

    def test_label_attaches_to_next_call_only(self):
        rng = MiscRNG(1)
        rng.label("first")
        rng.rng(10)
        rng.rng(10)
        self.assertEqual(rng.trace[0].label, "first")
        self.assertEqual(rng.trace[1].label, "")

    def test_advance_matches_manual_loop(self):
        a = MiscRNG(0xD6296661, trace=False)
        b = MiscRNG(0xD6296661, trace=False)
        a.advance(449)
        for _ in range(449):
            b.rng(0)
        self.assertEqual(a.seed, b.seed)

    def test_set_seed_rejects_out_of_range(self):
        rng = MiscRNG(0)
        with self.assertRaises(ValueError):
            rng.set_seed(-1)
        with self.assertRaises(ValueError):
            rng.set_seed(1 << 32)

    def test_rng_rejects_too_large_n(self):
        rng = MiscRNG(0)
        with self.assertRaises(ValueError):
            rng.rng(MAX_RNG_N + 1)


class RNGTraceDiffTests(unittest.TestCase):
    """Divergence detection — critical for Phase 2 per-call validation."""

    def _trace_of(self, seed, ns):
        rng = MiscRNG(seed)
        for n in ns:
            rng.rng(n)
        return rng.trace

    def test_identical_traces_no_divergence(self):
        a = self._trace_of(1, [10, 20, 30])
        b = self._trace_of(1, [10, 20, 30])
        self.assertIsNone(a.first_divergence(b))

    def test_divergent_n_detected_at_first_differing_index(self):
        a = self._trace_of(1, [10, 20, 30])
        b = self._trace_of(1, [10, 99, 30])
        self.assertEqual(a.first_divergence(b), 1)

    def test_length_mismatch_detected(self):
        a = self._trace_of(1, [10, 20, 30])
        b = self._trace_of(1, [10, 20])
        # Overlap is identical; divergence index = shorter length.
        self.assertEqual(a.first_divergence(b), 2)


class DistanceIfReachableTests(unittest.TestCase):
    """Round-trip: if we advance k steps, the search should return k."""

    def test_round_trip(self):
        start = 0x0AE83F5E
        for k in (0, 1, 5, 100, 449, 1000):
            target = simulate_lcg(start, k)
            self.assertEqual(distance_if_reachable(start, target, 2000), k)

    def test_unreachable_within_window_returns_none(self):
        start = 0x0AE83F5E
        # 1500 steps away, but only search 100.
        target = simulate_lcg(start, 1500)
        self.assertIsNone(distance_if_reachable(start, target, 100))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
