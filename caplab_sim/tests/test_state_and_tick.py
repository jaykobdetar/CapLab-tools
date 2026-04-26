"""Integration tests that actually touch the fixture .SAV files.

These tests verify:

- :func:`read_seed_from_blob0` dynamically locates tag 0x1067 and
  returns the same seed across the five fixture saves (Headquarters,
  Department, Factory, R&D, Warehouse), which are snapshots of the
  same game tick. The bug in ``caplab_save.parser.parse_rng_seed``
  gives wrong values on four of the five — this test guards against
  regressing to that behaviour.

- :class:`SimState` projection from a real save populates the count
  fields that ``one_day_rng_budget`` depends on.

- ``one_day_rng_budget`` produces a plausible total (>0, < engine
  assert-n limit * n_firms) and the component sum equals ``total_float``.

- ``replay_one_day`` produces a trace whose length == ``budget.total``
  and whose final seed equals ``expected_end_of_day_seed`` — the
  cornerstone invariant of Phase 1.

These run quickly (<2s total) because each save is ~100KB decompressed.
"""

from __future__ import annotations

import os
import unittest

from caplab_sim.state import load_sim_state, read_seed_from_blob0
from caplab_sim.tick import (
    expected_end_of_day_seed,
    one_day_rng_budget,
    replay_one_day,
)
from caplab_save.decompress import decompress_save


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PLAY_001 uses the 17-group session; PLAY_002-004 use the 22-group session.
# The fixture saves (Headquarters/Department/Factory/R&D/Warehouse) are all
# the same tick of the same 22-group game with one extra player firm per
# fixture — they MUST all report identical seeds.
FIXTURE_DIR = os.path.join(REPO_ROOT, "caplab_save", "saves")
FIXTURE_SAVES = [
    "Headquarters.SAV",
    "Department.SAV",
    "Factory.SAV",
    "R&D.SAV",
    "Warehouse.SAV",
]
PLAY_001 = os.path.join(REPO_ROOT, "PLAY_001_20260421_000834_000.SAV")
PLAY_002 = os.path.join(REPO_ROOT, "PLAY_002_20260421_000854_000.SAV")

# Consecutive-day validation pair captured late on 2026-04-21 — the first
# real day-N / day-N+1 pair we have. Their game_date counters advance by
# exactly 1 (the day field at FirmArray offset 0x1A26B8 reads 1 in _____001
# and 2 in _____002). This pair is what closes Phase 1.
CONSEC_DAY1 = os.path.join(REPO_ROOT, "_____001_20260421_054902_000.SAV")
CONSEC_DAY2 = os.path.join(REPO_ROOT, "_____002_20260421_054916_000.SAV")


def _skip_if_missing(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise unittest.SkipTest(f"Missing fixture file(s): {missing}")


class SeedExtractionTests(unittest.TestCase):
    """Regression guard for the hardcoded-offset bug in caplab_save."""

    def test_all_fixture_saves_have_same_seed(self):
        _skip_if_missing(*[os.path.join(FIXTURE_DIR, f) for f in FIXTURE_SAVES])
        seeds = []
        for name in FIXTURE_SAVES:
            path = os.path.join(FIXTURE_DIR, name)
            r = decompress_save(path)
            seed, offset = read_seed_from_blob0(r.blob0)
            seeds.append((name, seed, offset))
        # All five must report the same seed — they are snapshots of the
        # same game tick with one extra player firm each. If any one of
        # them disagrees, we are either reading a different field or the
        # offset search found the wrong 0x1067 occurrence.
        seed_values = {s for _, s, _ in seeds}
        self.assertEqual(
            len(seed_values), 1,
            f"fixture saves report different seeds (hardcoded-offset bug "
            f"regressed?): {seeds}"
        )

    def test_play_saves_seeds_match_session_identity(self):
        _skip_if_missing(PLAY_001, PLAY_002)
        s1 = load_sim_state(PLAY_001)
        s2 = load_sim_state(PLAY_002)
        # PLAY_001 is a 17-group session; PLAY_002 is a 22-group session.
        # Seeds must differ (otherwise we're reading a constant, not the seed).
        self.assertNotEqual(s1.seed, s2.seed,
                            "PLAY_001 and PLAY_002 are different sessions — "
                            "seeds should not match")
        self.assertEqual(s1.n_groups, 17)
        self.assertEqual(s2.n_groups, 22)


class BudgetSanityTests(unittest.TestCase):
    """Structural checks on :func:`one_day_rng_budget` output."""

    def test_total_float_equals_component_sum(self):
        _skip_if_missing(PLAY_002)
        s = load_sim_state(PLAY_002)
        b = one_day_rng_budget(s)
        expected = (
            b.S_stocks + b.S_persons + b.S_towns + b.S_parties + b.S_talents
            + b.S_firms_gate + b.S_firms_cascade + b.S_firms_rd_media
            + b.S_groupai + b.S_logistics + b.S_rawres + b.S_goal_ai
            + b.S_bond + b.S_monthly
        )
        self.assertAlmostEqual(b.total_float, expected, places=6)

    def test_budget_plausible_magnitude(self):
        # Reference pair (_____001 → _____002) empirical: 182 calls/day.
        # With the tuned defaults (site_spawn_enabled=False,
        # person_mean_rng_per_day=0.05) the formula predicts ~179 for the
        # 22-group / 173-firm / 1000-person reference scenario — comfortably
        # within ±25 of the observed 182. We assert a wide envelope here
        # to guard against accidental 10× regressions while tolerating
        # the small per-day jitter from month-end or slice edges.
        _skip_if_missing(PLAY_002)
        s = load_sim_state(PLAY_002)
        b = one_day_rng_budget(s)
        self.assertGreater(b.total, 50)
        self.assertLess(b.total, 1000)


class ReplayInvariantTests(unittest.TestCase):
    """The Phase 1 invariant: replay trace length == budget total, and
    replay's final seed == ``expected_end_of_day_seed``.

    If these hold, the stub replay is burning exactly the predicted
    number of LCG steps in the §5 order. When Phase 2 swaps the stub
    for the real per-subsystem replay, this invariant must continue to
    hold — the *ordering* of calls doesn't affect the final seed, only
    the *count* does.
    """

    def test_replay_length_equals_budget(self):
        _skip_if_missing(PLAY_002)
        s = load_sim_state(PLAY_002)
        b = one_day_rng_budget(s)
        _, trace = replay_one_day(s)
        self.assertEqual(len(trace), b.total)

    def test_replay_final_seed_equals_expected(self):
        _skip_if_missing(PLAY_002)
        s = load_sim_state(PLAY_002)
        rng, trace = replay_one_day(s)
        self.assertEqual(rng.seed, expected_end_of_day_seed(s))
        self.assertEqual(trace.last_seed, expected_end_of_day_seed(s))


class ValidationHarnessTests(unittest.TestCase):
    """Smoke test for the validate.py flow (without a real consecutive pair)."""

    def test_same_save_against_itself_yields_zero_observed(self):
        # When both paths are the same save, observed delta should be 0
        # — we're asking "how far does the LCG advance from X to X?".
        from caplab_sim.validate import validate_day_pair
        _skip_if_missing(PLAY_002)
        result = validate_day_pair(PLAY_002, PLAY_002)
        self.assertEqual(result.seed_before, result.seed_after)
        self.assertTrue(result.same_seed)
        self.assertEqual(result.observed_steps, 0)
        # Predicted should still be the budget total.
        self.assertGreater(result.predicted_steps, 0)


class PhaseOneCloseTests(unittest.TestCase):
    """The acceptance test that closes Phase 1.

    Runs :func:`validate_day_pair` on the real day-N / day-N+1 pair
    ``_____001`` / ``_____002`` and asserts the observed LCG distance is
    within ≤25 steps of the formula's prediction — the success criterion
    in ``RNG_Call_Map.txt §9``.

    A tighter match (≤5) is currently expected with the tuned defaults
    (site_spawn_enabled=False, person_mean_rng_per_day=0.05); we assert
    the looser ≤25 bound here so that small per-day jitter (e.g. a
    different slice alignment or a one-off AI event) doesn't cause
    spurious regressions. The harness's ``result.delta`` is logged in
    the assertion message so any drift shows up immediately.
    """

    def test_consecutive_day_pair_within_residual(self):
        from caplab_sim.validate import validate_day_pair
        _skip_if_missing(CONSEC_DAY1, CONSEC_DAY2)
        result = validate_day_pair(CONSEC_DAY1, CONSEC_DAY2)
        self.assertIsNotNone(result.observed_steps,
                             "observed LCG distance unreachable within "
                             f"{result.max_steps} steps — suggests saves "
                             "are not actually consecutive, or seeds "
                             "mis-read.")
        self.assertFalse(result.same_seed,
                         "consecutive-day saves must have different seeds")
        delta = result.delta
        self.assertIsNotNone(delta)
        self.assertLessEqual(
            abs(delta), 25,
            f"Phase 1 residual exceeded: predicted "
            f"{result.predicted_steps} vs observed {result.observed_steps} "
            f"(gap {delta:+d}). See RNG_Call_Map.txt §9.\n"
            f"{result.as_text()}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
