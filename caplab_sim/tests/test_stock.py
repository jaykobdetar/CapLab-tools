"""Phase 2 stage 1 tests — per-stock daily price predictor.

Covers:

* Pure-formula unit tests for :func:`predict_one_stock` — every branch
  (unlisted, foreign, case 0-4, CPI floor, EPS shock) exercised on
  hand-built ``StockInput`` fixtures so a bug in any individual case
  surfaces with a clear name.

* The end-to-end integration test against the consecutive-day save
  pair ``_____001`` → ``_____002``: at pre-stock offset ``k == 50``
  with descending iteration order, every one of the 21 listed
  domestic stocks matches day-N+1's ``base_stock_price`` and
  ``sentiment`` byte-for-byte.  This is the correctness floor for
  Phase 2 — if this test regresses, the stock predictor is wrong
  and no downstream work is trustworthy.

* :func:`validate_stock_day_pair` reports ``full_match=True`` and the
  expected offset / LCG distances (``k=50``, stock_rng_calls=80,
  post_stock_distance=52, total_day_distance=182) on the same pair.
"""

from __future__ import annotations

import os
import unittest

from caplab_sim.rng import MiscRNG, lcg_step
from caplab_sim.stock import (
    DEFAULT_CPI,
    ScenarioState,
    StockInput,
    ai_tail_advance,
    base_inter_group_advance,
    compute_eps,
    dlc_inter_group_advance,
    predict_all_stocks,
    predict_one_stock,
    stock_inputs_from_save,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONSEC_DAY1 = os.path.join(REPO_ROOT, "_____001_20260421_054902_000.SAV")
CONSEC_DAY2 = os.path.join(REPO_ROOT, "_____002_20260421_054916_000.SAV")


def _skip_if_missing(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise unittest.SkipTest(f"Missing fixture file(s): {missing}")


# ---------------------------------------------------------------------------
# Pure-formula unit tests
# ---------------------------------------------------------------------------

def _base_input(**overrides) -> StockInput:
    """A vanilla fresh-IPO-like stock: price 10, sent 0, pe 0, bvps 0."""
    defaults = dict(
        group_recno=99,
        is_listed=True,
        is_foreign=False,
        group_type=3,
        hq_type=0,
        shares_outstanding=10_000_000.0,
        base_stock_price=10.0,
        book_value_per_share=0.0,
        eps_basic=0.0,
        eps_adjusted_stored=0.0,
        pe_ratio_stored=0.0,
        sentiment=0.0,
    )
    defaults.update(overrides)
    return StockInput(**defaults)


class DispatcherShortCircuitTests(unittest.TestCase):
    """Unlisted / foreign stocks must consume zero RNG."""

    def test_unlisted_stock_consumes_no_rng(self):
        rng = MiscRNG(0xDEADBEEF, trace=True)
        s = _base_input(is_listed=False, book_value_per_share=5.0)
        out = predict_one_stock(rng, s)
        self.assertEqual(out.rng_calls, 0)
        self.assertEqual(len(rng.trace), 0)
        # Unlisted price = max(0.1, bvps).
        self.assertAlmostEqual(out.base_stock_price, 5.0)

    def test_unlisted_zero_bvps_floors_at_0_1(self):
        rng = MiscRNG(0xDEADBEEF, trace=True)
        s = _base_input(is_listed=False, book_value_per_share=0.0)
        out = predict_one_stock(rng, s)
        self.assertAlmostEqual(out.base_stock_price, 0.1)
        self.assertEqual(out.rng_calls, 0)

    def test_foreign_stock_consumes_no_rng(self):
        rng = MiscRNG(0xDEADBEEF, trace=True)
        s = _base_input(is_foreign=True)
        out = predict_one_stock(rng, s)
        self.assertEqual(out.rng_calls, 0)
        self.assertEqual(len(rng.trace), 0)

    def test_deleted_group_skipped_in_predict_all(self):
        rng = MiscRNG(0xDEADBEEF, trace=True)
        stocks = [_base_input(group_recno=1, group_type=0), _base_input(group_recno=2)]
        out = predict_all_stocks(rng, stocks)
        # Only one stock processed — group_type==0 is skipped.
        self.assertEqual(len(out.per_stock), 1)
        self.assertEqual(out.per_stock[0].group_recno, 2)


class FreshIPORngBudgetTests(unittest.TestCase):
    """Per-stock RNG call count by case on a fresh-IPO stock.

    On the reference scenario (price=10, sent=0, eps=0, bvps=0, pe=0):
      * case 0 (SELL gate passes with pe<=0): 5 or 6 calls
          - sentiment_rand(0), market_rand(200), case_selector(5),
            r_pen(4), r_gate(3), optional r_crash(10).
      * case 1 (BUY gate fails with pe<=0): 3 calls
      * case 2 (FUND gate fails with bvps==0): 3 calls
      * case 3 (sentiment reversion, sent=0 → random(0)): 4 calls
      * case 4 (noise): 4 calls
    Plus the CPI floor path (not triggered here).
    """

    def _case(self, case_wanted: int) -> tuple:
        """Find a seed where random(5) returns ``case_wanted``, with the
        sentiment_rand+market_rand prefix preceding it.  Returns
        (seed, rng_calls).
        """
        for probe in range(2**12):
            rng = MiscRNG(probe, trace=True)
            s = _base_input()
            out = predict_one_stock(rng, s)
            if out.case_selector == case_wanted:
                return probe, out.rng_calls
        self.fail(f"No seed found in probe range producing case {case_wanted}")

    def test_case_1_fresh_ipo_3_calls(self):
        _, calls = self._case(1)
        self.assertEqual(calls, 3)

    def test_case_2_fresh_ipo_3_calls(self):
        _, calls = self._case(2)
        self.assertEqual(calls, 3)

    def test_case_3_fresh_ipo_4_calls(self):
        _, calls = self._case(3)
        self.assertEqual(calls, 4)

    def test_case_4_fresh_ipo_4_calls(self):
        _, calls = self._case(4)
        self.assertEqual(calls, 4)


class EpsShockDivideByZeroTests(unittest.TestCase):
    """Engine semantics for the ``old_eps == 0`` edge case.

    When ``eps_adjusted_stored == 0.0`` and ``compute_eps()`` returns a
    non-zero value, the engine's x87 ``fdiv`` produces ±infinity rather
    than raising — the subsequent clamp to ±100 yields exactly ±100 and
    a ``random(100)`` call fires.  Our Python port used to silently
    skip the RNG call here, producing a one-time seed offset whenever a
    group's trailing profit first went positive.  Regression guard
    against reverting to the pre-2026-04-22c behaviour.
    """

    def _input_with_eps_shock(self, *, eps_basic: float,
                              new_eps: float) -> StockInput:
        # Force case=3 with sentiment=0 so no RNG fires before EPS shock:
        #   sent_rand(0), market_rand(200), case_selector(5). Case 3 with
        #   sentiment==0 hits the "pos" branch with random(0).
        return _base_input(
            eps_basic=eps_basic,
            eps_adjusted_stored=0.0,
            eps_adjusted_new=new_eps,
        )

    def test_positive_new_eps_fires_random_100(self):
        """new > 0, old == 0, eps_basic >= 0 → eps_delta = +100."""
        rng = MiscRNG(0xCAFEBABE, trace=True)
        s = self._input_with_eps_shock(eps_basic=1.0, new_eps=5.0)
        out = predict_one_stock(rng, s)
        # Must have fired the EPS-shock RNG call.
        self.assertTrue(out.fired_eps_shock,
                        "old_eps==0 with new_eps!=0 must still fire "
                        "random(100) — engine semantics from x87 fdiv/inf")
        # And the last RNG call must have been for n=100 (the ±100 clamp).
        self.assertEqual(rng.trace[-1].n, 100,
                         "Expected random(100) as the final RNG call "
                         "when clamp saturates at ±100")

    def test_negative_new_eps_fires_random_100(self):
        """new < 0, old == 0, eps_basic >= 0 → eps_delta = -100."""
        rng = MiscRNG(0xCAFEBABE, trace=True)
        s = self._input_with_eps_shock(eps_basic=1.0, new_eps=-5.0)
        out = predict_one_stock(rng, s)
        self.assertTrue(out.fired_eps_shock)
        self.assertEqual(rng.trace[-1].n, 100)

    def test_negative_eps_basic_inverts_sign(self):
        """eps_basic < 0 → mult = -50, sign flips.

        new > 0 with mult=-50 → delta*mult/old = -inf → clamp -100 → fire
        with neg branch.
        """
        rng = MiscRNG(0xCAFEBABE, trace=True)
        s = self._input_with_eps_shock(eps_basic=-1.0, new_eps=5.0)
        out = predict_one_stock(rng, s)
        self.assertTrue(out.fired_eps_shock)
        self.assertEqual(rng.trace[-1].n, 100)


class Case4NoiseTests(unittest.TestCase):
    """Explicit byte-exact check of case 4 (noise) output."""

    def test_case4_sentiment_noise_is_r_minus_5(self):
        # Find a case-4 configuration and verify sentiment = r_noise - 5.
        for probe in range(2**12):
            rng = MiscRNG(probe, trace=True)
            s = _base_input()
            out = predict_one_stock(rng, s)
            if out.case_selector == 4:
                # trace[0]=sent_rand(0), [1]=market, [2]=case_sel, [3]=noise.
                r_noise = rng.trace[3].out
                self.assertAlmostEqual(out.sentiment, r_noise - 5, places=6)
                return
        self.fail("No case-4 configuration found in probe range")


# ---------------------------------------------------------------------------
# Integration: the consecutive-day reference pair
# ---------------------------------------------------------------------------

class ConsecutiveDayStockPredictorTests(unittest.TestCase):
    """The correctness floor: _____001 → _____002 must match 21/21.

    If this test regresses, either the formula, iteration order, or
    input binding is broken.  The raw seed math is already validated
    by Phase 1's seed-delta test — what's new here is per-call output.
    """

    EXPECTED_K = 50
    EXPECTED_STOCK_RNG_CALLS = 80
    EXPECTED_POST_DISTANCE = 52
    EXPECTED_TOTAL_DISTANCE = 182

    def setUp(self):
        _skip_if_missing(CONSEC_DAY1, CONSEC_DAY2)

    def test_full_match_at_k50_descending(self):
        from caplab_sim.validate_stocks import validate_stock_day_pair
        result = validate_stock_day_pair(CONSEC_DAY1, CONSEC_DAY2)
        self.assertEqual(result.pre_stock_offset, self.EXPECTED_K,
                         f"pre-stock offset changed: got {result.pre_stock_offset}")
        self.assertEqual(result.stock_rng_calls, self.EXPECTED_STOCK_RNG_CALLS)
        self.assertEqual(result.post_stock_distance, self.EXPECTED_POST_DISTANCE)
        self.assertEqual(result.total_day_distance, self.EXPECTED_TOTAL_DISTANCE)
        self.assertTrue(result.full_match,
                        f"expected full match, got {result.n_matched}/{result.n_total}\n"
                        f"{result.as_text()}")

    def test_predict_all_descending_is_the_default(self):
        """The default iteration order in predict_all_stocks MUST be
        descending recno. If someone flips this to ascending (or adds
        a ``reorder=False`` default) the reference pair breaks."""
        from caplab_save.parser import CapLabSave
        save1 = CapLabSave.load(CONSEC_DAY1)
        save2 = CapLabSave.load(CONSEC_DAY2)
        inputs = [si for si in stock_inputs_from_save(save1) if si.group_type != 0]
        observed = {g.recno: g.stock for g in save2.groups if g.stock.is_listed}

        # Advance to offset 50.
        seed = save1.rng.seed
        for _ in range(50):
            seed = lcg_step(seed)
        rng = MiscRNG(seed, trace=True)
        # Pass inputs in ascending recno order — the predictor's own
        # reorder=True default flips it.
        out = predict_all_stocks(rng, inputs, cpi=DEFAULT_CPI)
        by_recno = {p.group_recno: p for p in out.per_stock}

        for recno, obs in observed.items():
            p = by_recno.get(recno)
            self.assertIsNotNone(p, f"group {recno} missing from prediction")
            self.assertAlmostEqual(p.base_stock_price, obs.base_stock_price, places=4,
                                   msg=f"g{recno} price mismatch")
            self.assertAlmostEqual(p.sentiment, obs.sentiment, places=2,
                                   msg=f"g{recno} sentiment mismatch")

    def test_ascending_order_is_wrong(self):
        """Flipping ``reorder=False`` with ascending inputs MUST fail
        to match — that confirms the iteration order is semantically
        meaningful and not just a cosmetic detail."""
        from caplab_save.parser import CapLabSave
        save1 = CapLabSave.load(CONSEC_DAY1)
        save2 = CapLabSave.load(CONSEC_DAY2)
        inputs = sorted(
            (si for si in stock_inputs_from_save(save1) if si.group_type != 0),
            key=lambda s: s.group_recno,  # ASCENDING
        )
        observed = {g.recno: g.stock for g in save2.groups if g.stock.is_listed}

        seed = save1.rng.seed
        for _ in range(50):
            seed = lcg_step(seed)
        rng = MiscRNG(seed, trace=True)
        out = predict_all_stocks(rng, inputs, cpi=DEFAULT_CPI, reorder=False)
        by_recno = {p.group_recno: p for p in out.per_stock}

        matched = 0
        for recno, obs in observed.items():
            p = by_recno.get(recno)
            if p is None:
                continue
            if (abs(p.base_stock_price - obs.base_stock_price) < 5e-4
                    and abs(p.sentiment - obs.sentiment) < 1e-2):
                matched += 1
        # Ascending iteration should NOT match the majority — a few
        # coincidental matches are possible (e.g. for a stock whose
        # case-2 gate fails, the output is unchanged from input, so any
        # iteration order appears to "match").  The bar is: descending
        # gives 21/21, ascending gives at most a handful of flukes.
        self.assertLess(matched, 5,
                        f"ascending iteration matched {matched} stocks — "
                        f"descending must clearly outperform")


# ---------------------------------------------------------------------------
# Inter-group AI-tail advance (2026-04-22)
# ---------------------------------------------------------------------------
#
# These tests pin the Ghidra-derived gating behaviour documented in
# Stock_Tick_InterGroup_RNG_Analysis.md §§3.1-3.2 and the conservative
# ship posture in §4.3:
#
#   - dlc_inter_group_advance: +1 iff DLC flags on AND hq_type ∈ {0,10}
#     AND dlc_build_flag on AND ai_leader_slot!=0 AND group_type==3
#     AND cadence matches.
#   - base_inter_group_advance: 0 (always, until the AI-startup gate
#     and game_end_state predicate are reverse-engineered).  The
#     "AI-competitor → +1" version was tried on 2026-04-22 and broke
#     live accuracy from 93.8% → 7.9%, confirming the base-tail is
#     not firing on the current save — see the function's docstring
#     for the full k-histogram analysis.
#   - ai_tail_advance: sum of the two (= dlc only, in practice).
#
# The scalar-sweep fallback in live_monitor.sweep_best_k is also
# exercised indirectly: predict_all_stocks with scenario_state=None
# must preserve the pre-2026-04-22 behaviour byte-for-byte, AND so
# must scenario_state=<base-game state> (the DLC branch contributes
# 0 on base saves and the base branch is disabled).


def _ai_group(**overrides) -> StockInput:
    """Base-game AI competitor: group_type=3, ai_leader_slot=1, hq_type=0."""
    defaults = dict(
        group_recno=99,
        group_type=3,
        ai_leader_slot=1,
        dlc_build_flag=1,
    )
    defaults.update(overrides)
    return _base_input(**defaults)


class DlcInterGroupAdvanceTests(unittest.TestCase):
    """DLC-scenario single-shot advance — the 23% regression fix."""

    def _state(self, **kw) -> ScenarioState:
        defaults = dict(
            game_date=0,
            dlc_economy_mode=True,
            dlc_stock_market_active=True,
        )
        defaults.update(kw)
        return ScenarioState(**defaults)

    def test_dlc_off_returns_zero(self):
        """Base-game scenarios must never hit the DLC path."""
        st = self._state(dlc_economy_mode=False, dlc_stock_market_active=False)
        g = _ai_group()
        self.assertEqual(dlc_inter_group_advance(g, st), 0)

    def test_one_dlc_flag_off_returns_zero(self):
        """BOTH flags must be on — either alone is not enough."""
        g = _ai_group()
        self.assertEqual(
            dlc_inter_group_advance(
                g, self._state(dlc_stock_market_active=False)
            ),
            0,
        )
        self.assertEqual(
            dlc_inter_group_advance(
                g, self._state(dlc_economy_mode=False)
            ),
            0,
        )

    def test_non_ai_group_returns_zero(self):
        """Player group (ai_leader_slot==0) never runs the tail."""
        st = self._state()
        g = _ai_group(ai_leader_slot=0)
        self.assertEqual(dlc_inter_group_advance(g, st), 0)

    def test_non_competitor_type_returns_zero(self):
        """group_type != 3 (e.g. player, bank-NPC) short-circuits."""
        st = self._state()
        self.assertEqual(dlc_inter_group_advance(_ai_group(group_type=1), st), 0)
        self.assertEqual(dlc_inter_group_advance(_ai_group(group_type=4), st), 0)

    def test_hq_type_outside_qualifying_set_returns_zero(self):
        """Only hq_type ∈ {0, 10} qualifies; banks/insurance/etc skip."""
        st = self._state()
        for bad_hq in (1, 2, 6, 7, 8, 11, 20):
            self.assertEqual(
                dlc_inter_group_advance(_ai_group(hq_type=bad_hq), st),
                0,
                f"hq_type={bad_hq} must not qualify",
            )

    def test_hq_type_zero_qualifies(self):
        """hq_type=0 + cadence match → +1 advance."""
        g = _ai_group(group_recno=90, hq_type=0)
        # Period 90 for hq_type=0: game_date % 90 == group_recno % 90
        st = self._state(game_date=90)  # 90 % 90 == 0, 90 % 90 == 0 → match
        self.assertEqual(dlc_inter_group_advance(g, st), 1)

    def test_hq_type_ten_uses_60_day_cadence(self):
        """hq_type=10 switches the period from 90 to 60."""
        g = _ai_group(group_recno=60, hq_type=10)
        # Period 60 for hq_type=10: cadence gate (60 % 60 == 60 % 60).
        st = self._state(game_date=60)
        self.assertEqual(dlc_inter_group_advance(g, st), 1)
        # Same group at a non-matching date: 0.
        self.assertEqual(
            dlc_inter_group_advance(g, self._state(game_date=59)), 0,
        )

    def test_dlc_build_flag_off_returns_zero(self):
        """Per-group DLC build flag must be on."""
        st = self._state(game_date=90)
        g = _ai_group(group_recno=90, dlc_build_flag=0)
        self.assertEqual(dlc_inter_group_advance(g, st), 0)

    def test_cadence_mismatch_returns_zero(self):
        """When game_date % period != recno % period, no advance."""
        g = _ai_group(group_recno=1, hq_type=0)  # period 90
        # game_date=0: 0 % 90 == 0, recno % 90 == 1 → mismatch.
        self.assertEqual(
            dlc_inter_group_advance(g, self._state(game_date=0)), 0,
        )


class BaseInterGroupAdvanceTests(unittest.TestCase):
    """Base-game daily Bernoulli tail — §3.1 rows 1-2.

    **Current predicate returns 0 unconditionally** — the AI-startup
    gate and game_end_state predicate are not yet traced, and the
    empirical k-histogram {0:70, 5:1, 17:1, 20:1, 25:1} on the 74-day
    base-game run proves the tail is not firing on the current save.
    See ``base_inter_group_advance``'s docstring for the full rationale.

    These tests therefore pin the *conservative posture*: always 0,
    regardless of group shape.  When the missing gates are wired up,
    flip these to expect 1 for the AI-competitor happy path.
    """

    def _state(self, **kw) -> ScenarioState:
        return ScenarioState(
            game_date=kw.get("game_date", 0),
            dlc_economy_mode=False,
            dlc_stock_market_active=False,
        )

    def test_player_group_returns_zero(self):
        """Player (ai_leader_slot==0) never runs the AI tail."""
        g = _ai_group(ai_leader_slot=0)
        self.assertEqual(base_inter_group_advance(g, self._state()), 0)

    def test_non_competitor_type_returns_zero(self):
        """group_type != 3 skips the tail."""
        g = _ai_group(group_type=1)
        self.assertEqual(base_inter_group_advance(g, self._state()), 0)

    def test_ai_competitor_returns_zero_pending_startup_gate(self):
        """Standard AI group: returns 0 until the startup gate is traced.

        Previously returned 1; that drove live accuracy 93.8% → 7.9%
        because the startup-gate-elapsed predicate isn't modelled.
        See Stock_Tick_InterGroup_RNG_Analysis.md §4.3 — the ship posture
        is DLC-only until the base-game gates are reverse-engineered.
        """
        g = _ai_group(hq_type=0)
        self.assertEqual(base_inter_group_advance(g, self._state()), 0)

    def test_bank_insurance_hq_types_return_zero(self):
        """hq_type ∈ {6,7,8} skips site 1 (the startup-gate path)."""
        for hq in (6, 7, 8):
            self.assertEqual(
                base_inter_group_advance(_ai_group(hq_type=hq), self._state()),
                0,
                f"hq_type={hq} must skip the base tail",
            )


class AiTailAdvanceIntegrationTests(unittest.TestCase):
    """`ai_tail_advance` = base + dlc, and wires into predict_all_stocks.

    Current base-tail predicate returns 0 (pending the AI-startup-gate
    trace); these integration tests therefore drive the RNG-injection
    proofs through the DLC branch, which is the production ship path
    documented in Stock_Tick_InterGroup_RNG_Analysis.md §4.3.
    """

    def test_base_only_scenario(self):
        """DLC off + AI competitor → 0 advances (base-tail disabled)."""
        st = ScenarioState(
            game_date=0, dlc_economy_mode=False, dlc_stock_market_active=False,
        )
        g = _ai_group()
        self.assertEqual(ai_tail_advance(g, st), 0)

    def test_dlc_scenario_cadence_match(self):
        """DLC on + cadence match → 1 advance (dlc only; base disabled)."""
        st = ScenarioState(
            game_date=90, dlc_economy_mode=True, dlc_stock_market_active=True,
        )
        g = _ai_group(group_recno=90, hq_type=0)
        self.assertEqual(ai_tail_advance(g, st), 1)

    def test_dlc_scenario_cadence_miss(self):
        """DLC on, cadence miss → 0 advances (nothing fires)."""
        st = ScenarioState(
            game_date=1, dlc_economy_mode=True, dlc_stock_market_active=True,
        )
        g = _ai_group(group_recno=90, hq_type=0)
        self.assertEqual(ai_tail_advance(g, st), 0)

    def test_predict_all_stocks_with_scenario_state_advances_seed(self):
        """Predictor with a DLC scenario_state advances by the tail amount.

        Uses two UNLISTED AI groups (0 RNG calls per stock tick) so
        the only RNG consumed is the inter-group advance itself —
        that lets us assert the seed delta exactly.  The first group
        iterated (descending by recno → s1 at recno=90) passes the
        DLC cadence gate (game_date=90, period=90, 90%90==90%90==0),
        so ai_tail_advance returns exactly 1 between s1 and s2.
        """
        s1 = _ai_group(group_recno=90, is_listed=False,
                       hq_type=0, dlc_build_flag=1,
                       book_value_per_share=1.0)
        s2 = _ai_group(group_recno=9,  is_listed=False,
                       hq_type=0, dlc_build_flag=1,
                       book_value_per_share=1.0)
        state = ScenarioState(
            game_date=90, dlc_economy_mode=True,
            dlc_stock_market_active=True,
        )

        # Without scenario_state: unlisted stocks consume 0 RNG calls,
        # so rng_a.seed remains at the initial seed.
        rng_a = MiscRNG(0xDEADBEEF)
        predict_all_stocks(rng_a, [s1, s2], scenario_state=None)

        # With DLC scenario_state: exactly 1 LCG step injected between
        # s1 and s2, otherwise still 0 stock-tick calls.
        rng_b = MiscRNG(0xDEADBEEF)
        predict_all_stocks(rng_b, [s1, s2], scenario_state=state)

        # Because neither stock consumes RNG in its tick, the b-seed
        # must be EXACTLY one lcg_step ahead of the a-seed (and the
        # a-seed equals the initial seed unchanged).
        self.assertEqual(rng_a.seed, 0xDEADBEEF,
                         "control run must not advance the seed when "
                         "both stocks are unlisted")
        self.assertEqual(lcg_step(rng_a.seed), rng_b.seed,
                         "ai_tail_advance must inject exactly 1 LCG step "
                         "between the two AI-competitor groups when DLC "
                         "cadence matches")

    def test_predict_all_stocks_base_scenario_is_passthrough(self):
        """Base-game scenario_state must not inject any inter-group steps.

        This is the ship-blocker guarantee — base-game saves (the
        _____001/_____002 reference pair) must see identical RNG
        consumption regardless of whether scenario_state is passed.
        """
        s1 = _ai_group(group_recno=10, is_listed=True, base_stock_price=10.0)
        s2 = _ai_group(group_recno=9,  is_listed=True, base_stock_price=10.0)
        base_state = ScenarioState(
            game_date=0, dlc_economy_mode=False,
            dlc_stock_market_active=False,
        )

        rng_a = MiscRNG(0xDEADBEEF)
        predict_all_stocks(rng_a, [s1, s2], scenario_state=None)

        rng_b = MiscRNG(0xDEADBEEF)
        predict_all_stocks(rng_b, [s1, s2], scenario_state=base_state)

        # Identical seed progression — no AI-tail injection on base.
        self.assertEqual(rng_a.seed, rng_b.seed,
                         "base-game scenario_state must not inject any "
                         "inter-group RNG advance (base_inter_group_advance "
                         "returns 0 until the startup gate is traced)")

    def test_predict_all_stocks_scenario_state_none_is_backward_compatible(self):
        """Legacy callers (scenario_state=None) must see no behaviour change.

        This is the critical compatibility guarantee — the 98.2%-match
        save-file replay must keep passing.  Checked by confirming that
        scenario_state=None produces identical output to the pre-fix
        inter_group_advance=0 path.
        """
        s1 = _ai_group(group_recno=10, is_listed=True, base_stock_price=10.0)
        s2 = _ai_group(group_recno=9, is_listed=True, base_stock_price=10.0)

        rng_a = MiscRNG(0xDEADBEEF)
        out_a = predict_all_stocks(rng_a, [s1, s2], scenario_state=None)

        rng_b = MiscRNG(0xDEADBEEF)
        out_b = predict_all_stocks(rng_b, [s1, s2], inter_group_advance=0)

        self.assertEqual(rng_a.seed, rng_b.seed)
        self.assertEqual(out_a.total_rng_calls, out_b.total_rng_calls)
        for pa, pb in zip(out_a.per_stock, out_b.per_stock):
            self.assertEqual(pa.group_recno, pb.group_recno)
            self.assertAlmostEqual(pa.base_stock_price, pb.base_stock_price)
            self.assertAlmostEqual(pa.sentiment, pb.sentiment)


class ComputeEpsTests(unittest.TestCase):
    """Unit tests for the ``Group::compute_eps(1)`` port.

    The port at ``caplab_sim.stock.compute_eps`` implements the simple
    branch of ``FUN_0051c390`` (skipping the ``FUN_0051d740``
    subsidiary-adjustment branch).  These tests cover the numeric
    identity, division-by-zero edge cases, and the critical wiring
    guarantee that :func:`stock_inputs_from_save` populates
    ``eps_adjusted_new`` via this port.
    """

    def test_fresh_ipo_group_has_zero_eps(self):
        """Jan 1 1990 case — no trailing profit yet."""
        s = _base_input(trailing_12m_net_profit=0.0,
                        trailing_12m_special_revenue=0.0)
        self.assertEqual(compute_eps(s), 0.0)

    def test_profit_only_matches_formula(self):
        """trailing_12m_net_profit / shares_outstanding."""
        s = _base_input(shares_outstanding=10_000_000.0,
                        trailing_12m_net_profit=50_000_000.0,
                        trailing_12m_special_revenue=0.0)
        # 50M / 10M = 5.0
        self.assertAlmostEqual(compute_eps(s), 5.0)

    def test_special_revenue_is_added(self):
        """Both fields sum before dividing."""
        s = _base_input(shares_outstanding=10_000_000.0,
                        trailing_12m_net_profit=30_000_000.0,
                        trailing_12m_special_revenue=20_000_000.0)
        # (30M + 20M) / 10M = 5.0
        self.assertAlmostEqual(compute_eps(s), 5.0)

    def test_negative_profit_produces_negative_eps(self):
        s = _base_input(shares_outstanding=10_000_000.0,
                        trailing_12m_net_profit=-20_000_000.0,
                        trailing_12m_special_revenue=5_000_000.0)
        self.assertAlmostEqual(compute_eps(s), -1.5)

    def test_zero_shares_outstanding_returns_zero_no_crash(self):
        """Defensive: shares==0 must not raise ZeroDivisionError.

        The engine would return NaN/Inf here via float10 division; we
        coerce to 0.0 because the EPS-shock branch skips zero deltas
        anyway (no_change branch).
        """
        s = _base_input(shares_outstanding=0.0,
                        trailing_12m_net_profit=100.0,
                        trailing_12m_special_revenue=0.0)
        self.assertEqual(compute_eps(s), 0.0)


class StockInputsFromSaveEpsTests(unittest.TestCase):
    """Regression guards for the compute_eps wiring in stock_inputs_from_save.

    This is the critical plumbing test — if the save-file builder
    regresses and stops populating ``eps_adjusted_new``, the EPS-shock
    branch in :func:`predict_one_stock` silently no-ops on every
    firm-owning group.  The symptom is ~95% → 46% match rate on the
    player group (SESSION_LOG 2026-04-22c).
    """

    def test_save_builder_populates_eps_adjusted_new(self):
        _skip_if_missing(CONSEC_DAY1)
        from caplab_save.parser import CapLabSave
        save = CapLabSave.load(CONSEC_DAY1)
        inputs = stock_inputs_from_save(save)
        # Every StockInput must have a non-None eps_adjusted_new after
        # the 2026-04-22 wiring.  None would silently revert the
        # EPS-shock branch to its pre-fix no-op behaviour.
        for si in inputs:
            self.assertIsNotNone(
                si.eps_adjusted_new,
                f"group_recno={si.group_recno}: eps_adjusted_new is None "
                "— stock_inputs_from_save regression (see SESSION_LOG "
                "2026-04-22c).  Must be populated via compute_eps(si)."
            )

    def test_save_builder_eps_matches_compute_eps_formula(self):
        _skip_if_missing(CONSEC_DAY1)
        from caplab_save.parser import CapLabSave
        save = CapLabSave.load(CONSEC_DAY1)
        inputs = stock_inputs_from_save(save)
        # Cross-check: eps_adjusted_new must equal compute_eps(si)
        # recomputed from the StockInput's own fields.  This is the
        # primary contract binding.
        for si in inputs:
            self.assertEqual(
                si.eps_adjusted_new,
                compute_eps(si),
                f"group_recno={si.group_recno}: eps_adjusted_new "
                "diverges from compute_eps(si).  Save builder must use "
                "compute_eps() — no shortcuts, no caching bugs."
            )


if __name__ == "__main__":
    unittest.main()
