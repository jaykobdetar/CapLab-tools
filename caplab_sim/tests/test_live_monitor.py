"""Unit tests for :mod:`caplab_sim.live_monitor`.

Focus: the pure-data pieces (change detection, sweep-k scoring, stats
accumulation, JSONL logging, plain-text rendering).  The live-attach
path is exercised integration-style by running ``python -m
caplab_sim.live_monitor --once`` against a real process; those scenarios
are too environment-dependent to test in-unit, so we stub or bypass the
``LiveGameReader`` in these cases.

The end-to-end sweep logic is verified against the known-good
``_____001`` → ``_____002`` pair — if this test regresses, the monitor
would mis-attribute day transitions for the only scenario we've proven
the predictor matches 21/21 on.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

from caplab_save.parser import CapLabSave

from caplab_sim import live_monitor as lm
from caplab_sim.rng import MiscRNG, lcg_step, simulate_lcg
from caplab_sim.stock import (
    DEFAULT_CPI,
    StockInput,
    predict_all_stocks,
    stock_inputs_from_save,
)


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONSEC_DAY1 = os.path.join(REPO_ROOT, "_____001_20260421_054902_000.SAV")
CONSEC_DAY2 = os.path.join(REPO_ROOT, "_____002_20260421_054916_000.SAV")


def _skip_if_missing(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise unittest.SkipTest(f"Missing fixture file(s): {missing}")


class _FakeReader:
    """Minimal ``LiveGameReader`` stub for renderer tests.

    Satisfies the attributes the :class:`Renderer` reads (``pid``,
    ``base``, ``misc_addr``) without touching ``/proc``.
    """

    pid = 4242
    base = 0x00400000
    misc_addr = 0x009753d8


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


class ChangeDetectionTests(unittest.TestCase):
    """``_state_changed`` — seed move alone is insufficient; prices must move too."""

    def _mk_snapshot(self, seed: int, price: float, sent: float) -> lm.Snapshot:
        s = StockInput(
            group_recno=1,
            is_listed=True,
            is_foreign=False,
            group_type=1,
            base_stock_price=price,
            sentiment=sent,
        )
        forecast = predict_all_stocks(MiscRNG(seed), [s], cpi=DEFAULT_CPI)
        return lm.Snapshot(
            wall_time=0.0, seed=seed, inputs=[s],
            listed_recnos=[1], forecast=forecast,
        )

    def test_seed_unchanged_returns_false(self):
        snap = self._mk_snapshot(0x12345678, 100.0, 0.0)
        self.assertFalse(
            lm._state_changed(snap, snap.inputs, snap.seed)
        )

    def test_seed_changed_but_prices_identical_returns_false(self):
        """Menu-Misc writebacks move the seed without any stock-tick work."""
        snap = self._mk_snapshot(0x12345678, 100.0, 0.0)
        self.assertFalse(
            lm._state_changed(snap, snap.inputs, lcg_step(snap.seed))
        )

    def test_seed_and_price_changed_returns_true(self):
        snap = self._mk_snapshot(0x12345678, 100.0, 0.0)
        new_inputs = [
            StockInput(
                group_recno=1, is_listed=True, is_foreign=False,
                group_type=1, base_stock_price=101.0, sentiment=0.0,
            ),
        ]
        self.assertTrue(
            lm._state_changed(snap, new_inputs, lcg_step(snap.seed))
        )

    def test_seed_and_sentiment_changed_returns_true(self):
        snap = self._mk_snapshot(0x12345678, 100.0, 0.0)
        new_inputs = [
            StockInput(
                group_recno=1, is_listed=True, is_foreign=False,
                group_type=1, base_stock_price=100.0, sentiment=5.0,
            ),
        ]
        self.assertTrue(
            lm._state_changed(snap, new_inputs, lcg_step(snap.seed))
        )


# ---------------------------------------------------------------------------
# Sweep + match stats against the reference day pair
# ---------------------------------------------------------------------------


class SweepAgainstReferencePairTests(unittest.TestCase):
    """``sweep_best_k`` against _____001 → _____002 finds k=50, 21/21."""

    def test_sweep_finds_k50_full_match(self):
        _skip_if_missing(CONSEC_DAY1, CONSEC_DAY2)
        save_a = CapLabSave.load(CONSEC_DAY1)
        save_b = CapLabSave.load(CONSEC_DAY2)

        inputs_a = stock_inputs_from_save(save_a)
        # The monitor builds "observed inputs" from live memory; the
        # save-B equivalent is stock_inputs_from_save(save_B), which
        # carries the same post-tick prices and sentiments.
        inputs_b = stock_inputs_from_save(save_b)

        prev = lm.Snapshot(
            wall_time=0.0,
            seed=save_a.rng.seed,
            inputs=inputs_a,
            listed_recnos=sorted(
                (s.group_recno for s in inputs_a if s.is_listed and not s.is_foreign),
                reverse=True,
            ),
            forecast=predict_all_stocks(
                MiscRNG(save_a.rng.seed), inputs_a, cpi=DEFAULT_CPI,
            ),
        )

        res = lm.sweep_best_k(prev, inputs_b, cpi=DEFAULT_CPI, max_k=200)
        self.assertEqual(res.best_k, 50)
        self.assertEqual(res.n_matched, res.n_total)
        self.assertEqual(res.n_matched, 21)
        # All 21 diffs come back as matches.
        self.assertTrue(all(d.matches for d in res.diffs))

    def test_sweep_records_rng_calls(self):
        _skip_if_missing(CONSEC_DAY1, CONSEC_DAY2)
        save_a = CapLabSave.load(CONSEC_DAY1)
        save_b = CapLabSave.load(CONSEC_DAY2)
        inputs_a = stock_inputs_from_save(save_a)
        inputs_b = stock_inputs_from_save(save_b)
        prev = lm.Snapshot(
            wall_time=0.0,
            seed=save_a.rng.seed,
            inputs=inputs_a,
            listed_recnos=[],
            forecast=predict_all_stocks(MiscRNG(save_a.rng.seed), inputs_a),
        )
        res = lm.sweep_best_k(prev, inputs_b, cpi=DEFAULT_CPI, max_k=200)
        # The reference pair consumes 80 stock RNG calls on day 1 →
        # day 2.  If this ever changes we've either broken stock.py or
        # changed the iteration order.
        self.assertEqual(res.best_rng_calls, 80)


# ---------------------------------------------------------------------------
# Stats accumulation
# ---------------------------------------------------------------------------


class RunningStatsTests(unittest.TestCase):
    """Cumulative stats track transitions, per-group hits, and k histogram."""

    def _res(self, k: int, diffs, inter_advance: int = 0):
        return lm.TransitionResult(
            wall_time=0.0, prev_seed=0, new_seed=0, seed_distance=None,
            best_k=k, best_inter_advance=inter_advance, best_rng_calls=0,
            diffs=diffs,
        )

    def _diff(self, recno: int, matches: bool) -> lm.TransitionDiff:
        # Build a diff that evaluates to `matches` via the real
        # tolerance logic, so the test exercises the same code path as
        # production.
        if matches:
            return lm.TransitionDiff(
                group_recno=recno,
                predicted_price=1.0, observed_price=1.0,
                predicted_sentiment=0.0, observed_sentiment=0.0,
            )
        return lm.TransitionDiff(
            group_recno=recno,
            predicted_price=1.0, observed_price=2.0,
            predicted_sentiment=0.0, observed_sentiment=10.0,
        )

    def test_ingest_updates_totals(self):
        stats = lm.RunningStats()
        stats.ingest(self._res(50, [self._diff(1, True), self._diff(2, False)]))
        stats.ingest(self._res(50, [self._diff(1, True), self._diff(2, True)]))
        self.assertEqual(stats.transitions, 2)
        self.assertEqual(stats.total_matched, 3)
        self.assertEqual(stats.total_observed, 4)
        self.assertAlmostEqual(stats.fraction, 0.75)

    def test_per_group_and_k_histogram(self):
        stats = lm.RunningStats()
        stats.ingest(self._res(50, [self._diff(1, True)]))
        stats.ingest(self._res(51, [self._diff(1, True)]))
        stats.ingest(self._res(50, [self._diff(1, False)]))
        self.assertEqual(stats.per_group[1], [2, 3])  # 2/3 matched
        self.assertEqual(stats.k_histogram, {50: 2, 51: 1})


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------


class EventLoggerTests(unittest.TestCase):
    """``EventLogger`` writes one JSON object per line, flushes eagerly."""

    def test_writes_snapshot_and_transition(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "log.jsonl")
            logger = lm.EventLogger(path)
            try:
                # Minimal synthetic snapshot — only the fields the
                # logger reads matter here.
                s = StockInput(
                    group_recno=1, is_listed=True, is_foreign=False,
                    group_type=1, base_stock_price=10.0, sentiment=5.0,
                )
                forecast = predict_all_stocks(
                    MiscRNG(0xDEADBEEF), [s], cpi=DEFAULT_CPI,
                )
                snap = lm.Snapshot(
                    wall_time=1.0, seed=0xDEADBEEF, inputs=[s],
                    listed_recnos=[1], forecast=forecast,
                )
                logger.snapshot(snap)

                res = lm.TransitionResult(
                    wall_time=2.0, prev_seed=0xDEADBEEF, new_seed=0xCAFEBABE,
                    seed_distance=182, best_k=50, best_inter_advance=0,
                    best_rng_calls=80,
                    diffs=[
                        lm.TransitionDiff(
                            group_recno=1,
                            predicted_price=10.0, observed_price=10.0,
                            predicted_sentiment=5.0, observed_sentiment=5.0,
                        ),
                    ],
                )
                logger.transition(res)
            finally:
                logger.close()

            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2)
            snap_rec = json.loads(lines[0])
            trans_rec = json.loads(lines[1])
            self.assertEqual(snap_rec["kind"], "snapshot")
            self.assertEqual(snap_rec["seed"], "0xdeadbeef")
            self.assertEqual(snap_rec["listed_recnos"], [1])
            self.assertEqual(trans_rec["kind"], "transition")
            self.assertEqual(trans_rec["best_k"], 50)
            self.assertEqual(trans_rec["matched"], 1)
            self.assertEqual(trans_rec["total"], 1)

    def test_logger_with_no_path_is_silent(self):
        logger = lm.EventLogger(None)
        # Should accept writes without raising; no file is created.
        s = StockInput(
            group_recno=1, is_listed=True, is_foreign=False, group_type=1,
        )
        forecast = predict_all_stocks(MiscRNG(1), [s])
        snap = lm.Snapshot(
            wall_time=0.0, seed=1, inputs=[s], listed_recnos=[1],
            forecast=forecast,
        )
        logger.snapshot(snap)  # no-op
        logger.close()


# ---------------------------------------------------------------------------
# Renderer (plain-text path — no rich dependency for correctness)
# ---------------------------------------------------------------------------


class PlainRendererTests(unittest.TestCase):
    """The plain-text renderer should never crash on empty or partial data."""

    def _render_to_string(self, **kwargs) -> str:
        renderer = lm.Renderer(use_tui=False)
        # Redirect stdout; the renderer writes directly via print.
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            renderer.render(**kwargs)
        renderer.close()
        return buf.getvalue()

    def test_renders_first_snapshot_without_transition(self):
        s = StockInput(
            group_recno=1, is_listed=True, is_foreign=False, group_type=1,
            base_stock_price=10.0, sentiment=2.0,
        )
        forecast = predict_all_stocks(MiscRNG(0x12345), [s])
        snap = lm.Snapshot(
            wall_time=0.0, seed=0x12345, inputs=[s], listed_recnos=[1],
            forecast=forecast,
        )
        out = self._render_to_string(
            reader=_FakeReader(), last=snap,
            stats=lm.RunningStats(), last_transition=None,
            status="idle",
        )
        self.assertIn("seed=0x00012345", out)
        self.assertIn("g1", out)
        # No transition yet, so the diff columns are dashes.
        self.assertIn("—", out)

    def test_renders_with_no_last_snapshot(self):
        out = self._render_to_string(
            reader=_FakeReader(), last=None,
            stats=lm.RunningStats(), last_transition=None,
            status="attaching",
        )
        self.assertIn("status=attaching", out)


# ---------------------------------------------------------------------------
# _compute_ia_vector — deprecated, returns None
# ---------------------------------------------------------------------------


class ComputeIaVectorTests(unittest.TestCase):
    """``_compute_ia_vector`` is deprecated and always returns None.

    The old formula pre-computed a per-group ia dict from game_date.  It
    had two bugs (startup-gate probability ignored, indexing off-by-one) that
    caused Run 5 regression (73.3% → 64.1%).  Reverted (2026-04-21) to the
    scalar 2-D sweep baseline (73.3%).  This function is now a no-op stub.
    """

    def _stocks(self, recnos):
        return [
            StockInput(
                group_recno=r,
                is_listed=True,
                is_foreign=False,
                group_type=1,
            )
            for r in recnos
        ]

    def test_unknown_game_date_returns_none(self):
        """Always returns None regardless of game_date."""
        stocks = self._stocks(range(2, 23))
        self.assertIsNone(lm._compute_ia_vector(stocks, 0))

    def test_known_game_date_also_returns_none(self):
        """Returns None even for a valid game_date (deprecated stub)."""
        stocks = self._stocks(range(2, 23))
        self.assertIsNone(lm._compute_ia_vector(stocks, 13))

    def test_all_game_dates_return_none(self):
        """Exhaustive: every gd 0..60 returns None."""
        stocks = self._stocks(range(2, 23))
        for gd in range(61):
            self.assertIsNone(lm._compute_ia_vector(stocks, gd),
                              f"Expected None for gd={gd}")


if __name__ == "__main__":
    unittest.main()
