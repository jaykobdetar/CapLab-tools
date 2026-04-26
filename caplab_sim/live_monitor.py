"""Live stock-market monitor and prediction-match tracker.

Attaches to a running ``CapMain.exe`` via :class:`caplab_sim.rng_reader
.LiveGameReader`, repeatedly snapshots the RNG seed and per-stock state,
runs :func:`caplab_sim.stock.predict_all_stocks` off the current seed,
and — when the game advances a day — compares the *previous* snapshot's
prediction against the *new* snapshot's observation.

What this script answers
------------------------
*   "Does ``stock.py`` predict the game's next day correctly, live?"
*   "How often do predictions match per day, and for which groups?"
*   "Which pre-stock RNG offset ``k`` lines up with the observed post-
    tick seed on each real day?"  (Same sweep logic as
    ``validate_stocks.py``.)
*   "Where is the gap — is it the 0-firm-group pattern from the
    2026-04-21 very-very-late diff (save-replay limitation caused by
    pre-tick FirmArray asset mutation), or something new?"

Usage
-----
    python -m caplab_sim.live_monitor                 # default: 0.5s poll, no log
    python -m caplab_sim.live_monitor --poll 1.0      # one poll per second
    python -m caplab_sim.live_monitor --log run.jsonl # append per-event JSON
    python -m caplab_sim.live_monitor --no-tui        # plain-text output
    python -m caplab_sim.live_monitor --once          # snapshot + forward prediction once

Design
------
The monitor keeps two pieces of state in memory:

*   ``last`` — the most-recent-successful snapshot (``seed``, ``inputs``,
    and a cached forward prediction at ``k=0`` for the user-facing
    "what will happen if the game ticks now?" view).
*   ``stats`` — cumulative per-day tally (day transitions observed,
    listed stocks matched, and per-group hit counts keyed by recno).

On each poll:

1.  Read ``seed`` and ``inputs`` (listed-domestic groups only, to stay
    apples-to-apples with ``_____001/_____002`` validation and to keep
    the TUI narrow).
2.  Build a forward prediction from ``inputs`` and the current ``seed``
    — this is the "next-day forecast" shown in the UI.
3.  Compare the new ``inputs`` against the *previous snapshot's*
    forward prediction to detect whether a day transition happened:
    if the seed changed AND at least one listed stock's price or
    sentiment moved, treat it as a day transition.
4.  On a day transition: run the same sweep-``k`` loop
    :mod:`caplab_sim.validate_stocks` uses.  The best ``k`` and its
    per-stock diff list are fed into the stats and rendered.

The sweep uses a 2-D grid over
``k_initial × inter_group_advance ∈ [0, max_inter_advance]``
(grid size 30 × 6 = 180 candidates).  The empirically validated best value
for the reference scenario is ``k_initial=0, inter_advance=0``
(98.2% live match rate confirmed 2026-04-21 Run 8).

Limitations (as of 2026-04-21)
------------------------------
*   Case-2 asset block (Group+0x39D8..+0x3A28) **IS now read** from live
    memory.  :meth:`LiveGameReader.stock_inputs` calls
    :func:`caplab_sim.rng_reader._read_group_assets` for every group,
    which reads corp_cash (Group+0x0CC0), the 10-double account_balance
    block (Group+0x39D8..+0x3A28), and earnings_metric_driver
    (Group+0x3B38).  Firm-owning groups that previously yielded zero
    weighted-BVPS and thus always failed case-2 prediction now receive
    correct asset values.  See SESSION_LOG
    "ENTRY — 2026-04-21 case-2 asset block live read" for results.
*   CPI defaults to 1.0.  Day-1 saves are on-the-money; later
    scenarios where CPI has drifted need an Economy+0x68 read.
*   HQ-type is 0 everywhere — the pe_ratio gate in cases 0/1 uses the
    stored ``pe_ratio`` field so this only bites the bank/insurance
    special case once we get to firm-owning runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

from caplab_sim.rng import MiscRNG, lcg_step, simulate_lcg, distance_if_reachable
from caplab_sim.rng_reader import LiveGameReader, ScenarioFlags
from caplab_sim.stock import (
    DEFAULT_CPI,
    MarketOutput,
    ScenarioState,
    StockInput,
    StockOutput,
    predict_all_stocks,
)


# ---------------------------------------------------------------------------
# Match tolerances — match validate_stocks exactly so a "21/21 here" means
# the same thing as "21/21 there".
# ---------------------------------------------------------------------------

PRICE_TOLERANCE: float = 5e-4
SENTIMENT_TOLERANCE: float = 1e-2


# ---------------------------------------------------------------------------
# In-memory records
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """One moment-in-time readout of the game's live state.

    Captured once per poll — the monitor keeps only the most recent
    snapshot for comparison against the next one.  ``forecast`` is the
    prediction this snapshot's seed+inputs produce at ``k=0``; once the
    game ticks we compare ``forecast`` (from snapshot N) against the
    observed inputs in snapshot N+1.  If that comparison misses, we
    sweep ``k`` to find the actual pre-stock offset.

    ``game_date`` is read from ``DAT_00a459b0`` at snapshot time.  The
    stock tick and GroupAI functions run BEFORE the engine increments
    game_date, so this value is the date under which the upcoming tick
    executes — correct for the 30-day stagger check.  0 means the read
    failed (game exited / unmapped).

    ``game_difficulty`` is the startup-gate modulus read from
    ``DAT_00a449a8`` via :meth:`LiveGameReader.read_game_difficulty`.
    Stored for logging/diagnostics only; ``sweep_best_k`` always uses
    the 2-D scalar sweep regardless of this value.

    ``ia_vector`` is deprecated and always None; kept for backward
    compatibility with old JSONL replays.
    """

    wall_time: float
    seed: int
    inputs: List[StockInput]
    listed_recnos: List[int]       # sorted descending (matches engine order)
    forecast: MarketOutput         # predict_all_stocks(MiscRNG(seed), inputs)
    game_date: int = 0             # DAT_00a459b0; 0 = unknown
    game_difficulty: int = 0       # startup_gate_modulus (2=Normal, 4=Easy, 0=unknown)
    scenario_flags: Optional[ScenarioFlags] = None  # DLC flag snapshot (2026-04-22)
    ia_vector: Optional[dict] = None  # Dict[int,int] or None (deprecated; use game_difficulty)

    def scenario_state(self) -> ScenarioState:
        """Build a :class:`ScenarioState` for ``ai_tail_advance``.

        Merges the snapshot's ``game_date`` with the cached scenario
        flags.  When ``scenario_flags`` is None (old-style snapshot
        from a JSONL replay that pre-dates the 2026-04-22 change) the
        returned state has both DLC flags false — which degrades
        cleanly to the base-game AI-tail path inside ai_tail_advance.
        """
        flags = self.scenario_flags or ScenarioFlags()
        return ScenarioState(
            game_date=self.game_date,
            dlc_economy_mode=flags.dlc_economy_mode,
            dlc_stock_market_active=flags.dlc_stock_market_active,
        )

    def price_of(self, recno: int) -> float:
        for s in self.inputs:
            if s.group_recno == recno:
                return s.base_stock_price
        return float("nan")

    def sentiment_of(self, recno: int) -> float:
        for s in self.inputs:
            if s.group_recno == recno:
                return s.sentiment
        return float("nan")

    def firms_of(self, recno: int) -> int:
        # StockInput doesn't carry firm_count; live reader doesn't
        # expose it yet.  Returns -1 meaning "unknown".  If a future
        # rng_reader upgrade populates this we can surface it in the UI.
        return -1


@dataclass
class TransitionDiff:
    """Per-group diff for a single observed day transition.

    Mirrors :class:`caplab_sim.validate_stocks.StockDiff` in shape so
    downstream consumers can use the same match/mismatch logic.
    """

    group_recno: int
    predicted_price: float
    observed_price: float
    predicted_sentiment: float
    observed_sentiment: float

    @property
    def price_error(self) -> float:
        if math.isnan(self.predicted_price) or math.isnan(self.observed_price):
            return float("inf")
        return abs(self.predicted_price - self.observed_price)

    @property
    def sentiment_error(self) -> float:
        if math.isnan(self.predicted_sentiment) or math.isnan(self.observed_sentiment):
            return float("inf")
        return abs(self.predicted_sentiment - self.observed_sentiment)

    @property
    def matches(self) -> bool:
        return (
            self.price_error < PRICE_TOLERANCE
            and self.sentiment_error < SENTIMENT_TOLERANCE
        )


@dataclass
class TransitionResult:
    """Outcome of a single day transition: best-(k, inter_advance) sweep + per-group diff."""

    wall_time: float
    prev_seed: int
    new_seed: int
    seed_distance: Optional[int]       # LCG steps prev -> new, if reachable
    best_k: int
    best_inter_advance: int            # best inter-group GroupAI RNG advance (steps)
    best_rng_calls: int
    diffs: List[TransitionDiff]

    @property
    def n_matched(self) -> int:
        return sum(1 for d in self.diffs if d.matches)

    @property
    def n_total(self) -> int:
        return len(self.diffs)

    @property
    def fraction(self) -> float:
        return self.n_matched / self.n_total if self.n_total else 0.0


@dataclass
class RunningStats:
    """Cumulative match stats across every transition this run."""

    transitions: int = 0
    total_matched: int = 0
    total_observed: int = 0
    # per-recno: (matched_count, observed_count)
    per_group: Dict[int, List[int]] = field(default_factory=dict)
    # best-k sweep histogram — key = k value, value = count.  Useful for
    # spotting whether the pre-stock offset drifts day to day.
    k_histogram: Dict[int, int] = field(default_factory=dict)
    # inter-group advance histogram — key = inter_advance, value = count.
    inter_advance_histogram: Dict[int, int] = field(default_factory=dict)

    def ingest(self, res: TransitionResult) -> None:
        self.transitions += 1
        self.total_matched += res.n_matched
        self.total_observed += res.n_total
        self.k_histogram[res.best_k] = self.k_histogram.get(res.best_k, 0) + 1
        ia = res.best_inter_advance
        self.inter_advance_histogram[ia] = self.inter_advance_histogram.get(ia, 0) + 1
        for d in res.diffs:
            slot = self.per_group.setdefault(d.group_recno, [0, 0])
            slot[1] += 1
            if d.matches:
                slot[0] += 1

    @property
    def fraction(self) -> float:
        return (
            self.total_matched / self.total_observed
            if self.total_observed else 0.0
        )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _listed_domestic(inputs: Sequence[StockInput]) -> List[StockInput]:
    """Filter to listed-domestic stocks (the subset the predictor validates)."""
    return [s for s in inputs if s.is_listed and not s.is_foreign]


def _state_changed(prev: Snapshot, cur_inputs: Sequence[StockInput],
                   cur_seed: int) -> bool:
    """True iff ``prev`` and ``(cur_inputs, cur_seed)`` look like different days.

    We require **both** a seed change and at least one listed-stock
    state field to move.  Seed alone is noisy — menu Misc instances
    share the seed field via the writeback at the end of
    ``Misc::random`` — but paired with any price/sentiment move it's
    a reliable day-boundary signal.
    """
    if prev.seed == cur_seed:
        return False
    prev_by_recno = {s.group_recno: s for s in _listed_domestic(prev.inputs)}
    for s in _listed_domestic(cur_inputs):
        p = prev_by_recno.get(s.group_recno)
        if p is None:
            # New listed stock appeared — definitely a state change.
            return True
        if abs(p.base_stock_price - s.base_stock_price) > PRICE_TOLERANCE:
            return True
        if abs(p.sentiment - s.sentiment) > SENTIMENT_TOLERANCE:
            return True
    return False


# ---------------------------------------------------------------------------
# Sweep — same logic validate_stocks uses, factored for reuse.
# ---------------------------------------------------------------------------

def _compute_ia_vector(
    stocks: Sequence[StockInput], game_date: int
) -> Optional[Dict[int, int]]:
    """[DEPRECATED] No-op stub — always returns None.

    Previously computed a per-group ``inter_group_advance`` dict from
    ``game_date`` (loan-management 30-day stagger).  The attempt had two bugs
    (gate probability ignored, indexing off-by-one) that caused a regression
    from 73.3% → 64.1% (Run 5).

    Reverted (2026-04-21) to the scalar 2-D sweep baseline (73.3%).  The
    call site in ``_capture_snapshot`` is retained so the code compiles
    unchanged; the return value is never used.
    """
    return None


def sweep_best_k(
    prev: Snapshot,
    cur_inputs: Sequence[StockInput],
    *,
    cpi: float,
    max_k: int,
    max_inter_advance: int = 5,
    hint_k: Optional[int] = None,
    hint_ia: Optional[int] = None,
) -> TransitionResult:
    """Sweep over ``k_initial`` to find the best pre-stock seed offset.

    Uses a 2-D grid over
    ``k_initial ∈ [0, max_k)`` × ``inter_group_advance ∈ [0, max_inter_advance]``
    (grid size ``max_k × (max_inter_advance + 1)``).

    ``hint_k`` / ``hint_ia`` — warm-start hint (previous transition's
    best values).  Tried first with zero overhead; on a perfect match the
    function returns immediately, skipping the full grid scan.

    Ties are broken: fewer mismatches first, then smaller cumulative error.
    A perfect match (0 mismatches) short-circuits immediately.
    """
    obs_listed = _listed_domestic(cur_inputs)
    obs_by_recno = {s.group_recno: s for s in obs_listed}

    diffs_best: List[TransitionDiff] = []
    best_k = 0
    best_inter_advance = 0
    best_rng_calls = 0
    best_score: Tuple[int, float] = (10**9, float("inf"))

    # Scenario state for ai_tail_advance (inside predict_all_stocks).
    # Pulled from the PREVIOUS snapshot because the transition is
    # "what did the engine do under last snapshot's conditions?"; the
    # game_date field on prev is the date the tick just executed under.
    scenario_state = prev.scenario_state()

    # Warm-start: try the last known-good (k, ia) before scanning the full
    # grid.  On stable days this is a single predict_all_stocks call (~0.1 ms)
    # instead of up to 180 (~24 ms).
    if hint_k is not None and hint_ia is not None:
        hint_seed = simulate_lcg(prev.seed, hint_k) if hint_k > 0 else prev.seed
        rng = MiscRNG(hint_seed)
        out = predict_all_stocks(rng, prev.inputs, cpi=cpi,
                                 inter_group_advance=hint_ia,
                                 scenario_state=scenario_state)
        pred_by_recno = {p.group_recno: p for p in out.per_stock}
        diffs = _build_diffs(obs_by_recno, pred_by_recno)
        score = _score(diffs)
        if score[0] == 0:
            # Perfect match on hint — return immediately, no grid scan needed.
            return TransitionResult(
                wall_time=time.time(),
                prev_seed=prev.seed,
                new_seed=0,
                seed_distance=None,
                best_k=hint_k,
                best_inter_advance=hint_ia,
                best_rng_calls=out.total_rng_calls,
                diffs=diffs,
            )
        # Not perfect — use hint result as initial best before grid scan.
        best_score = score
        diffs_best = diffs
        best_k = hint_k
        best_inter_advance = hint_ia
        best_rng_calls = out.total_rng_calls

    found_perfect = False
    for ia_val in range(max_inter_advance + 1):
        seed = prev.seed
        for k in range(max_k):
            rng = MiscRNG(seed)
            out = predict_all_stocks(
                rng, prev.inputs, cpi=cpi,
                inter_group_advance=ia_val,
                scenario_state=scenario_state,
            )
            pred_by_recno = {p.group_recno: p for p in out.per_stock}
            diffs = _build_diffs(obs_by_recno, pred_by_recno)
            score = _score(diffs)
            if score < best_score:
                best_score = score
                diffs_best = diffs
                best_k = k
                best_inter_advance = ia_val
                best_rng_calls = out.total_rng_calls
                if score[0] == 0:
                    found_perfect = True
                    break
            seed = lcg_step(seed)
        if found_perfect:
            break

    return TransitionResult(
        wall_time=time.time(),
        prev_seed=prev.seed,
        new_seed=0,                 # filled in by caller (needs live seed read)
        seed_distance=None,         # filled in by caller
        best_k=best_k,
        best_inter_advance=best_inter_advance,
        best_rng_calls=best_rng_calls,
        diffs=diffs_best,
    )


def _build_diffs(
    obs_by_recno: Dict[int, StockInput],
    pred_by_recno: Dict[int, StockOutput],
) -> List[TransitionDiff]:
    """Pair observation with prediction; NaN-fill any missing side.

    A group that the predictor dropped (deleted / foreign / unlisted)
    is reported with NaN prediction so the diff is visibly a mismatch
    instead of silently dropping it.
    """
    diffs: List[TransitionDiff] = []
    for recno in sorted(obs_by_recno.keys(), reverse=True):
        obs = obs_by_recno[recno]
        pred = pred_by_recno.get(recno)
        if pred is None:
            diffs.append(TransitionDiff(
                group_recno=recno,
                predicted_price=float("nan"),
                observed_price=obs.base_stock_price,
                predicted_sentiment=float("nan"),
                observed_sentiment=obs.sentiment,
            ))
            continue
        diffs.append(TransitionDiff(
            group_recno=recno,
            predicted_price=pred.base_stock_price,
            observed_price=obs.base_stock_price,
            predicted_sentiment=pred.sentiment,
            observed_sentiment=obs.sentiment,
        ))
    return diffs


def _score(diffs: Sequence[TransitionDiff]) -> Tuple[int, float]:
    """Sort key: (fewer mismatches, smaller cumulative error)."""
    mismatches = sum(1 for d in diffs if not d.matches)
    total_err = sum(
        (d.price_error if d.price_error != float("inf") else 1e9)
        + 0.01 * (d.sentiment_error if d.sentiment_error != float("inf") else 1e9)
        for d in diffs
    )
    return (mismatches, total_err)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

class Renderer:
    """Output strategy — rich TUI if available, plain text otherwise.

    The TUI draws a single live table in place; the plain-text
    renderer prints a fresh block per poll.  Both render the same
    content so the CLI flag is cosmetic.
    """

    def __init__(self, use_tui: bool) -> None:
        self.use_tui = use_tui
        self._live = None
        self._console = None
        self._table_builder = None
        if use_tui:
            try:
                from rich.console import Console
                from rich.live import Live
                self._console = Console()
                # auto_refresh=False stops Rich's background thread from
                # painting placeholder frames before the first real
                # update (otherwise each early frame draws a narrower
                # Panel than the next, leaving stacked top-border
                # artifacts because cursor-return-and-overwrite can't
                # handle width-growing content cleanly).  We call
                # refresh() explicitly after each update().
                # vertical_overflow="visible" prevents Rich from
                # silently truncating the table when the terminal is
                # shorter than the rendered content.
                self._live = Live(
                    console=self._console,
                    auto_refresh=False,
                    screen=False,
                    transient=False,
                    vertical_overflow="visible",
                )
                self._live.__enter__()
            except ImportError:
                self.use_tui = False

    def render(
        self,
        *,
        reader: LiveGameReader,
        last: Optional[Snapshot],
        stats: RunningStats,
        last_transition: Optional[TransitionResult],
        status: str,
    ) -> None:
        if self.use_tui:
            self._render_tui(reader, last, stats, last_transition, status)
        else:
            self._render_plain(reader, last, stats, last_transition, status)

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass

    # -- rich TUI ----------------------------------------------------------

    def _render_tui(
        self,
        reader: LiveGameReader,
        last: Optional[Snapshot],
        stats: RunningStats,
        last_transition: Optional[TransitionResult],
        status: str,
    ) -> None:
        from rich.console import Group as RichGroup
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # -- top header ----------------------------------------------------
        header_lines = [
            f"pid={reader.pid}  base=0x{reader.base:08x}  misc=0x{reader.misc_addr:08x}",
        ]
        if last is not None:
            header_lines.append(
                f"seed=0x{last.seed:08x}  listed-domestic={len(last.listed_recnos)}  "
                f"status={status}"
            )
        else:
            header_lines.append(f"status={status}")

        if last_transition is not None:
            d = last_transition
            ia_s = "sim" if d.best_inter_advance == -1 else str(d.best_inter_advance)
            header_lines.append(
                f"last transition: k={d.best_k} ia={ia_s} "
                f"stock_rng_calls={d.best_rng_calls} "
                f"{d.n_matched}/{d.n_total} match "
                f"seed_distance={d.seed_distance}"
            )

        ia_hist = dict(sorted(stats.inter_advance_histogram.items()))
        vector_days = ia_hist.pop(-1, 0)
        ia_mode = (f"sim={vector_days}d" if vector_days else
                   f"scalar={ia_hist}")
        header_lines.append(
            f"cumulative: transitions={stats.transitions}  "
            f"matched={stats.total_matched}/{stats.total_observed} "
            f"({stats.fraction:.1%})  "
            f"k_hist={dict(sorted(stats.k_histogram.items()))}  "
            f"ia={ia_mode}"
        )
        header = Text("\n".join(header_lines))

        # -- main table ----------------------------------------------------
        # expand=True pins the Table to the full terminal width so early
        # empty-table frames aren't narrower than later populated ones —
        # otherwise Rich's overwrite can't cleanly erase the previous
        # narrower frame and the top-border char stacks.
        table = Table(show_header=True, header_style="bold",
                      title="Live stock monitor", title_style="bold",
                      expand=True)
        table.add_column("recno", justify="right", style="dim")
        table.add_column("price", justify="right")
        table.add_column("sent", justify="right")
        table.add_column("pred+1 price", justify="right")
        table.add_column("pred+1 sent", justify="right")
        table.add_column("last Δprice", justify="right")
        table.add_column("last Δsent", justify="right")
        table.add_column("last match", justify="center")
        table.add_column("hit-rate", justify="right")

        per_recno_diff: Dict[int, TransitionDiff] = {}
        if last_transition is not None:
            per_recno_diff = {d.group_recno: d for d in last_transition.diffs}

        if last is not None:
            forecast_by_recno = {p.group_recno: p for p in last.forecast.per_stock}
            for recno in last.listed_recnos:  # already descending
                s = next(
                    (x for x in last.inputs if x.group_recno == recno), None
                )
                if s is None:
                    continue
                pred = forecast_by_recno.get(recno)
                pred_price_s = f"{pred.base_stock_price:.4f}" if pred else "—"
                pred_sent_s = f"{pred.sentiment:+.2f}" if pred else "—"
                td = per_recno_diff.get(recno)
                if td is None:
                    dp_s, ds_s, m_s = "—", "—", "—"
                else:
                    dp_s = f"{td.predicted_price - td.observed_price:+.4f}"
                    ds_s = f"{td.predicted_sentiment - td.observed_sentiment:+.2f}"
                    m_s = "OK" if td.matches else "MISS"
                pg = stats.per_group.get(recno)
                if pg and pg[1]:
                    hr_s = f"{pg[0]}/{pg[1]}"
                else:
                    hr_s = "—"
                row_style = None
                if td is not None and not td.matches:
                    row_style = "yellow"
                table.add_row(
                    f"g{recno}",
                    f"{s.base_stock_price:.4f}",
                    f"{s.sentiment:+.2f}",
                    pred_price_s,
                    pred_sent_s,
                    dp_s,
                    ds_s,
                    m_s,
                    hr_s,
                    style=row_style,
                )

        assert self._live is not None
        # expand=True on the Panel so its border snaps to terminal width
        # from the very first frame — with auto_refresh=False the only
        # frames Rich draws are the ones we drive here, but width must
        # still be stable frame-to-frame because the renderable grows
        # from 0 rows (pre-snapshot) to N rows (post-snapshot).
        self._live.update(
            RichGroup(
                Panel(header, border_style="cyan", expand=True),
                table,
            ),
            refresh=True,
        )

    # -- plain text --------------------------------------------------------

    def _render_plain(
        self,
        reader: LiveGameReader,
        last: Optional[Snapshot],
        stats: RunningStats,
        last_transition: Optional[TransitionResult],
        status: str,
    ) -> None:
        # ANSI clear-screen + home (matches `clear` behaviour without
        # invoking a subprocess).  Only emitted when stdout is a tty,
        # so piping to a file still produces a linear log.
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")

        print(
            f"pid={reader.pid} base=0x{reader.base:08x} "
            f"misc=0x{reader.misc_addr:08x}  status={status}"
        )
        if last is not None:
            print(
                f"seed=0x{last.seed:08x}  listed-domestic={len(last.listed_recnos)}"
            )
        if last_transition is not None:
            d = last_transition
            ia_s = "sim" if d.best_inter_advance == -1 else str(d.best_inter_advance)
            print(
                f"last transition: k={d.best_k} ia={ia_s} "
                f"stock_rng_calls={d.best_rng_calls} "
                f"{d.n_matched}/{d.n_total} match seed_distance={d.seed_distance}"
            )
        ia_hist_display = dict(sorted(stats.inter_advance_histogram.items()))
        ia_vec_d = ia_hist_display.pop(-1, 0)
        ia_mode = (f"sim={ia_vec_d}d" if ia_vec_d else
                   f"scalar={ia_hist_display}")
        print(
            f"cumulative: transitions={stats.transitions}  "
            f"matched={stats.total_matched}/{stats.total_observed} "
            f"({stats.fraction:.1%})  "
            f"k_hist={dict(sorted(stats.k_histogram.items()))}  "
            f"ia={ia_mode}"
        )

        if last is None:
            return

        per_recno_diff: Dict[int, TransitionDiff] = {}
        if last_transition is not None:
            per_recno_diff = {d.group_recno: d for d in last_transition.diffs}
        forecast_by_recno = {p.group_recno: p for p in last.forecast.per_stock}

        print(
            f"\n{'recno':>5} {'price':>10} {'sent':>8} "
            f"{'pred+1 price':>14} {'pred+1 sent':>12} "
            f"{'Δprice':>9} {'Δsent':>8} match"
        )
        print("-" * 90)
        for recno in last.listed_recnos:
            s = next((x for x in last.inputs if x.group_recno == recno), None)
            if s is None:
                continue
            pred = forecast_by_recno.get(recno)
            pp = f"{pred.base_stock_price:10.4f}" if pred else f"{'—':>10}"
            ps = f"{pred.sentiment:+8.2f}" if pred else f"{'—':>8}"
            td = per_recno_diff.get(recno)
            if td is None:
                dp = f"{'—':>9}"
                ds = f"{'—':>8}"
                m = "—"
            else:
                dp = f"{td.predicted_price - td.observed_price:+9.4f}"
                ds = f"{td.predicted_sentiment - td.observed_sentiment:+8.2f}"
                m = "OK" if td.matches else "MISS"
            print(
                f"g{recno:<4d} {s.base_stock_price:10.4f} {s.sentiment:+8.2f} "
                f"{pp} {ps} {dp} {ds} {m}"
            )


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

class EventLogger:
    """Append-only JSONL writer — one record per event.

    Two event kinds are emitted: ``snapshot`` (one per successful poll)
    and ``transition`` (one per detected day change with full diff).
    Flushed on every write so a crash mid-run still leaves a usable
    log behind.
    """

    def __init__(self, path: Optional[str]) -> None:
        self._fp: Optional[TextIO] = None
        if path:
            self._fp = open(path, "a", buffering=1)  # line-buffered

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _write(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._fp is None:
            return
        rec = {"kind": kind, "ts": time.time(), **payload}
        self._fp.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def snapshot(self, snap: Snapshot) -> None:
        self._write("snapshot", {
            "seed": f"0x{snap.seed:08x}",
            "game_date": snap.game_date,
            "game_difficulty": snap.game_difficulty,
            "listed_recnos": snap.listed_recnos,
            "prices": {
                s.group_recno: s.base_stock_price
                for s in _listed_domestic(snap.inputs)
            },
            "sentiments": {
                s.group_recno: s.sentiment
                for s in _listed_domestic(snap.inputs)
            },
            "forecast_prices": {
                p.group_recno: p.base_stock_price for p in snap.forecast.per_stock
            },
            "forecast_sentiments": {
                p.group_recno: p.sentiment for p in snap.forecast.per_stock
            },
            "forecast_rng_calls": snap.forecast.total_rng_calls,
        })

    def transition(self, res: TransitionResult) -> None:
        ia_mode = "exact_sim" if res.best_inter_advance == -1 else "scalar"
        self._write("transition", {
            "prev_seed": f"0x{res.prev_seed:08x}",
            "new_seed": f"0x{res.new_seed:08x}",
            "seed_distance": res.seed_distance,
            "best_k": res.best_k,
            "best_inter_advance": res.best_inter_advance,
            "ia_mode": ia_mode,
            "stock_rng_calls": res.best_rng_calls,
            "matched": res.n_matched,
            "total": res.n_total,
            "diffs": [
                {
                    "recno": d.group_recno,
                    "pred_price": d.predicted_price,
                    "obs_price": d.observed_price,
                    "pred_sent": d.predicted_sentiment,
                    "obs_sent": d.observed_sentiment,
                    "match": d.matches,
                }
                for d in res.diffs
            ],
        })


# ---------------------------------------------------------------------------
# Core poll loop
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    poll_seconds: float = 0.1           # 0.1s default; use 0.05 for speed 10+
    max_k: int = 30                     # pre-first-group k; ~0 in practice
    max_inter_advance: int = 5          # inter-group GroupAI budget sweep
    cpi: float = DEFAULT_CPI
    log_path: Optional[str] = None
    use_tui: bool = True
    run_once: bool = False
    max_seed_distance: int = 1000       # cap for distance_if_reachable


def _capture_snapshot(reader: LiveGameReader, cfg: MonitorConfig) -> Snapshot:
    """One read-and-forecast cycle.

    Reads the seed first so we're pinning "the seed the forecast is
    made from"; the inputs and game_date come next.  The forecast itself
    is built immediately with ``reorder=True`` so the per-stock order
    in ``forecast.per_stock`` matches the engine's descending-recno
    firing order — keeps the UI stable frame to frame.

    ``game_date`` is read from ``DAT_00a459b0`` (via
    :meth:`LiveGameReader.read_game_date`) and stored in the snapshot.
    The stock tick fires BEFORE the engine increments game_date, so the
    value captured here is correct for the ia vector calculation on the
    NEXT detected transition (which is derived from the *previous*
    snapshot's game_date).

    ``game_difficulty`` is read from ``DAT_00a449a8`` and stored for
    logging/diagnostics only.  ``sweep_best_k`` always uses the scalar
    2-D grid regardless of this value.

    ``ia_vector`` is always None (deprecated stub; see
    ``_compute_ia_vector``).
    """
    seed = reader.read_rng_seed()
    inputs = reader.stock_inputs()
    game_date = reader.read_game_date()
    game_difficulty = reader.read_game_difficulty()
    scenario_flags = reader.read_scenario_flags()
    listed = _listed_domestic(inputs)
    listed_recnos = sorted((s.group_recno for s in listed), reverse=True)
    ia_vector = _compute_ia_vector(inputs, game_date)   # returns None (deprecated)
    rng = MiscRNG(seed)
    scenario_state = ScenarioState(
        game_date=game_date,
        dlc_economy_mode=scenario_flags.dlc_economy_mode,
        dlc_stock_market_active=scenario_flags.dlc_stock_market_active,
    )
    forecast = predict_all_stocks(
        rng, inputs, cpi=cfg.cpi, scenario_state=scenario_state,
    )
    return Snapshot(
        wall_time=time.time(),
        seed=seed,
        inputs=inputs,
        listed_recnos=listed_recnos,
        forecast=forecast,
        game_date=game_date,
        game_difficulty=game_difficulty,
        scenario_flags=scenario_flags,
        ia_vector=ia_vector,
    )


def _handle_transition(
    prev: Snapshot,
    cur_seed: int,
    cur_inputs: Sequence[StockInput],
    cfg: MonitorConfig,
    *,
    hint_k: Optional[int] = None,
    hint_ia: Optional[int] = None,
) -> TransitionResult:
    """Build a full TransitionResult for a detected day transition.

    Runs the 2-D scalar sweep over ``(k_initial, inter_group_advance)``.
    ``hint_k`` / ``hint_ia`` — warm-start from the previous transition's
    best values.
    """
    res = sweep_best_k(
        prev, cur_inputs,
        cpi=cfg.cpi,
        max_k=cfg.max_k,
        max_inter_advance=cfg.max_inter_advance,
        hint_k=hint_k,
        hint_ia=hint_ia,
    )
    res.new_seed = cur_seed
    res.seed_distance = distance_if_reachable(
        prev.seed, cur_seed, cfg.max_seed_distance
    )
    return res


def run(cfg: MonitorConfig) -> int:
    try:
        reader = LiveGameReader.attach()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stats = RunningStats()
    logger = EventLogger(cfg.log_path)
    renderer = Renderer(use_tui=cfg.use_tui)

    # Ctrl+C should tear down rich/the log cleanly — signal handler
    # flips a flag that the main loop checks on each iteration.
    stop = {"flag": False}

    def _on_sigint(_signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    last: Optional[Snapshot] = None
    last_transition: Optional[TransitionResult] = None
    last_best_k: Optional[int] = None    # warm-start hint for next sweep
    last_best_ia: Optional[int] = None
    status = "attaching"

    try:
        while not stop["flag"]:
            try:
                seed = reader.read_rng_seed()
                inputs = reader.stock_inputs()
            except OSError as exc:
                status = f"read error: {exc}"
                renderer.render(reader=reader, last=last, stats=stats,
                                last_transition=last_transition, status=status)
                if cfg.run_once:
                    return 3
                time.sleep(cfg.poll_seconds)
                continue

            # Detect transition BEFORE overwriting `last`.  Compare against
            # the cached prediction: the prior snapshot's forecast was
            # generated from a seed-at-that-moment, and the observed post-
            # tick seed has now moved forward by some k+stock_rng_calls.
            if last is not None and _state_changed(last, inputs, seed):
                status = "transition"
                res = _handle_transition(
                    last, seed, inputs, cfg,
                    hint_k=last_best_k, hint_ia=last_best_ia,
                )
                stats.ingest(res)
                logger.transition(res)
                last_transition = res
                last_best_k = res.best_k
                last_best_ia = res.best_inter_advance
            else:
                status = "idle" if last is not None else "first snapshot"

            snap = _capture_snapshot(reader, cfg)
            # Re-use the just-read seed/inputs when possible to avoid
            # double-reading the process memory — we already have them.
            # (_capture_snapshot re-reads to get an atomic pair; for a
            # busy poll loop the cost is negligible, and the re-read
            # guarantees the forecast is derived from the seed the
            # caller will see in the UI.)
            logger.snapshot(snap)
            last = snap

            renderer.render(reader=reader, last=last, stats=stats,
                            last_transition=last_transition, status=status)

            if cfg.run_once:
                break

            # Sleep in small chunks so Ctrl+C is responsive even at fast
            # poll rates.  Chunk size is 25 ms so the stop flag is checked
            # ~40× per second regardless of the poll interval.
            _CHUNK = 0.025
            elapsed = 0.0
            while elapsed < cfg.poll_seconds and not stop["flag"]:
                chunk = min(_CHUNK, cfg.poll_seconds - elapsed)
                time.sleep(chunk)
                elapsed += chunk
    finally:
        renderer.close()
        logger.close()

    # Final summary to stdout (separate from the in-place TUI rendering).
    print()
    print("--- monitor summary ---")
    print(f"transitions observed: {stats.transitions}")
    print(f"cumulative match:     {stats.total_matched}/{stats.total_observed} "
          f"({stats.fraction:.1%})")
    if stats.k_histogram:
        print(f"pre-stock k histogram:          "
              f"{dict(sorted(stats.k_histogram.items()))}")
    if stats.inter_advance_histogram:
        ia_hist = dict(sorted(stats.inter_advance_histogram.items()))
        print(f"inter-group advance histogram:  {ia_hist}")
    if stats.per_group:
        # Sorted by lowest hit-rate first so problems are at the top.
        print("per-group hit rates (worst first):")
        rows = []
        for recno, (hit, total) in stats.per_group.items():
            rate = hit / total if total else 0.0
            rows.append((rate, recno, hit, total))
        for rate, recno, hit, total in sorted(rows):
            print(f"  g{recno:>3d}: {hit}/{total} ({rate:.1%})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str]) -> MonitorConfig:
    p = argparse.ArgumentParser(
        prog="python -m caplab_sim.live_monitor",
        description="Live stock predictor + match tracker for Capitalism Lab.",
    )
    p.add_argument(
        "--poll", type=float, default=0.1, metavar="SECONDS",
        help="polling interval in seconds (default: 0.1).  "
             "Use 0.05 for game speed 10+, 0.025 for speed 15+.  "
             "The warm-start sweep cuts per-transition cost to <1 ms "
             "on stable days, so very fast polling is cheap.",
    )
    p.add_argument(
        "--max-k", type=int, default=30,
        help="upper bound on pre-first-group RNG offset sweep (default: 30)",
    )
    p.add_argument(
        "--max-inter-advance", type=int, default=5,
        help="upper bound on inter-group GroupAI advance sweep (default: 5)",
    )
    p.add_argument(
        "--cpi", type=float, default=DEFAULT_CPI,
        help=f"CPI to feed the predictor (default: {DEFAULT_CPI})",
    )
    p.add_argument(
        "--log", type=str, default=None, metavar="PATH",
        help="append one-JSON-per-line event log to PATH",
    )
    p.add_argument(
        "--no-tui", action="store_true",
        help="disable the rich live table; use plain-text output",
    )
    p.add_argument(
        "--once", action="store_true",
        help="take a single snapshot, print the forward prediction, exit",
    )
    args = p.parse_args(argv)
    return MonitorConfig(
        poll_seconds=args.poll,
        max_k=args.max_k,
        max_inter_advance=args.max_inter_advance,
        cpi=args.cpi,
        log_path=args.log,
        use_tui=not args.no_tui,
        run_once=args.once,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(list(sys.argv[1:]) if argv is None else list(argv))
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
