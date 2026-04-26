"""Validation harness for Phase 1 seed replay.

This module answers the Phase 1 yes/no question:

    *Given a save on day N and a save on day N+1, does the RNG seed
    advance by the number of LCG steps we predicted?*

If the answer is yes (or within ≤25 calls of yes — see RNG_Call_Map.txt §9
residual notes), Phase 1 is done. If it's no, the gap tells us which
subsystem's budget estimate is off and by how much.

The harness has three entry points:

1. :func:`observed_lcg_distance` — given two seeds, find ``k`` such that
   ``k`` LCG steps map ``seed0`` to ``seed1``. Bounded search because the
   full period is 2³² and we only care about small single-day deltas
   (formula predicts ~400-500 calls/day for the reference save).

2. :func:`validate_day_pair` — combine :func:`observed_lcg_distance` with
   :func:`caplab_sim.tick.one_day_rng_budget` and return a
   :class:`DayValidationResult` summarising predicted vs observed.

3. :func:`main` — CLI for quick triage: ``python -m caplab_sim.validate
   save_day_N.SAV save_day_N1.SAV`` prints a human-readable diff.

Phase 1 acceptance — a real consecutive-day save pair
(``_____001_20260421_054902_000.SAV`` → ``_____002_20260421_054916_000.SAV``)
currently validates at +3 LCG steps (observed 182 vs predicted 179),
well within the ≤25-call/day residual documented in RNG_Call_Map.txt §9.
Additional pairs are welcome to keep the tuned defaults honest across
calendar positions (month-end, site-spawn-on, etc.). See
``caplab_sim/SESSION_LOG.txt`` for the closure entry and the specific
constant values that were tuned.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from caplab_sim.constants import SimConstants
from caplab_sim.rng import distance_if_reachable, simulate_lcg
from caplab_sim.state import SimState, load_sim_state, read_seed_from_save
from caplab_sim.tick import DailyBudget, one_day_rng_budget


# Search window for ``distance_if_reachable``. The reference save
# predicts ~400-500 calls/day; month-end boundaries can add a few tens
# more. 5000 is ~10× the upper end — comfortably large enough to find
# the answer if one exists without being so large we'd confuse a
# genuine mismatch with a collision on the 2³²-period LCG.
DEFAULT_MAX_STEPS: int = 5000


@dataclass
class DayValidationResult:
    """Outcome of comparing predicted vs observed one-day seed advance.

    Attributes
    ----------
    seed_before : int
    seed_after : int
        The two seeds being compared.
    predicted_steps : int
        Output of :func:`one_day_rng_budget` on the "before" state.
    observed_steps : Optional[int]
        Output of :func:`distance_if_reachable` — ``None`` if the target
        seed isn't reachable within ``max_steps`` of ``seed_before`` (in
        which case something much bigger than one day happened between
        the saves, or one of the seeds was mis-read).
    budget : DailyBudget
        Full per-subsystem breakdown. Useful for attributing any gap to
        a specific contributor.
    max_steps : int
        The search bound used. Reported so readers can distinguish
        "not reachable at all" (unlikely — LCG has full period) from
        "not reachable within the search window we tried".
    same_seed : bool
        Shortcut for ``seed_before == seed_after`` (which is the
        uninteresting case where the saves are from the same tick).
    """

    seed_before: int
    seed_after: int
    predicted_steps: int
    observed_steps: Optional[int]
    budget: DailyBudget
    max_steps: int

    @property
    def same_seed(self) -> bool:
        return self.seed_before == self.seed_after

    @property
    def delta(self) -> Optional[int]:
        """Observed - predicted (None if observed is unknown)."""
        if self.observed_steps is None:
            return None
        return self.observed_steps - self.predicted_steps

    @property
    def matches(self) -> bool:
        """True iff observed == predicted exactly.

        Phase 1's strict success criterion. Phase 2 needs this true
        before per-call output validation makes any sense.
        """
        return self.observed_steps is not None and self.observed_steps == self.predicted_steps

    def as_text(self) -> str:
        """Human-readable summary for CLI output and SESSION_LOG entries."""
        lines = [
            f"seed_before:    0x{self.seed_before:08x}",
            f"seed_after:     0x{self.seed_after:08x}",
        ]
        if self.same_seed:
            lines.append("state:          saves have identical seeds (same tick)")
        lines.append(f"predicted:      {self.predicted_steps} LCG steps")
        if self.observed_steps is None:
            lines.append(
                f"observed:       unreachable within {self.max_steps} steps "
                f"(either saves skipped >1 day, or seed extraction is wrong)"
            )
        else:
            lines.append(f"observed:       {self.observed_steps} LCG steps")
            gap = self.delta
            assert gap is not None
            if gap == 0:
                lines.append("result:         EXACT MATCH")
            else:
                sign = "+" if gap > 0 else ""
                lines.append(f"result:         gap {sign}{gap} calls (observed - predicted)")
                if abs(gap) <= 25:
                    lines.append(
                        "                within the documented ≤25-call/day residual "
                        "(RNG_Call_Map.txt §9)"
                    )
        lines.append("")
        lines.append("budget breakdown:")
        lines.append(self.budget.as_text())
        return "\n".join(lines)


# ---- primitive wrappers ----------------------------------------------------


def observed_lcg_distance(
    seed_before: int,
    seed_after: int,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Optional[int]:
    """Find ``k`` such that ``k`` LCG steps carry ``seed_before`` to ``seed_after``.

    Thin wrapper around :func:`caplab_sim.rng.distance_if_reachable` —
    named in game-engine terms so the validation log reads naturally.
    """
    return distance_if_reachable(seed_before, seed_after, max_steps)


# ---- main validation routine -----------------------------------------------


def validate_day_pair(
    path_before: str,
    path_after: str,
    constants: SimConstants = SimConstants(),
    max_steps: int = DEFAULT_MAX_STEPS,
) -> DayValidationResult:
    """Compare predicted vs observed seed advance for a pair of saves.

    ``path_before`` must be a save from day ``N``, ``path_after`` from
    day ``N+1`` of the same game — otherwise the observed delta isn't a
    one-day delta and the comparison is meaningless (the function
    doesn't enforce this; the caller owns the scenario).

    The "before" save is projected into a :class:`SimState` (giving the
    entity counts that feed :func:`one_day_rng_budget`). The "after"
    save is only read for its seed — no need to parse the whole record.
    """
    state_before = load_sim_state(path_before)
    seed_after = read_seed_from_save(path_after)

    budget = one_day_rng_budget(state_before, constants)
    observed = observed_lcg_distance(state_before.seed, seed_after, max_steps)

    return DayValidationResult(
        seed_before=state_before.seed,
        seed_after=seed_after,
        predicted_steps=budget.total,
        observed_steps=observed,
        budget=budget,
        max_steps=max_steps,
    )


# ---- single-save smoke check (works without a consecutive pair) -----------


@dataclass
class SingleSaveReport:
    """Predicted-only report for a single save (no observed delta).

    Useful for smoke-testing the pipeline when no day-N+1 save is
    available: shows the seed that was read and what the formula thinks
    tomorrow's seed would be, so at least the prediction path is
    exercised end-to-end. Pair with the real :func:`validate_day_pair`
    once a consecutive-day pair is captured.
    """

    state: SimState
    budget: DailyBudget
    predicted_next_seed: int

    def as_text(self) -> str:
        s = self.state
        lines = [
            f"seed:           0x{s.seed:08x}  (tag 0x1067 at offset 0x{s.seed_offset:x})",
            f"game date:      {s.year}-{s.month_of_year:02d}-{s.day_of_month:02d}  "
            f"(game_date={s.game_date}, month_end={s.is_month_end})",
            f"entities:       {s.n_groups} groups ({s.n_ai_groups} AI), "
            f"{s.n_firms} firms, {s.n_listed_stocks} listed stocks",
            f"predicted next: 0x{self.predicted_next_seed:08x}  "
            f"(+{self.budget.total} LCG steps)",
            "",
            "budget breakdown:",
            self.budget.as_text(),
        ]
        return "\n".join(lines)


def inspect_save(path: str, constants: SimConstants = SimConstants()) -> SingleSaveReport:
    """One-save variant — no observed side, just prediction."""
    state = load_sim_state(path)
    budget = one_day_rng_budget(state, constants)
    predicted = simulate_lcg(state.seed, budget.total)
    return SingleSaveReport(state=state, budget=budget, predicted_next_seed=predicted)


# ---- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 seed validation harness. Given a pair of consecutive-day "
            "saves, reports whether the observed seed advance matches the "
            "RNG_Call_Map.txt §4 formula prediction. With a single argument, "
            "runs in inspect mode (prediction only, no comparison)."
        )
    )
    parser.add_argument(
        "path_before",
        help="Path to the day-N save (used to read seed + entity counts)",
    )
    parser.add_argument(
        "path_after",
        nargs="?",
        default=None,
        help="Path to the day-N+1 save (optional; inspect-only if omitted)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=(
            "Upper bound on the LCG distance search. The LCG has full period "
            f"2^32 so any two seeds are reachable, but we cap the search at "
            f"{DEFAULT_MAX_STEPS} by default to fail fast on mismatched or "
            "non-consecutive saves."
        ),
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.path_after is None:
        report = inspect_save(args.path_before)
        print(report.as_text())
        return 0

    result = validate_day_pair(
        args.path_before,
        args.path_after,
        max_steps=args.max_steps,
    )
    print(result.as_text())
    # Exit code reflects success:
    #   0 — exact match
    #   1 — observed within the documented residual (≤25 calls)
    #   2 — larger gap
    #   3 — unreachable within window (likely bad pair)
    if result.observed_steps is None:
        return 3
    if result.matches:
        return 0
    gap = result.delta
    assert gap is not None
    return 1 if abs(gap) <= 25 else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
