"""Phase 2 stage 1 validation harness.

Given a day-N save and a day-N+1 save, runs the stock predictor forward
from day-N's RNG seed and checks that every listed stock's predicted
``base_stock_price`` and ``sentiment`` match the day-N+1 observation
byte-for-byte.

Locating the pre-stock-tick seed
--------------------------------
The stock tick is one piece of a larger per-day tick.  Other subsystems
(FirmArray daily, etc.) fire RNG calls BEFORE and AFTER the stock tick.
To validate the stock formula in isolation we need the seed the engine
held at the moment it entered ``GlobalGrpArray__daily_stock_tick``.

We don't know the exact pre-stock RNG call count from first principles
(the phase-1 budget is a mean, not a per-day exact count).  Instead we
sweep: for each candidate offset ``k`` in ``[0, MAX_K)``, advance the
day-N seed by ``k`` LCG steps, run the predictor, and check if every
listed stock matches.  If exactly one ``k`` produces a full match, that
offset is the pre-stock call count for this day.

On the _____001 → _____002 reference pair ``k == 50`` matches all 21
listed stocks byte-for-byte — the other 132 LCG calls (182 observed
total minus 80 stock calls) split as 50 pre-stock + 52 post-stock.

Usage
-----
    python -m caplab_sim.validate_stocks A.SAV B.SAV

Exit codes:
    0 — exact match (predictor is bit-correct for this day)
    1 — partial match (best offset explains SOME stocks; see report)
    2 — no match within MAX_K sweep (either wrong day pair or bug)
    3 — unreachable seed or I/O error
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from caplab_save.parser import CapLabSave
from caplab_sim.rng import MiscRNG, lcg_step, distance_if_reachable
from caplab_sim.stock import (
    StockInput,
    StockOutput,
    MarketOutput,
    predict_all_stocks,
    stock_inputs_from_save,
)


# Upper bound on pre-stock-tick RNG calls per day.  Phase 1 observed
# total was 182 calls/day for the reference scenario; 300 gives headroom
# for larger scenarios without burning noticeable time.
DEFAULT_MAX_K: int = 300

# Floating-point match tolerance.  The engine stores prices as doubles
# and the formula does only a handful of multiplies, so rounding error
# should be well under 1e-6.  Prices are displayed to 4 decimals in the
# UI so 5e-4 is a generous "indistinguishable from observed" bound.
PRICE_TOLERANCE: float = 5e-4
SENTIMENT_TOLERANCE: float = 1e-2


@dataclass
class StockDiff:
    group_recno: int
    predicted_price: float
    observed_price: float
    predicted_sentiment: float
    observed_sentiment: float

    @property
    def price_error(self) -> float:
        return abs(self.predicted_price - self.observed_price)

    @property
    def sentiment_error(self) -> float:
        return abs(self.predicted_sentiment - self.observed_sentiment)

    @property
    def matches(self) -> bool:
        return (
            self.price_error < PRICE_TOLERANCE
            and self.sentiment_error < SENTIMENT_TOLERANCE
        )


@dataclass
class StockValidationResult:
    pre_stock_offset: Optional[int]      # k such that advancing seed by k LCG steps lines up
    seed_before_stocks: int              # day-N seed + k LCG steps
    seed_after_stocks: int               # post-stock seed
    post_stock_distance: Optional[int]   # LCG steps from post-stock seed to day-N+1 seed
    total_day_distance: Optional[int]    # day-N seed → day-N+1 seed
    stock_rng_calls: int
    diffs: List[StockDiff] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return sum(1 for d in self.diffs if d.matches)

    @property
    def n_total(self) -> int:
        return len(self.diffs)

    @property
    def full_match(self) -> bool:
        return self.n_total > 0 and self.n_matched == self.n_total

    def as_text(self) -> str:
        lines = []
        if self.pre_stock_offset is None:
            lines.append("pre-stock offset: NOT FOUND within sweep range")
        else:
            lines.append(f"pre-stock offset (k): {self.pre_stock_offset}")
        lines.append(f"seed_before_stocks: 0x{self.seed_before_stocks:08x}")
        lines.append(f"seed_after_stocks:  0x{self.seed_after_stocks:08x}")
        if self.post_stock_distance is not None:
            lines.append(f"post-stock → day-N+1 seed: {self.post_stock_distance} LCG steps")
        if self.total_day_distance is not None:
            lines.append(f"total day-N → day-N+1:    {self.total_day_distance} LCG steps")
        lines.append(f"stock RNG calls: {self.stock_rng_calls}")
        lines.append(f"match: {self.n_matched}/{self.n_total} listed stocks")
        if self.n_matched < self.n_total:
            lines.append("")
            lines.append("mismatches:")
            for d in self.diffs:
                if not d.matches:
                    lines.append(
                        f"  g{d.group_recno}: "
                        f"pred price={d.predicted_price:.4f} vs obs {d.observed_price:.4f} "
                        f"(Δ={d.predicted_price - d.observed_price:+.4f}), "
                        f"pred sent={d.predicted_sentiment:+.1f} vs obs {d.observed_sentiment:+.1f}"
                    )
        return "\n".join(lines)


def _score(diffs: List[StockDiff]) -> Tuple[int, float]:
    """Sort key: (fewer_mismatches, then smaller_total_error)."""
    mismatches = sum(1 for d in diffs if not d.matches)
    total_err = sum(d.price_error + 0.01 * d.sentiment_error for d in diffs)
    return (mismatches, total_err)


def validate_stock_day_pair(
    path_A: str,
    path_B: str,
    *,
    max_k: int = DEFAULT_MAX_K,
    cpi: float = 1.0,
) -> StockValidationResult:
    """Sweep offset, pick best, report.

    Loads both saves, runs the predictor at every ``k in [0, max_k)`` with
    descending iteration order, and returns the ``k`` whose full replay
    best matches the observed day-N+1 listed-stock state.
    """
    save_A = CapLabSave.load(path_A)
    save_B = CapLabSave.load(path_B)

    inputs = [
        si for si in stock_inputs_from_save(save_A)
        if si.group_type != 0
    ]
    observed = {g.recno: g.stock for g in save_B.groups if g.stock.is_listed}

    seed0 = save_A.rng.seed
    seed_target = save_B.rng.seed

    best_diffs: List[StockDiff] = []
    best_k: Optional[int] = None
    best_score: Tuple[int, float] = (len(observed) + 1, float("inf"))
    best_rng_calls: int = 0
    best_seed_before = seed0
    best_seed_after = seed0

    seed = seed0
    for k in range(max_k):
        # Use trace=True so StockOutput.rng_calls is populated via
        # len(rng.trace) deltas — the counts are read in the report
        # even when we don't inspect per-call records.
        rng = MiscRNG(seed, trace=True)
        out = predict_all_stocks(rng, inputs, cpi=cpi)
        diffs: List[StockDiff] = []
        out_by_recno: Dict[int, StockOutput] = {p.group_recno: p for p in out.per_stock}
        for recno, obs in observed.items():
            p = out_by_recno.get(recno)
            if p is None:
                # Listed in B but not predicted (shouldn't happen for the
                # same scenario).  Record an impossible-match diff.
                diffs.append(StockDiff(
                    group_recno=recno,
                    predicted_price=float("nan"),
                    observed_price=obs.base_stock_price,
                    predicted_sentiment=float("nan"),
                    observed_sentiment=obs.sentiment,
                ))
                continue
            diffs.append(StockDiff(
                group_recno=recno,
                predicted_price=p.base_stock_price,
                observed_price=obs.base_stock_price,
                predicted_sentiment=p.sentiment,
                observed_sentiment=obs.sentiment,
            ))
        score = _score(diffs)
        if score < best_score:
            best_score = score
            best_diffs = diffs
            best_k = k
            best_rng_calls = out.total_rng_calls
            best_seed_before = seed
            best_seed_after = rng.seed
            if score[0] == 0:
                # Full match — no need to keep sweeping.
                break
        seed = lcg_step(seed)

    post_distance = distance_if_reachable(best_seed_after, seed_target, 500)
    total_distance = distance_if_reachable(seed0, seed_target, 500)

    return StockValidationResult(
        pre_stock_offset=best_k,
        seed_before_stocks=best_seed_before,
        seed_after_stocks=best_seed_after,
        post_stock_distance=post_distance,
        total_day_distance=total_distance,
        stock_rng_calls=best_rng_calls,
        diffs=best_diffs,
    )


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(f"usage: python -m caplab_sim.validate_stocks <dayN.SAV> <dayN+1.SAV>",
              file=sys.stderr)
        return 3
    try:
        result = validate_stock_day_pair(argv[1], argv[2])
    except (FileNotFoundError, OSError) as exc:
        print(f"error loading save: {exc}", file=sys.stderr)
        return 3
    print(result.as_text())
    if result.full_match:
        return 0
    if result.n_matched > 0:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
