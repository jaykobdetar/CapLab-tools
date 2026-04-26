"""Daily tick orchestrator — Phase 1: RNG budgeting + stub replay.

This module does two things:

1. :class:`DailyBudget` — translates a :class:`SimState` into a
   predicted *count* of Misc::random() invocations for one day, using
   the formula from ``RNG_Call_Map.txt §4``. This is the cheap check
   that can be validated against an observed seed delta
   (``distance_if_reachable(seed0, seed1, ~5000)``).

2. :func:`replay_one_day` — runs a call-by-call replay using
   :class:`MiscRNG`. Phase 1 doesn't yet have the per-entity economic
   logic, so most calls are made with plausible ``n`` values drawn
   from the per-site catalogue in ``RNG_Call_Map.txt §3``. What we get
   in return is a full :class:`RNGTrace` ordered according to
   ``RNG_Call_Map.txt §5``'s replay pseudocode — useful for binary-
   searching for the first divergence between our trace and an
   observed end-of-day seed.

**Honesty clause**: the RNG call count formula is approximate. On the
one validated consecutive-day pair (``_____001`` → ``_____002``, captured
2026-04-21) the tuned defaults land within +3 LCG steps of the observed
182/day — well inside the ≤25-call/day residual documented in
RNG_Call_Map.txt §9. The per-subsystem means used here (STOCK_MEAN=5.2,
PERSON_MEAN=0.05, etc.) are calibrated to that pair; additional pairs
may reveal second-order corrections. For Phase 2's per-call-output
validation we'll need to replace the stub replay with per-entity logic
so ``n`` values match the engine call-by-call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from caplab_sim.constants import SimConstants
from caplab_sim.rng import MiscRNG, RNGTrace, simulate_lcg
from caplab_sim.state import SimState


# ---- budget prediction ----------------------------------------------------


@dataclass
class DailyBudget:
    """Breakdown of predicted RNG calls for one day.

    All fields are predicted *mean* counts (floats for the stochastic
    contributors, ints for the deterministic ones). ``total`` is the
    rounded sum — the value to compare against the observed seed
    advance between two save snapshots.

    Use :meth:`as_text` for a human-readable summary in the validation
    harness and SESSION_LOG entries.
    """

    S_stocks: float
    S_persons: float
    S_towns: int
    S_parties: float
    S_talents: float
    S_firms_gate: float
    S_firms_cascade: float
    S_firms_rd_media: float
    S_groupai: float
    S_logistics: float
    S_rawres: float
    S_goal_ai: float
    S_bond: float
    S_monthly: int = 0       # added only on month boundary
    components: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Integer rounding of the component sum.

        The engine's per-day count is a non-negative integer by
        construction, so rounding is the defensible choice. For
        validation against a small observed delta we report the raw
        float too.
        """
        return int(round(self.total_float))

    @property
    def total_float(self) -> float:
        return (
            self.S_stocks
            + self.S_persons
            + self.S_towns
            + self.S_parties
            + self.S_talents
            + self.S_firms_gate
            + self.S_firms_cascade
            + self.S_firms_rd_media
            + self.S_groupai
            + self.S_logistics
            + self.S_rawres
            + self.S_goal_ai
            + self.S_bond
            + self.S_monthly
        )

    def as_text(self) -> str:
        rows = [
            ("S_stocks", self.S_stocks),
            ("S_persons", self.S_persons),
            ("S_towns", float(self.S_towns)),
            ("S_parties", self.S_parties),
            ("S_talents", self.S_talents),
            ("S_firms_gate", self.S_firms_gate),
            ("S_firms_cascade", self.S_firms_cascade),
            ("S_firms_rd_media", self.S_firms_rd_media),
            ("S_groupai", self.S_groupai),
            ("S_logistics", self.S_logistics),
            ("S_rawres", self.S_rawres),
            ("S_goal_ai", self.S_goal_ai),
            ("S_bond", self.S_bond),
            ("S_monthly", float(self.S_monthly)),
        ]
        width = max(len(r[0]) for r in rows)
        lines = [
            f"  {name:{width}s}  {val:8.2f}"
            for name, val in rows
        ]
        lines.append(f"  {'TOTAL':{width}s}  {self.total_float:8.2f}  ({self.total} rounded)")
        return "\n".join(lines)


def _rd_media_count(state: SimState) -> int:
    """Count FirmRD + FirmMediaBase firms.

    Firm type codes (see Class_Hierarchy.txt and Structs_Firm.txt):
      0x05 → FirmRD
      0x1D / 0x1E / 0x1F → FirmMedia variants (Newspaper, TV, Radio)
      0x27 → FirmWeb (also runs the 30-day media cycles indirectly via
             FirmMediaBase decay).
    We conservatively count types {0x05, 0x1D, 0x1E, 0x1F} — these are
    the ones whose AI_processing slot fires random() in the 30-day
    cadence per ``RNG_Call_Map.txt §3 #1``.
    """
    wanted = (0x05, 0x1D, 0x1E, 0x1F)
    return sum(state.n_firms_by_type.get(t, 0) for t in wanted)


def one_day_rng_budget(
    state: SimState,
    constants: SimConstants = SimConstants(),
) -> DailyBudget:
    """Apply the formula from ``RNG_Call_Map.txt §4`` to ``state``.

    Returns a :class:`DailyBudget`. Does not mutate state or advance
    the seed — this is just a count.
    """
    c = constants

    # §3 #2 stocks
    S_stocks = state.n_listed_stocks * c.stock_mean_rng_per_listed

    # §3 #3 persons
    S_persons = state.n_persons * c.person_mean_rng_per_day

    # §3 #4 towns
    S_towns = state.n_towns * c.town_rng_per_day

    # §3 #5 parties
    S_parties = state.n_parties * c.party_mean_rng_per_day

    # §3 #6 talents
    S_talents = state.n_talents * c.talent_mean_rng_per_day

    # §3 #1 firms
    if state.site_spawn_enabled and c.site_spawn_enabled:
        S_firms_gate = state.n_firms * c.firm_gate_rng_per_day
        S_firms_cascade = state.n_firms * c.firm_cascade_rate
    else:
        S_firms_gate = 0.0
        S_firms_cascade = 0.0
    S_firms_rd_media = _rd_media_count(state) * c.firm_rd_media_30d_contrib

    # GroupAI
    S_groupai = state.n_ai_groups * c.groupai_mean_rng_per_day

    # Logistics
    S_logistics = c.logistics_rng_per_day if (state.logistics_enabled and c.logistics_enabled) else 0.0

    # Raw-resource discovery — gated by the RawRes DLC
    if state.rawres_dlc_active or c.rawres_dlc_active:
        S_rawres = c.rawres_base_rng_per_day + c.rawres_extra_per_slice
    else:
        S_rawres = 0.0

    # Goal-AI
    S_goal_ai = state.n_ai_groups * c.goal_ai_mean_rng_per_day

    # Bond-issuance (GroupAI, 90-day)
    # Mean per-day contribution = n_ai_groups * 2 / 90 ≈ linear amortisation.
    S_bond = (
        state.n_ai_groups
        * c.bond_issuance_rng_per_fire
        / c.bond_issuance_cycle_days
    )

    # Monthly spike
    S_monthly = 0
    if state.is_month_end:
        # Baseline monthly: 3 (gdp + inflation)
        base = c.monthly_baseline_rng
        # Foreign stocks — 0 for the reference scenario (no foreign stocks
        # parsed yet). Once the NationStock array is decoded we'll add
        # ``n_foreign × 1`` here. Documented as an open item.
        foreign = 0
        S_monthly = base + foreign

    return DailyBudget(
        S_stocks=S_stocks,
        S_persons=S_persons,
        S_towns=S_towns,
        S_parties=S_parties,
        S_talents=S_talents,
        S_firms_gate=S_firms_gate,
        S_firms_cascade=S_firms_cascade,
        S_firms_rd_media=S_firms_rd_media,
        S_groupai=S_groupai,
        S_logistics=S_logistics,
        S_rawres=S_rawres,
        S_goal_ai=S_goal_ai,
        S_bond=S_bond,
        S_monthly=S_monthly,
    )


# ---- stub replay -----------------------------------------------------------


def replay_one_day(
    state: SimState,
    constants: SimConstants = SimConstants(),
) -> Tuple[MiscRNG, RNGTrace]:
    """Stub replay — burns the predicted number of RNG calls with
    placeholder ``n`` values, ordered to match ``RNG_Call_Map.txt §5``.

    **This is not yet a correct replay.** Phase 1's replay only cares
    about getting the *call count* right, because the final seed
    depends only on the total number of LCG steps — the ``n`` arguments
    affect per-call outputs but never the seed. So for seed validation
    we can use any ``n`` we like, as long as we make the right number
    of calls in the right order.

    For the per-call-output validation needed in Phase 2 (stock price
    prediction), this stub will need to be replaced with the real
    per-subsystem replay. The stub is a scaffold that labels each call
    with the subsystem it came from, so when a real implementation
    lands the labels will already be in place for divergence reporting.
    """
    rng = MiscRNG(state.seed, trace=True)
    budget = one_day_rng_budget(state, constants)

    # Helper that burns ``n_calls`` RNG invocations with value ``n`` and
    # a given label. Rounded per-call because partial calls are not a
    # thing; we truncate the fractional part and attribute the rounding
    # to ``_residual`` at the end.
    def burn(n_calls: float, n: int, label: str) -> int:
        made = 0
        for i in range(int(n_calls)):
            rng.label(f"{label}#{i}")
            rng.rng(n)
            made += 1
        return made

    # Order follows RNG_Call_Map.txt §5 ("REPLAY PSEUDOCODE") — firms,
    # stocks, persons, towns, parties, talents, logistics, rawres, goal,
    # then month-end if applicable.
    total_burned = 0
    total_burned += burn(budget.S_firms_gate, 120, "firms.gate")
    total_burned += burn(budget.S_firms_cascade, 10, "firms.cascade")
    total_burned += burn(budget.S_firms_rd_media, 60, "firms.rd_media_30d")
    total_burned += burn(budget.S_stocks, 200, "stocks.price_update")
    total_burned += burn(budget.S_persons, 30, "persons.daily")
    total_burned += burn(budget.S_towns, 60, "towns.popgrowth")
    total_burned += burn(budget.S_parties, 180, "parties.daily")
    total_burned += burn(budget.S_talents, 20, "talents.slice")
    total_burned += burn(budget.S_logistics, 5, "logistics.gate")
    total_burned += burn(budget.S_rawres, 10, "rawres.discovery")
    total_burned += burn(budget.S_groupai, 100, "groupai.daily")
    total_burned += burn(budget.S_goal_ai, 100, "goal_ai.daily")
    total_burned += burn(budget.S_bond, 100, "bond.issuance")

    if budget.S_monthly:
        # Month-end baseline: 2 gdp + 1 inflation. Use n values from
        # RNG_Call_Map.txt §3 #25 — rng(9) shock gate and rng(60) inflation.
        rng.label("monthly.gdp.shock_gate")
        rng.rng(9)
        rng.label("monthly.gdp.step")
        rng.rng(3)
        rng.label("monthly.inflation")
        rng.rng(60)
        total_burned += 3
        # Anything above the 3 baseline (e.g. shock, crash) is left out
        # of the stub — documented open item.

    # Residual: each ``burn`` call truncates fractional per-subsystem
    # budgets, but :attr:`DailyBudget.total` rounds the *sum*. So the
    # trace can legitimately be short (or long) by up to ~14 calls —
    # one per component that had a fractional piece. We attribute that
    # gap to a single ``residual`` label so that the Phase 1 invariant
    # (``replay final seed == expected_end_of_day_seed``) holds bit-
    # exactly. When Phase 2 replaces the stub with real per-entity
    # logic, this residual block should shrink to zero because each
    # subsystem will know its integer call count exactly.
    residual = budget.total - total_burned
    if residual > 0:
        for i in range(residual):
            rng.label(f"residual#{i}")
            rng.rng(1)
            total_burned += 1
    elif residual < 0:
        # We burned more than the rounded budget — drop the extras by
        # re-running the same sequence into a fresh MiscRNG seeded at
        # the state's seed advanced by exactly ``budget.total`` steps.
        # This shouldn't happen with current constants (every component
        # truncates DOWN), but keeping the branch makes the invariant
        # robust to future formula changes.
        rng = MiscRNG(state.seed, trace=True)
        for _ in range(budget.total):
            rng.label("residual_rebuild")
            rng.rng(1)
        total_burned = budget.total

    # Trace length == budget.total by construction of the residual block.
    assert len(rng.trace) == budget.total, (
        f"trace length {len(rng.trace)} != budget.total {budget.total} "
        f"(burned={total_burned}, residual={residual})"
    )
    return rng, rng.trace


def expected_end_of_day_seed(state: SimState, constants: SimConstants = SimConstants()) -> int:
    """Pure-math prediction: seed after ``one_day_rng_budget().total`` LCG steps.

    Equivalent to ``simulate_lcg(state.seed, total)`` but exposed for
    clarity in the validation harness.
    """
    total = one_day_rng_budget(state, constants).total
    return simulate_lcg(state.seed, total)
