"""Phase 2 stage 1 — per-stock daily price predictor.

This module implements a pure-Python replay of the engine's
``NationStock__daily_price_update`` function (0x00647d30) and the
surrounding per-stock dispatch in ``NationStock__daily_tick``
(0x00647530) / ``GlobalGrpArray__daily_stock_tick`` (0x00509570).

Goal
----
Given a save-file snapshot (``SimState``) and a starting RNG seed,
produce the ``base_stock_price`` (NationStock+0x18) that every listed
domestic stock would have on the next game day, along with the updated
sentiment (NationStock+0x98) and the RNG trace of the stock tick.

This is the correctness floor for Phase 2: if the formula is right, the
predicted prices will match the actual next-day prices byte-for-byte
when we feed in the correct pre-stock-tick seed.

Scope — what this file OMITS intentionally
------------------------------------------
* ``NationStock__check_reverse_split`` — has no RNG unless price < $1
  and shares > 10,000, which never fires on the reference scenario.
* ``NationStock__annual_dividend`` — fires on day 2 of month 1 ONLY
  when ``starting_game_year < current_game_year``.  Both _____001 and
  _____002 are in 1990 with start_year 1990, so the gate is false.
* ``NationStock__update_financial_metrics`` — no RNG; rewrites
  book-value / P/E / earnings-yield but does not affect today's price.
* Foreign stocks (+0x03 != 0) — updated monthly, not daily; no daily RNG.
* Unlisted stocks (+0x02 == 0) — no RNG (price = max(0.1, bvps)).

Everything above runs only for domestic listed stocks.  We match the
Ghidra-confirmed call sequence exactly:

    1.  random(|sentiment|)          (sentiment_random)
    2.  random(200)                   (market_random)
        price += price * (market_random - 100 + signed_sentiment_random) / 10000
    3.  if price < CPI * 0.1:
            random(10)                (price recovery)
            random(30)                (sentiment recovery)
        price = max(price, 0.01)
    4.  earnings_metric = Group+0x3B38 * 3 / (CPI * 500000)
    5.  random(5) -> case selector
        case 0: sentiment -= random(4); if random(3)==0: price -= price*random(10)/100
        case 1: sentiment += random(4); if random(3)==0: price += price*random(20)/100
        case 2: if bvps*0.7 > price:
                    r = random(100)
                    undervaluation = (bvps*0.7 - price) * r / price
                    if undervaluation > 0:
                        sentiment += undervaluation (clamp to 100)
                        if random(3)==0: price += price*random(10)/100
        case 3: if sentiment<=0: sentiment += random(int(|sentiment|/5))
                else:              sentiment -= random(int(sentiment/5))
        case 4: sentiment += random(11) - 5
    6.  if hq_type != 0xB and new_eps != old_eps:
            eps_delta = (new_eps - old_eps) * (+50 if eps_basic>=0 else -50) / old_eps
            eps_delta = clamp(eps_delta, -100, +100)
            random(|eps_delta|) -> sentiment shock

Group iteration in the engine is *slot-ascending* in
``GlobalGrpArray__daily_stock_tick`` (``for iVar6 = 1..n_groups``,
deleted groups skipped).  **Empirically validated 2026-04-21**: the
slot-to-recno mapping is REVERSED in the GlobalGrpArray — slot 1 holds
the highest recno, slot 2 the next-highest, etc.  So in terms of the
``recno`` field we parse out of each Group, the stock tick fires in
**DESCENDING** recno order.  Iterating ``save.groups`` (which our
parser returns in ascending recno order) directly gives the WRONG
answer; callers should pass ``sorted(stocks, key=-recno)`` or use
:func:`predict_all_stocks`'s default which does the reverse.

Reference for this correction: the _____001 → _____002 consecutive-day
pair validates bit-exactly only under descending iteration — ASC
matches 0/21 observed outputs, DESC matches 21/21.  See the forward-
simulation harness in caplab_sim/validate_stocks.py and the "Phase 2
stage 1 close" entry in SESSION_LOG.txt.

Only one NationStock exists per Group (Group+0x70).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from caplab_sim.rng import MiscRNG, simulate_lcg


# Default CPI value used when economy state isn't available.  The engine's
# ``eco__price_level_CPI`` starts at 1.0 and drifts with inflation; for
# the reference scenario on Jan 1 1990 it's still 1.0.
DEFAULT_CPI: float = 1.0


@dataclass
class StockInput:
    """Pre-tick snapshot for one NationStock, in engine units.

    Built from :class:`caplab_save.structs.NationStock` plus a few
    fields pulled from its owning ``Group``.  We keep ``StockInput``
    decoupled from the save-file types so tests can drive the predictor
    with hand-built fixtures.
    """

    group_recno: int
    is_listed: bool                     # NationStock+0x02 (byte)
    is_foreign: bool                    # NationStock+0x03 (byte)
    group_type: int                     # Group+0x74 (short: 0/1/3/4)
    hq_type: int = 0                    # vtable[1]()->+0x4528 short; 10=bank, 11=insurance
    shares_outstanding: float = 0.0
    base_stock_price: float = 0.0       # +0x18
    book_value_per_share: float = 0.0   # +0x28
    eps_basic: float = 0.0              # +0x30
    eps_adjusted_stored: float = 0.0    # +0x38 (last tick's compute_eps(1))
    pe_ratio_stored: float = 0.0        # +0x43A0 (double — gate for case 0/1)
    sentiment: float = 0.0              # +0x98

    # Fields sourced from the owning Group for case-2 fundamental valuation.
    # Offsets below are Group-relative bytes, and their doubles are:
    #   0x39D8 bank_deposits
    #   0xCC0  corp_cash
    #   0x39E0 inventory
    #   0x39E8 business_assets
    #   0x39F0 land
    #   0x39F8 technology
    #   0x3A00 stocks_held
    #   0x3A10 bank_net_assets
    #   0x3A18 insurance_net_assets
    #   0x3A20 loans
    #   0x3A28 bond_interest_liability
    #   0x3B38 driver for earnings_metric
    bank_deposits: float = 0.0
    corp_cash: float = 0.0
    inventory: float = 0.0
    business_assets: float = 0.0
    land: float = 0.0
    technology: float = 0.0
    stocks_held: float = 0.0
    bank_net_assets: float = 0.0
    insurance_net_assets: float = 0.0
    loans: float = 0.0
    bond_interest_liability: float = 0.0
    earnings_metric_driver: float = 0.0   # Group+0x3B38

    # New EPS for this tick — computed by Group::compute_eps(1).  For Jan 1 1990
    # with zero trailing-12m profit this is exactly zero, so defaulting to
    # ``eps_adjusted_stored`` yields the right no-shock behaviour.  Callers
    # that know the fresh value should set this explicitly.
    eps_adjusted_new: Optional[float] = None

    # Number of firms owned by this group (Group+0xE4).  Used by the live
    # monitor to determine whether a group is in 5-day startup mode (firm_count
    # == 0) or the normal 25-day cycle for GroupAI__select_firm_type_25day.
    # Defaults to 0 (conservative: startup mode), overridden to the live value
    # by the memory reader.
    firm_count: int = 0

    # --- Group::compute_eps inputs (2026-04-22) ---------------------------
    # Group+0xCE0 and Group+0xCE8 feed ``Group::compute_eps(1)`` at
    # 0x0051c390.  The engine recomputes EPS fresh each tick from these two
    # doubles (plus shares_outstanding) and compares the result against
    # the stored ``eps_adjusted`` at NationStock+0x38; when they differ,
    # the EPS-shock RNG branch at the tail of
    # ``NationStock__daily_price_update`` fires.
    #
    # Without populating these fields, ``eps_adjusted_new`` falls back
    # to ``eps_adjusted_stored`` and the shock branch NEVER fires in
    # the predictor.  That drift is the dominant source of per-group
    # miss rate on firm-owning groups (observed 2026-04-22: the player
    # group drops from ~95% match → 46% once trailing profit
    # accumulates; see SESSION_LOG 2026-04-22c for the full trace).
    trailing_12m_net_profit: float = 0.0       # Group+0xCE0 (double)
    trailing_12m_special_revenue: float = 0.0  # Group+0xCE8 (double)

    # --- AI-tail gating fields (2026-04-22) --------------------------------
    # Used by ai_tail_advance() to determine how many Misc::random calls
    # GroupAI__daily_update fires AFTER this group's NationStock tick and
    # BEFORE the next group's vtable+0x18 dispatch.  See
    # Stock_Tick_InterGroup_RNG_Analysis.md §§3.1-3.2 for derivations.
    #
    # ai_leader_slot  — Group+0xdc (short).  The group's leader PersonRes
    #                   index.  The early-return gate at
    #                   `GroupAI__daily_update+0x14` is `if (group[0xdc] == 0)`
    #                   — i.e. it fires only when the group has NO leader
    #                   assigned (destroyed / uninitialised).  CORRECTION
    #                   2026-04-22d: the player's conglomerate does NOT
    #                   have ai_leader_slot == 0 — the live diagnostic on
    #                   g2 (player) reports `ai_ldr = 82` (a valid
    #                   PersonRes slot for the player CEO).  The previous
    #                   assumption that this gate filtered out the player
    #                   was wrong; the real player/AI filter is at
    #                   `+0x34c` (`if (group[0x74] != 3) return;`) —
    #                   i.e. `group_type != 3`.  `ai_leader_slot == 0` is
    #                   kept as an additional predicate for defence in
    #                   depth but is not the primary filter.
    # dlc_build_flag  — Group+0x21f (byte).  Per-group "DLC build-from-city"
    #                   enable flag.  Pairs with the global ai_enabled flag
    #                   to gate the DLC town-scoring RNG path at 0x0042e78b.
    ai_leader_slot: int = 0     # Group+0xdc (short)
    dlc_build_flag: int = 0     # Group+0x21f (byte)


@dataclass
class StockOutput:
    """Predicted post-tick state for one NationStock."""

    group_recno: int
    base_stock_price: float             # predicted next-day price (+0x18)
    sentiment: float                    # predicted next-day sentiment (+0x98)
    rng_calls: int                      # number of RNG steps consumed
    case_selector: int = -1             # -1 if case never evaluated (unlisted/foreign)
    fired_cpi_floor: bool = False
    fired_case_surge: bool = False
    fired_eps_shock: bool = False


@dataclass
class MarketOutput:
    """Full-market prediction."""

    per_stock: List[StockOutput] = field(default_factory=list)
    total_rng_calls: int = 0
    starting_seed: int = 0
    ending_seed: int = 0


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def compute_eps(s: StockInput) -> float:
    """Port of ``Group::compute_eps(1)`` at 0x0051c390 (FUN_0051c390).

    Engine source (decompiled 2026-04-22)::

        dVar4 = *(double *)(param_1 + 0xce8) + *(double *)(param_1 + 0xce0);
        local_8 = dVar4;
        if ((param_2 != '\\0') && (iVar2 = FUN_0051d740(), iVar2 != 0)) {
            fVar3 = (float10)FUN_00491360();        // subsidiary earnings
            dVar1 = (double)((fVar3 * (float10)70.0) / (float10)100.0);
            if (dVar1 <= 0.0) {
                if (dVar4 < 0.0) {
                    local_8 = 0.0;
                    if (dVar4 - dVar1 <= 0.0) local_8 = dVar4 - dVar1;
                }
            } else {
                local_8 = dVar4 - dVar1;
            }
        }
        return (float10)local_8 / (float10)*(double *)(*(int *)(param_1 + 0x70) + 8);

    ``param_1 + 0x70`` dereferences to the owning ``NationStock`` record,
    whose ``+0x8`` is ``shares_outstanding``.  The subsidiary adjustment
    branch (inside ``if (iVar2 != 0)``) fires only when ``FUN_0051d740``
    returns non-zero — which happens when the group owns an HQ firm whose
    vtable[0x58]() indicates an active subsidiary with 12-month interpolated
    earnings history (FUN_00491360).  For groups without an HQ firm
    (Group+0xcac == 0) or with a brand-new HQ firm (no 12-month history
    yet) the branch is a no-op and this function reduces to::

        (trailing_12m_net_profit + trailing_12m_special_revenue) / shares_outstanding

    That simple formula is what we ship.  The subsidiary-adjustment branch
    (FUN_0051d740 → FUN_00491360 with 12-iter GameDate interpolation over
    Group+0x3040) is left as future work — empirically the player group
    drifts from ~95% → 46% once the trailing-12m profit accumulator starts
    moving, so getting this simple formula right is the dominant fix.

    Returns ``0.0`` when ``shares_outstanding == 0`` to avoid division by
    zero; the engine would return a NaN/Inf float10 there, but the caller
    path (EPS-shock branch in ``predict_one_stock``) already skips zeros.
    """
    if s.shares_outstanding == 0.0:
        return 0.0
    profit = s.trailing_12m_net_profit + s.trailing_12m_special_revenue
    return profit / s.shares_outstanding


def _compute_weighted_bvps(s: StockInput) -> float:
    """Case-2 weighted book value per share. Returns NaN if shares == 0."""
    if s.shares_outstanding == 0.0:
        return float("nan")
    assets = (
        s.bank_deposits * 1.0
        + s.corp_cash * 1.0
        + s.inventory * 0.5
        + s.business_assets * 0.5
        + s.land * 1.0
        + s.technology * 0.2
        + s.stocks_held * 0.7
        + s.bank_net_assets * 1.0
        + s.insurance_net_assets * 0.7
    )
    liabilities = s.loans + s.bond_interest_liability
    return (assets - liabilities) / s.shares_outstanding


def predict_one_stock(
    rng: MiscRNG, s: StockInput, cpi: float = DEFAULT_CPI
) -> StockOutput:
    """Predict next-day state for one stock, consuming RNG calls in place.

    Mirrors ``NationStock__daily_price_update`` byte-for-byte.  For
    unlisted or foreign stocks the function is a no-op (consumes no RNG)
    and returns the input state unchanged — matching the engine's
    dispatcher in ``NationStock__daily_tick``.
    """
    calls_before = len(rng.trace)

    if not s.is_listed:
        # Unlisted: price = max(0.1, bvps).  No RNG.
        new_price = max(0.1, s.book_value_per_share)
        return StockOutput(
            group_recno=s.group_recno,
            base_stock_price=new_price,
            sentiment=s.sentiment,
            rng_calls=0,
        )
    if s.is_foreign:
        # Foreign: daily tick is a no-op; monthly foreign_price_update runs elsewhere.
        return StockOutput(
            group_recno=s.group_recno,
            base_stock_price=s.base_stock_price,
            sentiment=s.sentiment,
            rng_calls=0,
        )

    price = float(s.base_stock_price)
    sentiment = _clamp(float(s.sentiment), -100.0, 100.0)
    hq_type = s.hq_type

    # ---- Step 2: sentiment_random + market_random --------------------------
    # The engine calls random(int(|sentiment|)) FIRST (even when sentiment is
    # zero — random(0) still advances the seed), then random(200).
    rng.label(f"stock#{s.group_recno}.sentiment_rand")
    sent_mag = int(abs(sentiment))
    r_sent = rng.rng(sent_mag)
    signed_sent_rand = -r_sent if sentiment <= 0.0 else r_sent

    rng.label(f"stock#{s.group_recno}.market_rand")
    r_market = rng.rng(200)

    price_change_pct = ((r_market - 100) + signed_sent_rand) / 10000.0
    price = price + price * price_change_pct

    # ---- Step 3: CPI floor -------------------------------------------------
    fired_cpi_floor = False
    if price < cpi * 0.1:
        fired_cpi_floor = True
        rng.label(f"stock#{s.group_recno}.cpi_price_recover")
        r_price_boost = rng.rng(10)
        price += r_price_boost / 100.0

        rng.label(f"stock#{s.group_recno}.cpi_sent_recover")
        r_sent_boost = rng.rng(30)
        sentiment += r_sent_boost

    # Absolute floor (no RNG).
    if price <= 0.01:
        price = 0.01

    # ---- Step 4: 5-way NPC investor dispatch -------------------------------
    earnings_metric = (s.earnings_metric_driver * 3.0) / (cpi * 500_000.0)

    rng.label(f"stock#{s.group_recno}.case_selector")
    case = rng.rng(5)

    fired_surge = False
    if case == 0:
        # SELL gate: NOT bank/insurance AND (pe <= 0 OR pe > metric + 20)
        if hq_type not in (10, 11) and (
            s.pe_ratio_stored <= 0.0 or s.pe_ratio_stored > earnings_metric + 20.0
        ):
            rng.label(f"stock#{s.group_recno}.case0.sentiment_penalty")
            r_pen = rng.rng(4)
            sentiment -= r_pen
            rng.label(f"stock#{s.group_recno}.case0.surge_gate")
            r_gate = rng.rng(3)
            if r_gate == 0:
                fired_surge = True
                rng.label(f"stock#{s.group_recno}.case0.price_crash")
                r_crash = rng.rng(10)
                price -= price * r_crash / 100.0

    elif case == 1:
        # BUY gate: NOT bank/insurance AND pe > 0 AND pe < metric + 10
        if (
            hq_type not in (10, 11)
            and s.pe_ratio_stored > 0.0
            and s.pe_ratio_stored < earnings_metric + 10.0
        ):
            rng.label(f"stock#{s.group_recno}.case1.sentiment_boost")
            r_boost = rng.rng(4)
            sentiment += r_boost
            rng.label(f"stock#{s.group_recno}.case1.surge_gate")
            r_gate = rng.rng(3)
            if r_gate == 0:
                fired_surge = True
                rng.label(f"stock#{s.group_recno}.case1.price_surge")
                r_surge = rng.rng(20)
                price += price * r_surge / 100.0

    elif case == 2:
        # FUNDAMENTAL VALUATION CHECK
        bvps_w = _compute_weighted_bvps(s) * 0.7
        # Engine: if (bvps_w < price || bvps_w == price) break;
        # We use strict > because == breaks too; that's the engine semantics.
        if bvps_w > price:
            rng.label(f"stock#{s.group_recno}.case2.undervaluation")
            r_under = rng.rng(100)
            under_pct = ((bvps_w - price) * r_under) / price
            if under_pct > 0.0:
                sentiment += under_pct
                if sentiment > 100.0:
                    sentiment = 100.0
                rng.label(f"stock#{s.group_recno}.case2.surge_gate")
                r_gate = rng.rng(3)
                if r_gate == 0:
                    fired_surge = True
                    rng.label(f"stock#{s.group_recno}.case2.price_surge")
                    r_surge = rng.rng(10)
                    price += price * r_surge / 100.0

    elif case == 3:
        # SENTIMENT MEAN REVERSION
        if sentiment <= 0.0:
            n = int(-sentiment / 5.0)
            rng.label(f"stock#{s.group_recno}.case3.revert_neg")
            r = rng.rng(n)
            sentiment += r
        else:
            n = int(sentiment / 5.0)
            rng.label(f"stock#{s.group_recno}.case3.revert_pos")
            r = rng.rng(n)
            sentiment -= r

    elif case == 4:
        # RANDOM NOISE
        rng.label(f"stock#{s.group_recno}.case4.noise")
        r = rng.rng(11)
        sentiment += r - 5

    # ---- Step 5: EPS shock -------------------------------------------------
    fired_eps_shock = False
    if hq_type != 11:
        new_eps = s.eps_adjusted_new if s.eps_adjusted_new is not None else s.eps_adjusted_stored
        old_eps = s.eps_adjusted_stored
        if new_eps != old_eps:
            delta = new_eps - old_eps
            mult = 50.0 if s.eps_basic >= 0.0 else -50.0
            if old_eps != 0.0:
                eps_delta = (delta * mult) / old_eps
            else:
                # Engine semantics (2026-04-22c update): x87 fdiv by zero on a
                # non-zero dividend produces ±infinity without raising — the
                # subsequent clamp to ±100 yields exactly ±100.  Previous
                # iteration of this block silently skipped the RNG call on
                # old_eps==0, which is a bug: the engine fires random(100)
                # in that case.  The sign is (sign of delta) * (sign of mult)
                # — same as the x87 fdiv sign rules.
                sign = 1.0 if (delta >= 0.0) == (mult >= 0.0) else -1.0
                eps_delta = 100.0 * sign
            eps_delta = _clamp(eps_delta, -100.0, 100.0)
            if eps_delta > 0.0:
                rng.label(f"stock#{s.group_recno}.eps_shock_pos")
                r = rng.rng(int(eps_delta))
                sentiment += r
                fired_eps_shock = True
            elif eps_delta < 0.0:
                rng.label(f"stock#{s.group_recno}.eps_shock_neg")
                r = rng.rng(int(-eps_delta))
                sentiment -= r
                fired_eps_shock = True

    calls_after = len(rng.trace)

    return StockOutput(
        group_recno=s.group_recno,
        base_stock_price=price,
        sentiment=sentiment,
        rng_calls=calls_after - calls_before,
        case_selector=case,
        fired_cpi_floor=fired_cpi_floor,
        fired_case_surge=fired_surge,
        fired_eps_shock=fired_eps_shock,
    )


# ---------------------------------------------------------------------------
# Inter-group AI-tail advance (2026-04-22)
# ---------------------------------------------------------------------------
#
# Between one group's NationStock__daily_tick and the next, the engine runs
# the GroupAI post-amble (GroupAI__daily_update @ 0x00438180) on every
# AI-competitor group.  That post-amble contains a handful of Misc::random
# call sites that fire on gated daily cadences.
#
# The predictor has historically swept a scalar inter_group_advance ∈ [0, 5]
# and taken the best match — which works on stable days but drifts by 1-2/21
# on base-game runs (93.8%) and crashes to 23% on DLC-scenario saves where
# an additional RNG site (FUN_0042e2d0 @ 0x0042e78b) fires.
#
# Instead of sweeping, we now EMULATE the post-amble's two highest-frequency
# RNG sites directly, using the gating predicates reverse-engineered in
# Stock_Tick_InterGroup_RNG_Analysis.md §§3.1-3.2:
#
#   base game  — 0x004383cd + 0x00438425 daily Bernoullis (1-2/day/group)
#   DLC        — 0x0042e78b town-scoring index rotation  (1/day/qualifying)
#
# These two sources alone account for the observed inter-group k histogram
# {0:70, 5:1, 17:1, 20:1, 25:1} on the 74-day base run and the ia=1-on-10/15
# pattern in the 15-day DLC run.  The remaining ~36 cycle-gated dispatchers
# (§3.3 of the analysis) fire far less often and are left to the scalar
# sweep fallback path — see ai_tail_advance's docstring for the strategy.


@dataclass
class ScenarioState:
    """Snapshot of global game state feeding :func:`ai_tail_advance`.

    Fed either from :meth:`caplab_sim.rng_reader.LiveGameReader.read_scenario_flags`
    (for live monitoring) or hand-built for unit tests / save-file replays.

    The one field that actually matters for a 2026-04-22 ship is
    ``dlc_stock_market_active`` — when it's false (every base-game save,
    _____001/_____002 included), the ai_tail_advance function degrades
    to the base-game daily Bernoulli path.  When it's true, the DLC
    town-scoring advance fires on qualifying groups.

    ``game_date`` is required for cadence gating.  It should match
    ``DAT_00a459b0`` as read by
    :meth:`caplab_sim.rng_reader.LiveGameReader.read_game_date`.
    """

    game_date: int = 0
    dlc_economy_mode: bool = False
    dlc_stock_market_active: bool = False


def dlc_inter_group_advance(
    group: StockInput, state: ScenarioState
) -> int:
    """Ship-blocker fix for the DLC 23% regression — §4.3 of the analysis.

    Returns the number of ``Misc::random`` calls that fire inside
    ``FUN_0042e2d0`` at ``0x0042e78b`` for this group on this day.  At
    most one (the callee has a single RNG site — the town-scoring loop
    start-index rotation).  Returns 0 if any gate fails.

    Gate chain (read top-down in Ghidra at ``GroupAI__daily_update``
    around ``LAB_00438506``):

      1.  DLC flags: ``dlc__economy_mode_flag1 != 0 && dlc__stock_market_active != 0``
      2.  Group type: ``group.hq_type ∈ {0, 10}`` (bank/insurance excluded)
      3.  Group build flag: ``group.dlc_build_flag != 0`` (OR the global
          ``bVar8`` AI-enabled flag is on — we treat the per-group flag
          as the primary gate because the global is always on in practice
          on DLC saves).
      4.  Group has an AI leader: ``group.ai_leader_slot != 0`` (the
          GroupAI post-amble doesn't run otherwise — this is the
          ``group[0xdc] == 0 → return`` gate at 0x00438194).
      5.  Group is an AI competitor: ``group.group_type == 3`` (the
          ``group[0x74] != 3 → return`` gate at 0x004384dc).
      6.  Cadence: ``game_date % period == group_recno % period`` with
          ``period = 60`` for ``hq_type == 10`` else ``90``.  The
          Ghidra decompile shows a third 180-day path gated on
          ``FUN_0051d5f0()`` — we conservatively use 90 because that
          produces the larger firing set (DLC regression is driven by
          too FEW advances, not too many; the 180-day gate is a strict
          subset of the 90-day one).  Refine to the true
          FUN_0051d5f0 predicate when the helper is disassembled.

    The function deliberately does NOT emulate the sub-gates 4-7 of the
    callee (wealth, profit, DAT_a44b88, cash-threshold, build-slot).
    Those would reduce the false-positive rate by gating the advance
    on game-state fields the live reader doesn't yet expose
    (FUN_00458d00, FUN_00458a80).  Given the ship-blocker is that we
    return 0 when we should return 1 (driving the 23% regression), an
    over-eager +1 is strictly safer than an under-eager 0 — the
    scalar-sweep fallback in ``live_monitor.sweep_best_k`` will still
    find the right answer if ai_tail_advance over-fires.

    Returns 0 | 1.
    """
    if not (state.dlc_economy_mode and state.dlc_stock_market_active):
        return 0
    # Player / non-AI groups never run the tail.
    if group.ai_leader_slot == 0 or group.group_type != 3:
        return 0
    if group.hq_type not in (0, 10):
        return 0
    if group.dlc_build_flag == 0:
        return 0

    period = 60 if group.hq_type == 10 else 90
    if (state.game_date % period) != (group.group_recno % period):
        return 0

    return 1


def base_inter_group_advance(
    group: StockInput, state: ScenarioState
) -> int:
    """Base-game daily Bernoulli tail — §3.1 rows 1-2 of the analysis.

    Returns the number of ``Misc::random`` calls fired in the
    ``GroupAI__daily_update`` tail AFTER the NationStock tick for this
    group, covering the two daily-conditional direct-RNG sites:

      * ``0x004383cd`` — site 1.  Fires only when the AI-startup gate
        has elapsed and ``hq_type ∉ {6,7,8}`` and the game_end_state
        predicate holds (see §3.1 row 1 for the full gate chain).
      * ``0x00438425`` — site 2.  Gated on a production-snapshot feature
        flag and ``hq_type == 0``.

    **Current status: returns 0 (disabled).**

    Historical context (critical — do not re-enable without re-tracing):

    An earlier iteration of this function returned ``1`` for every
    AI-competitor group with ``hq_type ∉ {6,7,8}``.  That broke the
    live-monitor accuracy from 93.8% → 7.9% (5/63 match, with every
    group g2-g21 at 0/3 per-group hit rate).  The observed pre-fix
    scalar-sweep k-histogram on the 74-day base-game run was::

        {0: 70, 5: 1, 17: 1, 20: 1, 25: 1}

    k=0 on 70/74 transitions means the base-tail RNG is almost
    always NOT firing in practice on the current save — consistent
    with the AI-startup gate (365/90 days gated on
    ``DAT_00a449a8``, see §3.1 row 1) still blocking site 1 in the
    current game state.  Firing ``+1`` per AI competitor (21
    competitors × 74 days = ~1550 injected steps) drowns out the
    actual stock-tick signal.

    The correct conservative posture is to return 0 for base-game
    saves and rely entirely on:

      1.  The DLC branch (``dlc_inter_group_advance``), which is
          independently gated on ``dlc__stock_market_active`` so it
          contributes 0 on base-game saves and is a strict improvement
          on DLC saves.
      2.  The scalar-sweep fallback in ``live_monitor.sweep_best_k``,
          which absorbs the rare ``{5, 17, 20, 25}`` outliers via its
          2-D grid over ``(k, inter_group_advance)``.

    Re-enabling this path requires wiring up at least:

      * ``DAT_00a449a8`` (scenario-specific startup period toggle).
      * ``DAT_00a44c35`` + ``DAT_00a44c21`` + ``DAT_0090e140+0x1650``
        (game_end_state predicate from §3.1 row 1).
      * A proof that the per-group +1 matches the live k-histogram
        bucket distribution (e.g. k=N on a majority of transitions
        where N is the group count).

    Until those are traced and validated, this function returns 0.
    See ``Stock_Tick_InterGroup_RNG_Analysis.md §4.3`` ("recommended
    minimal patch — ship-blocker fix") which specifies DLC-only as
    the ship posture.

    Returns 0 (always, for now).
    """
    # Player group (ai_leader_slot == 0) never runs the tail regardless.
    if group.ai_leader_slot == 0 or group.group_type != 3:
        return 0
    # Bank/insurance HQ types skip site 1 entirely (0x004383a1 gate).
    if group.hq_type in (6, 7, 8):
        return 0
    # Site 1 is further gated on an AI-startup period check and a
    # game_end_state predicate that we don't yet read from the live
    # process. Empirically on the current save those gates block the
    # RNG call 70/74 days, so the safest emulation is to assume the
    # block and defer to the scalar sweep for residuals.
    return 0


def ai_tail_advance(
    group: StockInput, state: ScenarioState
) -> int:
    """Combined inter-group RNG budget for one group's AI post-amble.

    This is the "Option A" predictor-side emulator from
    Stock_Tick_InterGroup_RNG_Analysis.md §4.1, scoped to the two
    dominant RNG sources (§3.1 + §3.2).  The ~36 less-frequent
    cycle-gated dispatchers in §3.3 are deliberately omitted — they
    fire on multi-day recno-offset cadences and don't stack up to the
    6%/group daily drift that §3.1 produces.

    A single-row §3.3 prototype for Row 12
    (``GroupAI__loan_management_30day``) was built and live-tested on
    2026-04-22; the minimum-bound predicate over-fired and caused a
    12pp regression in overall match rate (see SESSION_LOG 2026-04-22f
    and analysis doc §3.3b).  The Row 12 code was reverted and the
    scalar-sweep + DLC baseline remains the ship posture.

    Returns the total number of ``Misc::random`` calls the predictor
    should advance the seed by between this group and the next.
    """
    return (
        base_inter_group_advance(group, state)
        + dlc_inter_group_advance(group, state)
    )


# ---------------------------------------------------------------------------
# Market-wide prediction
# ---------------------------------------------------------------------------

def predict_all_stocks(
    rng: MiscRNG,
    stocks: Sequence[StockInput],
    cpi: float = DEFAULT_CPI,
    *,
    reorder: bool = True,
    inter_group_advance: int = 0,
    scenario_state: Optional["ScenarioState"] = None,
) -> MarketOutput:
    """Replay the per-group stock tick.

    By default (``reorder=True``) sorts ``stocks`` by ``-group_recno``
    before iterating, which matches the engine's empirical firing order
    (see module docstring).  Pass ``reorder=False`` only if the caller
    has already applied the correct ordering and wants to control it
    explicitly (e.g. for unit tests with hand-built fixtures, or for
    counterfactual "what if the engine iterated differently" studies).

    ``stocks`` may include non-listed / foreign / deleted groups; they
    consume no RNG (matching the engine).  Deleted groups (group_type
    == 0) are skipped before dispatch, same as the engine's deleted
    guard at ``iVar1 == 0``.

    ``inter_group_advance`` — scalar RNG steps to inject between every
    consecutive active-group pair.  Skipped for the very first active
    group.  Used by the live monitor's 2-D scalar sweep.  Defaults to 0.

    ``scenario_state`` — optional ScenarioState for per-group AI-tail
    emulation (2026-04-22).  When provided, each group's inter-group
    advance is computed via :func:`ai_tail_advance` using the PREVIOUS
    group's ``StockInput`` — covering the base-game daily Bernoullis
    and the DLC town-scoring branch.  The scalar ``inter_group_advance``
    is ADDED on top of the per-group amount (so the monitor can still
    sweep a small scalar residual for the §3.3 periodic dispatchers).
    When ``scenario_state`` is None, the legacy scalar-only behaviour
    is preserved exactly.
    """
    start_seed = rng.seed
    out = MarketOutput(starting_seed=start_seed)

    if reorder:
        iter_stocks = sorted(stocks, key=lambda s: -s.group_recno)
    else:
        iter_stocks = list(stocks)

    first_active = True
    prev_stock: Optional[StockInput] = None
    for s in iter_stocks:
        if s.group_type == 0:
            # Deleted group — skip entirely, no RNG.
            continue
        if not first_active:
            # Advance over GroupAI RNG calls that fired after the
            # previous group's stock tick and before this one's.
            advance = inter_group_advance
            if scenario_state is not None and prev_stock is not None:
                advance += ai_tail_advance(prev_stock, scenario_state)
            if advance > 0:
                rng.set_seed(simulate_lcg(rng.seed, advance))
        first_active = False
        out.per_stock.append(predict_one_stock(rng, s, cpi=cpi))
        prev_stock = s

    out.ending_seed = rng.seed
    out.total_rng_calls = sum(p.rng_calls for p in out.per_stock)
    return out


# ---------------------------------------------------------------------------
# Save-file binding
# ---------------------------------------------------------------------------

def stock_inputs_from_save(save) -> List[StockInput]:
    """Build ``StockInput`` records from a ``CapLabSave`` in recno order.

    Pulls per-Group fields through the already-parsed ``Group`` dataclass
    and per-stock fields through ``Group.stock`` (lazy NationStock decode).

    ``hq_type`` is not currently populated from the save; it's a vtable-
    dispatched short at runtime and stays 0 on Jan 1 1990 (no HQ firm yet).
    For downstream use in active saves we'll need to trace it from the
    Group's type + attached firms.
    """
    import struct as _s

    result: List[StockInput] = []
    for g in save.groups:
        ns = g.stock
        # Pull the Case-2 asset block directly from the Group's raw_base blob
        # because the `Group` dataclass only exposes account_balance[14]
        # starting at +0x39D0 — the offsets we care about are at +0x39D8,
        # +0x39E0, +0x39E8, +0x39F0, +0x39F8, +0x3A00, +0x3A10, +0x3A18,
        # +0x3A20 (loan principal), +0x3A28 (bond interest).  Most of those
        # already appear as account_balance slots, but indexing through
        # the typed attribute is noisier than reading directly.
        rb = g.raw_base
        def _d(off: int) -> float:
            if len(rb) < off + 8:
                return 0.0
            return _s.unpack_from("<d", rb, off)[0]

        # AI-tail gating fields (see StockInput docstring).  Offsets read
        # directly from raw_base since the Group dataclass does not expose
        # ai_leader_slot (+0xdc) or the DLC build flag (+0x21f).
        def _h(off: int) -> int:
            if len(rb) < off + 2:
                return 0
            return _s.unpack_from("<h", rb, off)[0]
        def _b(off: int) -> int:
            if len(rb) < off + 1:
                return 0
            return rb[off]

        si = StockInput(
            group_recno=g.recno,
            is_listed=bool(ns.is_listed),
            is_foreign=bool(ns.is_foreign),
            group_type=g.group_type,
            hq_type=0,                      # placeholder, see docstring
            shares_outstanding=ns.shares_outstanding,
            base_stock_price=ns.base_stock_price,
            book_value_per_share=ns.book_value_per_share,
            eps_basic=ns.eps_basic,
            eps_adjusted_stored=ns.eps_adjusted,
            pe_ratio_stored=ns.earnings_yield_ratio,  # +0x43A0
            sentiment=ns.sentiment,
            bank_deposits=_d(0x39D8),
            corp_cash=g.corp_cash,          # Group+0xCC0
            inventory=_d(0x39E0),
            business_assets=_d(0x39E8),
            land=_d(0x39F0),
            technology=_d(0x39F8),
            stocks_held=_d(0x3A00),
            bank_net_assets=_d(0x3A10),
            insurance_net_assets=_d(0x3A18),
            loans=g.loan_principal,
            bond_interest_liability=g.bond_interest_liability,
            earnings_metric_driver=_d(0x3B38),
            trailing_12m_net_profit=g.trailing_12m_net_profit,
            trailing_12m_special_revenue=g.trailing_12m_special_revenue,
            eps_adjusted_new=None,          # populated below
            ai_leader_slot=_h(0xdc),
            dlc_build_flag=_b(0x21f),
        )
        # Populate fresh EPS via Group::compute_eps(1) port — this is what
        # the EPS-shock branch in predict_one_stock compares against
        # ``eps_adjusted_stored``.  Without this, new == stored always and
        # the shock branch never fires.  See compute_eps() docstring for
        # the skipped subsidiary-adjustment branch (safe on Jan 1 1990
        # since no group owns an HQ firm yet with 12-month history).
        si.eps_adjusted_new = compute_eps(si)
        result.append(si)
    return result
