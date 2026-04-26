"""Formula constants swappable for counterfactual simulations.

Phase 1 only exposes constants directly tied to the RNG / tick order —
enough for the seed-replay harness. Later phases will add brand decay,
ad score caps, AI thresholds, etc. (see architecture plan in project
kickoff brief, Phase 6).

Every value here is taken from the reverse-engineered reference docs
(``capmath.txt``, ``RNG_Call_Map.txt``, ``Engine_*.txt``) with the
source note in a comment. When changing a value for a counterfactual,
clone the module and swap the import at the call site — do NOT mutate
the module in place (keeps counterfactuals reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- RNG primitive (see RNG_Call_Map.txt §1) ------------------------------

LCG_MULTIPLIER: int = 0x015A4E35
LCG_INCREMENT: int = 1

# ---- feature flags driving the daily RNG budget ---------------------------
# These mirror engine globals (e.g. DAT_00a458d8) that the game sets from
# INI parameters. For a clean-start base-only save they default as below;
# override per-save via the SimConstants factory once we parse them out of
# the blob.

# Site-spawn defaults off: empirically verified against the _____001 →
# _____002 consecutive-day pair (captured 2026-04-21). With this flag
# on, the formula predicts ``n_firms * (1 + 5/120) ≈ 180`` extra LCG
# calls per day — but the observed one-day advance for that pair is
# only 182 LCG steps total. The earlier ``True`` default assumed the
# engine global DAT_00a458d8 was set for the reference scenario; it
# isn't. Override via ``SimConstants(site_spawn_enabled=True)`` once a
# save with the DLC toggle on is captured.
DEFAULT_SITE_SPAWN_ENABLED: bool = False      # DAT_00a458d8 != 0 (empirically off)
DEFAULT_LOGISTICS_ENABLED: bool = True        # DAT_00a44982 != 0
DEFAULT_RAWRES_DLC_ACTIVE: bool = False       # DAT_00a44a7f == 0
DEFAULT_HARD_MODE: bool = False               # game__hard_mode
DEFAULT_CITY_SIM_ECONOMY: bool = False        # DAT_00a44a7e != 0 (Mode 2 inflation)

# ---- calendar slice moduli (RNG_Call_Map.txt §7) --------------------------

FIRM_SLICE_MOD: int = 30        # firm_recno % 30 == game_date % 30
PERSON_SLICE_MOD: int = 30      # person_recno % 30 == game_date % 30

# ---- per-subsystem RNG call mean counts (RNG_Call_Map.txt §4) -------------
# These are the *mean* per-entity daily RNG budgets derived from the docs.
# Used by :class:`caplab_sim.tick.DailyBudget` to predict the total daily
# seed advance without doing a full entity-by-entity replay. When the
# formula is a plain linear function of entity counts, this is the cheap
# check; the real replay is built on top in later phases.

# §3 #2 NationStock__daily_price_update: mean ~5.2 calls per listed stock.
STOCK_MEAN_RNG_PER_LISTED: float = 5.2
STOCK_MIN_RNG_PER_LISTED: int = 4
STOCK_MAX_RNG_PER_LISTED: int = 9

# §3 #3 PersonArray — mean ~0.05/day/person. The 2026-04-21 revision to
# 0.10 (after the FUN_006150f0 correction) was reverted later the same
# day when the _____001 → _____002 consecutive-day pair was captured and
# the observed 182 LCG steps/day required a smaller per-person rate.
# 0.05 is consistent with the person_recno % 30 slice firing 1-2 calls
# per sliced person (~33 persons/day × ~1.5 calls ≈ 50 calls ≈ 0.05 ×
# 1000 persons). See RNG_Call_Map.txt §3 #3 and §9 residual notes.
PERSON_MEAN_RNG_PER_DAY: float = 0.05

# §3 #4 Town__daily_population_growth: exactly 1 RNG/town/day.
TOWN_RNG_PER_DAY: int = 1

# §3 #5 PartyArray per-party: 1 always + 0-2 conditional; mean ~1.3.
PARTY_MEAN_RNG_PER_DAY: float = 1.3

# §3 #6 TalentArray slice: (n/30) * P(morale<20); negligible in most saves.
TALENT_MEAN_RNG_PER_DAY: float = 0.003  # ≈ per-talent contribution

# §3 #1 FirmArray — gate RNG dominant term.
FIRM_GATE_RNG_PER_DAY: int = 1                    # when site-spawn flag on
FIRM_CASCADE_RATE: float = 5.0 / 120.0            # on gate==0, ~5 further calls

# FirmRD / FirmMedia 30-day cycles contribute ~1/30 per RD/Media firm.
FIRM_RD_MEDIA_30D_CONTRIB: float = 1.0 / 30.0

# §3 #GroupAI — per-AI-group mean (revised 2026-04-21 after distinct-site walk).
GROUPAI_MEAN_RNG_PER_DAY: float = 0.7

# §3 #10 Logistics scheduler — gate call plus rare cascade.
LOGISTICS_RNG_PER_DAY: float = 1.0

# §3 #16 Raw-resource discovery — base plus per-product/town slice.
RAWRES_BASE_RNG_PER_DAY: float = 1.0
RAWRES_EXTRA_PER_SLICE: float = 2.0     # ≈ n_products / DAT_009dd3dc rough mean

# §3 #18 GoalA daily — bursty; mean ~0.05 per AI group (rare rejection loop).
GOAL_AI_MEAN_RNG_PER_DAY: float = 0.05

# 90-day GroupAI bond-issuance (FUN_00433830) — 2 RNG per trigger per AI group.
BOND_ISSUANCE_CYCLE_DAYS: int = 90
BOND_ISSUANCE_RNG_PER_FIRE: int = 2


# ---- monthly contributions (non-zero only on month-end) --------------------
# Baseline (no shock, no crash, Mode 1) — see RNG_Call_Map.txt §3 #25.
MONTHLY_BASELINE_RNG: int = 3           # 2 gdp_growth + 1 adjust_stock_index
MONTHLY_WORSTCASE_RNG: int = 11         # shock+crash+Mode2+rate roll

# Foreign stocks — 1 always + 1 conditional per foreign stock.
FOREIGN_STOCK_MEAN_RNG_PER_MONTH: float = 1.0


# ---- save layout constants (empirical) ------------------------------------

# GAMESTATE tag — its payload's first uint32 is the Misc RNG seed. The
# upstream ``caplab_save.parser.parse_rng_seed`` was patched on 2026-04-21
# to find this tag dynamically (it used to hardcode blob0 offset 0x1159A
# from the PLAY_001/PLAY_002 layout, which was wrong for any save whose
# GameInfo payload had grown). ``caplab_sim.state.read_seed_from_blob0``
# does the same dynamic lookup and is the recommended entry point.
TAG_GAMESTATE: int = 0x1067


# ---- container dataclass (for counterfactual mode) ------------------------

@dataclass(frozen=True)
class SimConstants:
    """Bundle of swappable constants for counterfactual simulation runs.

    Construct with the defaults above, then ``dataclasses.replace`` to
    create variants for A/B comparisons. See Phase 6 in the project
    brief — ``caplab_sim.validate`` will use this to fork a save and
    run N days under each set.
    """

    lcg_multiplier: int = LCG_MULTIPLIER
    lcg_increment: int = LCG_INCREMENT

    site_spawn_enabled: bool = DEFAULT_SITE_SPAWN_ENABLED
    logistics_enabled: bool = DEFAULT_LOGISTICS_ENABLED
    rawres_dlc_active: bool = DEFAULT_RAWRES_DLC_ACTIVE
    hard_mode: bool = DEFAULT_HARD_MODE
    city_sim_economy: bool = DEFAULT_CITY_SIM_ECONOMY

    stock_mean_rng_per_listed: float = STOCK_MEAN_RNG_PER_LISTED
    person_mean_rng_per_day: float = PERSON_MEAN_RNG_PER_DAY
    town_rng_per_day: int = TOWN_RNG_PER_DAY
    party_mean_rng_per_day: float = PARTY_MEAN_RNG_PER_DAY
    talent_mean_rng_per_day: float = TALENT_MEAN_RNG_PER_DAY
    firm_gate_rng_per_day: int = FIRM_GATE_RNG_PER_DAY
    firm_cascade_rate: float = FIRM_CASCADE_RATE
    firm_rd_media_30d_contrib: float = FIRM_RD_MEDIA_30D_CONTRIB
    groupai_mean_rng_per_day: float = GROUPAI_MEAN_RNG_PER_DAY
    logistics_rng_per_day: float = LOGISTICS_RNG_PER_DAY
    rawres_base_rng_per_day: float = RAWRES_BASE_RNG_PER_DAY
    rawres_extra_per_slice: float = RAWRES_EXTRA_PER_SLICE
    goal_ai_mean_rng_per_day: float = GOAL_AI_MEAN_RNG_PER_DAY
    bond_issuance_cycle_days: int = BOND_ISSUANCE_CYCLE_DAYS
    bond_issuance_rng_per_fire: int = BOND_ISSUANCE_RNG_PER_FIRE
    monthly_baseline_rng: int = MONTHLY_BASELINE_RNG
    monthly_worstcase_rng: int = MONTHLY_WORSTCASE_RNG
    foreign_stock_mean_rng_per_month: float = FOREIGN_STOCK_MEAN_RNG_PER_MONTH
