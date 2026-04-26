"""Dataclasses that hold parsed Capitalism Lab save state.

These are lightweight value types — they do no parsing themselves.  The
``parser`` module owns the byte-level logic and populates these records.

Policy:
    * All fields that we have confidently mapped are exposed as typed
      attributes.  For fields that are in the save blob but not yet understood,
      we store the raw byte slice in ``.raw``.  This preserves round-trip
      capability and lets downstream code poke at unmapped ranges without
      touching the parser.
    * Monthly arrays are always ``tuple[float, ...]`` of length 36, indexed
      such that element ``i`` corresponds to the month the engine treats as
      ``DAT_00a45a1c == i``.  The calling code is responsible for rotating
      them relative to the current month index if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---- calendar / header ----------------------------------------------------

@dataclass
class GameInfo:
    """Top-level game state header.

    Holds the live calendar (year / month / day / day_counter) plus the
    raw 0x71E8 GameInfo payload (tag 0x1065) for downstream analysis.

    Important: despite the name, the calendar fields here are read from
    the **ScenarioCfg** payload (tag 0x1068), not the GameInfo payload.
    The 0x1065 payload only holds historical per-month/per-year stamp
    arrays; scanning it for "the current date" matches stale snapshot
    values (see SESSION_LOG 2026-04-21 for the forensic story).

    ``game_date`` is the Julian Day Number (JDN) of the current game day
    — e.g. JDN 2,447,893 = 1990-01-01.  It advances by exactly 1 per
    game tick.  Use :func:`caplab_save.parser.jdn_to_gregorian` to
    decode it into a (year, month, day) triple that matches
    ``(self.year, self.month_of_year, self.day_of_month)``.
    """

    game_date: int                    # Julian Day Number (JDN); +1 per game day
    day_of_month: int                 # 1..31
    month_index: int                  # 0..35 (DAT_00a45a1c) or 1..12 depending on slot
    month_of_year: int                # 1..12
    year: int                         # absolute year (e.g. 1990)
    raw: bytes = field(repr=False)    # full 0x71E8 payload

    # -- new fields from ScenarioCfg --------------------------------------
    # Defaults preserve backward-compatibility for any external caller
    # that constructs a GameInfo with positional args only.
    day_of_year: int = 0              # 1..365 (1-based; Jan 1 = 1)
    days_since_start: int = 0         # 0-based; 0 on the scenario's start day
    start_day_counter: int = 0        # JDN of scenario start (ScenarioCfg+0x10)
    start_year: int = 0               # Year of scenario start (ScenarioCfg+0x14)
    scenario_cfg_offset: int = -1     # byte offset of tag 0x1068 header in blob0


# ---- economy --------------------------------------------------------------

@dataclass
class EconomyState:
    """Mid-game macroeconomic state (subset of the Economy struct).

    Offsets are relative to the Economy struct base.  See
    ``caplab_save.constants`` for the map.
    """

    base_interest_rate: float
    gdp_growth_rate: float
    cycle_phase: int                  # 1..7
    interest_rate_level: int          # 1..9
    stock_index_target: float
    annual_inflation_rate: float
    price_level_CPI: float
    price_level_PPP: float
    raw: bytes = field(repr=False)    # at least 0x78 bytes of Economy struct
    found_at_blob: int = -1           # byte offset into blob0 (debug aid)


# ---- group ----------------------------------------------------------------

@dataclass
class Group:
    """One Group record (tag 0x10CA).

    A Group is a business conglomerate / holding company.  The record has a
    0x4A18-byte flat base plus two heap-referenced sub-blocks (FirmRes array
    and NationStock block) that we keep as raw bytes for now.
    """

    recno: int                                 # 1-based position in GroupArray
    group_type: int                            # 0 = deleted/empty slot
    group_recno_field: int                     # Group+0x06
    person_recno: int                          # Group+0xDC
    firm_count: int                            # Group+0xE4
    active_count: int                          # Group+0xE8

    corp_cash: float                           # Group+0xCC0
    net_worth: float                           # Group+0xCC8
    strategy_budget: int                       # Group+0xCD4
    trailing_12m_net_flow: float               # Group+0xCD8
    trailing_12m_net_profit: float             # Group+0xCE0
    trailing_12m_special_revenue: float        # Group+0xCE8
    tax_reserve: float                         # Group+0xCF0

    monthly_net_flow: Tuple[float, ...]        # Group+0xCF8  double[36]
    monthly_net_profit: Tuple[float, ...]      # Group+0xE18  double[36]
    monthly_gross_field: Tuple[float, ...]     # Group+0xF38  double[36]
    monthly_special_revenue: Tuple[float, ...] # Group+0x1058 double[36]
    monthly_business_revenue: Tuple[float, ...]# Group+0x1178 double[36]
    monthly_expense_total: Tuple[float, ...]   # Group+0x1298 double[36]
    monthly_net_worth_snap: Tuple[float, ...]  # Group+0x13B8 double[36]
    monthly_sales_flow: Tuple[float, ...]      # Group+0x15F8 double[36]

    account_balance: Tuple[float, ...]         # Group+0x39D0 double[14]
    loan_principal: float                      # Group+0x3A20 (account_balance[10])
    bond_interest_liability: float             # Group+0x3A28 (account_balance[11])
    account_balance_2: Tuple[float, ...]       # Group+0x3A40 double[14]

    dept_revenue_current: Tuple[float, ...]    # Group+0x3610 double[24]
    dept_revenue_ytd: Tuple[float, ...]        # Group+0x3790 double[24]
    dept_revenue_alltime: Tuple[float, ...]    # Group+0x3910 double[24]

    # FirmRes subarray (Group+0x394 heap block): DAT_0090e13c shorts,
    # one per firm-resource descriptor.  Semantics TBD; NOT an ownership
    # list of firms owned by this group.
    firmres_shorts: Tuple[int, ...]

    # firm_recnos is populated by the top-level loader after both the
    # Group array and the Firm array have been parsed: it lists the
    # firm recnos whose `group_recno` (Firm+0xF8) points back to this
    # Group.  Empty until linkage runs.
    firm_recnos: Tuple[int, ...] = field(default_factory=tuple)

    raw_base: bytes = field(repr=False, default=b"")        # full 0x4A18 base
    raw_firmres_subarray: bytes = field(repr=False, default=b"")
    raw_nationstock: bytes = field(repr=False, default=b"")

    @property
    def is_deleted(self) -> bool:
        return self.group_type == 0

    @property
    def stock(self) -> "NationStock":
        """Typed view of :attr:`raw_nationstock`.

        The save file stores one NationStock record (0x43B0 bytes) per
        Group as a heap-referenced sub-block (Group+0x70 points to it at
        runtime).  Lazily decoded on access so Group remains a plain
        dataclass.  The ``raw_nationstock`` attribute is the source of
        truth either way.
        """
        from caplab_save.structs import NationStock as _NS  # local to avoid cycles
        return _NS.from_bytes(self.raw_nationstock)


# ---- nation stock ---------------------------------------------------------

@dataclass
class NationStock:
    """Per-Group stock record (heap block at Group+0x70, size 0x43B0).

    Only the fields driving ``NationStock__daily_price_update``
    (0x00647d30) and ``NationStock__daily_tick`` (0x00647530) are decoded
    explicitly; the rest — history arrays, per-person share ownership,
    overvaluation accumulators — remain in :attr:`raw` for now.

    Field layout is anchored by the Ghidra decompile of
    ``NationStock__daily_price_update`` (confirmed 2026-04-21 evening);
    see ``engine/Engine_StockMarket.txt`` for the full table.
    """

    group_recno: int                   # +0x00 (short)
    is_listed: int                     # +0x02 (byte: 0 = unlisted)
    is_foreign: int                    # +0x03 (byte: nonzero = foreign)
    day_initialized: int               # +0x04 (int, JDN — set on first daily tick)
    shares_outstanding: float          # +0x08 (double)
    initial_stock_price: float         # +0x10 (double, IPO price)
    base_stock_price: float            # +0x18 (double, core engine price)
    market_cap_base: float             # +0x20 (double)
    book_value_per_share: float        # +0x28 (double, Group+0xCC8 / shares)
    eps_basic: float                   # +0x30 (double; P/E display gate)
    eps_adjusted: float                # +0x38 (double; used for EPS-shock delta)
    pe_ratio: float                    # +0x40 (double, stored)
    sentiment: float                   # +0x98 (double, [-100, +100] accumulator)
    earnings_yield_ratio: float        # +0x43A0 (double, used in case 0/1 gates)
    raw: bytes = field(repr=False, default=b"")

    @classmethod
    def from_bytes(cls, blob: bytes) -> "NationStock":
        """Decode a 0x43B0-byte NationStock record.

        Returns an all-zero record if ``blob`` is empty or too short;
        this matches the parser's behaviour for Groups with no stock
        subarray (shouldn't normally happen, but is defensive).
        """
        import struct as _s
        if len(blob) < 0x43A8:
            return cls(
                group_recno=0, is_listed=0, is_foreign=0,
                day_initialized=0, shares_outstanding=0.0,
                initial_stock_price=0.0, base_stock_price=0.0,
                market_cap_base=0.0, book_value_per_share=0.0,
                eps_basic=0.0, eps_adjusted=0.0, pe_ratio=0.0,
                sentiment=0.0, earnings_yield_ratio=0.0,
                raw=bytes(blob),
            )
        return cls(
            group_recno           = _s.unpack_from("<H", blob, 0x00)[0],
            is_listed             = blob[0x02],
            is_foreign            = blob[0x03],
            day_initialized       = _s.unpack_from("<i", blob, 0x04)[0],
            shares_outstanding    = _s.unpack_from("<d", blob, 0x08)[0],
            initial_stock_price   = _s.unpack_from("<d", blob, 0x10)[0],
            base_stock_price      = _s.unpack_from("<d", blob, 0x18)[0],
            market_cap_base       = _s.unpack_from("<d", blob, 0x20)[0],
            book_value_per_share  = _s.unpack_from("<d", blob, 0x28)[0],
            eps_basic             = _s.unpack_from("<d", blob, 0x30)[0],
            eps_adjusted          = _s.unpack_from("<d", blob, 0x38)[0],
            pe_ratio              = _s.unpack_from("<d", blob, 0x40)[0],
            sentiment             = _s.unpack_from("<d", blob, 0x98)[0],
            earnings_yield_ratio  = _s.unpack_from("<d", blob, 0x43A0)[0],
            raw                   = bytes(blob),
        )


# ---- firm -----------------------------------------------------------------

@dataclass
class Firm:
    """One Firm record (tag 0x10CB).

    Firms are polymorphic (CFirm base + per-type subclass).  We store the base
    blob plus the subclass blob raw.  Common base fields are decoded.

    ``group_recno`` (= Firm+0xF8) is the *owning* Group's recno — the
    canonical firm→group link used by the game's revenue/expense
    accounting.  Zero when the firm is unowned.
    """

    recno: int
    firm_type: int                             # Firm+0x08; 0 = deleted
    firm_subtype: int                          # Firm+0x0A

    ai_capacity_target: float                  # Firm+0x88
    utilization: float                         # Firm+0x8C
    demand_bonus: float                        # Firm+0x11C

    monthly_net_profit: Tuple[float, ...]      # Firm+0x180 double[36]
    monthly_revenue: Tuple[float, ...]         # Firm+0x2A0 double[36]
    monthly_expense: Tuple[float, ...]         # Firm+0x3C0 double[36]

    dept_block_1: bytes = field(repr=False)    # 0x4E0
    dept_block_2: bytes = field(repr=False)    # 0x540
    dept_block_3: bytes = field(repr=False)    # 0x5A0
    dept_block_4: bytes = field(repr=False)    # 0x660

    raw_base: bytes = field(repr=False)        # 0x718 base blob
    raw_subclass: bytes = field(repr=False)    # variable, per firm_type
    raw_unitgroups: bytes = field(repr=False)  # N * 0x48 bytes
    raw_dynamic: bytes = field(repr=False, default=b"")  # type 0x25 only

    @property
    def is_deleted(self) -> bool:
        return self.firm_type == 0

    @property
    def unit_group_count(self) -> int:
        # 0x48 bytes per unit group (see Structs_Serialization.txt)
        return len(self.raw_unitgroups) // 0x48

    @property
    def group_recno(self) -> int:
        """Owning Group recno; parsed from Firm+0xF8 (int).

        Lazily decoded from ``raw_base`` so Firm remains a plain dataclass
        (no bespoke __init__ logic).
        """
        import struct as _s
        return _s.unpack_from("<i", self.raw_base, 0xF8)[0]

    @property
    def subclass(self):
        """Typed view of :attr:`raw_subclass` when a decoder is registered.

        Returns ``None`` for firm types without a typed decoder — see
        :pymod:`caplab_save.firm_subclass` for the coverage list.  The
        ``raw_subclass`` attribute is the source of truth either way.
        """
        from caplab_save.firm_subclass import decode_firm_subclass
        return decode_firm_subclass(self.firm_type, self.raw_subclass)


# ---- nation ---------------------------------------------------------------

@dataclass
class Nation:
    """One Nation record (tag 0x10CE).

    Base is 0x3C9B8 bytes.  Four subarrays follow for active nations (those
    with Nation+0x3A == 0).  We leave the subarrays raw pending further RE.
    """

    recno: int
    active: bool                               # Nation+0x3A == 0
    raw_base: bytes = field(repr=False)        # 0x3C9B8 bytes
    raw_per_firm: bytes = field(repr=False, default=b"")
    raw_per_group: bytes = field(repr=False, default=b"")
    raw_per_firmres: bytes = field(repr=False, default=b"")
    raw_per_firmtype: bytes = field(repr=False, default=b"")


# ---- RNG ------------------------------------------------------------------

@dataclass
class RNGState:
    """Snapshot of the game's deterministic RNG.

    The Capitalism Lab RNG is a Borland-C LCG (multiplier 0x15A4E35,
    increment 1) stored at Misc+0x7C (uint32 seed) and an independent global
    call counter at DAT_00de5bd8 (uint32).  The seed is persisted into the
    save blob at ``constants.RNG_SEED_OFFSET_BLOB0``.
    """

    seed: int                                  # uint32
    counter: Optional[int] = None              # if we can locate it; else None
    source_offset: int = -1                    # byte offset within blob0
