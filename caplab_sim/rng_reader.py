"""Phase 2 stage 2 — live memory reader for Capitalism Lab.

Reads two things from a running ``CapMain.exe`` via ``/proc/<pid>/mem``:

1.  The RNG seed at ``Misc+0x7C``.  The Misc pointer is discovered via
    the call-site scan in :func:`_find_misc_pointer`.

2.  The live Group population, walked out of whichever ``GroupArray``
    singleton is populated for the active scenario (see below).
    Dereferencing ``Group+0x70`` yields the heap-allocated NationStock
    record (0x43B0 bytes) for each group.

GroupArray singletons
---------------------
Capitalism Lab v11.1.2 has **two** global ``GroupArray`` instances, each
used by a different daily update path.  They have identical header
layout (chunked array with ``stride``/``n_groups``/``entry_size``/
``n_chunks``/``chunk_stride``/``chunks_base`` at the documented offsets)
but live at different static VAs and hold different populations:

* ``0x00a46be8`` — "base game" GroupArray.  Populated in every scenario
  with the player's conglomerate plus the AI-controlled competitors.
  Pinned from ``SaveLoad__group_array_load`` (``Gfile3.cpp``), which
  walks it via the inlined field accesses
  ``DAT_00a46bec`` (stride), ``DAT_00a46bf4`` (n_groups),
  ``DAT_00a46bf8`` (entry_size), ``DAT_00a46c08`` (n_chunks),
  ``DAT_00a46c0c`` (chunk_stride), ``DAT_00a46c18`` (chunks_base).
  Per-group NationStock ticks happen through
  ``GroupAI__daily_update → Group__daily_update → NationStock__daily_tick``.

* ``0x009dd328`` — Economy-Mode DLC "NPC stock market" GroupArray.
  Pinned from ``GameTick__daily`` disassembly:
  ``MOV ECX, 0x9dd328 / CALL GlobalGrpArray__daily_stock_tick`` and
  gated on ``dlc__economy_mode_flag1``.  Holds the five industry NPC
  groups (``Healthcare``, ``Industrial``, ``Energy``, ``Consumer``,
  ``Stocks.DBF``) and is empty unless Economy-Mode is enabled.

The reader probes both on ``attach`` and walks every array whose header
passes :meth:`GroupArrayInfo.is_plausible`, so the result is correct
regardless of scenario/DLC combination.  Slots from different arrays
are yielded with distinct ``(array_tag, slot, group_ptr)`` tuples —
callers that just want stock inputs can ignore the tag.

The public surface mirrors :func:`caplab_sim.stock.stock_inputs_from_save`
so the live reader can feed straight into ``predict_all_stocks``::

    from caplab_sim.rng import MiscRNG
    from caplab_sim.stock import predict_all_stocks, DEFAULT_CPI
    from caplab_sim.rng_reader import LiveGameReader

    reader = LiveGameReader.attach()
    seed = reader.read_rng_seed()
    inputs = reader.stock_inputs()
    out = predict_all_stocks(MiscRNG(seed), inputs, cpi=DEFAULT_CPI)

Caveats
-------
* **Case-2 asset block IS now read** (2026-04-21 very-very late).
  ``_read_group_assets`` pulls corp_cash (Group+0x0CC0), the
  ``account_balance`` array (Group+0x39D8..+0x3A30) and the earnings
  metric driver (Group+0x3B38) with two targeted reads per group.
  These feed the case-2 weighted-BVPS formula and close the
  firm-owning-group gap documented in the 2026-04-21 SESSION_LOG
  "stock.py ↔ engine decomp diff" entry.  The old default-zeros behaviour
  remains available (and is still what the unit-test ``_FakeReader``
  stubs inject); the live path now populates these fields so
  ``predict_all_stocks`` has the same asset inputs the engine reads
  mid-tick.

* **HQ type defaults to 0.**  This is a vtable-dispatched short at
  runtime; reading it properly requires walking the Firm arrays.  Not
  needed on Jan 1 1990 (no HQ firms exist yet).

* **eps_adjusted_new populated via compute_eps().**  ``Group::compute_eps(1)``
  at 0x0051c390 is ported in :func:`caplab_sim.stock.compute_eps`.  The
  live reader pulls ``trailing_12m_net_profit`` (Group+0xCE0) and
  ``trailing_12m_special_revenue`` (Group+0xCE8) from memory and feeds
  them through the port; the simple formula
  ``(t12m_net_profit + t12m_special_rev) / shares_outstanding`` matches
  the engine exactly for groups without an HQ firm (Group+0xCAC == 0).
  For groups with an HQ firm, the subsidiary-adjustment branch in the
  engine (``FUN_0051d740 → FUN_00491360`` with 12-iter GameDate
  interpolation over Group+0x3040) is NOT ported — it's empirically
  rare on Jan 1 1990 saves since no group owns an HQ firm yet with a
  12-month history.  Adding the subsidiary branch would be an iterative
  improvement if per-group hit rates remain below target.

  Historical note (superseded 2026-04-22): the previous iteration
  of this file left ``eps_adjusted_new=None`` — causing the
  EPS-shock branch in :func:`caplab_sim.stock.predict_one_stock`
  to compare ``stored vs stored`` (always equal) and never fire.
  That produced the documented 46% match rate on the player
  group while other stocks held at ~95% (SESSION_LOG 2026-04-22c).

* **Legacy eps fallback shim.**  ``Group::compute_eps(1)``
  (@ 0x0051c390) is a pure function of trailing-12-month profit
  history.  On non-quarter-boundary days it produces exactly
  ``eps_adjusted_stored`` (NationStock+0x38) so the EPS-shock branch
  short-circuits — leaving ``eps_adjusted_new=None`` is correct for
  those days.  On quarter boundaries it differs and the monitor will
  currently under-match firm-owning groups on that one day per
  quarter.  Can be added later by re-implementing compute_eps in
  Python against the trailing-profit fields.

* **Deleted groups are skipped** — matches the engine's own iteration
  which drops ``group_ptr == 0`` slots.

Environment
-----------
* Linux, game running under Wine/Lutris.
* Run as same UID as the game (or root).  Uses ``/proc/{pid}/mem`` —
  does not attach via ptrace, so the game keeps running undisturbed.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from caplab_sim.stock import StockInput


# ---------------------------------------------------------------------------
# Known addresses from Ghidra analysis of CapMain.exe v11.1.2
# ---------------------------------------------------------------------------

# 32-bit PE default image base (no ASLR for this binary).
DEFAULT_IMAGE_BASE: int = 0x00400000

# Global RNG call counter (not used by this reader; kept for
# future cross-checks against live seed progression).
RNG_COUNTER_VA: int = 0x00de5bd8
RNG_COUNTER_OFFSET: int = RNG_COUNTER_VA - DEFAULT_IMAGE_BASE  # 0x9e5bd8

# Misc::random function.  Callers do ``MOV ECX, [global_misc_ptr]`` then
# ``CALL FUN_006b0900``; we scan .text for that pattern to find the Misc
# pointer variable.
MISC_RAND_VA: int = 0x006b0900
MISC_SEED_OFFSET: int = 0x7C  # Misc+0x7C holds the uint32 LCG seed.

# Base-game GroupArray — the per-scenario singleton holding the
# player conglomerate plus all AI competitors.  Pinned from
# ``SaveLoad__group_array_load`` (``Gfile3.cpp``), which accesses the
# header via inlined static-var reads:
#   DAT_00a46bec  stride        (GGA+0x04)
#   DAT_00a46bf4  n_groups      (GGA+0x0C)
#   DAT_00a46bf8  entry_size    (GGA+0x10)
#   DAT_00a46c08  n_chunks      (GGA+0x20)
#   DAT_00a46c0c  chunk_stride  (GGA+0x24)
#   DAT_00a46c18  chunks_base   (GGA+0x30)
# so the object starts at 0x00a46be8.  Populated by every scenario
# (base game, classic, default, et al.) — this is the one we want for
# reading player/AI stock state in non-Economy-Mode saves.
BASE_GROUP_ARRAY_VA: int = 0x00a46be8

# Economy-Mode GroupArray — the NPC-stock-market singleton iterated in
# ``GlobalGrpArray__daily_stock_tick`` (gated on ``dlc__economy_mode_flag1``).
# Pinned from ``GameTick__daily`` disassembly:
#   004e3f7a  B928D39D00      MOV ECX, 0x9dd328
#   004e3f7f  E8EC550200      CALL 0x00509570
# Empty in base-game scenarios; holds the five industry NPC groups when
# the Economy-Mode DLC is active.
ECONOMY_GROUP_ARRAY_VA: int = 0x009dd328

# Backwards-compatible alias — older callers (and the initial Phase 2
# stage 2 draft) used this name for the Economy-Mode array before we
# knew about the base-game one.  New code should prefer the explicit
# ``BASE_GROUP_ARRAY_VA`` / ``ECONOMY_GROUP_ARRAY_VA`` constants, or
# iterate :data:`KNOWN_GROUP_ARRAYS` directly.
GLOBAL_GROUP_ARRAY_VA: int = ECONOMY_GROUP_ARRAY_VA
GLOBAL_GROUP_ARRAY_OFFSET: int = GLOBAL_GROUP_ARRAY_VA - DEFAULT_IMAGE_BASE

# All statically-known GroupArray singletons, in the order we try them.
# ``base`` comes first because it's populated in every scenario; the
# ``economy`` array is only populated when the Economy-Mode DLC is on.
KNOWN_GROUP_ARRAYS: Tuple[Tuple[str, int], ...] = (
    ("base",    BASE_GROUP_ARRAY_VA),
    ("economy", ECONOMY_GROUP_ARRAY_VA),
)

# GroupArray header layout (shared by every instance — both singletons
# above plus any others a future scenario might introduce):
#   +0x04  stride      (groups per chunk)
#   +0x0C  n_groups    (total populated slot count)
#   +0x10  entry_size  (bytes per group-pointer slot — should be 4)
#   +0x20  n_chunks    (chunks allocated)
#   +0x24  chunk_stride (bytes per chunk-header entry)
#   +0x30  chunks_base (address of the chunk-header table)
GGA_STRIDE:       int = 0x04
GGA_N_GROUPS:     int = 0x0C
GGA_ENTRY_SIZE:   int = 0x10
GGA_N_CHUNKS:     int = 0x20
GGA_CHUNK_STRIDE: int = 0x24
GGA_CHUNKS_BASE:  int = 0x30

# NationStock pointer stored at Group+0x70 (live memory only — in the
# save file the 0x43B0 bytes are inlined).
GROUP_NATION_STOCK_PTR_OFFSET: int = 0x70

# NationStock size — kept in sync with caplab_save.structs.NationStock.
NATION_STOCK_SIZE: int = 0x43B0

# ---------------------------------------------------------------------------
# Group asset-block offsets (for case-2 weighted-BVPS fundamental value)
# ---------------------------------------------------------------------------
# Every offset below is a Group-relative byte offset holding a little-endian
# IEEE 754 double (8 bytes).  Confirmed byte-for-byte against the case-2
# inline code of NationStock__daily_price_update (0x00647d30) — task #27.
# There is NO separate Group::compute_weighted_bvps function; the formula is
# entirely inlined.  See RNG_Call_Map.txt §2.6 for the full coefficient table.
#
# The asset fields at 0x39D8..0x3A28 form a contiguous 0x58-byte block
# (the `account_balance` array at Group+0x39D0 with index 0 at 0x39D0 —
# we skip the first two slots at 0x39D0/0x39D8 are BOTH reals; the first
# we actually want is bank_deposits at 0x39D8).  corp_cash at 0x0CC0 is
# separate — 11,288 bytes earlier in the struct — so we do two reads
# per group (cheap at /proc/<pid>/mem kernel-side buffering).
#
# earnings_metric_driver at 0x3B38 drives step 4b of the case dispatcher
# (not the weighted-BVPS formula); pulled in the same second read by
# widening it to 0x168 bytes from 0x39D8.
GROUP_CORP_CASH_OFFSET:              int = 0x0CC0
GROUP_ASSETS_BLOCK_OFFSET:           int = 0x39D8
GROUP_ASSETS_BLOCK_SIZE:             int = 0x168   # covers +0x39D8..+0x3B40
# Offsets *within* the 0x168 block (indexing the buffer, not the Group).
# Each is ``Group_offset - GROUP_ASSETS_BLOCK_OFFSET``, pre-computed.
_BLK_BANK_DEPOSITS:            int = 0x00    # Group+0x39D8
_BLK_INVENTORY:                int = 0x08    # Group+0x39E0
_BLK_BUSINESS_ASSETS:          int = 0x10    # Group+0x39E8
_BLK_LAND:                     int = 0x18    # Group+0x39F0
_BLK_TECHNOLOGY:               int = 0x20    # Group+0x39F8
_BLK_STOCKS_HELD:              int = 0x28    # Group+0x3A00
_BLK_BANK_NET_ASSETS:          int = 0x38    # Group+0x3A10
_BLK_INSURANCE_NET_ASSETS:     int = 0x40    # Group+0x3A18
_BLK_LOANS:                    int = 0x48    # Group+0x3A20
_BLK_BOND_INTEREST_LIABILITY:  int = 0x50    # Group+0x3A28
_BLK_EARNINGS_METRIC_DRIVER:   int = 0x160   # Group+0x3B38

# Optional diagnostic field (not consumed by stock.py but helpful for
# logging which groups own firms).  A Group+0xE4 uint32.
GROUP_FIRM_COUNT_OFFSET:       int = 0xE4

# Trailing-12-month profit fields — inputs to Group::compute_eps(1) at
# 0x0051c390.  Rolling sums recomputed monthly in FUN_0051F9A0 from the
# monthly_net_profit[36] (Group+0xE18) and monthly_special_revenue[36]
# (Group+0x1058) ring buffers.  Both are doubles.  See
# ``caplab_sim.stock.compute_eps`` for the formula.  Without these, the
# EPS-shock branch in the predictor never fires — empirically dropping
# the player group's match rate from ~95% → 46% as soon as earnings
# accumulate (2026-04-22).
GROUP_T12M_NET_PROFIT_OFFSET:       int = 0xCE0
GROUP_T12M_SPECIAL_REVENUE_OFFSET:  int = 0xCE8

# Group+0x06: group_recno (short).  Used for the 25-day GroupAI cycle check:
#   game_date % cycle == group_recno % cycle → GroupAI__select_firm_type_25day fires
# where cycle = 5 (firm_count == 0, startup mode) or 25 (firm_count > 0).
GROUP_RECNO_OFFSET:            int = 0x06

# ---------------------------------------------------------------------------
# FirmArray — runtime array of every live Firm pointer
# ---------------------------------------------------------------------------
#
# Pinned from the 2-instruction thunk at 0x008453d0
# (``MOV ECX, 0xa45ac8 / JMP FirmArray__dtor``) and cross-checked against
# ``game_deinit_sequence`` (0x004dfd10).  The container uses the same
# paged-array layout as :data:`BASE_GROUP_ARRAY_VA` (stride at +0x04,
# total_count at +0x0C, entry_size at +0x10, chunks at +0x20/+0x24/+0x30),
# with ``entry_size == 4`` because the slots store ``Firm*`` pointers.
# We deliberately reuse :class:`GroupArrayInfo` / :func:`_iter_group_pointers`
# instead of cloning them — the layout is identical.
FIRM_ARRAY_VA: int = 0x00a45ac8

# current_month_index global (DAT_00a45a1c).  Signed int 0..35 indexing the
# rolling 3-year window used by every Firm and Group monthly array (revenue,
# expense, net_profit, etc.).  Required to read ``Firm+0x180 + idx*8`` etc.
# Rebase path mirrors :data:`GAME_DATE_OFFSET`.
CURRENT_MONTH_INDEX_VA:     int = 0x00a45a1c
CURRENT_MONTH_INDEX_OFFSET: int = CURRENT_MONTH_INDEX_VA - DEFAULT_IMAGE_BASE

# Firm base-class field offsets — Structs_Firm.txt "Firm Base Class Field
# Map".  Every offset below is relative to a ``Firm*`` pointer returned by
# :meth:`LiveGameReader.iter_firm_pointers`.  All firms share this base
# layout regardless of subclass; subclass data starts at +0x718, so a
# single 0x720-byte read is enough for everything the dashboard surfaces.
FIRM_TYPE_OFFSET:            int = 0x08   # short, 1-based ProductClass index
FIRM_BUILD_ID_OFFSET:        int = 0x0A   # short, 1-based FirmBuild index
FIRM_RECNO_OFFSET:           int = 0x0C   # int
FIRM_NATION_OFFSET:          int = 0x10   # int, nation_recno
FIRM_LOC_X1_OFFSET:          int = 0x14   # short, map tile top-left X
FIRM_LOC_Y1_OFFSET:          int = 0x16   # short, map tile top-left Y
FIRM_LOC_X2_OFFSET:          int = 0x18   # short, map tile bottom-right X
FIRM_LOC_Y2_OFFSET:          int = 0x1A   # short, map tile bottom-right Y
FIRM_TECH_LEVEL_OFFSET:      int = 0x64   # int, per-factory tech level
FIRM_CREATION_DATE_OFFSET:   int = 0x70   # int, game-date JDN
FIRM_ALIVE_STATUS_OFFSET:    int = 0x86   # byte, 0=alive, 1=pending-delete, 2+=dying
FIRM_CORP_RECNO_OFFSET:      int = 0xF8   # int, owning Group recno
# Monthly financial arrays — double[36], indexed by ``current_month_index``.
FIRM_MONTHLY_NET_PROFIT_OFFSET: int = 0x180    # double[36] — revenue − expense
FIRM_MONTHLY_REVENUE_OFFSET:    int = 0x2A0    # double[36] — gross revenue
FIRM_MONTHLY_EXPENSE_OFFSET:    int = 0x3C0    # double[36] — total expenses
# 0x720 covers the entire base class (subclass data starts at +0x718 and
# we do not need it for the dashboard).
FIRM_BASE_BLOCK_SIZE: int = 0x720

# ---------------------------------------------------------------------------
# ProductClass resource table — source of firm_type name strings
# ---------------------------------------------------------------------------
#
# Pointer to the ProductClass array lives at ``0x009265e8`` (a module global
# pointer, NOT a direct struct — the one-level indirection is because the
# array is allocated at game-load time and then its base pointer is written
# into the slot).  Entries are 0x360 bytes apart; the name is a null-
# terminated ASCII string at +0x0B.  Confirmed in ``Parsers_UI.txt`` under
# ``CFirmRes__FindByName``.
PRODUCT_CLASS_DATA_PTR_VA: int = 0x009265e8
PRODUCT_CLASS_COUNT_VA:    int = 0x009265cc
PRODUCT_CLASS_STRIDE:      int = 0x360
PRODUCT_CLASS_NAME_OFFSET: int = 0x0B
PRODUCT_CLASS_NAME_MAX:    int = 0x20     # generous cap — in-game names are ≤ 24

# game_date global: DAT_00a459b0 (absolute VA).  Current game day number (Julian
# day number from Ghidra annotations).  Used in every modular-scheduling gate
# inside GroupAI and FirmArray.  Image-base-relative offset is also provided
# for the rebased-address path used by read_game_date().
GAME_DATE_VA:    int = 0x00a459b0
GAME_DATE_OFFSET: int = GAME_DATE_VA - DEFAULT_IMAGE_BASE  # 0x6459b0

# game_difficulty global: DAT_00a449a8 (absolute VA).  uint32 difficulty setting.
# 0 = Easy, 1 = Normal, 2+ = Hard/Custom.  Determines the startup-gate modulus
# inside GroupAI::daily_update (0x004383a1–0x004383d4):
#   difficulty == 0 → PUSH 0x4 → Misc::random(4)
#   difficulty == 1 → PUSH 0x2 → Misc::random(2)
# Used by read_game_difficulty() to return the correct startup_gate_modulus for
# exact post-tick simulation in predict_all_stocks().
GAME_DIFFICULTY_VA:     int = 0x00a449a8
GAME_DIFFICULTY_OFFSET: int = GAME_DIFFICULTY_VA - DEFAULT_IMAGE_BASE  # 0x649a8

# ---------------------------------------------------------------------------
# DLC scenario flags (for ai_tail_advance — 2026-04-22)
# ---------------------------------------------------------------------------
# Located via Ghidra in the scenario-flag cluster at 0x00a449xx..0x00a44cxx.
# The task brief's user-supplied names (DAT_00a44a7e, DAT_00a44996,
# DAT_00a44b79, DAT_00a44bd8) resolve within ±8 bytes of the canonical
# Ghidra symbols dlc__economy_mode_flag1 and dlc__stock_market_active.
# Disambiguation against the user's exact addresses is OPEN (task #X in
# RNG_Call_Map.txt); the addresses below match the Ghidra-symbolized
# references used inside FUN_0042e2d0 at 0x0042e78b.  If the DLC flag
# disambiguation resolves to different VAs we update these constants.
#
# Both flags are bytes.  Non-zero = feature active.  In base-game saves
# both read as 0, so read_scenario_flags() returns an all-zero
# ScenarioFlags and ai_tail_advance() skips the DLC branch entirely —
# making the fix backward-compatible with the 98.2%-match reference run.
DLC_ECONOMY_MODE_VA:       int = 0x00a44a7e
DLC_ECONOMY_MODE_OFFSET:   int = DLC_ECONOMY_MODE_VA - DEFAULT_IMAGE_BASE

DLC_STOCK_MARKET_VA:       int = 0x00a44b79
DLC_STOCK_MARKET_OFFSET:   int = DLC_STOCK_MARKET_VA - DEFAULT_IMAGE_BASE

# Group-level flag at Group+0x21f (byte): per-group DLC "can build from
# city" enable.  Non-zero iff the DLC town-scoring RNG path in
# FUN_0042e2d0 is allowed for this group.
GROUP_DLC_BUILD_FLAG_OFFSET: int = 0x21f

# Group+0xdc (short): AI leader slot.  Non-zero iff the group has an AI
# leader assigned and the GroupAI post-amble (hence the inter-group
# RNG tail) runs.  Equal to 0 on the player's conglomerate, which is
# why the player group never contributes inter-group drift.
GROUP_AI_LEADER_SLOT_OFFSET: int = 0xdc

# Group+0x4528 (short): hq_type.  Used by the DLC branch gate (0 or 10
# qualify).  Currently always read as 0 in base-game saves (vtable call
# returns the stub zero) — documented in the live reader.  We read the
# raw short so downstream code can see the true value once the
# vtable-dispatch gap is closed.
GROUP_HQ_TYPE_OFFSET:        int = 0x4528


# ---------------------------------------------------------------------------
# Raw /proc/<pid>/mem primitives
# ---------------------------------------------------------------------------

def _find_capmain_pid() -> Optional[int]:
    """Return the PID of the running CapMain.exe, or ``None`` if not found.

    Scans ``/proc/<pid>/comm`` first (fast, often conclusive on Wine),
    then falls back to ``/proc/<pid>/cmdline`` which includes the Wine
    wrapper invocation for some launcher configurations.
    """
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                if "CapMain" in f.read():
                    return pid
            with open(f"/proc/{pid}/cmdline", "r") as f:
                if "CapMain" in f.read():
                    return pid
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return None


# Matches a ``/proc/<pid>/maps`` entry of the form
# ``<start>-<end> <perms> <offset> <dev> <inode> <path>``.
_MAPS_LINE_RE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(.*)$"
)


def _find_capmain_base(pid: int) -> Optional[int]:
    """Return the base mapping address of ``CapMain.exe`` for ``pid``.

    Picks the first executable or read-only mapping whose path contains
    ``CapMain`` — the image is laid out contiguously starting from that
    base address.
    """
    with open(f"/proc/{pid}/maps", "r") as f:
        for line in f:
            m = _MAPS_LINE_RE.match(line.strip())
            if not m:
                continue
            start_hex, _end_hex, perms, path = m.groups()
            if "CapMain" in path and ("r-xp" in perms or "r--p" in perms):
                return int(start_hex, 16)
    return None


def _read_mem(pid: int, address: int, size: int) -> bytes:
    """Raw byte read from ``/proc/<pid>/mem``.

    Raises ``OSError`` if the address is unmapped or the process has
    exited.  The caller typically handles those as "not connected".
    """
    with open(f"/proc/{pid}/mem", "rb") as f:
        f.seek(address)
        return f.read(size)


def _read_u32(pid: int, address: int) -> int:
    return struct.unpack("<I", _read_mem(pid, address, 4))[0]


# ---------------------------------------------------------------------------
# Misc pointer discovery
# ---------------------------------------------------------------------------

def _find_misc_pointer(pid: int, base: int) -> Optional[int]:
    """Locate the gameplay ``Misc`` singleton by scanning calls to Misc::random.

    The engine invokes ``Misc::random(n)`` via thiscall, loading ECX with
    the Misc pointer just before the CALL.  Two x86 patterns show up:

        B9 XX XX XX XX              MOV ECX, imm32       ; direct address
        8B 0D XX XX XX XX           MOV ECX, [imm32]     ; indirect through pointer var

    For this build the gameplay Misc is a static global at ``0x009753d8``,
    loaded via the direct form (``B9``).  Other Misc instances (menu-time,
    per-subsystem temporaries) use the indirect form — so a scan that
    only checks ``8B 0D`` will latch onto the wrong object, find a seed
    that happens to contain ASCII text or other garbage, and silently
    produce nonsense downstream.

    We walk backwards from each Misc::random call site up to 30 bytes,
    checking BOTH opcodes at each position and returning whichever lands
    closer to the CALL.  That mirrors the engine's actual ECX-set order
    (the last instruction before the CALL is the one that matters).

    Sanity filter: the resolved Misc address must be inside the
    ``0x00400000..0x10000000`` range (image + heap on 32-bit Wine),
    AND reading ``Misc+0x7C`` must succeed.  Any uint32 is a legal LCG
    seed so we can't test it structurally; if this heuristic ever
    mis-fires on a future build, widen the 30-byte backscan window
    or cross-check against a second call site.
    """
    rng_func_va = base + (MISC_RAND_VA - DEFAULT_IMAGE_BASE)

    # .text begins at base + 0x1000 for this binary; 3MB is enough to
    # contain Misc::random and any early caller.
    try:
        text = _read_mem(pid, base + 0x1000, 0x300000)
    except OSError:
        return None

    text_start = base + 0x1000
    n = len(text)
    # E8 rel32 CALL is 5 bytes; scan up to n-5.
    for i in range(n - 5):
        if text[i] != 0xE8:
            continue
        rel = struct.unpack_from("<i", text, i + 1)[0]
        target = (text_start + i + 5 + rel) & 0xFFFFFFFF
        if target != rng_func_va:
            continue
        # Walk backwards up to 30 bytes, checking both MOV ECX forms at
        # each position.  First hit wins (nearest set-ECX to the CALL).
        for back in range(1, 31):
            pos = i - back
            if pos < 0:
                break
            # MOV ECX, [imm32] — 8B 0D XX XX XX XX (6 bytes).  Requires
            # one more byte of headroom for the imm32.
            if (pos + 5 < n and text[pos] == 0x8B and text[pos + 1] == 0x0D):
                ptr_var_va = struct.unpack_from("<I", text, pos + 2)[0]
                try:
                    misc_obj = _read_u32(pid, ptr_var_va)
                    if 0x00400000 < misc_obj < 0x10000000:
                        # Confirm +0x7C is mapped (not strictly a seed
                        # validation — any uint32 is valid — but catches
                        # obviously-unmapped garbage pointers).
                        _ = _read_u32(pid, misc_obj + MISC_SEED_OFFSET)
                        return misc_obj
                except OSError:
                    pass
            # MOV ECX, imm32 — B9 XX XX XX XX (5 bytes).
            if pos + 4 < n and text[pos] == 0xB9:
                misc_obj = struct.unpack_from("<I", text, pos + 1)[0]
                if 0x00400000 < misc_obj < 0x10000000:
                    try:
                        _ = _read_u32(pid, misc_obj + MISC_SEED_OFFSET)
                        return misc_obj
                    except OSError:
                        pass
    return None


# ---------------------------------------------------------------------------
# GlobalGrpArray walker
# ---------------------------------------------------------------------------

@dataclass
class GroupArrayInfo:
    """Parsed GlobalGrpArray header — all four counts in one place."""

    stride: int
    n_groups: int
    entry_size: int
    n_chunks: int
    chunk_stride: int
    chunks_base: int

    def is_plausible(self) -> bool:
        """Sanity-check the header against the expected invariants.

        Returns True when:
        * ``n_groups`` is positive but not absurd (< 1e6),
        * ``stride`` is positive (chunked arrays need at least one per
          chunk — the engine uses 1024 in practice),
        * ``entry_size`` equals 4 (we're walking pointer arrays),
        * ``chunks_base`` points into a heap-ish address range.

        Useful as a "did we attach to the right process / right version"
        check before walking.
        """
        return (
            0 < self.n_groups < 1_000_000
            and self.stride > 0
            and self.entry_size == 4
            and 0x00400000 < self.chunks_base < 0x80000000
        )


def _read_group_array_header(pid: int, gga_addr: int) -> GroupArrayInfo:
    """Read the six header uint32s from the GlobalGrpArray object."""
    return GroupArrayInfo(
        stride       = _read_u32(pid, gga_addr + GGA_STRIDE),
        n_groups     = _read_u32(pid, gga_addr + GGA_N_GROUPS),
        entry_size   = _read_u32(pid, gga_addr + GGA_ENTRY_SIZE),
        n_chunks     = _read_u32(pid, gga_addr + GGA_N_CHUNKS),
        chunk_stride = _read_u32(pid, gga_addr + GGA_CHUNK_STRIDE),
        chunks_base  = _read_u32(pid, gga_addr + GGA_CHUNKS_BASE),
    )


def _iter_group_pointers(
    pid: int, info: GroupArrayInfo
) -> Iterator[Tuple[int, int]]:
    """Yield ``(slot, group_ptr)`` for every non-zero slot, slot-ascending.

    Mirrors the engine's own loop in ``GlobalGrpArray__daily_stock_tick``:

        for slot in 1..n_groups:
            chunk_idx    = (slot - 1) // stride
            slot_in_chunk = (slot - 1) %  stride
            chunk_body   = *(chunks_base + chunk_stride * chunk_idx)
            group_ptr    = *(chunk_body + entry_size * slot_in_chunk)
            if group_ptr != 0:
                yield slot, group_ptr

    Deleted slots (group_ptr == 0) are skipped, same as the engine.
    """
    for slot in range(1, info.n_groups + 1):
        chunk_idx = (slot - 1) // info.stride
        slot_in_chunk = (slot - 1) % info.stride
        if chunk_idx >= info.n_chunks:
            # Malformed header — the engine would crash too; we just stop.
            return
        chunk_header_addr = info.chunks_base + info.chunk_stride * chunk_idx
        chunk_body = _read_u32(pid, chunk_header_addr)
        if chunk_body == 0:
            continue
        slot_addr = chunk_body + info.entry_size * slot_in_chunk
        group_ptr = _read_u32(pid, slot_addr)
        if group_ptr == 0:
            continue
        yield slot, group_ptr


@dataclass
class ScenarioFlags:
    """Global DLC scenario flags relevant to the AI-tail inter-group RNG.

    Read once per snapshot by :meth:`LiveGameReader.read_scenario_flags`
    and passed into :func:`caplab_sim.stock.ai_tail_advance` so it knows
    whether the DLC town-scoring RNG path in FUN_0042e2d0 can fire.

    In base-game saves both flags are 0 and every DLC gate in
    ai_tail_advance short-circuits, preserving the 98.2% match rate of
    the scalar-sweep reference run.  Any scenario that flips these
    flags on (Economy Mode, Classic+Stocks, etc.) will enable the
    single-shot +1-advance-per-qualifying-group-day fix documented in
    Stock_Tick_InterGroup_RNG_Analysis.md §4.3.
    """

    dlc_economy_mode: bool = False
    dlc_stock_market_active: bool = False


@dataclass
class GroupAssets:
    """Case-2 weighted-BVPS asset block, pulled from live Group memory.

    Fields match the 11 doubles the engine reads during case 2 of
    ``NationStock__daily_price_update`` (see RNG_Call_Map.txt §2.6 and
    SESSION_LOG "ENTRY — 2026-04-21 very-very late (stock.py ↔ engine
    decomp diff; live-pair gap isolated)").  ``firm_count`` and
    ``earnings_metric_driver`` are included as diagnostics / inputs for
    the case-2 surge branch.
    """

    corp_cash: float = 0.0               # Group+0x0CC0
    bank_deposits: float = 0.0           # Group+0x39D8
    inventory: float = 0.0               # Group+0x39E0
    business_assets: float = 0.0         # Group+0x39E8
    land: float = 0.0                    # Group+0x39F0
    technology: float = 0.0              # Group+0x39F8
    stocks_held: float = 0.0             # Group+0x3A00
    bank_net_assets: float = 0.0         # Group+0x3A10
    insurance_net_assets: float = 0.0    # Group+0x3A18
    loans: float = 0.0                   # Group+0x3A20
    bond_interest_liability: float = 0.0 # Group+0x3A28
    earnings_metric_driver: float = 0.0  # Group+0x3B38
    firm_count: int = 0                  # Group+0xE4 (uint32)
    trailing_12m_net_profit: float = 0.0        # Group+0xCE0
    trailing_12m_special_revenue: float = 0.0   # Group+0xCE8


def _read_group_assets(pid: int, group_ptr: int) -> GroupAssets:
    """Read the case-2 asset block + firm_count from a live Group.

    Does three targeted ``/proc/<pid>/mem`` reads rather than a single
    wide one because the three regions of interest are far apart
    (0xE4, 0xCC0, and 0x39D8) and reading the intervening 11KB of
    monthly-history doubles would waste ~98% of the bytes.  Returns a
    zero-filled :class:`GroupAssets` if any read fails so the caller
    gets a silent-degrade path instead of a hard crash — match rates
    will drop visibly in that case, which is the right failure mode.
    """
    try:
        corp_cash_bytes = _read_mem(pid, group_ptr + GROUP_CORP_CASH_OFFSET, 8)
        block_bytes = _read_mem(
            pid,
            group_ptr + GROUP_ASSETS_BLOCK_OFFSET,
            GROUP_ASSETS_BLOCK_SIZE,
        )
        firm_count_bytes = _read_mem(
            pid, group_ptr + GROUP_FIRM_COUNT_OFFSET, 4
        )
        # Trailing-12-month profit doubles at Group+0xCE0 and +0xCE8 —
        # feed Group::compute_eps(1).  Fetched as one 16-byte read; if it
        # fails we fall through to a zero-filled GroupAssets below.
        t12m_bytes = _read_mem(
            pid, group_ptr + GROUP_T12M_NET_PROFIT_OFFSET, 16
        )
    except OSError:
        return GroupAssets()

    def _d(off: int) -> float:
        if len(block_bytes) < off + 8:
            return 0.0
        return struct.unpack_from("<d", block_bytes, off)[0]

    t12m_net_profit, t12m_special_rev = struct.unpack("<dd", t12m_bytes)

    return GroupAssets(
        corp_cash               = struct.unpack("<d", corp_cash_bytes)[0],
        bank_deposits           = _d(_BLK_BANK_DEPOSITS),
        inventory               = _d(_BLK_INVENTORY),
        business_assets         = _d(_BLK_BUSINESS_ASSETS),
        land                    = _d(_BLK_LAND),
        technology              = _d(_BLK_TECHNOLOGY),
        stocks_held             = _d(_BLK_STOCKS_HELD),
        bank_net_assets         = _d(_BLK_BANK_NET_ASSETS),
        insurance_net_assets    = _d(_BLK_INSURANCE_NET_ASSETS),
        loans                   = _d(_BLK_LOANS),
        bond_interest_liability = _d(_BLK_BOND_INTEREST_LIABILITY),
        earnings_metric_driver  = _d(_BLK_EARNINGS_METRIC_DRIVER),
        firm_count              = struct.unpack("<I", firm_count_bytes)[0],
        trailing_12m_net_profit      = t12m_net_profit,
        trailing_12m_special_revenue = t12m_special_rev,
    )


def _read_nation_stock_input(
    pid: int, group_ptr: int, recno: int
) -> Optional[StockInput]:
    """Read one Group's NationStock (via Group+0x70 pointer) and decode it.

    Returns ``None`` if the group has no stock subobject (pointer is
    zero) or the stock record is too short — matches the defensive
    behaviour of ``NationStock.from_bytes`` when fed an empty blob.

    The case-2 asset block (corp_cash, bank_deposits, inventory,
    business_assets, land, technology, stocks_held, bank_net_assets,
    insurance_net_assets, loans, bond_interest_liability, and the
    earnings_metric_driver) IS read here via :func:`_read_group_assets`.
    This is the live-memory counterpart to the save-file reader's
    :func:`caplab_sim.stock.stock_inputs_from_save` — it lets the
    predictor score firm-owning groups correctly (see SESSION_LOG
    "ENTRY — 2026-04-21 very-very late (stock.py ↔ engine decomp
    diff; live-pair gap isolated)" for why zero-filling these fields
    caused the original 6/21 match rate on mature saves).
    """
    from caplab_save.structs import NationStock

    ns_ptr = _read_u32(pid, group_ptr + GROUP_NATION_STOCK_PTR_OFFSET)
    if ns_ptr == 0:
        return None
    try:
        blob = _read_mem(pid, ns_ptr, NATION_STOCK_SIZE)
    except OSError:
        return None
    ns = NationStock.from_bytes(blob)

    # Group+0x74 is the group_type short.  Reading 2 bytes rather than
    # going through the save parser keeps this module free of the full
    # Group-blob read.
    try:
        group_type = struct.unpack_from(
            "<H", _read_mem(pid, group_ptr + 0x74, 2), 0
        )[0]
    except OSError:
        group_type = 0

    # Case-2 fundamental-value asset block — required for firm-owning
    # groups to match predictions post-tick.  For fresh-IPO / 0-firm
    # groups all these fields are zero and the prediction is unchanged.
    assets = _read_group_assets(pid, group_ptr)

    # AI-tail gating fields — Group+0xdc (ai_leader_slot short) and
    # Group+0x21f (dlc_build_flag byte).  Both feed ai_tail_advance()
    # via :class:`StockInput`.  Safe failure: zero on OSError so the
    # predicate skips the AI tail entirely for groups that read cleanly
    # elsewhere (a reasonable conservative default that matches the
    # old scalar-sweep baseline).
    try:
        ai_leader_slot = struct.unpack_from(
            "<h", _read_mem(pid, group_ptr + GROUP_AI_LEADER_SLOT_OFFSET, 2), 0
        )[0]
    except OSError:
        ai_leader_slot = 0
    try:
        dlc_build_flag = _read_mem(
            pid, group_ptr + GROUP_DLC_BUILD_FLAG_OFFSET, 1
        )[0]
    except OSError:
        dlc_build_flag = 0

    si = StockInput(
        group_recno=recno,
        is_listed=bool(ns.is_listed),
        is_foreign=bool(ns.is_foreign),
        group_type=group_type,
        hq_type=0,
        # hq_type note: Group.vtable[+4] is Job__vtable_stub_return_zero (always
        # returns 0) for every Group in the base game.  Therefore hq_type=0
        # is correct for all groups; bank/insurance NPC-investment special-
        # casing in cases 0/1 never fires.  See 2026-04-21 Ghidra decompile
        # of NationStock__daily_price_update and GroupAI__daily_update.
        shares_outstanding=ns.shares_outstanding,
        base_stock_price=ns.base_stock_price,
        book_value_per_share=ns.book_value_per_share,
        eps_basic=ns.eps_basic,
        eps_adjusted_stored=ns.eps_adjusted,
        pe_ratio_stored=ns.earnings_yield_ratio,
        sentiment=ns.sentiment,
        bank_deposits=assets.bank_deposits,
        corp_cash=assets.corp_cash,
        inventory=assets.inventory,
        business_assets=assets.business_assets,
        land=assets.land,
        technology=assets.technology,
        stocks_held=assets.stocks_held,
        bank_net_assets=assets.bank_net_assets,
        insurance_net_assets=assets.insurance_net_assets,
        loans=assets.loans,
        bond_interest_liability=assets.bond_interest_liability,
        earnings_metric_driver=assets.earnings_metric_driver,
        trailing_12m_net_profit=assets.trailing_12m_net_profit,
        trailing_12m_special_revenue=assets.trailing_12m_special_revenue,
        firm_count=assets.firm_count,
        ai_leader_slot=ai_leader_slot,
        dlc_build_flag=dlc_build_flag,
        # eps_adjusted_new populated below via compute_eps() — simple
        # Group::compute_eps(1) port skipping the subsidiary-adjustment
        # branch.  See caplab_sim.stock.compute_eps docstring.
    )
    # Fresh EPS for this tick — the engine recomputes via
    # Group::compute_eps(1) inside NationStock__daily_price_update just
    # before the EPS-shock branch, so the predictor must also provide a
    # fresh value.  Falling back to eps_adjusted_stored (the old None
    # behaviour) short-circuits the shock branch and is the dominant
    # source of per-group drift on firm-owning groups once trailing
    # profit starts moving.
    from caplab_sim.stock import compute_eps
    si.eps_adjusted_new = compute_eps(si)
    return si


# ---------------------------------------------------------------------------
# LiveGameReader public API
# ---------------------------------------------------------------------------

@dataclass
class LiveGameReader:
    """Attached view onto a running ``CapMain.exe``.

    Instantiate via :meth:`attach` (which auto-discovers the PID, base,
    Misc pointer, and both GroupArray singleton addresses) or pass the
    fields directly for testing.

    ``group_arrays`` is a tuple of ``(tag, absolute_address)`` pairs —
    one entry per :data:`KNOWN_GROUP_ARRAYS` singleton, already
    rebased to the process's actual image load address.  See module
    docstring for which tags are populated under which scenario.

    ``gga_addr`` is preserved as a backwards-compatible accessor that
    returns the first populated array (or the base-game array if none
    are populated); code written against the original single-array API
    continues to work.
    """

    pid: int
    base: int
    misc_addr: int
    group_arrays: Tuple[Tuple[str, int], ...]

    @classmethod
    def attach(cls) -> "LiveGameReader":
        """Find the running CapMain.exe and resolve all required addresses.

        Raises ``RuntimeError`` with a human-readable reason if any step
        fails (no process, unmapped base, Misc pointer not located).
        """
        pid = _find_capmain_pid()
        if pid is None:
            raise RuntimeError(
                "CapMain.exe not found.  Is the game running under Wine?"
            )
        base = _find_capmain_base(pid)
        if base is None:
            raise RuntimeError(
                f"Could not locate CapMain.exe mapping for PID {pid}.  "
                "Check /proc/{pid}/maps — the image may be using a "
                "non-default base."
            )
        misc_addr = _find_misc_pointer(pid, base)
        if misc_addr is None:
            raise RuntimeError(
                "Could not locate the Misc singleton pointer via the "
                "Misc::random call-site scan.  The binary may have "
                "diverged from v11.1.2."
            )
        group_arrays = tuple(
            (tag, base + (va - DEFAULT_IMAGE_BASE))
            for tag, va in KNOWN_GROUP_ARRAYS
        )
        return cls(pid=pid, base=base, misc_addr=misc_addr,
                   group_arrays=group_arrays)

    # -- Backwards-compatible primary-array accessor -----------------------

    @property
    def gga_addr(self) -> int:
        """Return the "primary" GroupArray address (first populated one).

        Preserves the v1 single-array API for older callers.  Picks the
        first entry in ``group_arrays`` whose header is plausible; if
        none are, falls back to the base-game array so that failure
        diagnostics still have a concrete address to report.
        """
        for _tag, addr in self.group_arrays:
            try:
                info = _read_group_array_header(self.pid, addr)
            except OSError:
                continue
            if info.is_plausible():
                return addr
        return self.group_arrays[0][1]

    # -- RNG ---------------------------------------------------------------

    def read_rng_seed(self) -> int:
        """Return the current uint32 LCG seed at ``Misc+0x7C``."""
        return _read_u32(self.pid, self.misc_addr + MISC_SEED_OFFSET)

    def read_game_date(self) -> int:
        """Return the current game date from ``DAT_00a459b0`` (rebased).

        The game date is a Julian Day Number stored as a 32-bit int at
        ``base + GAME_DATE_OFFSET``.  It is incremented AFTER the daily
        GroupArray / stock tick, so a snapshot taken before the tick sees
        the date under which the upcoming tick will execute — exactly
        what the per-group ia vector computation needs.

        Returns 0 on OSError (unmapped / game exited) so callers can
        treat a zero date as "unknown, fall back to scalar ia".
        """
        try:
            return _read_u32(self.pid, self.base + GAME_DATE_OFFSET)
        except OSError:
            return 0

    def read_game_difficulty(self) -> int:
        """Return the startup-gate modulus derived from the game difficulty setting.

        Reads ``DAT_00a449a8`` (difficulty uint32) and maps to the modulus
        used by ``Misc::random`` in ``GroupAI::daily_update`` at 0x004383cd:

          * difficulty == 0 (Easy)   → modulus 4  (``random(4)``)
          * difficulty == 1 (Normal) → modulus 2  (``random(2)``)
          * difficulty >= 2 (Hard+)  → modulus 0  (gate behaviour unknown)

        Returns 0 on OSError (game exited / address unmapped).  Callers
        should treat 0 as "unknown difficulty; fall back to scalar sweep".
        """
        try:
            diff = _read_u32(self.pid, self.base + GAME_DIFFICULTY_OFFSET)
            if diff == 0:
                return 4
            elif diff == 1:
                return 2
            return 0   # difficulty 2+: gate code path not confirmed
        except OSError:
            return 0

    def read_scenario_flags(self) -> ScenarioFlags:
        """Return the DLC scenario flags used by ``ai_tail_advance``.

        Reads the two image-rebased DLC flag bytes:

          * ``dlc__economy_mode_flag1`` at :data:`DLC_ECONOMY_MODE_OFFSET`
          * ``dlc__stock_market_active`` at :data:`DLC_STOCK_MARKET_OFFSET`

        Both fields are single bytes in Ghidra; non-zero means "feature
        on".  On ``OSError`` we return an all-false flags object so the
        caller's behaviour degrades to the base-game path (which is the
        correct conservative default — the 98.2%-match scalar sweep).
        """
        try:
            eco_byte = _read_mem(
                self.pid, self.base + DLC_ECONOMY_MODE_OFFSET, 1
            )[0]
            stk_byte = _read_mem(
                self.pid, self.base + DLC_STOCK_MARKET_OFFSET, 1
            )[0]
        except OSError:
            return ScenarioFlags()
        return ScenarioFlags(
            dlc_economy_mode=bool(eco_byte),
            dlc_stock_market_active=bool(stk_byte),
        )

    def read_rng_counter(self) -> int:
        """Return the global RNG call counter (``DAT_00de5bd8``).

        Useful for "did the game advance between reads?" checks, though
        the counter also ticks on non-gameplay Misc instances (see
        ``RNG_Call_Map.txt §6``) so it is NOT a reliable budget for
        replay purposes.
        """
        return _read_u32(self.pid, self.base + RNG_COUNTER_OFFSET)

    # -- GroupArray / NationStock ------------------------------------------

    def read_group_array_info(
        self, tag: Optional[str] = None
    ) -> GroupArrayInfo:
        """Return the parsed header for one GroupArray.

        ``tag`` selects which known array to read ("base", "economy");
        if omitted, reads whichever is currently the primary (see
        :attr:`gga_addr`).  Callers can use
        :meth:`GroupArrayInfo.is_plausible` to confirm the array is
        populated before walking its ``n_groups`` slots.
        """
        if tag is None:
            return _read_group_array_header(self.pid, self.gga_addr)
        for t, addr in self.group_arrays:
            if t == tag:
                return _read_group_array_header(self.pid, addr)
        raise KeyError(f"unknown GroupArray tag: {tag!r}")

    def iter_group_array_infos(
        self,
    ) -> Iterator[Tuple[str, int, GroupArrayInfo]]:
        """Yield ``(tag, addr, info)`` for each known GroupArray.

        Plausibility is NOT filtered — callers get every array that
        read cleanly, including empty ones.  Useful for the ``--dump``
        CLI path and for logging which arrays are populated.
        """
        for tag, addr in self.group_arrays:
            try:
                info = _read_group_array_header(self.pid, addr)
            except OSError:
                continue
            yield tag, addr, info

    def iter_groups(self) -> Iterator[Tuple[str, int, int]]:
        """Yield ``(array_tag, slot, group_ptr)`` for every live group.

        Walks every populated GroupArray (base-game, Economy-Mode, any
        future singleton that gets added to :data:`KNOWN_GROUP_ARRAYS`)
        in :data:`KNOWN_GROUP_ARRAYS` order, slot-ascending inside each.
        Arrays with implausible headers are skipped silently — that's
        the normal state for the Economy-Mode array in base-game saves.

        The header is re-read for each array on every call so the
        caller can retry after the engine mutates the structure.
        """
        for tag, addr, info in self.iter_group_array_infos():
            if not info.is_plausible():
                continue
            for slot, group_ptr in _iter_group_pointers(self.pid, info):
                yield tag, slot, group_ptr

    # -- FirmArray ---------------------------------------------------------

    def read_firm_array_info(self) -> GroupArrayInfo:
        """Return the :data:`FIRM_ARRAY_VA` paged-array header.

        FirmArray uses the exact same container layout as GlobalGrpArray
        (see :data:`GGA_STRIDE` etc.), so we reuse :class:`GroupArrayInfo`.
        ``entry_size`` should always be 4 on a healthy process — the
        slots hold ``Firm*`` pointers.

        Raises ``OSError`` if the VA has not been mapped yet (caller
        should treat this as "no firms in this scenario").
        """
        addr = self.base + (FIRM_ARRAY_VA - DEFAULT_IMAGE_BASE)
        return _read_group_array_header(self.pid, addr)

    def iter_firm_pointers(self) -> Iterator[Tuple[int, int]]:
        """Yield ``(recno, firm_ptr)`` for every non-null FirmArray slot.

        The engine's own recno is 1-based, matching ``slot`` here.  Slots
        with ``firm_ptr == 0`` are holes left by deleted firms (the paged
        container never compacts) and are skipped, identical to the
        engine's own ``FirmArray__next_alive`` loop.

        On a malformed header we yield nothing instead of raising so
        callers iterating during a save-load transition stay quiet.
        """
        try:
            info = self.read_firm_array_info()
        except OSError:
            return
        if not info.is_plausible():
            return
        yield from _iter_group_pointers(self.pid, info)

    def read_firm_base_block(self, firm_ptr: int) -> bytes:
        """Read the 0x720-byte Firm base-class block.

        Covers the full base class (subclass data starts at +0x718).  For
        the dashboard's revenue/profit/identity view that is sufficient;
        callers needing subclass-specific fields should issue their own
        :func:`_read_mem` request with the subclass allocation size from
        ``Structs_Firm.txt``.
        """
        return _read_mem(self.pid, firm_ptr, FIRM_BASE_BLOCK_SIZE)

    def read_current_month_index(self) -> int:
        """Return ``DAT_00a45a1c`` — the rolling month index 0..35.

        Indexes the per-firm and per-group ``double[36]`` monthly arrays.
        Returns 0 on ``OSError`` so callers can fall back to slot 0
        rather than crash; legitimate games always have a value in
        ``0..35`` once the first month boundary has passed.
        """
        try:
            raw = _read_u32(
                self.pid, self.base + CURRENT_MONTH_INDEX_OFFSET
            )
        except OSError:
            return 0
        # Defensive clamp — corrupted mid-read reads can return garbage
        # that would later crash the unpack on Firm+0x180+raw*8.
        if 0 <= raw <= 35:
            return raw
        return 0

    def read_product_class_names(self) -> Dict[int, str]:
        """Return ``{firm_type → product_class_name}`` for the whole table.

        Reads the base pointer from :data:`PRODUCT_CLASS_DATA_PTR_VA`
        (one-level indirection because the array itself is heap-
        allocated), walks every entry at ``base + (i-1) * 0x360``, and
        decodes the null-terminated ASCII name at +0x0B.  The index key
        is the 1-based ``firm_type`` so callers can do::

            names = reader.read_product_class_names()
            label = names.get(firm.firm_type, f"type {firm.firm_type}")

        Returns ``{}`` on any read failure so callers degrade to numeric
        labels rather than crash.
        """
        base_ptr_va = self.base + (
            PRODUCT_CLASS_DATA_PTR_VA - DEFAULT_IMAGE_BASE
        )
        count_va = self.base + (
            PRODUCT_CLASS_COUNT_VA - DEFAULT_IMAGE_BASE
        )
        try:
            data_base = _read_u32(self.pid, base_ptr_va)
            count     = _read_u32(self.pid, count_va)
        except OSError:
            return {}
        if data_base == 0 or count <= 0 or count > 256:
            return {}

        # Single wide read is ~1.6 KB for ~50 product classes.  Slicing
        # in Python is vastly cheaper than issuing one ReadProcessMemory
        # per entry.
        try:
            blob = _read_mem(
                self.pid, data_base, count * PRODUCT_CLASS_STRIDE
            )
        except OSError:
            return {}

        out: Dict[int, str] = {}
        for i in range(count):
            off = i * PRODUCT_CLASS_STRIDE + PRODUCT_CLASS_NAME_OFFSET
            end = off + PRODUCT_CLASS_NAME_MAX
            raw = blob[off:end]
            # Trim at first NUL or non-printable byte.
            cut = 0
            for cut in range(len(raw)):
                b = raw[cut]
                if b == 0 or b < 0x20 or b > 0x7E:
                    break
            else:
                cut = len(raw)
            if cut == 0:
                continue
            try:
                name = raw[:cut].decode("ascii")
            except UnicodeDecodeError:
                continue
            # firm_type is 1-based; index 0 in the array → firm_type 1.
            out[i + 1] = name
        return out

    def stock_inputs(self) -> List[StockInput]:
        """Build :class:`StockInput` records for every live listed group.

        Drop-in replacement for
        :func:`caplab_sim.stock.stock_inputs_from_save` — ordering,
        field set, and defaults match, with the known Case-2 caveat
        documented at module level.  Order is
        (array in KNOWN_GROUP_ARRAYS order, slot ascending inside each);
        ``predict_all_stocks`` will re-sort into descending-recno
        internally when ``reorder=True``.

        When multiple arrays are populated (Economy-Mode DLC plus
        base-game), the recno space is disjoint — base-game groups use
        1..N_base and Economy-Mode NPC groups use 1..N_npc — so the
        downstream predictor must either consume a single array's worth
        at a time or handle the collision.  For the common case
        (base-game only, or Economy-Mode only) this is not a concern.
        """
        result: List[StockInput] = []
        for _tag, slot, group_ptr in self.iter_groups():
            si = _read_nation_stock_input(self.pid, group_ptr, recno=slot)
            if si is not None:
                result.append(si)
        return result


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GroupArray discovery by vtable scan (fallback / diagnostic)
# ---------------------------------------------------------------------------

# Sentinel value at GroupArray+0x00 for the Economy-Mode instance.  Not
# a "real" C++ vtable — the Ghidra view at this address is actually a
# pointer table into the DBF-column string pool ("Stocks.DBF",
# "Healthcare", "Industrial", "Energy", "Consumer"), embedded at the
# start of the GroupArray object.  It still works as a scan needle
# because it's a stable process-wide constant that every Economy-Mode
# GroupArray carries.  The base-game GroupArray at
# :data:`BASE_GROUP_ARRAY_VA` has a DIFFERENT value at +0x00 (its own
# vtable/class-data pointer), so a scan using only this constant will
# miss it.  :func:`find_group_arrays` is therefore a fallback, used
# mainly for diagnosing unexpected extra instances — the two
# statically-known singletons in :data:`KNOWN_GROUP_ARRAYS` are the
# primary path.
GROUP_ARRAY_VTABLE: int = 0x0085ea38


def _iter_rw_ranges(pid: int):
    """Yield ``(start, end)`` for each readable mapping in ``/proc/pid/maps``.

    We only need readable ranges — even read-only mappings (like the
    binary's .rdata) could in principle hold GroupArray pointers, so
    the filter is ``'r' in perms``.  Non-file anonymous heap mappings
    are included (that's where the Wine heap lives).
    """
    with open(f"/proc/{pid}/maps", "r") as f:
        for line in f:
            m = _MAPS_LINE_RE.match(line.strip())
            if not m:
                continue
            start_hex, end_hex, perms, _path = m.groups()
            if "r" not in perms:
                continue
            start = int(start_hex, 16)
            end = int(end_hex, 16)
            if start == end:
                continue
            yield start, end


def find_group_arrays(pid: int, vtable: int = GROUP_ARRAY_VTABLE,
                      max_candidates: int = 32) -> List[Tuple[int, GroupArrayInfo]]:
    """Scan process memory for all ``GroupArray`` instances by vtable.

    Walks every readable mapping, reads it in chunks, and collects every
    4-byte-aligned offset whose uint32 value equals ``vtable``.  For each
    hit, reads the candidate's six header fields and returns the list of
    ``(address, GroupArrayInfo)`` pairs — caller can filter by
    ``info.is_plausible()`` and inspect ``info.n_groups`` to identify
    the live array for the current scenario.

    Bounded by ``max_candidates`` (default 32) — normal builds have
    a handful of instances; hitting the cap means the vtable is being
    used generically and we should raise the filter.
    """
    out: List[Tuple[int, GroupArrayInfo]] = []
    needle = struct.pack("<I", vtable)
    for start, end in _iter_rw_ranges(pid):
        size = end - start
        # Large heaps can be huge; chunk reads to keep memory usage sane.
        chunk = 0x100000  # 1 MB
        for base in range(start, end, chunk):
            read_size = min(chunk, end - base)
            try:
                data = _read_mem(pid, base, read_size)
            except OSError:
                continue
            pos = 0
            while True:
                idx = data.find(needle, pos)
                if idx < 0:
                    break
                # Require 4-byte alignment (object pointers always are).
                if idx % 4 == 0:
                    addr = base + idx
                    try:
                        info = _read_group_array_header(pid, addr)
                        out.append((addr, info))
                        if len(out) >= max_candidates:
                            return out
                    except OSError:
                        pass
                pos = idx + 1
    return out


def _hex_dump(pid: int, address: int, size: int) -> str:
    """Pretty-print ``size`` bytes from ``address`` as 16-byte rows.

    Used by the ``--dump`` diagnostic to help confirm struct layout
    when the header reports something unexpected.
    """
    data = _read_mem(pid, address, size)
    out = []
    for row in range(0, size, 16):
        chunk = data[row:row + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        u32s = " ".join(
            f"{struct.unpack_from('<I', chunk, i)[0]:08x}"
            for i in range(0, len(chunk) - 3, 4)
        )
        out.append(f"  +0x{row:02x}  {hex_part:<48}  [{u32s}]")
    return "\n".join(out)


def _main(argv: List[str]) -> int:
    """Dump seed + group-array header + one StockInput sample.

    Pass ``--dump`` for extra diagnostic output (raw bytes at the GGA
    object and the first chunk body).  Useful when the reported
    ``n_groups`` doesn't line up with what's on screen in the game.
    """
    want_dump = "--dump" in argv
    want_scan = "--scan" in argv

    try:
        reader = LiveGameReader.attach()
    except RuntimeError as exc:
        print(f"attach failed: {exc}")
        return 1

    if want_scan:
        # Hunt for all GroupArray-like objects in memory.  Useful when
        # the static GlobalGrpArray is empty (e.g. base-game scenarios
        # that don't use Economy-Mode NPC stocks) and the real Group
        # array lives elsewhere.
        print(
            f"Scanning process memory for vtable=0x{GROUP_ARRAY_VTABLE:08x} "
            f"(this may take a few seconds)..."
        )
        hits = find_group_arrays(reader.pid)
        print(f"Found {len(hits)} candidate(s):")
        for addr, info in hits:
            tag = " (known GlobalGrpArray)" if addr == reader.gga_addr else ""
            print(
                f"  0x{addr:08x}{tag}: n_groups={info.n_groups:<5d} "
                f"n_chunks={info.n_chunks} stride={info.stride} "
                f"entry_size={info.entry_size} "
                f"chunks_base=0x{info.chunks_base:08x} "
                f"plausible={info.is_plausible()}"
            )
        return 0

    seed = reader.read_rng_seed()
    counter = reader.read_rng_counter()
    print(f"pid={reader.pid} base=0x{reader.base:08x}")
    print(f"misc=0x{reader.misc_addr:08x}  seed=0x{seed:08x}  counter={counter:,}")

    # Print one line per known GroupArray singleton so it's obvious which
    # one is populated (base-game vs Economy-Mode).  Both addresses and
    # plausibility are always shown even when empty — useful when the
    # user asks "is Economy Mode actually on?".
    for tag, addr, info in reader.iter_group_array_infos():
        print(
            f"GGA[{tag:<7s}] @ 0x{addr:08x}: "
            f"stride={info.stride} n_groups={info.n_groups} "
            f"entry_size={info.entry_size} n_chunks={info.n_chunks} "
            f"chunk_stride={info.chunk_stride} "
            f"chunks_base=0x{info.chunks_base:08x}  "
            f"plausible={info.is_plausible()}"
        )
        if want_dump:
            print(f"\nRaw GGA[{tag}] header bytes (first 0x40):")
            print(_hex_dump(reader.pid, addr, 0x40))
            if info.n_chunks > 0 and info.chunks_base:
                try:
                    body_ptr = _read_u32(reader.pid, info.chunks_base)
                except OSError:
                    body_ptr = 0
                print(f"\nchunks_base[0] = 0x{body_ptr:08x}")
                if body_ptr:
                    print(f"\nFirst 8 slot entries at 0x{body_ptr:08x}:")
                    print(_hex_dump(reader.pid, body_ptr, 0x20))
            print()

    inputs = reader.stock_inputs()
    listed = [s for s in inputs if s.is_listed and not s.is_foreign]
    print(f"\ngroups walked: {len(inputs)}  listed-domestic: {len(listed)}")
    if listed:
        s = listed[0]
        print(
            f"  sample g{s.group_recno}: price={s.base_stock_price:.4f} "
            f"sent={s.sentiment:+.1f} pe={s.pe_ratio_stored:.2f}"
        )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
