"""Constants for the Capitalism Lab save file parser.

All numbers come from reverse-engineering Capitalism Lab 11.1.2 with Ghidra and
are cross-referenced with the research notes in this repo:

    - structs/Structs_Serialization.txt   (master section layout, tag sequence)
    - structs/Structs_Group.txt           (Group struct offsets)
    - structs/Structs_GroupAI.txt         (GroupAI / Firm extensions)
    - engine/Engine_Economy.txt           (Economy struct offsets)
    - engine/Engine_TimeIntegration.txt   (monthly array semantics)

Two important empirical facts established by inspection of blob00.bin:

    1. Version tags on disk are **4 bytes** (little-endian int), even though
       `CFile__read_short` is the I/O primitive used at load time.  Distances
       between successive tags match the documented payload sizes only when we
       treat the tag as a 4-byte header.

    2. The first zlib stream (blob 0) opens with a legacy preamble that uses
       tags 0x1001..~0x1004 — this is the scenario / environment block.  The
       documented "Section 1" starts at tag **0x1065** (observed at offset
       0x939e in PLAY_001 blob 0).

All integers are little-endian.
"""

from __future__ import annotations

# ---- tag header width -----------------------------------------------------

TAG_SIZE = 4  # bytes; verified empirically (see module docstring)

# ---- Section 1: legacy (tags 0x1065..0x1079) ------------------------------

TAG_GAMEINFO      = 0x1065  # game_info struct, payload 0x71E8 bytes
TAG_SECONDARY_CFG = 0x1066  # DAT_00a44910 block, payload 0x1008 bytes
TAG_GAMESTATE     = 0x1067  # FUN_005006f0: game state fields (incl. RNG)
TAG_SCENARIO_CFG  = 0x1068  # DAT_00a45998, payload 0xA8 bytes
TAG_PLAYER_CFG    = 0x1069  # FUN_005001d0, 0xB4 bytes at +0x59E94
TAG_NATION_BULK   = 0x106A  # Nation bulk loads, 0x7968 bytes per nation
TAG_ADDITIONAL_1  = 0x106B
TAG_PLAYER_NATION = 0x106C  # DAT_0091c064 (player_nation_recno)
TAG_DBF_VALIDATE  = 0x106D  # DBF resource validation + legacy Firm flat array
TAG_CONFIG_1A     = 0x106E
TAG_BLOCK_5A8     = 0x106F  # tag 0x1070 block
TAG_GLOBAL_1      = 0x1070
TAG_DAT_009C4DDC  = 0x1071
TAG_LOADER_2      = 0x1072
TAG_LOADER_3      = 0x1073
TAG_SHORT_INT     = 0x1074
TAG_BLOCK_260     = 0x1075
TAG_LOADER_4      = 0x1076
TAG_DLC_COND      = 0x1077
TAG_LOADER_5      = 0x1078
TAG_LEGACY_END    = 0x1079

SECTION_1_TAGS = tuple(range(TAG_GAMEINFO, TAG_LEGACY_END + 1))

# ---- Section 2: new format (tags 0x10C9..0x10CE, 0x112D, 0x1191) ----------

TAG_ARRAY_META        = 0x10C9
TAG_GROUP_ARRAY       = 0x10CA  # per-record 0x4A18 base + subarrays
TAG_FIRM_ARRAY        = 0x10CB  # per-record 0x718 base + subclass + unit groups
TAG_RECORD_1C         = 0x10CC  # BuildArray (0x1C bytes/record)
# NB (2026-04-21): tag 0x10CD was previously believed to be the PersonArray
# tag, but a byte-pattern probe of PLAY_004 blob0 shows that the 0x10CD
# byte pattern appears 4 times in the blob and every occurrence is
# immediately adjacent (≤40 bytes) to a 0x10CE byte pattern — far too
# close to be a genuine tagged array (PersonArray records are 0x110 =
# 272 bytes each, so N persons requires N×272 bytes of payload between
# 0x10CD and the next tag). The 0x10CD patterns are actually record-type
# bytes inside a 28-byte-stride inline record array. Therefore 0x10CD
# is NOT the PersonArray tag in this save format; PersonArray is
# serialized via FUN_005042b0 (Gfile3.cpp) which reads
# `int active_count; for i..count: short flag; if flag!=0: block(0x110)`
# with NO explicit tag prefix — its byte position in the save is
# determined by the preceding array loaders' cumulative payload size.
# Locating the real PersonArray header requires walking the save
# dispatcher to resolve serialization order. Until that is done,
# the runtime PersonArray live-count estimate of ~1000 (from
# DAT_0091bf28+0xc at run time) is the best available number.
TAG_RECORD_110        = 0x10CD  # RESERVED — see note above, NOT PersonArray
TAG_NATION_ARRAY      = 0x10CE  # per-record 0x3C9B8 base + 4 subarrays
TAG_RECORD_50_VTABLE  = 0x112D  # JobArray (0x50+vtable per record)
TAG_METADATA_ONLY     = 0x1191

SECTION_2_TAGS = (
    TAG_ARRAY_META,
    TAG_GROUP_ARRAY,
    TAG_FIRM_ARRAY,
    TAG_RECORD_1C,
    TAG_RECORD_110,
    TAG_NATION_ARRAY,
    TAG_RECORD_50_VTABLE,
    TAG_METADATA_ONLY,
)

# ---- payload sizes (Section 1) --------------------------------------------

SIZE_GAMEINFO      = 0x71E8
SIZE_SECONDARY_CFG = 0x1008
SIZE_SCENARIO_CFG  = 0x00A8
SIZE_PLAYER_CFG    = 0x00B4
SIZE_NATION_LEGACY = 0x7968
SIZE_CONFIG_1A     = 0x001A
SIZE_BLOCK_260     = 0x0260

# ---- payload sizes (Section 2) --------------------------------------------

SIZE_GROUP_BASE       = 0x4A18   # Group flat blob for types 1/3/4
SIZE_FIRM_BASE        = 0x0718   # Firm base (before subclass + unit groups)
SIZE_FIRM_UNITGROUP   = 0x0048   # one unit group record
SIZE_RECORD_1C        = 0x001C
SIZE_RECORD_110       = 0x0110
SIZE_NATION_BASE      = 0x3C9B8
SIZE_NATIONSTOCK      = 0x43B0   # heap block referenced at Group+0x70

# FirmRes subarray on each Group: `DAT_0090e13c × 2 bytes` per
# Structs_Serialization.txt §179.  DAT_0090e13c is the *FirmRes count*
# (number of firm-resource descriptors in the game, NOT the firm count),
# so this block is a per-resource short for each group.  In the observed
# PLAY_001 saves it is 96 entries × 2 bytes = 0xC0 = 192 bytes.  This
# array is NOT the group's firm-ownership list; ownership is resolved via
# Firm+0xF8 (see FIRM_OFF_GROUP_RECNO below).
FIRMRES_ENTRY_SIZE    = 2

# Firm subclass extension sizes (keyed by firm_type).
#
# The serialized size is determined by the engine's
# ``FirmArray__firm_class_size(firm_type, firm_subtype)``.  The values here are
# documentation values from ``Structs_Serialization.txt`` **supplemented by
# empirical observations** from the PLAY_001..PLAY_004 fixture walks.  The
# parser in :pymod:`caplab_save.firm` does NOT rely on this table to size the
# subclass — it scans for sentinel ``0x82`` and infers the size from the gap
# between ``0x81`` and ``0x82``.  The table is retained for sanity-checking
# (see ``firm.FIRM_SUBCLASS_OBSERVED`` for the walk's observed truth).
FIRM_SUBCLASS_SIZES: dict[int, int] = {
    0x01: 0x08, 0x02: 0x08, 0x04: 0x08, 0x05: 0x08, 0x07: 0x08,   # basic
    0x03: 0xB0,                                                     # FirmService
    0x06: 0x20, 0x08: 0x20, 0x09: 0x20, 0x0A: 0x20,                 # extended basic
    0x0B: 0x18, 0x19: 0x18, 0x26: 0x18,
    0x0C: 0xAF0,
    0x0D: 0x150,                                                    # FirmTelecom
    0x0E: 0x1680,                                                   # FirmMall
    0x0F: 0x1F8,
    0x10: 0x3048,                                                   # FirmMedia
    0x11: 0x740,
    0x13: 0x600,
    0x14: 0x30, 0x16: 0x30,
    0x15: 0x28, 0x1A: 0x28,
    0x18: 0xB8,
    0x1B: 0x918, 0x1C: 0x918,
    0x1D: 0x659, 0x1E: 0x659, 0x1F: 0x659,   # observed 0x659 (docs said 0x638)
    0x20: 0x00, 0x21: 0x00, 0x22: 0x00, 0x23: 0x00, 0x24: 0x00,     # observed base-only
    0x25: 0x2E,     # observed 0x2E (docs said 0x10 + dynamic tail)
    0x27: 0x80,
    0x2B: 0x00, 0x2C: 0x00, 0x2E: 0x00, 0x2F: 0x00,                 # base-only
}
# Types with variable subclass sizing (resolved via FUN_004a6480 at load time).
# The parser uses sentinel-walk to size these at runtime.
FIRM_VARIABLE_SUBTYPES = {0x28, 0x29}
FIRM_DYNAMIC_SUBTYPES = {0x25}  # doc: has byte(+0x71B)*10 dynamic tail

# Plausible range for a valid firm_type (nonzero, <= 0x50).  Used by the
# sentinel walker's start/boundary heuristics.
FIRM_TYPE_MAX = 0x50

# Firm inline sentinel shorts (validated at load time).
FIRM_SENTINEL_81 = 0x0081
FIRM_SENTINEL_82 = 0x0082
FIRM_SENTINEL_83 = 0x0083

# ---- RNG ------------------------------------------------------------------

# Tag ID for the GAMESTATE record (Section 2 marker). The 32-bit RNG seed
# sits in the first 4 bytes of this record's payload. Location is NOT
# fixed across saves — adding a single player-owned firm shifts the tag
# forward. Parser MUST locate this dynamically via
# ``blob0.find(struct.pack('<I', RNG_SEED_TAG))`` — see
# ``parse_rng_seed`` and the regression tests in
# ``caplab_save/tests/test_rng_seed_offset.py``.
RNG_SEED_TAG = 0x1067

# Blob-0 byte offset of the 32-bit RNG seed for the PLAY_001 / PLAY_002
# reference saves.  Retained for diagnostic purposes only — DO NOT use this
# as the source of truth, because it is not layout-stable. Five of nine
# reference saves captured during Phase 1 have the tag at 0x115CE /
# 0x11606 / 0x1163E / 0x11676 / 0x116AE depending on how many player-owned
# firms exist. See ``caplab_sim/SESSION_LOG.txt`` "Upstream parser bug"
# section for the full analysis.
RNG_SEED_OFFSET_BLOB0_LEGACY = 0x0115_9A

# Back-compat alias — still references the legacy constant, but callers
# should migrate to the dynamic lookup. Flagged DeprecationWarning-class
# in code review until the last consumer (if any) is updated.
RNG_SEED_OFFSET_BLOB0 = RNG_SEED_OFFSET_BLOB0_LEGACY

# Borland-C-style LCG used by the game (FUN_006b0900).
RNG_MULTIPLIER = 0x015A_4E35  # 22,695,477
RNG_INCREMENT  = 0x0000_0001

# ---- Economy struct (offsets within the 0x1067 payload, tentative) --------

# Relative offsets inside the Economy struct itself.  The struct's base offset
# inside the save blob has to be located at parse time (see economy.py).
ECONOMY_OFF_BASE_INTEREST_RATE      = 0x10
ECONOMY_OFF_PREV_BASE_INTEREST_RATE = 0x18
ECONOMY_OFF_GDP_RAW_INDICATOR       = 0x20
ECONOMY_OFF_GDP_GROWTH_RATE         = 0x28
ECONOMY_OFF_CYCLE_PHASE             = 0x30  # int
ECONOMY_OFF_GDP_SUBPHASE            = 0x34  # int
ECONOMY_OFF_INTEREST_RATE_LEVEL     = 0x38  # int
ECONOMY_OFF_TARGET_RATE_PCT         = 0x40
ECONOMY_OFF_BASE_RATE_PCT           = 0x48  # 25.0 constant
ECONOMY_OFF_PREV_RATE_LEVEL         = 0x50  # int
ECONOMY_OFF_LAST_RATE_CHANGE_DATE   = 0x54  # int
ECONOMY_OFF_STOCK_INDEX_TARGET      = 0x58
ECONOMY_OFF_ANNUAL_INFLATION_RATE   = 0x60
ECONOMY_OFF_PRICE_LEVEL_CPI         = 0x68
ECONOMY_OFF_PRICE_LEVEL_PPP         = 0x70
ECONOMY_SIZE_TOTAL                  = 0x78  # conservative lower bound

# Economy runtime address (inside CapMain.exe, for reference)
ECONOMY_STATIC_VA = 0x00A47698

# ---- Group struct offsets (inside the 0x4A18 flat blob) -------------------

GROUP_OFF_RECNO                     = 0x06    # short
GROUP_OFF_NATION_PTR                = 0x70    # runtime pointer (will be stale in save)
GROUP_OFF_TYPE                      = 0x74    # short: 1/3/4 valid, 0 deleted
GROUP_OFF_PERSON_RECNO              = 0xDC    # short
GROUP_OFF_FIRM_COUNT                = 0xE4    # int
GROUP_OFF_ACTIVE_COUNT              = 0xE8    # int
GROUP_OFF_CORP_RECNO_SELF           = 0xF8    # int

GROUP_OFF_CORP_CASH                 = 0xCC0   # double
GROUP_OFF_NET_WORTH                 = 0xCC8   # double
GROUP_OFF_STRATEGY_BUDGET           = 0xCD4   # int
GROUP_OFF_T12M_NET_FLOW             = 0xCD8   # double
GROUP_OFF_T12M_NET_PROFIT           = 0xCE0   # double
GROUP_OFF_T12M_SPECIAL_REVENUE      = 0xCE8   # double
GROUP_OFF_TAX_RESERVE               = 0xCF0   # double

# All monthly arrays below are double[36] (288 bytes).
GROUP_OFF_MONTHLY_NET_FLOW          = 0x0CF8
GROUP_OFF_MONTHLY_NET_PROFIT        = 0x0E18
GROUP_OFF_ANNUAL_PROFIT_OFFSET_1    = 0x0F28  # double
GROUP_OFF_ANNUAL_PROFIT_OFFSET_2    = 0x0F30  # double
GROUP_OFF_MONTHLY_GROSS_FIELD       = 0x0F38
GROUP_OFF_MONTHLY_SPECIAL_REVENUE   = 0x1058
GROUP_OFF_MONTHLY_BUSINESS_REVENUE  = 0x1178
GROUP_OFF_MONTHLY_EXPENSE_TOTAL     = 0x1298
GROUP_OFF_MONTHLY_NET_WORTH_SNAP    = 0x13B8
GROUP_OFF_MONTHLY_UNKNOWN_14D8      = 0x14D8
GROUP_OFF_MONTHLY_SALES_FLOW        = 0x15F8

# Per-dept long-term arrays: each double[24].
GROUP_OFF_DEPT_REVENUE_CURRENT      = 0x3610
GROUP_OFF_DEPT_REVENUE_YTD          = 0x3790
GROUP_OFF_DEPT_REVENUE_ALLTIME      = 0x3910

# Running cash totals: per-dept double[14].
GROUP_OFF_ACCOUNT_BALANCE           = 0x39D0
GROUP_OFF_LOAN_PRINCIPAL            = 0x3A20  # double (account_balance[10])
GROUP_OFF_BOND_INTEREST_LIABILITY   = 0x3A28  # double (account_balance[11])
GROUP_OFF_ACCOUNT_BALANCE_2         = 0x3A40  # double[14]

MONTHLY_ARRAY_LEN = 36
DEPT_ARRAY_LEN_24 = 24
DEPT_ARRAY_LEN_14 = 14

# ---- Firm struct offsets (inside the 0x718 base flat blob) ----------------

FIRM_OFF_TYPE                       = 0x08   # short; 0 => deleted
FIRM_OFF_SUBTYPE                    = 0x0A   # short
FIRM_OFF_ACTIVE_VISITOR_COUNT       = 0x90   # int, transient (zeroed after load)
FIRM_OFF_UNIT_GROUP_COUNT           = 0xB0   # short count used for loop (1..3)
FIRM_OFF_VISITOR_ARRAY_1            = 0xC4   # 10 shorts, zeroed post-load
FIRM_OFF_VISITOR_ARRAY_2            = 0xD8   # 10 shorts, zeroed post-load

# Firm+0xF8 holds the owning group's recno.  Verified in
# Structs_Units_Items.txt §101 and Structs_Firm.txt §314:
# `GroupArray__operator_bracket(*(int*)(firm+0xF8))` is the canonical
# firm→group link used in revenue/expense accounting.  Stored as int.
FIRM_OFF_GROUP_RECNO                = 0xF8   # int; owning Group recno

FIRM_OFF_AI_CAPACITY_TARGET         = 0x88   # double (per user request)
FIRM_OFF_UTILIZATION                = 0x8C   # float/double? — treated as double
FIRM_OFF_DEMAND_BONUS               = 0x11C  # double
FIRM_OFF_MONTHLY_NET_PROFIT         = 0x180
FIRM_OFF_MONTHLY_REVENUE            = 0x2A0
FIRM_OFF_MONTHLY_EXPENSE            = 0x3C0

# Per-dept arrays — user-supplied base offsets; element stride unknown here,
# so we simply expose them as 0x60-byte blocks (12 doubles) per semantic group.
FIRM_OFF_DEPT_BLOCK_1               = 0x4E0
FIRM_OFF_DEPT_BLOCK_2               = 0x540
FIRM_OFF_DEPT_BLOCK_3               = 0x5A0
FIRM_OFF_DEPT_BLOCK_4               = 0x660

# ---- Time / calendar ------------------------------------------------------

MONTHS_PER_YEAR = 12
MONTHLY_ARRAY_YEARS = 3  # 36 months / 12 = 3-year rolling window
