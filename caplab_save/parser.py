"""Main Capitalism Lab save-file parser.

High-level flow::

    1. Decompress the .SAV (decompress.py) → plaintext header + N zlib blobs.
    2. Walk the version tags in blob 0 to locate Section 1 / Section 2 anchors.
    3. Extract the fields we understand; preserve unknown ranges as raw bytes.

Guiding principle: be **noisy on unexpected structure** so bugs are caught at
parse time rather than silently producing garbage.
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional

# ---- path bootstrap -------------------------------------------------------
# Support being run as a plain script from inside the package directory
# (``python parser.py``) in addition to the normal ``python -m
# caplab_save.parser`` invocation.  When run as a script, Python only adds
# the file's own directory to sys.path; we also need the parent so the
# ``caplab_save`` package is importable.
if __package__ in (None, ""):  # running as a script, not as a module
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)

from caplab_save import constants as C  # noqa: E402
from caplab_save.decompress import DecompressResult, decompress_save  # noqa: E402
from caplab_save.economy import locate_economy, parse_economy  # noqa: E402
from caplab_save.firm import parse_firm_array  # noqa: E402
from caplab_save.structs import (  # noqa: E402
    EconomyState,
    Firm,
    GameInfo,
    Group,
    Nation,
    RNGState,
)


# ---- tag search ------------------------------------------------------------

@dataclass
class TagSlot:
    tag: int
    offset: int       # byte offset of tag header in blob
    payload: int      # byte offset of payload (= offset + TAG_SIZE)


def find_tag_sequence(blob: bytes, expected_tags: tuple[int, ...]) -> list[TagSlot]:
    """Find the in-order occurrences of ``expected_tags`` in ``blob``.

    We use a monotonic cursor: each tag is searched starting after the previous
    tag's position, which filters out the spurious tag-like byte patterns that
    occur inside random double-array payloads.
    """
    out: list[TagSlot] = []
    cursor = 0
    for tag in expected_tags:
        pat = struct.pack("<I", tag)
        idx = blob.find(pat, cursor)
        if idx == -1:
            raise ValueError(
                f"tag 0x{tag:04x} not found in blob after offset 0x{cursor:x}"
            )
        out.append(TagSlot(tag=tag, offset=idx, payload=idx + C.TAG_SIZE))
        cursor = idx + C.TAG_SIZE
    return out


def _read_tag_at(blob: bytes, offset: int) -> int:
    if offset + C.TAG_SIZE > len(blob):
        raise ValueError(f"cannot read tag at 0x{offset:x}: past end")
    return struct.unpack_from("<I", blob, offset)[0]


def _require_tag(blob: bytes, offset: int, expected: int) -> None:
    got = _read_tag_at(blob, offset)
    if got != expected:
        raise ValueError(
            f"version tag mismatch at 0x{offset:x}: "
            f"expected 0x{expected:04x}, got 0x{got:04x}"
        )


# ---- Section 1 readers -----------------------------------------------------

def _unpack_monthly(blob: bytes, offset: int) -> tuple[float, ...]:
    return tuple(
        struct.unpack_from("<d", blob, offset + i * 8)[0]
        for i in range(C.MONTHLY_ARRAY_LEN)
    )


def _unpack_doubles(blob: bytes, offset: int, count: int) -> tuple[float, ...]:
    return tuple(
        struct.unpack_from("<d", blob, offset + i * 8)[0] for i in range(count)
    )


def parse_gameinfo(blob: bytes, payload_offset: int) -> GameInfo:
    """Parse the 0x71E8-byte GameInfo payload following tag 0x1065.

    The GameInfo payload itself is *not* where the live calendar lives.
    The real fields are in the ScenarioCfg payload (tag 0x1068) which
    follows GameInfo → SecondaryCfg → GameState in Section 1.  We locate
    it by searching forward from just past the GameInfo payload and read
    the eight calendar fields directly — no heuristics.

    Tag-0x1068 calendar layout (verified via byte-diff of a
    consecutive-day save pair on 2026-04-21; see SESSION_LOG task #8):

    =========  ======  ======================================================
    offset     type    meaning
    =========  ======  ======================================================
    +0x10      int32   start_day_counter (JDN of scenario start)
    +0x14      int32   start_year
    +0x18      int32   current_day_counter (JDN; +1 per game day)
    +0x1C      int32   day_of_year (1-based; Jan 1 = 1)
    +0x20      int32   day_of_month (1-based)
    +0x24      int32   month (1-based; Jan = 1)
    +0x28      int32   year
    +0x2C      int32   days_since_start (0-based; 0 on scenario start day)
    =========  ======  ======================================================

    ``game_date`` on the returned :class:`GameInfo` now carries the
    Julian Day Number (engine-native units); JDN 2,447,893 = 1990-01-01.
    Prior to this change the parser matched a stale 15-element stamp
    array inside the GameInfo payload that happened to sit in the old
    heuristic's value range — see SESSION_LOG 2026-04-21 entry.

    The raw 0x71E8 GameInfo payload is still preserved in ``.raw`` for
    downstream analysis.
    """
    raw = bytes(blob[payload_offset:payload_offset + C.SIZE_GAMEINFO])

    cal = _parse_scenario_calendar(blob, payload_offset + C.SIZE_GAMEINFO)

    return GameInfo(
        game_date=cal["day_counter"],
        day_of_month=cal["day_of_month"],
        month_index=(cal["month"] - 1) % C.MONTHLY_ARRAY_LEN,
        month_of_year=cal["month"],
        year=cal["year"],
        raw=raw,
        day_of_year=cal["day_of_year"],
        days_since_start=cal["days_since_start"],
        start_day_counter=cal["start_day_counter"],
        start_year=cal["start_year"],
        scenario_cfg_offset=cal["_tag_offset"],
    )


def _parse_scenario_calendar(blob: bytes, search_from: int) -> dict:
    """Locate tag 0x1068 (ScenarioCfg) and return its calendar fields.

    The search starts at ``search_from`` (normally the byte just past the
    GameInfo payload) and falls back to a whole-blob scan if the
    forward-only search misses — defensive against future layout
    variations.  Raises :class:`ValueError` if the tag is nowhere to be
    found.

    The eight int32 fields are at well-defined offsets relative to the
    tag payload; see :func:`parse_gameinfo` for the complete layout
    table.
    """
    pat = struct.pack("<I", C.TAG_SCENARIO_CFG)
    tag_off = blob.find(pat, search_from)
    if tag_off < 0:
        tag_off = blob.find(pat)
    if tag_off < 0:
        raise ValueError(
            f"tag 0x{C.TAG_SCENARIO_CFG:04x} (SCENARIO_CFG) not found in blob"
        )
    payload = tag_off + C.TAG_SIZE
    fields = struct.unpack_from("<iiiiiiii", blob, payload + 0x10)
    (start_day_counter, start_year,
     day_counter, day_of_year, day_of_month,
     month, year, days_since_start) = fields

    return {
        "start_day_counter": start_day_counter,
        "start_year": start_year,
        "day_counter": day_counter,
        "day_of_year": day_of_year,
        "day_of_month": day_of_month,
        "month": month,
        "year": year,
        "days_since_start": days_since_start,
        "_tag_offset": tag_off,
        "_payload_offset": payload,
    }


def jdn_to_gregorian(jdn: int) -> tuple[int, int, int]:
    """Convert a Julian Day Number (proleptic Gregorian) to (year, month, day).

    JDN 2,447,893 = 1990-01-01.  Used by the engine for
    ``GameInfo.game_date``.  Provided here so test code can cross-check
    the parsed ``(year, month, day_of_month)`` triple against the
    independently-parsed ``game_date``.
    """
    # Standard Fliegel-Van Flandern algorithm (Gregorian).
    L = jdn + 68569
    N = (4 * L) // 146097
    L = L - (146097 * N + 3) // 4
    I = (4000 * (L + 1)) // 1461001
    L = L - (1461 * I) // 4 + 31
    J = (80 * L) // 2447
    day = L - (2447 * J) // 80
    L = J // 11
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L
    return year, month, day


def parse_rng_seed(blob: bytes) -> RNGState:
    """Extract the RNG seed by locating the GAMESTATE tag (``0x1067``).

    The seed is the first 4 bytes of the tag-0x1067 payload. The tag's
    absolute offset in blob 0 depends on upstream record sizes (e.g.
    GameInfo grows when the player owns firms), so we search for the
    tag rather than trusting a hardcoded constant. See
    ``caplab_save/constants.RNG_SEED_OFFSET_BLOB0_LEGACY`` for the
    historical fixed offset and the comment there for why it's wrong.

    Raises ``ValueError`` if tag 0x1067 can't be found — that would
    indicate a save format we don't understand (e.g. a version skew),
    and downstream code needs to fail loudly rather than silently
    returning the bytes from some unrelated region.
    """
    pat = struct.pack("<I", C.RNG_SEED_TAG)
    off = blob.find(pat)
    if off == -1:
        raise ValueError(
            f"tag 0x{C.RNG_SEED_TAG:04x} (GAMESTATE) not found in blob 0; "
            "save format may differ from v11.1.2 expectations"
        )
    payload = off + C.TAG_SIZE
    if payload + 4 > len(blob):
        raise ValueError(
            f"tag 0x{C.RNG_SEED_TAG:04x} found at 0x{off:x} but payload "
            "runs past end of blob"
        )
    seed = struct.unpack_from("<I", blob, payload)[0]
    # ``source_offset`` is the tag offset (not the payload offset) — it's
    # the addressable landmark for debugging; callers that care about the
    # seed's raw position can add ``C.TAG_SIZE``.
    return RNGState(seed=seed, counter=None, source_offset=off)


# ---- Group array parser ----------------------------------------------------

@dataclass
class _GroupArrayMeta:
    active_count: int
    capacity: int
    next_recno: int
    payload_start: int
    first_record: int
    record_stride: int
    firmres_entries: int       # number of 2-byte shorts per group's FirmRes subarray


def _parse_group_array_metadata(blob: bytes, tag_offset: int) -> _GroupArrayMeta:
    """Parse the Group Array metadata header after tag 0x10CA.

    The array manager writes ~8 ints of bookkeeping data before the first
    record.  We don't rely on a fixed header size — instead we scan forward
    for the first plausible record and derive the stride from record pairs.
    """
    payload_start = tag_offset + C.TAG_SIZE
    # First 8 ints are the array manager fields (active, cap, next_recno, …).
    metadata_ints = [
        struct.unpack_from("<I", blob, payload_start + i * 4)[0] for i in range(8)
    ]
    active_count, capacity, next_recno, *_ = metadata_ints

    first_rec = _find_first_group_record(blob, payload_start + 0x20, limit=0x200)
    if first_rec is None:
        raise ValueError(
            f"could not locate first Group record after tag 0x10CA "
            f"at 0x{tag_offset:x}"
        )
    second_rec = _find_first_group_record(
        blob,
        first_rec + 2 + C.SIZE_GROUP_BASE,  # search begins past base blob
        limit=0x20000,
    )

    # Derive record stride.  Fall back to a conservative default if we can't
    # spot a second record (e.g. only one Group exists).
    if second_rec is not None:
        stride = second_rec - first_rec
    else:
        # Assume zero FirmRes subarray and a single record.  This is still
        # safe because we bound record count by active_count.
        stride = 2 + C.SIZE_GROUP_BASE + C.SIZE_NATIONSTOCK

    # FirmRes subarray fills the gap between the known base + NationStock and
    # the record stride.  Entries are **2-byte shorts** keyed by FirmRes recno
    # (DAT_0090e13c × 2 bytes per group) per Structs_Serialization.txt §179.
    fixed = 2 + C.SIZE_GROUP_BASE + C.SIZE_NATIONSTOCK
    firmres_bytes = stride - fixed
    if firmres_bytes < 0 or firmres_bytes % C.FIRMRES_ENTRY_SIZE != 0:
        raise ValueError(
            f"Group record stride 0x{stride:x} is inconsistent with fixed "
            f"blob sizes (diff=0x{firmres_bytes:x}); diff is not a multiple "
            f"of {C.FIRMRES_ENTRY_SIZE} bytes."
        )

    return _GroupArrayMeta(
        active_count=active_count,
        capacity=capacity,
        next_recno=next_recno,
        payload_start=payload_start,
        first_record=first_rec,
        record_stride=stride,
        firmres_entries=firmres_bytes // C.FIRMRES_ENTRY_SIZE,
    )


def _find_first_group_record(
    blob: bytes, start: int, limit: int
) -> Optional[int]:
    """Scan ``blob[start : start+limit]`` for a byte offset at which a Group
    record plausibly begins.

    The signature is: 2-byte prefix short whose value equals the 2-byte short
    at +2+0x74 (the `Group+0x74 = group_type` invariant).  Only types 1, 3, 4
    are considered valid (matches GroupArray__group_class_size returning
    0x4A18 only for these).
    """
    end = min(start + limit, len(blob) - (2 + C.SIZE_GROUP_BASE))
    for off in range(start, end):
        gt_prefix = struct.unpack_from("<H", blob, off)[0]
        if gt_prefix not in (1, 3, 4):
            continue
        gt_base = struct.unpack_from("<H", blob, off + 2 + C.GROUP_OFF_TYPE)[0]
        if gt_base == gt_prefix:
            return off
    return None


def parse_group(blob: bytes, record_start: int, meta: _GroupArrayMeta, recno: int) -> Group:
    """Decode a single Group record starting at ``record_start``."""
    # Skip the 2-byte prefix short; base begins 2 bytes in.
    base_start = record_start + 2
    base_end = base_start + C.SIZE_GROUP_BASE
    firmres_end = base_end + meta.firmres_entries * C.FIRMRES_ENTRY_SIZE
    nationstock_end = firmres_end + C.SIZE_NATIONSTOCK

    b = blob  # alias
    u = struct.unpack_from

    gt_prefix = u("<H", blob, record_start)[0]
    # gt_prefix is the same as base[+0x74]; we surface it as group_type.

    monthly_net_flow         = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_NET_FLOW)
    monthly_net_profit       = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_NET_PROFIT)
    monthly_gross_field      = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_GROSS_FIELD)
    monthly_special_revenue  = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_SPECIAL_REVENUE)
    monthly_business_revenue = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_BUSINESS_REVENUE)
    monthly_expense_total    = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_EXPENSE_TOTAL)
    monthly_net_worth_snap   = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_NET_WORTH_SNAP)
    monthly_sales_flow       = _unpack_monthly(b, base_start + C.GROUP_OFF_MONTHLY_SALES_FLOW)

    account_balance = _unpack_doubles(b, base_start + C.GROUP_OFF_ACCOUNT_BALANCE,   C.DEPT_ARRAY_LEN_14)
    account_balance_2 = _unpack_doubles(b, base_start + C.GROUP_OFF_ACCOUNT_BALANCE_2, C.DEPT_ARRAY_LEN_14)
    dept_rev_cur  = _unpack_doubles(b, base_start + C.GROUP_OFF_DEPT_REVENUE_CURRENT, C.DEPT_ARRAY_LEN_24)
    dept_rev_ytd  = _unpack_doubles(b, base_start + C.GROUP_OFF_DEPT_REVENUE_YTD,     C.DEPT_ARRAY_LEN_24)
    dept_rev_all  = _unpack_doubles(b, base_start + C.GROUP_OFF_DEPT_REVENUE_ALLTIME, C.DEPT_ARRAY_LEN_24)

    firmres_shorts = tuple(
        u("<H", b, base_end + i * C.FIRMRES_ENTRY_SIZE)[0]
        for i in range(meta.firmres_entries)
    )

    return Group(
        recno=recno,
        group_type=gt_prefix,
        group_recno_field          = u("<H", b, base_start + C.GROUP_OFF_RECNO)[0],
        person_recno               = u("<H", b, base_start + C.GROUP_OFF_PERSON_RECNO)[0],
        firm_count                 = u("<i", b, base_start + C.GROUP_OFF_FIRM_COUNT)[0],
        active_count               = u("<i", b, base_start + C.GROUP_OFF_ACTIVE_COUNT)[0],
        corp_cash                  = u("<d", b, base_start + C.GROUP_OFF_CORP_CASH)[0],
        net_worth                  = u("<d", b, base_start + C.GROUP_OFF_NET_WORTH)[0],
        strategy_budget            = u("<i", b, base_start + C.GROUP_OFF_STRATEGY_BUDGET)[0],
        trailing_12m_net_flow      = u("<d", b, base_start + C.GROUP_OFF_T12M_NET_FLOW)[0],
        trailing_12m_net_profit    = u("<d", b, base_start + C.GROUP_OFF_T12M_NET_PROFIT)[0],
        trailing_12m_special_revenue = u("<d", b, base_start + C.GROUP_OFF_T12M_SPECIAL_REVENUE)[0],
        tax_reserve                = u("<d", b, base_start + C.GROUP_OFF_TAX_RESERVE)[0],
        monthly_net_flow           = monthly_net_flow,
        monthly_net_profit         = monthly_net_profit,
        monthly_gross_field        = monthly_gross_field,
        monthly_special_revenue    = monthly_special_revenue,
        monthly_business_revenue   = monthly_business_revenue,
        monthly_expense_total      = monthly_expense_total,
        monthly_net_worth_snap     = monthly_net_worth_snap,
        monthly_sales_flow         = monthly_sales_flow,
        account_balance            = account_balance,
        loan_principal             = u("<d", b, base_start + C.GROUP_OFF_LOAN_PRINCIPAL)[0],
        bond_interest_liability    = u("<d", b, base_start + C.GROUP_OFF_BOND_INTEREST_LIABILITY)[0],
        account_balance_2          = account_balance_2,
        dept_revenue_current       = dept_rev_cur,
        dept_revenue_ytd           = dept_rev_ytd,
        dept_revenue_alltime       = dept_rev_all,
        firmres_shorts             = firmres_shorts,
        raw_base                   = bytes(b[base_start:base_end]),
        raw_firmres_subarray       = bytes(b[base_end:firmres_end]),
        raw_nationstock            = bytes(b[firmres_end:nationstock_end]),
    )


def parse_group_array(blob: bytes, tag_offset: int) -> list[Group]:
    """Parse the full GroupArray following tag 0x10CA at ``tag_offset``."""
    meta = _parse_group_array_metadata(blob, tag_offset)
    groups: list[Group] = []
    off = meta.first_record
    for i in range(meta.active_count):
        # Validate the prefix short before parsing — this catches drift early.
        gt_prefix = struct.unpack_from("<H", blob, off)[0]
        if gt_prefix not in (1, 3, 4):
            raise ValueError(
                f"expected Group record at 0x{off:x} but prefix short "
                f"is 0x{gt_prefix:04x} (record {i+1}/{meta.active_count})"
            )
        groups.append(parse_group(blob, off, meta, recno=i + 1))
        off += meta.record_stride
    return groups


# ---- Nation — best-effort (raw blob exposed) ------------------------------

def parse_nation_array_raw(blob: bytes, tag_offset: int, end_offset: int) -> bytes:
    return bytes(blob[tag_offset + C.TAG_SIZE : end_offset])


# ---- Group ↔ Firm linkage --------------------------------------------------

def link_groups_and_firms(groups: List[Group], firms: List[Firm]) -> None:
    """Populate ``Group.firm_recnos`` with the recnos of firms owned by that
    Group.

    Ownership is decoded from ``Firm+0xF8`` (see ``FIRM_OFF_GROUP_RECNO``).
    The engine treats this as the canonical firm→group link for revenue /
    expense accounting (verified in ``Structs_Firm.txt §314``).
    """
    by_group: dict[int, list[int]] = {}
    for f in firms:
        g = f.group_recno
        if g <= 0:
            continue  # unowned / deleted slot
        by_group.setdefault(g, []).append(f.recno)

    by_recno: dict[int, Group] = {g.recno: g for g in groups}
    for gr, firm_list in by_group.items():
        g = by_recno.get(gr)
        if g is None:
            # Stale / dangling pointer.  Don't crash — the engine tolerates
            # this too (it simply ignores firms whose group has been wiped).
            continue
        g.firm_recnos = tuple(sorted(firm_list))


# ---- top-level facade ------------------------------------------------------

@dataclass
class CapLabSave:
    """Parsed Capitalism Lab save file.

    Instantiate via :py:meth:`CapLabSave.load` — construction from raw fields
    is intended for tests.
    """

    path: str
    decompressed: DecompressResult
    game_info: GameInfo
    economy: Optional[EconomyState]
    groups: List[Group]
    firms: List[Firm]
    nation_raw: bytes
    rng: RNGState
    tags_section2: List[TagSlot]

    # ---- convenience accessors ----------------------------------------
    @property
    def rng_seed(self) -> int:
        return self.rng.seed

    @property
    def year(self) -> int:
        return self.game_info.year

    # ---- cross-reference helpers --------------------------------------
    def firms_of(self, group_recno: int) -> List[Firm]:
        """Return the Firm objects owned by the group with ``group_recno``."""
        return [f for f in self.firms if f.group_recno == group_recno]

    def group_of(self, firm_recno: int) -> Optional[Group]:
        """Return the Group that owns the firm with ``firm_recno`` (or None)."""
        firm = next((f for f in self.firms if f.recno == firm_recno), None)
        if firm is None or firm.group_recno <= 0:
            return None
        return next((g for g in self.groups if g.recno == firm.group_recno), None)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CapLabSave":
        """Load a .SAV file and return a fully parsed :class:`CapLabSave`."""
        path = os.fspath(path)
        dec = decompress_save(path)
        if len(dec.streams) < 1:
            raise ValueError(f"{path}: no zlib streams found")
        blob0 = dec.streams[0].data

        # -- Section 1 header -------------------------------------------
        gameinfo_tag_off = blob0.find(struct.pack("<I", C.TAG_GAMEINFO))
        if gameinfo_tag_off < 0:
            raise ValueError("tag 0x1065 (GameInfo) not found in blob 0")
        _require_tag(blob0, gameinfo_tag_off, C.TAG_GAMEINFO)

        game_info = parse_gameinfo(blob0, gameinfo_tag_off + C.TAG_SIZE)

        # Validate tag 0x1066 follows at the expected offset.
        exp_1066 = gameinfo_tag_off + C.TAG_SIZE + C.SIZE_GAMEINFO
        _require_tag(blob0, exp_1066, C.TAG_SECONDARY_CFG)

        # Tag 0x1067 payload is the game-state block that holds the RNG seed.
        exp_1067 = exp_1066 + C.TAG_SIZE + C.SIZE_SECONDARY_CFG
        _require_tag(blob0, exp_1067, C.TAG_GAMESTATE)

        rng = parse_rng_seed(blob0)

        # -- Section 2 sequence -----------------------------------------
        tags2 = find_tag_sequence(blob0, C.SECTION_2_TAGS)
        tag_by_id = {t.tag: t for t in tags2}

        # Groups
        groups = parse_group_array(blob0, tag_by_id[C.TAG_GROUP_ARRAY].offset)

        # Firms — sentinel-walked per record (see :pymod:`caplab_save.firm`).
        firms_start = tag_by_id[C.TAG_FIRM_ARRAY].offset
        firms_end   = tag_by_id[C.TAG_RECORD_1C].offset
        firms = parse_firm_array(blob0, firms_start, firms_end)

        # Cross-reference firm ownership back onto each Group.
        link_groups_and_firms(groups, firms)

        # Nations: raw blob for now (base block + 4 subarrays pending RE).
        nation_start = tag_by_id[C.TAG_NATION_ARRAY].offset
        nation_end   = tag_by_id[C.TAG_RECORD_50_VTABLE].offset
        nation_raw   = parse_nation_array_raw(blob0, nation_start, nation_end)

        # -- Economy (scans both blobs) ---------------------------------
        blobs = [s.data for s in dec.streams]
        loc = locate_economy(blobs)
        economy: Optional[EconomyState] = None
        if loc is not None:
            blob_idx, off = loc
            economy = parse_economy(blobs[blob_idx], off)

        return cls(
            path=path,
            decompressed=dec,
            game_info=game_info,
            economy=economy,
            groups=groups,
            firms=firms,
            nation_raw=nation_raw,
            rng=rng,
            tags_section2=tags2,
        )


# ---- CLI: python -m caplab_save.parser path/to/PLAY.SAV -------------------

def _format_summary(save: "CapLabSave") -> str:
    gi = save.game_info
    lines = [
        f"path:        {save.path}",
        f"blobs:       {len(save.decompressed.streams)} "
        + ", ".join(f"{len(s.data):,}" for s in save.decompressed.streams),
        f"calendar:    year={gi.year}  month={gi.month_of_year}  day={gi.day_of_month}  "
        f"(game_date={gi.game_date})",
        f"RNG seed:    0x{save.rng_seed:08x}  (at blob0+0x{save.rng.source_offset:06x})",
        f"groups:      {len(save.groups)}",
    ]
    for g in save.groups:
        owned = len(g.firm_recnos)
        lines.append(
            f"  rec {g.recno:2d}  type={g.group_type}  "
            f"cash=${g.corp_cash:>18,.2f}  net_worth=${g.net_worth:>18,.2f}  "
            f"firms={g.firm_count}/{g.active_count}  owned={owned}"
        )
    if save.economy is not None:
        e = save.economy
        lines.append(
            f"economy:     bir={e.base_interest_rate}  gdp={e.gdp_growth_rate}  "
            f"cycle={e.cycle_phase}  irl={e.interest_rate_level}  "
            f"cpi={e.price_level_CPI}  ppp={e.price_level_PPP}  "
            f"air={e.annual_inflation_rate}  sit={e.stock_index_target}  "
            f"(blob+0x{e.found_at_blob:x})"
        )
    else:
        lines.append("economy:     <not located>")

    # Firm summary by type.
    from collections import Counter
    type_hist = Counter(f.firm_type for f in save.firms)
    hist_str = ", ".join(f"0x{t:02x}×{n}" for t, n in sorted(type_hist.items()))
    lines.append(f"firms:       {len(save.firms)} records  [{hist_str}]")

    # Typed-subclass coverage: how many firms decode to a named subclass view
    # vs. are still held as raw bytes.
    from caplab_save.firm_subclass import supported_types
    covered = supported_types()
    typed   = sum(1 for f in save.firms if f.firm_type in covered)
    lines.append(
        f"subclass:    {typed}/{len(save.firms)} firms typed "
        f"(types: {', '.join(f'0x{t:02x}' for t in sorted(covered))})"
    )
    lines.append(f"nation_raw:  {len(save.nation_raw):,} bytes")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m caplab_save.parser SAVE.SAV [SAVE2.SAV ...]")
        return 2
    for i, path in enumerate(argv[1:]):
        if i > 0:
            print()
        try:
            save = CapLabSave.load(path)
        except Exception as exc:  # noqa: BLE001 (diagnostic CLI)
            print(f"{path}: FAILED to parse: {exc}")
            return 1
        print(_format_summary(save))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv))
