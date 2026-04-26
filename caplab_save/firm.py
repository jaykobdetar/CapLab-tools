"""Firm record parser.

Capitalism Lab serializes every firm with a strict, self-validating layout
(see ``structs/Structs_Serialization.txt`` §Firm Serialization)::

    [short] firm_type         (== firm+0x08 in-memory)
    [short] firm_subtype      (== firm+0x0A in-memory)
    [0x718] Firm BASE blob    (base+0x08 duplicates firm_type/subtype)
    [short] sentinel 0x81
    [N]     subclass blob     (size = firm_class_size(type, subtype) − 0x718)
    [short] sentinel 0x82
    For i in 0..base[+0xB0] − 1:
        [0x48] UnitGroup data
        [short] sentinel 0x83

The parser is **sentinel-first**: we do not trust any external size table for
the subclass.  Instead we find sentinel ``0x82`` by scanning bytes forward
from ``0x81`` (8-bit struct alignment means it can appear at any offset),
derive the true subclass size from the gap, and then validate each unit-group
tail ``0x83``.  This matches the engine's serializer and catches drift the
moment it happens.

Empirical facts (verified across PLAY_001..PLAY_004, 176 firms each):
    * Every record begins with a 4-byte prefix whose type/subtype
      duplicate base[+0x08]/base[+0x0A].  The signature used to locate
      the first record is (prefix == base-duplicate) AND (0x81 at
      record+0x71C).
    * Every record ends with sentinel ``0x82`` at (record_end − 2) when
      ``unit_group_count == 0``; else it ends with a ``0x83`` sentinel
      after the last UnitGroup block.
    * Firm+0xF8 (int) is the owning Group's recno; this is the
      canonical firm→group link used by the game's revenue/expense
      accounting (see ``FIRM_OFF_GROUP_RECNO`` in constants.py).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Tuple

from caplab_save import constants as C
from caplab_save.structs import Firm


# ---- low-level helpers ---------------------------------------------------

def _u(fmt: str, buf: bytes, off: int):
    return struct.unpack_from(fmt, buf, off)[0]


def _unpack_monthly(buf: bytes, off: int) -> Tuple[float, ...]:
    return tuple(_u("<d", buf, off + i * 8) for i in range(C.MONTHLY_ARRAY_LEN))


def _find_short(buf: bytes, start: int, target_lo: int, target_hi: int,
                limit: int) -> int:
    """Scan ``buf[start : start+limit]`` byte-by-byte for a 2-byte
    little-endian short matching ``target_lo, target_hi``.  Returns the
    absolute offset or -1.

    8-bit struct alignment means the sentinel can appear at odd offsets.
    """
    end = min(start + limit, len(buf) - 1)
    o = start
    while o < end:
        if buf[o] == target_lo and buf[o + 1] == target_hi:
            return o
        o += 1
    return -1


# ---- single-record parse -------------------------------------------------

@dataclass
class _FirmWalkResult:
    record_size: int
    subclass_size: int
    ug_count: int


def _validate_and_size(blob: bytes, off: int) -> _FirmWalkResult:
    """Validate sentinels at ``off`` and compute the record's total size.

    Raises ValueError on any sentinel mismatch — that is the whole point of
    sentinel-based walking.
    """
    if off + 4 + C.SIZE_FIRM_BASE + 2 > len(blob):
        raise ValueError(f"firm record at 0x{off:x} runs past end of blob")

    ft_prefix = _u("<H", blob, off)
    st_prefix = _u("<H", blob, off + 2)
    ft_base   = _u("<H", blob, off + 4 + C.FIRM_OFF_TYPE)
    st_base   = _u("<H", blob, off + 4 + C.FIRM_OFF_SUBTYPE)
    if ft_prefix != ft_base or st_prefix != st_base:
        raise ValueError(
            f"firm at 0x{off:x}: prefix/base type mismatch "
            f"(prefix=0x{ft_prefix:04x}/0x{st_prefix:04x}, "
            f"base=0x{ft_base:04x}/0x{st_base:04x})"
        )

    s81_off = off + 4 + C.SIZE_FIRM_BASE   # = off + 0x71C
    s81 = _u("<H", blob, s81_off)
    if s81 != C.FIRM_SENTINEL_81:
        raise ValueError(
            f"firm at 0x{off:x}: sentinel 0x81 missing at +0x71C "
            f"(got 0x{s81:04x})"
        )

    # Unit-group count lives at base+0xB0 (short).
    ug_count = _u("<h", blob, off + 4 + C.FIRM_OFF_UNIT_GROUP_COUNT)
    if ug_count < 0 or ug_count > 32:
        raise ValueError(
            f"firm at 0x{off:x}: implausible unit_group_count={ug_count}"
        )

    # Scan forward for the next 0x82 sentinel (subclass end).  8-bit struct
    # alignment means it can appear at any offset.  Limit search to a sane
    # upper bound well past the largest documented subclass (0x3048 for
    # FirmMedia) plus a generous margin.
    sub_start = s81_off + 2
    s82_off = _find_short(blob, sub_start, 0x82, 0x00, limit=0x8000)
    if s82_off < 0:
        raise ValueError(
            f"firm at 0x{off:x}: sentinel 0x82 not found after 0x81"
        )
    subclass_size = s82_off - sub_start

    # Each unit group = 0x48 bytes + 2-byte 0x83 sentinel.
    ug_start = s82_off + 2
    for i in range(ug_count):
        s83_off = ug_start + (i + 1) * C.SIZE_FIRM_UNITGROUP + i * 2
        if s83_off + 2 > len(blob):
            raise ValueError(
                f"firm at 0x{off:x}: UG #{i} 0x83 sentinel runs past blob"
            )
        s83 = _u("<H", blob, s83_off)
        if s83 != C.FIRM_SENTINEL_83:
            raise ValueError(
                f"firm at 0x{off:x}: UG #{i} sentinel 0x83 missing "
                f"(got 0x{s83:04x})"
            )

    total_size = ug_start + ug_count * (C.SIZE_FIRM_UNITGROUP + 2) - off
    return _FirmWalkResult(total_size, subclass_size, ug_count)


def parse_firm(blob: bytes, off: int, recno: int) -> Tuple[Firm, int]:
    """Decode a single firm record starting at byte offset ``off`` in
    ``blob``.

    Returns ``(firm, next_offset)`` — ``next_offset`` is where the caller
    should resume parsing (immediately after the final sentinel).
    """
    walk = _validate_and_size(blob, off)

    # Slice the record into its pieces.
    prefix_end     = off + 4
    base_end       = prefix_end + C.SIZE_FIRM_BASE
    sent81_end     = base_end + 2                         # after 0x81
    subclass_end   = sent81_end + walk.subclass_size
    sent82_end     = subclass_end + 2                     # after 0x82
    ugs_end        = off + walk.record_size

    raw_base     = bytes(blob[prefix_end:base_end])
    raw_subclass = bytes(blob[sent81_end:subclass_end])
    raw_ugs      = bytes(blob[sent82_end:ugs_end])

    # Validated fields from the base blob.
    ft = _u("<H", raw_base, C.FIRM_OFF_TYPE)
    st = _u("<H", raw_base, C.FIRM_OFF_SUBTYPE)

    ai_cap = _u("<d", raw_base, C.FIRM_OFF_AI_CAPACITY_TARGET)
    util   = _u("<d", raw_base, C.FIRM_OFF_UTILIZATION)
    demand_bonus = _u("<d", raw_base, C.FIRM_OFF_DEMAND_BONUS)

    net_profit = _unpack_monthly(raw_base, C.FIRM_OFF_MONTHLY_NET_PROFIT)
    revenue    = _unpack_monthly(raw_base, C.FIRM_OFF_MONTHLY_REVENUE)
    expense    = _unpack_monthly(raw_base, C.FIRM_OFF_MONTHLY_EXPENSE)

    dept_1 = bytes(raw_base[C.FIRM_OFF_DEPT_BLOCK_1:C.FIRM_OFF_DEPT_BLOCK_1 + 0x60])
    dept_2 = bytes(raw_base[C.FIRM_OFF_DEPT_BLOCK_2:C.FIRM_OFF_DEPT_BLOCK_2 + 0x60])
    dept_3 = bytes(raw_base[C.FIRM_OFF_DEPT_BLOCK_3:C.FIRM_OFF_DEPT_BLOCK_3 + 0x60])
    dept_4 = bytes(raw_base[C.FIRM_OFF_DEPT_BLOCK_4:C.FIRM_OFF_DEPT_BLOCK_4 + 0x60])

    # Note: ``raw_dynamic`` is only nonempty for firm_type 0x25 per the
    # documented format.  Today we fold the entire post-0x81 blob into
    # raw_subclass and leave raw_dynamic empty; if a later RE pass
    # identifies the dynamic tail's boundary within the 0x25 subclass
    # we can split it off without touching callers.
    raw_dynamic = b""

    firm = Firm(
        recno=recno,
        firm_type=ft,
        firm_subtype=st,
        ai_capacity_target=ai_cap,
        utilization=util,
        demand_bonus=demand_bonus,
        monthly_net_profit=net_profit,
        monthly_revenue=revenue,
        monthly_expense=expense,
        dept_block_1=dept_1,
        dept_block_2=dept_2,
        dept_block_3=dept_3,
        dept_block_4=dept_4,
        raw_base=raw_base,
        raw_subclass=raw_subclass,
        raw_unitgroups=raw_ugs,
        raw_dynamic=raw_dynamic,
    )
    return firm, ugs_end


# ---- array walk ----------------------------------------------------------

@dataclass
class _FirmArrayMeta:
    active_count: int
    payload_start: int
    first_record: int


def _find_first_firm(blob: bytes, start: int, end: int) -> int:
    """Locate the first firm record after the array-manager header.

    We scan for the first byte offset ``o`` where all of these hold:
      * short at ``o`` is a plausible firm_type (1..0x50, nonzero)
      * short at ``o+0x0C`` == short at ``o`` (type is duplicated in base[+0x08])
      * short at ``o+0x71C`` == 0x81 (sentinel lives right after the base)
    """
    o = start
    last = end - (4 + C.SIZE_FIRM_BASE + 2)
    while o < last:
        ft = _u("<H", blob, o)
        if 0 < ft < C.FIRM_TYPE_MAX:
            if _u("<H", blob, o + 4 + C.FIRM_OFF_TYPE) == ft:
                if _u("<H", blob, o + 4 + C.SIZE_FIRM_BASE) == C.FIRM_SENTINEL_81:
                    return o
        o += 1
    raise ValueError(
        f"first firm record not found between 0x{start:x} and 0x{end:x}"
    )


def parse_firm_array(blob: bytes, tag_offset: int, end_offset: int) -> List[Firm]:
    """Parse the full FirmArray following tag 0x10CB.

    ``tag_offset`` points at the tag header; ``end_offset`` is the offset
    of the next section-2 tag (typically 0x10CC).  The engine writes an
    array-manager header of ~32 bytes then ``active_count`` records.
    """
    payload = tag_offset + C.TAG_SIZE
    active_count = _u("<I", blob, payload)   # first int of the header

    first_rec = _find_first_firm(blob, payload + 0x20, end_offset)

    firms: List[Firm] = []
    off = first_rec
    for i in range(active_count):
        if off >= end_offset:
            raise ValueError(
                f"firm walk ran out of blob at record {i}/{active_count} "
                f"(expected more records before 0x{end_offset:x})"
            )
        firm, next_off = parse_firm(blob, off, recno=i + 1)
        firms.append(firm)
        off = next_off

    return firms
