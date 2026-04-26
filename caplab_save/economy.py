"""Economy struct locator and parser.

The Economy singleton lives at static VA 0x00A47698 in CapMain.exe.  Its
save-side location is not yet pinned to a specific version tag; empirically it
appears in the second zlib stream (blob1) of the .SAV file.

Rather than rely on a fixed offset we scan for the structure's signature, then
decode the fields we care about.  The signature we use:

    * int  at +0x30 in [1..7]                       (cycle_phase)
    * int  at +0x38 in [1..9]                       (interest_rate_level)
    * dbl  at +0x40 == interest_rate_level * 10     (target_rate_pct)
    * int  at +0x50 int                             (prev_interest_rate_level)
    * dbl  at +0x10 in (1, 30)                      (base_interest_rate)
    * dbl  at +0x28 in [-15, 15]                    (gdp_growth_rate)
    * dbl  at +0x60 in [-50, 50], finite            (annual_inflation_rate)
    * dbl  at +0x68 in [0.5, 100]                   (price_level_CPI)
    * dbl  at +0x70 in [0.5, 100]                   (price_level_PPP)

This combination has been observed to pick out exactly one structure per save
in testing (PLAY_001..PLAY_004).  We scan blob 1 before blob 0 because the
struct empirically lives in the second zlib stream; blob 0 contains
pattern-colliding byte sequences that would otherwise give false positives.
"""

from __future__ import annotations

import math
import struct
from typing import Optional

from caplab_save import constants as C
from caplab_save.structs import EconomyState


def _looks_like_economy(buf: bytes, base: int) -> bool:
    """Return True if bytes at ``base`` look like the Economy struct.

    Uses a multi-field signature tight enough to exclude the random
    double-array collisions that otherwise occur in blob 0.
    """
    if base < 0 or base + C.ECONOMY_SIZE_TOTAL > len(buf):
        return False
    try:
        cp  = struct.unpack_from('<i', buf, base + C.ECONOMY_OFF_CYCLE_PHASE)[0]
        irl = struct.unpack_from('<i', buf, base + C.ECONOMY_OFF_INTEREST_RATE_LEVEL)[0]
        if not (1 <= cp <= 7 and 1 <= irl <= 9):
            return False
        target_pct = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_TARGET_RATE_PCT)[0]
        # target_rate_pct = interest_rate_level × 10 (engine invariant, every month)
        if abs(target_pct - irl * 10.0) > 1e-6:
            return False
        bir = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_BASE_INTEREST_RATE)[0]
        # Base interest rate is a *nonzero* percent (engine never runs at 0%).
        if not (1.0 <= bir <= 30.0):
            return False
        gdp = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_GDP_GROWTH_RATE)[0]
        if not (math.isfinite(gdp) and -15.0 <= gdp <= 15.0):
            return False
        air = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_ANNUAL_INFLATION_RATE)[0]
        if not (math.isfinite(air) and -50.0 <= air <= 50.0):
            return False
        cpi = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_PRICE_LEVEL_CPI)[0]
        if not (math.isfinite(cpi) and 0.5 <= cpi <= 100.0):
            return False
        ppp = struct.unpack_from('<d', buf, base + C.ECONOMY_OFF_PRICE_LEVEL_PPP)[0]
        if not (math.isfinite(ppp) and 0.5 <= ppp <= 100.0):
            return False
        return True
    except struct.error:
        return False


def locate_economy(blobs: list[bytes]) -> tuple[int, int] | None:
    """Scan blobs for the Economy struct and return (blob_index, offset).

    Blob 1 is scanned before blob 0: the struct lives in the second zlib
    stream in every save observed so far, and blob 0 is pattern-heavy enough
    that even the strengthened signature may not fully eliminate false
    positives there.  Returns None if no plausible struct is found.
    """
    order = [1, 0] + list(range(2, len(blobs)))
    for idx in order:
        if idx >= len(blobs):
            continue
        off = _find_in_blob(blobs[idx])
        if off is not None:
            return idx, off
    return None


def _find_in_blob(buf: bytes) -> Optional[int]:
    # doubles are 8-byte aligned in the blob as written by the engine
    for base in range(0, len(buf) - C.ECONOMY_SIZE_TOTAL, 4):
        if _looks_like_economy(buf, base):
            return base
    return None


def parse_economy(blob: bytes, offset: int) -> EconomyState:
    """Decode an Economy struct at ``offset`` in ``blob``."""
    u = struct.unpack_from
    bir = u('<d', blob, offset + C.ECONOMY_OFF_BASE_INTEREST_RATE)[0]
    gdp = u('<d', blob, offset + C.ECONOMY_OFF_GDP_GROWTH_RATE)[0]
    cp  = u('<i', blob, offset + C.ECONOMY_OFF_CYCLE_PHASE)[0]
    irl = u('<i', blob, offset + C.ECONOMY_OFF_INTEREST_RATE_LEVEL)[0]
    sit = u('<d', blob, offset + C.ECONOMY_OFF_STOCK_INDEX_TARGET)[0]
    air = u('<d', blob, offset + C.ECONOMY_OFF_ANNUAL_INFLATION_RATE)[0]
    cpi = u('<d', blob, offset + C.ECONOMY_OFF_PRICE_LEVEL_CPI)[0]
    ppp = u('<d', blob, offset + C.ECONOMY_OFF_PRICE_LEVEL_PPP)[0]

    return EconomyState(
        base_interest_rate=bir,
        gdp_growth_rate=gdp,
        cycle_phase=cp,
        interest_rate_level=irl,
        stock_index_target=sit,
        annual_inflation_rate=air,
        price_level_CPI=cpi,
        price_level_PPP=ppp,
        raw=bytes(blob[offset:offset + C.ECONOMY_SIZE_TOTAL]),
        found_at_blob=offset,
    )
