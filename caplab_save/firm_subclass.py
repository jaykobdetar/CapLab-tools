"""Typed decoders for Firm subclass data regions.

The Firm base blob is 0x718 bytes; each subclass follows immediately with a
variable-length extension whose size is determined by ``firm_type`` (and for
a handful of types, ``firm_subtype``).  The serialization layout
(``[0x81 sentinel][subclass bytes][0x82 sentinel]``) is decoded by
:pymod:`caplab_save.firm`; this module takes the ``raw_subclass`` bytes it
produces and interprets them into typed dataclasses when the field map is
known.

Current coverage
----------------

* **Types 0x01 / 0x02 / 0x04 / 0x05 / 0x07** — FirmHq, FirmRetail,
  FirmFactory, FirmRD, FirmWarehouse.  8-byte subclass.  FirmRD's fields
  are documented (char ``focus_flag`` at +0x00, int ``product_recno`` at
  +0x04); the other four have no reverse-engineered field names yet and
  are exposed as four little-endian shorts for inspection.

* **Types 0x20..0x24** — FirmPublic variants (firm types 32..36:
  police, fire station, city hall, park, school building — the exact
  mapping is encoded by the type code itself).  Base-only records,
  zero-byte subclass; decoded to an empty :class:`FirmPublicSub`.

Types not yet covered return ``None`` from :func:`decode_firm_subclass`;
callers should fall back to ``Firm.raw_subclass`` for those.

Validation status
-----------------

All registered decoders are now empirically anchored to real save
fixtures captured from the running engine:

* **FirmPublic (0x20..0x24)** — PLAY_001..PLAY_004 each contain 4 firms
  of every FirmPublic type (20 per save).  Every one round-trips a
  0-byte subclass.
* **Basic types (0x01 / 0x02 / 0x04 / 0x05 / 0x07)** — anchored by the
  five cumulative saves under ``saves/`` (Headquarters, Department,
  Factory, R&D, Warehouse).  Each save is the previous game plus one
  newly-bought player firm; the observed raw_subclass bytes are frozen
  as regression contracts in ``TestBasicFirmFixtures``:

  ======  ==========================  ====================================
  type    raw_subclass (hex)          interpretation
  ======  ==========================  ====================================
  0x01    ``01 00 00 00 00 00 00 00`` HQ: first byte 0x01 (likely active /
                                      primary marker), rest zero.
  0x02    ``00 00 00 00 00 00 00 00`` Retail: all zero on fresh placement.
  0x04    ``00 ff ff ff 00 00 00 00`` Factory: first byte 0, next three
                                      0xFF (sentinel for no selection),
                                      trailing four zero.
  0x05    ``00 00 00 00 00 00 00 00`` R&D: focus_flag=0, product_recno=0.
                                      Fresh place → not focused on any
                                      specific product, which matches
                                      the field map exactly.
  0x07    ``00 ff ff ff 00 00 00 00`` Warehouse: identical byte pattern
                                      to Factory — strongly suggests a
                                      shared sub-structure shape (the
                                      0xFFFFFF triplet is almost certainly
                                      a "no item / no product" sentinel).
  ======  ==========================  ====================================

The field maps for FirmRD's ``focus_flag`` / ``product_recno`` are
transcribed from ``Class_Hierarchy.txt §Known Subclass Field Details``
(Ghidra RE of FirmRD ctor + vtable+0x140 / +0x144 serialization stubs);
the R&D fixture above confirms the default-state values match.  The
other four basic decoders still expose their 8 bytes as four little-
endian shorts (``words``) pending named-field RE — but the raw bytes
are now locked down.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


# ---- canonical type codes -------------------------------------------------

FIRM_TYPE_HQ        = 0x01
FIRM_TYPE_RETAIL    = 0x02
FIRM_TYPE_FACTORY   = 0x04
FIRM_TYPE_RD        = 0x05
FIRM_TYPE_WAREHOUSE = 0x07

#: Types 32..36 serialize with a zero-byte subclass (base-only records).
FIRM_PUBLIC_TYPES: frozenset[int] = frozenset(range(0x20, 0x25))

#: Firm types covered by the typed decoders registered in this module.
FIRM_TYPES_WITH_BASIC_SUBCLASS: frozenset[int] = frozenset({
    FIRM_TYPE_HQ,
    FIRM_TYPE_RETAIL,
    FIRM_TYPE_FACTORY,
    FIRM_TYPE_RD,
    FIRM_TYPE_WAREHOUSE,
})

#: Expected subclass byte lengths.
SUBCLASS_SIZE_BASIC: int = 0x08
SUBCLASS_SIZE_PUBLIC: int = 0x00


# ---- low-level helpers ----------------------------------------------------

def _u(fmt: str, buf: bytes, off: int):
    return struct.unpack_from(fmt, buf, off)[0]


def _require_size(raw: bytes, expected: int, label: str) -> None:
    if len(raw) != expected:
        raise ValueError(
            f"{label} subclass: expected {expected} bytes, got {len(raw)}"
        )


# ---- basic (8-byte) subclasses --------------------------------------------

@dataclass
class FirmHqSub:
    """Subclass payload for FirmHq (firm_type 0x01).

    The FirmHq ctor zeroes 8 bytes at +0x718; no named fields have been
    reverse-engineered yet.  We surface the payload as four little-endian
    shorts for inspection and keep the raw bytes for round-trip.
    """

    words: Tuple[int, int, int, int]
    raw: bytes = field(repr=False)


@dataclass
class FirmRetailSub:
    """Subclass payload for FirmRetail (firm_type 0x02).  8 bytes, fields TBD."""

    words: Tuple[int, int, int, int]
    raw: bytes = field(repr=False)


@dataclass
class FirmFactorySub:
    """Subclass payload for FirmFactory (firm_type 0x04).  8 bytes, fields TBD."""

    words: Tuple[int, int, int, int]
    raw: bytes = field(repr=False)


@dataclass
class FirmRDSub:
    """Subclass payload for FirmRD (firm_type 0x05).

    Documented fields (``Class_Hierarchy.txt §Known Subclass Field Details``):

    * ``focus_flag`` — unsigned byte at subclass+0x00.  Nonzero when the
      R&D firm is configured to focus on a single dedicated product
      rather than developing across the full technology tree.
    * ``product_recno`` — int32 at subclass+0x04.  The ProductType recno
      of the dedicated product (1-based; 0 when the firm is not
      focused).

    The three bytes at subclass+0x01..+0x03 are padding / unused by the
    decompiled reads traced so far.
    """

    focus_flag: int
    product_recno: int
    raw: bytes = field(repr=False)


@dataclass
class FirmWarehouseSub:
    """Subclass payload for FirmWarehouse (firm_type 0x07).  8 bytes, fields TBD."""

    words: Tuple[int, int, int, int]
    raw: bytes = field(repr=False)


@dataclass
class FirmPublicSub:
    """Subclass payload for FirmPublic variants (firm_types 0x20..0x24).

    Base-only records: zero bytes of subclass data.  The public-building
    kind (police, fire station, city hall, ...) is encoded entirely by
    the ``firm_type`` code itself.
    """

    raw: bytes = field(default=b"", repr=False)


#: Public union of every typed-subclass the dispatcher can return.
FirmSubclass = Union[
    FirmHqSub,
    FirmRetailSub,
    FirmFactorySub,
    FirmRDSub,
    FirmWarehouseSub,
    FirmPublicSub,
]


# ---- per-type decoders ----------------------------------------------------

def _decode_basic_words(raw: bytes) -> Tuple[int, int, int, int]:
    _require_size(raw, SUBCLASS_SIZE_BASIC, "basic firm")
    return struct.unpack_from("<HHHH", raw, 0)


def decode_firm_hq(raw: bytes) -> FirmHqSub:
    return FirmHqSub(words=_decode_basic_words(raw), raw=bytes(raw))


def decode_firm_retail(raw: bytes) -> FirmRetailSub:
    return FirmRetailSub(words=_decode_basic_words(raw), raw=bytes(raw))


def decode_firm_factory(raw: bytes) -> FirmFactorySub:
    return FirmFactorySub(words=_decode_basic_words(raw), raw=bytes(raw))


def decode_firm_rd(raw: bytes) -> FirmRDSub:
    _require_size(raw, SUBCLASS_SIZE_BASIC, "FirmRD")
    return FirmRDSub(
        focus_flag=_u("<B", raw, 0x00),
        product_recno=_u("<i", raw, 0x04),
        raw=bytes(raw),
    )


def decode_firm_warehouse(raw: bytes) -> FirmWarehouseSub:
    return FirmWarehouseSub(words=_decode_basic_words(raw), raw=bytes(raw))


def decode_firm_public(raw: bytes) -> FirmPublicSub:
    _require_size(raw, SUBCLASS_SIZE_PUBLIC, "FirmPublic")
    return FirmPublicSub(raw=bytes(raw))


# ---- dispatcher -----------------------------------------------------------

# Populated at module import; keep as a dict (not frozen) so test helpers
# can introspect it without jumping through private-name hoops.
_DECODERS: dict[int, "callable"] = {
    FIRM_TYPE_HQ:        decode_firm_hq,
    FIRM_TYPE_RETAIL:    decode_firm_retail,
    FIRM_TYPE_FACTORY:   decode_firm_factory,
    FIRM_TYPE_RD:        decode_firm_rd,
    FIRM_TYPE_WAREHOUSE: decode_firm_warehouse,
}
for _ft in FIRM_PUBLIC_TYPES:
    _DECODERS[_ft] = decode_firm_public


def decode_firm_subclass(
    firm_type: int, raw_subclass: bytes
) -> Optional[FirmSubclass]:
    """Dispatch to the typed decoder for ``firm_type``.

    Returns ``None`` for firm types that don't yet have a typed decoder —
    callers should fall back to ``Firm.raw_subclass``.
    """
    decoder = _DECODERS.get(firm_type)
    if decoder is None:
        return None
    return decoder(raw_subclass)


def supported_types() -> frozenset[int]:
    """Every firm_type with a typed subclass decoder registered."""
    return frozenset(_DECODERS.keys())


__all__ = [
    "FIRM_TYPE_HQ",
    "FIRM_TYPE_RETAIL",
    "FIRM_TYPE_FACTORY",
    "FIRM_TYPE_RD",
    "FIRM_TYPE_WAREHOUSE",
    "FIRM_PUBLIC_TYPES",
    "FIRM_TYPES_WITH_BASIC_SUBCLASS",
    "SUBCLASS_SIZE_BASIC",
    "SUBCLASS_SIZE_PUBLIC",
    "FirmHqSub",
    "FirmRetailSub",
    "FirmFactorySub",
    "FirmRDSub",
    "FirmWarehouseSub",
    "FirmPublicSub",
    "FirmSubclass",
    "decode_firm_hq",
    "decode_firm_retail",
    "decode_firm_factory",
    "decode_firm_rd",
    "decode_firm_warehouse",
    "decode_firm_public",
    "decode_firm_subclass",
    "supported_types",
]
