"""Dump per-town WorldMap cell matrices from a Capitalism Lab .SAV file.

Usage:
    python -m caplab_save.map_dump <path/to/file.SAV> <out_dir/>

What gets written
-----------------
For each town map found in the save:
    out_dir/town_<idx>_<WxH>.json   - per-cell decoded fields
And one summary file:
    out_dir/index.json              - list of all maps with shapes and locations

Cell record (32 bytes per cell in the save; in-memory cells are 40 bytes —
the 8-byte runtime tail at offsets 0x20-0x27 is NOT serialized):

    +0x00 u16   type_flags     - high nibble: tile type
                                 (0x200=tree/decoration, 0x300=firm-bld,
                                  0x400=firm-owned, 0x500=road,
                                  0x600=building, 0x700=vehicle?,
                                  0x900=land-owned)
                                 0x4000: purchasable flag
                                 0xc0: zone tier (0x40 urban-edge,
                                                  0x80 suburban-edge,
                                                  0xc0 town-interior)
    +0x02 u16   sprite_index   - tile graphic atlas index (1..605); the
                                 underlying ground graphic. NOT a category.
    +0x04 u8    walker_terrain - walking path geometry (0=non-walkable,
                                 1-15=walkable types; see walkres.cpp)
    +0x05 u8    (reserved)
    +0x06 u16   (reserved)
    +0x08 u16   owner_recno    - FirmArray / LandArray / GroupArray recno
                                 of whoever owns/occupies this cell
                                 (interpretation depends on tile_type)
    +0x0a u16   sub_recno      - secondary recno (rarely used)
    +0x0c u16   site_recno     - SiteArray recno of the decoration / site
                                 occupying this cell (e.g. trees set this).
                                 Roads do NOT use this — roads are
                                 identified purely by tile_type==5.
    +0x0e u8    flags_b        - misc flags (bit 7 = cell_lock)
    +0x0f u8    (reserved)
    +0x10 f64   land_value     - base land quality (0.0 – 50.0+)
    +0x18 u8    dev_level      - development level (0–100)
    +0x19 u8    traffic_index  - traffic / pedestrian load (0–100)
    +0x1a u8    congestion_index
    +0x1b u8    zone_type
    +0x1c..1f   4 bytes additional fields (partially mapped)

Per-map header (8 bytes immediately before the cell matrix):

    +0x00 i16   map_width      - cells in X dimension
    +0x02 i16   map_height     - cells in Y dimension
    +0x04 u32   unknown        - 4-byte filler (looks like a checksum or seed,
                                 differs per town; not yet decoded)

Discovery
---------
Maps are located by scanning every decompressed zlib stream for the dim+cell
signature: a (W:i16, H:i16) pair where 16 <= W,H <= 1024, followed by 4 bytes
of filler, followed by W*H*32 bytes of plausible cell data.

A cell is "plausible" when its land_value double parses as either 0.0 or a
finite value in [0, 1000].
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

# Allow running both as module and script
if __package__ in (None, ""):
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)

from caplab_save.decompress import decompress_save  # noqa: E402

CELL_BYTES = 32
HEADER_BYTES = 8
DIM_MIN = 16
DIM_MAX = 1024
LV_MAX_PLAUSIBLE = 1000.0


@dataclass
class MapLocation:
    """Where a town map lives in the decompressed save."""

    stream_index: int
    header_offset: int          # offset of (W, H, filler) header
    cell_array_offset: int      # offset of first cell (= header + 8)
    width: int
    height: int


def _looks_like_cell(blob: bytes, off: int) -> bool:
    if off + CELL_BYTES > len(blob):
        return False
    lv = struct.unpack_from("<d", blob, off + 0x10)[0]
    # NaN/inf check + plausible land_value
    if lv != lv or lv == float("inf") or lv == float("-inf"):
        return False
    if lv < 0 or lv > LV_MAX_PLAUSIBLE:
        return False
    return True


def _validate_map_region(blob: bytes, cell_off: int, n_cells: int) -> bool:
    """Strict validation that the region is a real cell array.

    Two checks:
      1. Every single cell must have a finite, non-negative land_value
         in [0, LV_MAX_PLAUSIBLE].
      2. At least 50% of cells must have land_value > 0 (real WorldMap
         cells are populated; spurious zero-padded matches fail this).
    """
    if cell_off + n_cells * CELL_BYTES > len(blob):
        return False
    nonzero_lv = 0
    for ci in range(n_cells):
        off = cell_off + ci * CELL_BYTES
        lv = struct.unpack_from("<d", blob, off + 0x10)[0]
        if lv != lv or lv == float("inf") or lv == float("-inf"):
            return False
        if lv < 0 or lv > LV_MAX_PLAUSIBLE:
            return False
        if lv > 0:
            nonzero_lv += 1
    return nonzero_lv * 2 >= n_cells  # >= 50%


_PREFIX_ZERO_BYTES = 8   # require 8 bytes of zeros immediately before dims


def find_map_locations(streams_data: List[bytes]) -> List[MapLocation]:
    """Locate every WorldMap section across all decompressed streams.

    Algorithm:
      1. Scan every byte offset; collect every (W, H) candidate where:
           - 8 bytes immediately before the dims are all zero (the
             WorldMap struct prefix is mostly empty in fresh saves)
           - proposed cell region passes strict validation
             (all cells parse, ≥50% nonzero land_value)
      2. Sort candidates by area (W*H), largest first.
      3. Greedily accept candidates that don't overlap with any already-
         accepted candidate's cell region. Rejects spurious smaller
         "subset" matches that fall inside a real map.
    """
    candidates: List[MapLocation] = []
    for stream_idx, blob in enumerate(streams_data):
        n = len(blob)
        i = _PREFIX_ZERO_BYTES
        while i + HEADER_BYTES + CELL_BYTES <= n:
            # Cheap pre-filter: the 8 bytes before the dim header must be zero
            if blob[i - _PREFIX_ZERO_BYTES : i] != b"\x00" * _PREFIX_ZERO_BYTES:
                i += 1
                continue
            w = struct.unpack_from("<h", blob, i)[0]
            h = struct.unpack_from("<h", blob, i + 2)[0]
            if DIM_MIN <= w <= DIM_MAX and DIM_MIN <= h <= DIM_MAX:
                cell_off = i + HEADER_BYTES
                n_cells = w * h
                cells_size = n_cells * CELL_BYTES
                if (cell_off + cells_size <= n
                        and _looks_like_cell(blob, cell_off)
                        and _validate_map_region(blob, cell_off, n_cells)):
                    candidates.append(MapLocation(
                        stream_index=stream_idx,
                        header_offset=i,
                        cell_array_offset=cell_off,
                        width=w,
                        height=h,
                    ))
            i += 1

    # Sort: largest area first; ties broken by smaller offset (earlier wins)
    candidates.sort(key=lambda c: (-(c.width * c.height),
                                    c.stream_index,
                                    c.cell_array_offset))

    # Per stream: list of (start, end) intervals of accepted maps
    accepted_by_stream: dict[int, List[tuple[int, int]]] = {}
    accepted: List[MapLocation] = []
    for cand in candidates:
        start = cand.cell_array_offset
        end = start + cand.width * cand.height * CELL_BYTES
        intervals = accepted_by_stream.setdefault(cand.stream_index, [])
        if any(not (end <= s or start >= e) for s, e in intervals):
            continue  # overlaps with already-accepted
        intervals.append((start, end))
        accepted.append(cand)

    # Return in (stream_index, offset) order for stable output
    accepted.sort(key=lambda c: (c.stream_index, c.cell_array_offset))
    return accepted


def decode_cell(blob: bytes, off: int) -> dict:
    """Decode one 32-byte cell into a dict of field name -> value."""
    type_flags     = struct.unpack_from("<H", blob, off + 0x00)[0]
    sprite_index   = struct.unpack_from("<H", blob, off + 0x02)[0]
    walker_terrain = blob[off + 0x04]            # 0 = no road, 1-15 = road sprite type
    # +0x08 is *not* the owner — it's the recno of whatever feature is on
    # the cell, indexed into a different array depending on tile_type:
    #   tile_type==3 (firm)       → FirmArray recno  (firm.owner_group is the real owner)
    #   tile_type==6 (building)   → BuildArray recno (residential/commercial building)
    #   tile_type==9 (nation-owned land) → LandArray recno (then LandEntry.owner_group)
    feature_recno  = struct.unpack_from("<H", blob, off + 0x08)[0]
    sub_recno      = struct.unpack_from("<H", blob, off + 0x0a)[0]
    site_recno     = struct.unpack_from("<H", blob, off + 0x0c)[0]
    flags_b        = blob[off + 0x0e]
    land_value     = struct.unpack_from("<d", blob, off + 0x10)[0]
    dev_level      = blob[off + 0x18]
    traffic        = blob[off + 0x19]
    congestion     = blob[off + 0x1a]
    zone_type      = blob[off + 0x1b]
    extra4         = blob[off + 0x1c : off + 0x20]

    tile_type = (type_flags & 0x0f00) >> 8

    # Predicates extracted verbatim from CapMain.exe FUN_006260f0
    # (the F1-debug overlay). These are canonical.
    walkable    = bool(type_flags & 0x01)
    sailable    = bool(type_flags & 0x02)        # → water
    is_coast    = bool(type_flags & 0x08)
    has_site    = (type_flags & 0x30) == 0x10
    is_harbor   = bool(type_flags & 0x1000)
    has_hill    = tile_type == 1
    is_plant    = tile_type == 2                  # tree / decoration
    is_firm     = tile_type == 3
    is_farm_ext = tile_type == 4
    is_building = tile_type == 6
    is_road     = walker_terrain != 0             # NOT a tile_type bit!
    purchasable = bool(type_flags & 0x4000)

    # Zone tier — from FUN_006260f0 "UB SU RU" computation:
    if   land_value >= 30.0 or (type_flags & 0xc0) == 0x40: zone_tier = "urban"
    elif land_value >= 15.0 or (type_flags & 0xc0) == 0x80: zone_tier = "suburban"
    else:                                                   zone_tier = "rural"

    # Single-word category for visualization. Order matters: roads/firms
    # take priority over the underlying terrain bits.
    if   is_road:     category = "road"
    elif is_plant:    category = "plant"
    elif is_firm:     category = "firm"
    elif is_farm_ext: category = "farm_ext"
    elif is_building: category = "building"
    elif has_hill:    category = "hill"
    elif sailable:    category = "water"
    elif is_coast:    category = "coast"
    else:             category = "land"

    return {
        "type_flags":      type_flags,
        "tile_type":       tile_type,
        "category":        category,
        "walkable":        walkable,
        "sailable":        sailable,
        "is_coast":        is_coast,
        "has_site":        has_site,
        "is_harbor":       is_harbor,
        "has_hill":        has_hill,
        "is_plant":        is_plant,
        "is_firm":         is_firm,
        "is_farm_ext":     is_farm_ext,
        "is_building":     is_building,
        "is_road":         is_road,
        "purchasable":     purchasable,
        "zone_tier":       zone_tier,
        "sprite_index":    sprite_index,
        "walker_terrain":  walker_terrain,
        "feature_recno":   feature_recno,
        "sub_recno":       sub_recno,
        "site_recno":      site_recno,
        "flags_b":         flags_b,
        "land_value":      land_value,
        "dev_level":       dev_level,
        "traffic_index":   traffic,
        "congestion_index": congestion,
        "zone_type":       zone_type,
        "extra_bytes":     extra4.hex(),
    }


def dump_map(blob: bytes, loc: MapLocation, out_path: Path,
             town_index: int,
             firms: Optional[Dict[int, dict]] = None,
             groups: Optional[Dict[int, dict]] = None,
             sites: Optional[Dict[int, dict]] = None) -> dict:
    """Serialize one map to JSON. Returns a brief summary dict.

    If ``firms`` and ``groups`` are supplied, they're embedded in the JSON
    so the viewer can resolve firm cells (feature_recno -> firm details ->
    owning Group) without extra files.
    """
    cells: List[dict] = []
    for y in range(loc.height):
        for x in range(loc.width):
            i = (y * loc.width + x) * CELL_BYTES
            c = decode_cell(blob, loc.cell_array_offset + i)
            c["x"] = x
            c["y"] = y
            cells.append(c)

    payload = {
        "town_index_in_save": town_index,
        "width": loc.width,
        "height": loc.height,
        "cell_count": loc.width * loc.height,
        "stream_index": loc.stream_index,
        "header_offset_hex": f"0x{loc.header_offset:08x}",
        "cell_array_offset_hex": f"0x{loc.cell_array_offset:08x}",
        "cells": cells,
    }
    if firms is not None:
        payload["firms"] = firms
    if groups is not None:
        payload["groups"] = groups
    if sites is not None:
        payload["sites"] = sites

    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return {
        "town_index": town_index,
        "width": loc.width,
        "height": loc.height,
        "stream_index": loc.stream_index,
        "header_offset_hex": payload["header_offset_hex"],
        "cell_array_offset_hex": payload["cell_array_offset_hex"],
        "json_path": str(out_path),
        "json_bytes": out_path.stat().st_size,
    }


# --- Firm / Group resolution -------------------------------------------------

# Offsets within Firm.raw_base (the 0x718-byte firm base block). Source:
# caplab-mcp/caplab_mcp/constants.py FIRM_* offsets (verified against live).
_FIRM_OFF_TYPE        = 0x08   # u16
_FIRM_OFF_NATION      = 0x10   # i32
_FIRM_OFF_LOC_X1      = 0x14   # i16
_FIRM_OFF_LOC_Y1      = 0x16   # i16
_FIRM_OFF_LOC_X2      = 0x18   # i16
_FIRM_OFF_LOC_Y2      = 0x1A   # i16
_FIRM_OFF_TECH_LEVEL  = 0x64   # i32
_FIRM_OFF_CORP_RECNO  = 0xF8   # i32 — owning Group recno
_GROUP_OFF_NAME       = 0x08   # 20-byte char[]

# Per-firm_subtype display name. firm_subtype is an index (1..142) into
# FirmRes (the build-resource table). For ALL firm types — not just public
# buildings — this is the **specific** name (e.g. "Airport", "Hospital",
# "City Hall", "Department Store"). Coarse FIRM_TYPE_NAMES below is just
# the C++ class category and is too coarse to display.
#
# Extracted from CapMain.exe live memory: FirmRes array @ 0x02c028ac,
# stride 0x1c8 bytes, name string at +0x0F. Confirmed by FUN_004a6480
# (FirmRes::firm_build) which reads the same field.
FIRM_BUILD_NAMES: Dict[int, str] = {
      1: "Headquarters",            2: "Headquarters",
      3: "General Store",           4: "Department Store",
      5: "Supermarket",             6: "Convenience Store",
      7: "Discount Megastore",      8: "Apparel Store",
      9: "Automobile Outlet",      10: "Computer Store",
     11: "Cosmetic Store",         12: "Drug Store",
     13: "Electronics Store",      14: "Footwear Store",
     15: "Furniture Store",        16: "Jewelry and Watch Shop",
     17: "Leather Store",          18: "Sports Store",
     19: "Toy Store",              20: "Factory",
     21: "Factory",                22: "Factory",
     23: "Factory",                24: "Factory",
     25: "Factory",                26: "R&D Center",
     27: "R&D Center",             28: "Farm",
     29: "Farm",                   30: "Warehouse",
     31: "Warehouse",              32: "Logging Camp",
     33: "Mine",                   34: "Oil Well",
     35: "Software Company",       36: "Internet Company",
     37: "Telecom",                38: "Bank Headquarters",
     39: "Bank Branch",            40: "Insurance Headquarters",
     41: "Insurance Front Office", 42: "Startup Accelerator",
     43: "Shopping Mall",          44: "Shopping Mall",
     45: "Shopping Mall",          46: "Shopping Mall",
     47: "Art Gallery",            48: "Art Museum",
     49: "Auction House",          50: "Import Company",
     51: "Export Company",         52: "Logistics Company",
     53: "Mansion",                54: "Mansion",
     55: "Mansion",                56: "Mansion",
     57: "Mansion",                58: "Mansion",
     59: "Mansion",                60: "Mansion",
     61: "Apartment",              62: "Apartment",
     63: "Apartment",              64: "Apartment",
     65: "Apartment",              66: "Apartment",
     67: "Commercial Building",    68: "Commercial Building",
     69: "Commercial Building",    70: "Commercial Building",
     71: "Commercial Building",    72: "Commercial Building",
     73: "TV Station",             74: "Newspaper Publisher",
     75: "Radio Station",          76: "Airport",
     77: "Investment Bank",        78: "Stock Exchange",
     79: "Bank",                   80: "Information Center",
     81: "Seaport",                82: "Seaport",
     83: "Seaport",                84: "Seaport",
     85: "Farm (Supplier)",        86: "City Hall",
     87: "Police Station",         88: "Fire Station",
     89: "Grade School",           90: "Middle School",
     91: "High School",            92: "University",
     93: "Library",                94: "Museum",
     95: "Hospital",               96: "Tennis Court",
     97: "Soccer Field",           98: "Golf Course",
     99: "Stadium",               100: "Statue of Liberty",
    101: "Leaning Tower of Pisa", 102: "Colosseum",
    103: "Sydney Opera House",    104: "Arc De Triomphe",
    105: "Pyramid of Giza",       106: "Temple of Heaven",
    107: "Saint Basil's Cathedral", 108: "Himeji Castle",
    109: "Taj Mahal",             110: "Totem",
    111: "House",                 112: "House",
    113: "House",                 114: "House",
    115: "House",                 116: "House",
    117: "House",                 118: "House",
    119: "Park",                  120: "Park",
    121: "Park",                  122: "Park",
    123: "Park",                  124: "Park",
    125: "Park",                  126: "Park",
    127: "Park",                  128: "Park",
    129: "Park",                  130: "Park",
    131: "Park",                  132: "Park",
    133: "Park",                  134: "Park",
    135: "Park",                  136: "Park",
    137: "Park",                  138: "Car Park - small",
    139: "Car Park - large",      140: "Cinema",
    141: "Amusement Park",        142: "Farm Extension",
}


# Coarse firm_type → C++ class category. From FirmArray::create_firm
# (CapMain.exe @ 0x004c4ab0). Useful as a fallback / category tag, but
# NOT specific enough to display — firm_type 0x28 / 0x29 / 0x20-0x24
# all umbrella over multiple distinct buildings (City Hall, Police, Fire,
# School, Hospital, Library, etc.) which are only distinguished via
# firm_subtype + FIRM_BUILD_NAMES above.
FIRM_TYPE_NAMES: Dict[int, str] = {
    0x01: "Headquarters",         0x02: "Retail",
    0x03: "Advertising Service",  0x04: "Factory",
    0x05: "R&D Center",           0x06: "Farm",
    0x07: "Warehouse",            0x08: "Forest",
    0x09: "Mine",                 0x0A: "Oil Field",
    0x0B: "Software",             0x0C: "Web",
    0x0D: "Telecom",              0x0E: "Bank HQ",
    0x0F: "Bank Branch",          0x10: "Insurance HQ",
    0x11: "Insurance Branch",     0x13: "Mall",
    0x14: "Art Gallery",          0x15: "Art Museum",
    0x16: "Auction",              0x17: "Import",
    0x18: "Export",               0x19: "Logistics",
    0x1A: "Mansion",              0x1B: "Apartment",
    0x1C: "Commercial",           0x1D: "TV Station",
    0x1E: "Publisher",            0x1F: "Radio Station",
    0x20: "Public Building (Gov)",
    0x21: "Public Building (variant 0x21)",
    0x22: "Public Building (variant 0x22)",
    0x23: "Public Building (variant 0x23)",
    0x24: "Public Building (variant 0x24)",
    0x25: "Seaport",              0x26: "Supplier",
    0x27: "Golf Course",          0x28: "School/University",
    0x29: "Hospital",             0x2A: "Landmark",
}


# Raw-material site item_type → display name. Verified Apr 2026 against
# CapMain.exe live SiteArray @ 0x0091c8b8: type byte at +0x03 is 0 for
# Forest/Timber sites (always item_type=10) and 1 for Mine deposits
# (item_types 1..9). Alphabetical order matches the 10 raw-material
# ItemRes entries (recnos 93..102):
#   ALUMINIU, CHEMICAL, COAL, GOLD, IRON, LITHIUM, OIL, SILICA, SILVER, TIMBER.
RAW_MATERIAL_NAMES: Dict[int, str] = {
    1:  "Aluminum",
    2:  "Chemical Minerals",
    3:  "Coal",
    4:  "Gold",
    5:  "Iron Ore",
    6:  "Lithium",
    7:  "Oil",
    8:  "Silica",
    9:  "Silver",
    10: "Timber",
}


def parse_sites(streams: List[bytes]) -> Dict[int, dict]:
    """Find SiteArray in the save and decode each site entry.

    The save embeds the SiteArray header (5 i32 fields) immediately before
    the entry block. We scan for the canonical signature:
      ``[i32 entries_per_block=100][i32 cursor][i32 total_count]``
      ``[i32 element_stride=64][i32 block_count=1]``
    then validate by checking that the first site has alive byte == 1
    and a small item_type (1..16). Returns ``{recno: site_dict}``.
    """
    SITE_SIZE = 64
    SIG_PREFIX = struct.pack("<i", 100)            # entries_per_block
    SIG_SUFFIX = struct.pack("<ii", 64, 1)         # element_stride, block_count

    found: Dict[int, dict] = {}
    for blob in streams:
        n = len(blob)
        i = 0
        while True:
            i = blob.find(SIG_PREFIX, i)
            if i < 0 or i + 20 > n:
                break
            # Header at i: [100][cursor][total_count][64][1]
            if blob[i + 12 : i + 20] == SIG_SUFFIX:
                cursor = struct.unpack_from("<i", blob, i + 4)[0]
                total  = struct.unpack_from("<i", blob, i + 8)[0]
                if 1 <= total <= 1000 and total <= cursor + 1000:
                    block_off = i + 20
                    if block_off + total * SITE_SIZE <= n:
                        # Validate first entry
                        first_alive = blob[block_off + 2]
                        first_item  = struct.unpack_from(
                            "<H", blob, block_off + 4)[0]
                        if first_alive == 1 and 1 <= first_item <= 16:
                            for k in range(total):
                                off = block_off + k * SITE_SIZE
                                rec = struct.unpack_from("<H", blob, off + 0)[0]
                                if rec == 0:
                                    continue
                                alive_byte = blob[off + 2]
                                type_byte  = blob[off + 3]
                                item_type  = struct.unpack_from(
                                    "<H", blob, off + 4)[0]
                                town_recno = struct.unpack_from(
                                    "<H", blob, off + 0x0c)[0]
                                x1 = struct.unpack_from("<H", blob, off + 0x0e)[0]
                                y1 = struct.unpack_from("<H", blob, off + 0x10)[0]
                                x2 = struct.unpack_from("<H", blob, off + 0x12)[0]
                                y2 = struct.unpack_from("<H", blob, off + 0x14)[0]
                                supply_rate  = struct.unpack_from(
                                    "<d", blob, off + 0x18)[0]
                                capacity     = struct.unpack_from(
                                    "<d", blob, off + 0x20)[0]
                                utilization  = struct.unpack_from(
                                    "<d", blob, off + 0x28)[0]
                                site_class = ("forest" if type_byte == 0
                                              else "mine"  if type_byte == 1
                                              else f"type_{type_byte}")
                                found[rec] = {
                                    "recno":      rec,
                                    "alive":      bool(alive_byte),
                                    "site_class": site_class,
                                    "item_type":  item_type,
                                    "item_name":  RAW_MATERIAL_NAMES.get(
                                        item_type, f"raw_{item_type}"),
                                    "town_recno": town_recno,
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "supply_rate":  supply_rate,
                                    "capacity":     capacity,
                                    "utilization":  utilization,
                                }
                            return found  # take the first valid match
            i += 1
    return found


def parse_firms_groups(sav_path: Path
                       ) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Parse FirmArray + GroupArray from the save and return JSON-friendly
    dicts keyed by recno. Returns ({} {}) on import / parse failure so map
    dumping never depends on this side data."""
    try:
        # Lazy import to keep map_dump usable without the full save parser.
        from caplab_save.parser import CapLabSave
        save = CapLabSave.load(str(sav_path))
    except Exception as e:
        print(f"  warning: firm/group parse skipped ({e})", file=sys.stderr)
        return {}, {}

    groups: Dict[int, dict] = {}
    for g in save.groups:
        if g.recno == 1:    label = "government"
        elif g.recno == 2:  label = "player"
        else:               label = "ai"
        name = ""
        try:
            buf = g.raw_base[_GROUP_OFF_NAME:_GROUP_OFF_NAME + 20]
            nul = buf.find(b"\x00")
            name = buf[: nul if nul >= 0 else len(buf)].decode("latin-1",
                                                                errors="ignore")
        except Exception:
            pass
        groups[g.recno] = {
            "recno": g.recno,
            "name":  name or f"group_{g.recno}",
            "label": label,
        }

    firms: Dict[int, dict] = {}
    for f in save.firms:
        rb = f.raw_base
        try:
            ft     = struct.unpack_from("<H", rb, _FIRM_OFF_TYPE)[0]
            nation = struct.unpack_from("<i", rb, _FIRM_OFF_NATION)[0]
            x1     = struct.unpack_from("<h", rb, _FIRM_OFF_LOC_X1)[0]
            y1     = struct.unpack_from("<h", rb, _FIRM_OFF_LOC_Y1)[0]
            x2     = struct.unpack_from("<h", rb, _FIRM_OFF_LOC_X2)[0]
            y2     = struct.unpack_from("<h", rb, _FIRM_OFF_LOC_Y2)[0]
            tech   = struct.unpack_from("<i", rb, _FIRM_OFF_TECH_LEVEL)[0]
            corp   = struct.unpack_from("<i", rb, _FIRM_OFF_CORP_RECNO)[0]
        except Exception:
            continue
        owner = groups.get(corp)
        # Specific name from FirmRes (City Hall, Police Station, etc.).
        # Falls back to the coarse class name if subtype unknown.
        build_name = FIRM_BUILD_NAMES.get(f.firm_subtype)
        coarse = FIRM_TYPE_NAMES.get(ft, f"type_{ft}")
        firms[f.recno] = {
            "recno":              f.recno,
            "firm_type":          ft,
            "firm_type_name":     build_name or coarse,
            "firm_class":         coarse,           # coarse C++ class
            "firm_build_name":    build_name or coarse,
            "firm_subtype":       f.firm_subtype,
            "owner_group_recno":  corp,
            "owner_group_name":   owner["name"]  if owner else f"group_{corp}",
            "owner_group_label":  owner["label"] if owner else "?",
            "nation_recno":       nation,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "tech_level":         tech,
        }
    return firms, groups


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        print("\nUsage: python -m caplab_save.map_dump <SAV path> <out dir>", file=sys.stderr)
        return 2

    sav_path = Path(argv[1])
    out_dir = Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Decompressing {sav_path.name}...", file=sys.stderr)
    res = decompress_save(sav_path)
    streams = [s.data for s in res.streams]
    for k, s in enumerate(streams):
        print(f"  stream[{k}]: {len(s):,} decompressed bytes", file=sys.stderr)

    print("Locating town maps...", file=sys.stderr)
    locs = find_map_locations(streams)
    print(f"  {len(locs)} town map(s) found", file=sys.stderr)

    print("Parsing FirmArray + GroupArray...", file=sys.stderr)
    firms, groups = parse_firms_groups(sav_path)
    print(f"  {len(firms)} firms, {len(groups)} groups", file=sys.stderr)

    print("Parsing SiteArray (raw-material deposits)...", file=sys.stderr)
    sites = parse_sites(streams)
    print(f"  {len(sites)} sites", file=sys.stderr)

    summary = []
    for idx, loc in enumerate(locs, start=1):
        out_file = out_dir / f"town_{idx:02d}_{loc.width}x{loc.height}.json"
        print(f"  dumping town {idx} ({loc.width}x{loc.height}) -> {out_file.name}",
              file=sys.stderr)
        s = dump_map(streams[loc.stream_index], loc, out_file, idx,
                     firms=firms, groups=groups, sites=sites)
        summary.append(s)

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps({
        "save_file": str(sav_path),
        "save_bytes": sav_path.stat().st_size,
        "stream_sizes": [len(s) for s in streams],
        "maps": summary,
    }, indent=2))
    print(f"\nIndex written to {index_path}", file=sys.stderr)
    print(f"Total: {len(locs)} maps, "
          f"{sum(s['width'] * s['height'] for s in summary):,} cells",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
