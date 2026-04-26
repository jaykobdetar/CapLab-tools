"""Simulation state — the slice of a parsed ``CapLabSave`` that the
replay engine actually needs.

``caplab_save.CapLabSave`` carries every parseable field of the save
format. ``SimState`` is a thinner projection: just the entities and
counts that drive the daily tick's RNG budget. We keep it simple — a
dataclass of counts and a few typed arrays — because Phase 1 of the
simulation engine only needs entity *cardinalities*, not full
balance sheets. Later phases will add richer per-entity state as
those subsystems come online.

This module also carries the replacement ``read_seed_from_blob0`` that
works around the hardcoded-offset bug in ``caplab_save.parser.parse_rng_seed``
(see SESSION_LOG.txt for the full story — the upstream parser assumes
tag 0x1067 sits at blob-0 offset 0x11596, which is only true for two
of our nine reference saves).
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Bootstrap so ``caplab_save`` is importable whether this file is run as
# a script or imported from the package. Mirrors the pattern used in
# caplab_save/parser.py so developers get the same ergonomics.
if __package__ in (None, ""):  # pragma: no cover - only triggers when run directly
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)

from caplab_save import CapLabSave  # noqa: E402
from caplab_save.decompress import decompress_save  # noqa: E402

from caplab_sim.constants import TAG_GAMESTATE


# ---- seed extraction (bug-workaround for caplab_save hardcoded offset) -----


def read_seed_from_blob0(blob0: bytes) -> Tuple[int, int]:
    """Locate tag 0x1067 dynamically and return ``(seed, tag_offset)``.

    The GAMESTATE record (tag 0x1067) carries the RNG seed as the first
    uint32 of its payload. Its byte position in blob 0 depends on the
    size of the GameInfo payload that precedes it, which is not constant
    across saves — adding player-owned firms, for instance, shifts every
    tag downstream of GameInfo. The ``RNG_SEED_OFFSET_BLOB0`` constant
    in ``caplab_save/constants.py`` was pinned to a single observed
    layout and is wrong for the fixture saves captured this session.

    See ``caplab_sim/SESSION_LOG.txt`` for the analysis — this function
    is the correct way to read the seed until the upstream parser is
    patched.
    """
    pat = struct.pack("<I", TAG_GAMESTATE)
    idx = blob0.find(pat)
    if idx == -1:
        raise ValueError(
            f"tag 0x{TAG_GAMESTATE:04x} (GAMESTATE) not found in blob 0; "
            "save format may differ from v11.1.2 expectations"
        )
    payload = idx + 4
    if payload + 4 > len(blob0):
        raise ValueError(
            f"tag 0x{TAG_GAMESTATE:04x} found at 0x{idx:x} but payload "
            "runs past end of blob"
        )
    seed = struct.unpack_from("<I", blob0, payload)[0]
    return seed, idx


def read_seed_from_save(path: str) -> int:
    """Decompress ``path`` and read the correctly-located RNG seed.

    Convenience wrapper — equivalent to ``decompress_save(path).blob0``
    then :func:`read_seed_from_blob0`.
    """
    r = decompress_save(path)
    seed, _ = read_seed_from_blob0(r.blob0)
    return seed


# ---- sim state container ---------------------------------------------------


@dataclass
class SimState:
    """Minimal state needed to estimate or replay one daily tick.

    This is intentionally sparse. Full balance sheets, monthly arrays,
    and per-firm production are either computed on the fly or loaded
    from the underlying ``CapLabSave`` attached via ``save``. The
    ``SimState`` exists so the RNG replay code doesn't have to reach
    through the save object every time it wants an entity count.

    Attributes
    ----------
    seed : int
        The game's Misc RNG seed at load time (read via
        :func:`read_seed_from_blob0`, not the potentially-stale
        ``save.rng_seed``).
    seed_offset : int
        Byte offset of the tag-0x1067 header in blob 0. Useful for
        sanity-checking which layout variant the save uses.
    game_date : int
        Days since year 0 (same convention as ``GameInfo.game_date``).
    day_of_month : int
    month_of_year : int
    year : int
    day_of_month_index : int
        Zero-based day index (``day_of_month - 1``) — convenient for
        slice-modulo checks that the replay code does.
    n_groups : int
        Total groups, including the synthetic Government / player / AI.
    n_ai_groups : int
        Subset of ``n_groups`` that trigger ``GroupAI__daily_update``.
        Derived from ``group_type`` (see ``Structs_Group.txt`` — type 3
        is AI, type 2 is player, type 4 is government).
    n_firms : int
        Total active firms.
    n_firms_by_type : dict[int,int]
        Type histogram (key = ``firm_type``, value = count).
    n_listed_stocks : int
        Count of domestic, listed stocks that call
        ``NationStock__daily_price_update`` each tick. In Phase 1 we
        approximate as ``n_ai_groups + 1`` (player + AI each own one
        listed national stock) until we parse the NationStock array.
    n_towns : int
        Town count — needed for ``S_towns = n_towns * 1`` and for the
        raw-resource discovery slice.
    n_persons : int
        Total persons in the PersonArray. Not currently serialized in a
        tag we can locate reliably (see
        ``RESEARCH_NOTES.md §18``), so this is a runtime-inferred
        estimate defaulting to 1000 for the 22-group reference save.
    n_parties : int
        Political parties (``PartyArray``). Defaulted to 3 for the
        reference save until we parse the ``PartyArray`` tag.
    n_talents : int
        Talents (``TalentArray``). Defaulted to 60 until parsed.
    site_spawn_enabled : bool
        Mirrors the engine's ``DAT_00a458d8 != 0`` flag — the single
        biggest lever on the daily RNG budget. Parsed from INI-level
        feature flags when we locate them; defaults ``False`` in both
        ``SimState`` and ``SimConstants`` because empirical validation
        against the _____001 → _____002 pair requires it off. Override
        per-save when the DLC toggle is known to be on.
    save : CapLabSave | None
        Reference back to the full parse, for ad-hoc inspection.
    """

    seed: int
    seed_offset: int
    game_date: int
    day_of_month: int
    month_of_year: int
    year: int

    n_groups: int
    n_ai_groups: int
    n_firms: int
    n_firms_by_type: dict
    n_listed_stocks: int
    n_towns: int
    n_persons: int
    n_parties: int
    n_talents: int

    site_spawn_enabled: bool = False      # see attribute docstring above
    logistics_enabled: bool = True
    rawres_dlc_active: bool = False

    save: Optional[CapLabSave] = field(default=None, repr=False)

    # -- derived helpers ---------------------------------------------------

    @property
    def day_of_month_index(self) -> int:
        return self.day_of_month - 1

    @property
    def is_month_end(self) -> bool:
        """True if today's tick will fire the ``EconomyUpdate__month_end`` chain.

        Heuristic for the 28/30/31-day Gregorian calendar used by the game.
        Will be replaced with an exact check once ``current_month_index``
        handling is nailed down — current impl is good enough for Phase 1
        scope.
        """
        # Gregorian month lengths for a non-leap year.
        month_lens = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if not (1 <= self.month_of_year <= 12):
            return False
        return self.day_of_month == month_lens[self.month_of_year - 1]

    @property
    def is_firm_slice_day(self) -> bool:
        """True if firm_recno % 30 slice fires for firms with recno % 30 == today."""
        return True  # every day has some slice match; kept for future clarity

    @property
    def firm_slice_index(self) -> int:
        return self.game_date % 30


# ---- loaders ---------------------------------------------------------------


# Group type codes used by the game (see Structs_Group.txt and
# RESEARCH_NOTES §18 "groups"). Only group_type == 3 is AI-controlled.
GROUP_TYPE_DELETED = 0
GROUP_TYPE_CLEARED = 1       # slot wiped; not in use
GROUP_TYPE_PLAYER = 2
GROUP_TYPE_AI = 3
GROUP_TYPE_GOVERNMENT = 4


def load_sim_state(path: str) -> SimState:
    """Parse a save and project it into a :class:`SimState`.

    Uses :func:`read_seed_from_blob0` for the seed to work around the
    upstream parser bug described in the module docstring. All other
    fields come from the regular ``CapLabSave`` object.
    """
    save = CapLabSave.load(path)
    # Re-read seed at the correct offset (bypass the hardcoded one).
    r = decompress_save(path)
    seed, seed_offset = read_seed_from_blob0(r.blob0)

    # AI groups are group_type == 3.
    n_ai_groups = sum(1 for g in save.groups if g.group_type == GROUP_TYPE_AI)

    firm_types: dict = {}
    for f in save.firms:
        firm_types[f.firm_type] = firm_types.get(f.firm_type, 0) + 1

    # Phase-1 approximation: one listed stock per AI group (+ the player's).
    # The actual NationStock array is nested inside each Group record and
    # isn't split out into its own tag — decoding it is a post-Phase-2
    # task. For the 22-group reference save this gives
    # n_listed_stocks ≈ 22, which matches the RNG_Call_Map budget.
    n_listed_stocks = n_ai_groups + (1 if any(g.group_type == GROUP_TYPE_PLAYER for g in save.groups) else 0)

    # Town / person / party / talent counts are not yet parsed — default to
    # the reference scenario's empirical ceilings so the formula works
    # on PLAY_*/fixture saves. See SESSION_LOG.txt blockers section.
    n_towns = 4           # reference save
    n_persons = 1000      # runtime estimate (DAT_0091bf28+0xc at game tick)
    n_parties = 3         # reference save (3 political parties)
    n_talents = 60        # reference save

    return SimState(
        seed=seed,
        seed_offset=seed_offset,
        game_date=save.game_info.game_date,
        day_of_month=save.game_info.day_of_month,
        month_of_year=save.game_info.month_of_year,
        year=save.game_info.year,
        n_groups=len(save.groups),
        n_ai_groups=n_ai_groups,
        n_firms=len(save.firms),
        n_firms_by_type=firm_types,
        n_listed_stocks=n_listed_stocks,
        n_towns=n_towns,
        n_persons=n_persons,
        n_parties=n_parties,
        n_talents=n_talents,
        save=save,
    )
