"""Smoke tests for ``caplab_save``.

These tests load the real PLAY_001..PLAY_004 save files captured during the
2026-04-21 RE session and verify that the parser extracts the fields we have
verified by hand against the engine disassembly.  The expected values below
are **empirical** — established by the initial parser runs and cross-checked
against ``/proc/{pid}/mem`` reads at save time — and are the regression
contract for the parser.

In addition to the PLAY_* bulk-scenario fixtures, a second set of saves
lives in ``saves/`` (``Headquarters.SAV``, ``Department.SAV``,
``Factory.SAV``, ``R&D.SAV``, ``Warehouse.SAV``).  These were captured by
the user in a single fresh game by buying one basic-type firm per save, in
the order HQ → Department store → Factory → R&D → Warehouse; they
cumulatively exercise the five basic-subclass decoders
(0x01/0x02/0x04/0x05/0x07) that no PLAY_* fixture touches.  See
``TestBasicFirmFixtures`` below.

Any of the following invocations work:

    # from repo root
    python -m unittest caplab_save.tests.test_load -v
    pytest caplab_save/tests/test_load.py -v

    # direct script execution (from anywhere, including inside the package)
    python caplab_save/tests/test_load.py
    cd caplab_save/tests && python test_load.py

The save files are expected to live at the repo root (i.e. at
``/home/jaykob/caplab-research/PLAY_*.SAV``) and in the ``saves/``
subdirectory.  If they are missing, the tests skip cleanly rather than
failing.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# ---- path bootstrap -------------------------------------------------------
# Make `from caplab_save import ...` work even when this file is run directly
# as a script (e.g. `python tests/test_load.py`), in which case Python only
# puts the enclosing dir on sys.path.  The repo root is two levels above
# this file: .../caplab-research/caplab_save/tests/test_load.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caplab_save import CapLabSave  # noqa: E402  (import after sys.path tweak)


REPO_ROOT = _REPO_ROOT

SAVE_PLAY_001 = os.path.join(REPO_ROOT, "PLAY_001_20260421_000834_000.SAV")
SAVE_PLAY_002 = os.path.join(REPO_ROOT, "PLAY_002_20260421_000854_000.SAV")
SAVE_PLAY_003 = os.path.join(REPO_ROOT, "PLAY_003_20260421_000930_000.SAV")
SAVE_PLAY_004 = os.path.join(REPO_ROOT, "PLAY_004_20260421_000950_000.SAV")

# Expected values. The fixture saves were all captured within ~75 s of each
# other during a paused game, so calendar / economy fields are identical
# across the set.  PLAY_001 was saved slightly earlier in the setup flow and
# therefore has fewer AI groups than PLAY_002..PLAY_004.

EXPECTED = {
    SAVE_PLAY_001: {
        "rng_seed": 0x0AE83F5E,
        "group_count": 17,
        "player_cash": 200_000_000.0,
        "player_net_worth": 200_000_000.0,
        "firm_count": 176,
    },
    SAVE_PLAY_002: {
        "rng_seed": 0xD6296661,
        "group_count": 22,
        "player_cash": 200_000_000.0,
        "player_net_worth": 200_000_000.0,
        "firm_count": 176,
    },
    SAVE_PLAY_003: {
        "rng_seed": 0xD6296661,
        "group_count": 22,
        "player_cash": 200_000_000.0,
        "player_net_worth": 200_000_000.0,
        "firm_count": 176,
    },
    SAVE_PLAY_004: {
        "rng_seed": 0xD6296661,
        "group_count": 22,
        "player_cash": 200_000_000.0,
        "player_net_worth": 200_000_000.0,
        "firm_count": 176,
    },
}

# Observed firm-type histograms (consistent across the 4 reference saves —
# all 4 were captured during the same paused fresh-start scenario).
EXPECTED_FIRM_TYPE_HIST = {
    0x1B: 58, 0x1C:  6,
    0x1D:  4, 0x1E:  4, 0x1F:  4,
    0x20:  4, 0x21:  4, 0x22:  4, 0x23:  4, 0x24:  4,
    0x25: 16,
    0x28: 43, 0x29: 21,
}

EXPECTED_CALENDAR = {
    # PLAY_002..PLAY_004 are one day after PLAY_001.  Values below
    # correspond to Jan 2, 1990 in the engine's proleptic-Gregorian
    # calendar.  ``game_date`` is the Julian Day Number (JDN) stored at
    # ScenarioCfg+0x18; JDN 2,447,894 = 1990-01-02.  See SESSION_LOG
    # entry for 2026-04-21 task #8 — the old heuristic reported 708_407
    # (a stale stamp in the GameInfo payload), which has been retired.
    "game_date": 2_447_894,
    "year": 1990,
    "month_of_year": 1,
    "day_of_month": 2,
    "day_of_year": 2,
    "days_since_start": 1,
    "start_day_counter": 2_447_893,
    "start_year": 1990,
}

# PLAY_001 is the "day 0" save (scenario start day, not yet advanced).
EXPECTED_CALENDAR_PLAY_001 = {
    "game_date": 2_447_893,
    "year": 1990,
    "month_of_year": 1,
    "day_of_month": 1,
    "day_of_year": 1,
    "days_since_start": 0,
    "start_day_counter": 2_447_893,
    "start_year": 1990,
}

EXPECTED_ECONOMY = {
    "base_interest_rate": 6.75,
    "gdp_growth_rate": 2.0,
    "cycle_phase": 1,
    "interest_rate_level": 5,
    "price_level_CPI": 1.0,
    "price_level_PPP": 1.0,
    "annual_inflation_rate": 3.0,
    "stock_index_target": 1.0,
}


# ---- basic-firm fixtures (saves/*.SAV) -----------------------------------
#
# Five cumulative saves captured by the user to close the basic-subclass
# validation gap.  Each save is the previous game state plus one
# newly-bought player firm, in order HQ → Department store → Factory →
# R&D → Warehouse.  All five saves share the same calendar and scenario
# seed; only the player firm roster grows.
#
# The regression contract for these saves is:
#
#   1. The player group (type=1, recno=2) holds exactly N firms, where N
#      is the cumulative index of that save (1, 2, 3, 4, 5).
#   2. The Nth firm in that list has the exact (firm_type, firm_subtype,
#      raw_subclass) tuple listed in ``BASIC_FIRM_STEPS`` below.
#   3. Every player firm decodes to the expected typed-subclass dataclass.
#
# The raw_subclass bytes below are the ground truth extracted from the
# first successful load of each save — they anchor the decoders to real
# game output instead of synthetic payloads.

SAVES_DIR = os.path.join(REPO_ROOT, "saves")

SAVE_HEADQUARTERS = os.path.join(SAVES_DIR, "Headquarters.SAV")
SAVE_DEPARTMENT   = os.path.join(SAVES_DIR, "Department.SAV")
SAVE_FACTORY      = os.path.join(SAVES_DIR, "Factory.SAV")
SAVE_RD           = os.path.join(SAVES_DIR, "R&D.SAV")
SAVE_WAREHOUSE    = os.path.join(SAVES_DIR, "Warehouse.SAV")

#: Ordered list of (save_path, cumulative_firm_count, step_metadata) where
#: ``step_metadata`` describes the *newly added* firm in that save.
BASIC_FIRM_STEPS = [
    (SAVE_HEADQUARTERS, 1, {
        "firm_type": 0x01,
        "firm_subtype": 0x01,
        "raw_subclass": bytes.fromhex("0100000000000000"),
        "typed_cls_name": "FirmHqSub",
        "label": "HQ",
    }),
    (SAVE_DEPARTMENT, 2, {
        "firm_type": 0x02,
        "firm_subtype": 0x04,
        "raw_subclass": bytes.fromhex("0000000000000000"),
        "typed_cls_name": "FirmRetailSub",
        "label": "Department store",
    }),
    (SAVE_FACTORY, 3, {
        "firm_type": 0x04,
        "firm_subtype": 0x14,
        "raw_subclass": bytes.fromhex("00ffffff00000000"),
        "typed_cls_name": "FirmFactorySub",
        "label": "Factory",
    }),
    (SAVE_RD, 4, {
        "firm_type": 0x05,
        "firm_subtype": 0x1B,
        "raw_subclass": bytes.fromhex("0000000000000000"),
        "typed_cls_name": "FirmRDSub",
        "label": "R&D center",
    }),
    (SAVE_WAREHOUSE, 5, {
        "firm_type": 0x07,
        "firm_subtype": 0x1F,
        "raw_subclass": bytes.fromhex("00ffffff00000000"),
        "typed_cls_name": "FirmWarehouseSub",
        "label": "Warehouse",
    }),
]

#: All five basic-firm saves share the same calendar & background firm mix.
#: They're captured on the scenario's very first day (Jan 1, 1990) before
#: the game has advanced any ticks — so game_date == start_day_counter and
#: days_since_start == 0.
BASIC_FIRM_CALENDAR = {
    "game_date": 2_447_893,
    "year": 1990,
    "month_of_year": 1,
    "day_of_month": 1,
    "day_of_year": 1,
    "days_since_start": 0,
    "start_day_counter": 2_447_893,
    "start_year": 1990,
}

#: Background (Government-owned) firm histogram common to all five saves.
#: The only cell that grows is the basic-firm type being added.
BASIC_FIRM_BACKGROUND_HIST = {
    0x1B: 38, 0x1C: 4,
    0x1D: 4, 0x1E: 4, 0x1F: 4,
    0x20: 4, 0x21: 4, 0x22: 4, 0x23: 4, 0x24: 4,
    0x25: 16,
    0x28: 56, 0x29: 26,
}


def _require_save(path: str) -> None:
    if not os.path.exists(path):
        raise unittest.SkipTest(f"fixture save not present: {path}")


# ---- the tests ------------------------------------------------------------


class TestDecompressAndHeader(unittest.TestCase):
    """Verify the .SAV decompressor produces the expected shape."""

    def test_play001_has_two_blobs(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        self.assertGreaterEqual(len(save.decompressed.streams), 2)
        self.assertGreater(len(save.decompressed.blob0), 1_000_000)
        self.assertGreater(len(save.decompressed.blob1), 500_000)

    def test_play001_header_contains_plaintext(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        # Header is the plaintext prefix before the first zlib stream.
        self.assertGreater(len(save.decompressed.header), 0)
        self.assertLess(len(save.decompressed.header), 1_000_000)


class TestGameInfo(unittest.TestCase):
    """Verify GameInfo / calendar extraction is consistent across saves."""

    def test_calendar_matches_across_saves(self) -> None:
        # PLAY_001 was taken on scenario-start day (Jan 1 1990); PLAY_002..004
        # all landed on the day after (Jan 2 1990) — the seed divergence
        # between PLAY_001 (0x0AE83F5E) and PLAY_002..004 (0xD6296661)
        # reflects that one-day advance.
        cal_by_save = {
            SAVE_PLAY_001: EXPECTED_CALENDAR_PLAY_001,
            SAVE_PLAY_002: EXPECTED_CALENDAR,
            SAVE_PLAY_003: EXPECTED_CALENDAR,
            SAVE_PLAY_004: EXPECTED_CALENDAR,
        }
        for path, expected in cal_by_save.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                gi = save.game_info
                self.assertEqual(gi.game_date, expected["game_date"])
                self.assertEqual(gi.year, expected["year"])
                self.assertEqual(gi.month_of_year, expected["month_of_year"])
                self.assertEqual(gi.day_of_month, expected["day_of_month"])
                self.assertEqual(gi.day_of_year, expected["day_of_year"])
                self.assertEqual(gi.days_since_start, expected["days_since_start"])
                self.assertEqual(gi.start_day_counter, expected["start_day_counter"])
                self.assertEqual(gi.start_year, expected["start_year"])
                self.assertEqual(len(gi.raw), 0x71E8)

    def test_calendar_game_date_decodes_to_same_gregorian_triple(self) -> None:
        """Sanity check: jdn_to_gregorian(game_date) == (year, month, day).

        Cross-verifies that the JDN field (ScenarioCfg+0x18) and the
        separate year/month/day fields (ScenarioCfg+0x20/+0x24/+0x28)
        agree with each other.  If this fails, either the layout has
        shifted between engine versions or one of the offsets is being
        read from the wrong field.
        """
        from caplab_save.parser import jdn_to_gregorian
        for path in (SAVE_PLAY_001, SAVE_PLAY_002, SAVE_PLAY_003, SAVE_PLAY_004):
            _require_save(path)
            save = CapLabSave.load(path)
            gi = save.game_info
            y, m, d = jdn_to_gregorian(gi.game_date)
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual((y, m, d),
                                 (gi.year, gi.month_of_year, gi.day_of_month))


class TestRngSeed(unittest.TestCase):
    """RNG seed — located dynamically via tag 0x1067 (GAMESTATE).

    History: this test used to assert ``source_offset == 0x01159A``
    against a hardcoded constant. That was wrong — the tag's position
    in blob 0 depends on earlier record sizes, and only the PLAY_001 /
    PLAY_002 layouts have the tag at payload offset 0x1159A. The
    fixture saves (Headquarters / Department / …) each add a player-
    owned firm, which shifts the tag forward by one firm stride. The
    new ``parse_rng_seed`` locates the tag dynamically and returns the
    tag offset (not the payload offset). See
    ``test_rng_seed_offset.py`` for the regression guard and
    ``caplab_sim/SESSION_LOG.txt`` for the bug write-up.
    """

    # Expected tag offsets for the PLAY_* fixtures. ``C.TAG_SIZE`` more
    # than this is the payload offset where the 4-byte seed sits.
    PLAY_TAG_OFFSET = 0x011596

    def test_seed_per_save(self) -> None:
        for path, exp in EXPECTED.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual(save.rng_seed, exp["rng_seed"])
                # All four PLAY_* saves share the same GameInfo layout
                # (no player-owned firms) — so tag is at the same offset.
                self.assertEqual(save.rng.source_offset, self.PLAY_TAG_OFFSET)
                # Counter is not yet located; should be None for now.
                self.assertIsNone(save.rng.counter)


class TestGroupArray(unittest.TestCase):
    """Group array walker produces the expected structure and values."""

    def test_group_counts(self) -> None:
        for path, exp in EXPECTED.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual(len(save.groups), exp["group_count"])

    def test_first_group_is_government(self) -> None:
        """Record 1 is the synthetic Government group (type=4)."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        gov = save.groups[0]
        self.assertEqual(gov.recno, 1)
        self.assertEqual(gov.group_type, 4)

    def test_player_group_is_type1_with_starting_cash(self) -> None:
        for path, exp in EXPECTED.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                # The player is the first type=1 group in the array.
                player = next((g for g in save.groups if g.group_type == 1), None)
                self.assertIsNotNone(player, "no player (type=1) group in save")
                self.assertAlmostEqual(player.corp_cash, exp["player_cash"], places=2)
                self.assertAlmostEqual(player.net_worth, exp["player_net_worth"], places=2)

    def test_all_groups_have_valid_types(self) -> None:
        """Only types 1/3/4 are valid for GroupArray 0x4A18 records."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        for g in save.groups:
            self.assertIn(g.group_type, (1, 3, 4),
                          f"unexpected group_type={g.group_type} at recno={g.recno}")

    def test_monthly_arrays_are_36_long(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        g = save.groups[1]  # player
        self.assertEqual(len(g.monthly_net_flow), 36)
        self.assertEqual(len(g.monthly_net_profit), 36)
        self.assertEqual(len(g.monthly_business_revenue), 36)
        self.assertEqual(len(g.monthly_expense_total), 36)

    def test_account_balance_arrays(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        g = save.groups[1]  # player
        self.assertEqual(len(g.account_balance), 14)
        self.assertEqual(len(g.account_balance_2), 14)
        # Loan principal / bond interest liability should be finite.
        self.assertTrue(math.isfinite(g.loan_principal))
        self.assertTrue(math.isfinite(g.bond_interest_liability))

    def test_raw_base_length(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        for g in save.groups:
            self.assertEqual(len(g.raw_base), 0x4A18,
                             f"raw_base length mismatch at recno={g.recno}")


class TestEconomy(unittest.TestCase):
    """Economy struct should be located in blob 1 and decoded cleanly."""

    def test_economy_present_and_values_stable(self) -> None:
        for path in EXPECTED:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                e = save.economy
                self.assertIsNotNone(e, "economy struct was not located")
                for field, expected in EXPECTED_ECONOMY.items():
                    actual = getattr(e, field)
                    if isinstance(expected, float):
                        self.assertAlmostEqual(actual, expected, places=6,
                                               msg=f"economy.{field}")
                    else:
                        self.assertEqual(actual, expected, f"economy.{field}")

    def test_target_rate_invariant(self) -> None:
        """Engine invariant: target_rate_pct == interest_rate_level * 10."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        e = save.economy
        self.assertIsNotNone(e)
        # target_rate_pct isn't exposed on the dataclass; re-derive for the
        # invariant check to stay honest.
        import struct
        from caplab_save import constants as C
        blob = save.decompressed.streams[1].data
        # `found_at_blob` stores the base offset inside the host blob.
        base = e.found_at_blob
        target = struct.unpack_from("<d", blob, base + C.ECONOMY_OFF_TARGET_RATE_PCT)[0]
        self.assertAlmostEqual(target, e.interest_rate_level * 10.0, places=6)


class TestSection2TagSequence(unittest.TestCase):
    """The Section 2 tag sequence should appear in order in blob 0."""

    def test_tag_sequence_monotonic(self) -> None:
        from caplab_save import constants as C
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        offsets = [t.offset for t in save.tags_section2]
        tags = [t.tag for t in save.tags_section2]
        self.assertEqual(tuple(tags), C.SECTION_2_TAGS)
        self.assertEqual(offsets, sorted(offsets),
                         f"section 2 tags not monotonic: {offsets}")


class TestFirmArray(unittest.TestCase):
    """Firm array walker + linkage against groups."""

    def test_firm_counts_match_active_count(self) -> None:
        for path, exp in EXPECTED.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual(len(save.firms), exp["firm_count"])

    def test_firm_type_histogram_stable(self) -> None:
        """All 4 fixture saves share the same firm-type mix."""
        from collections import Counter
        for path in EXPECTED:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                hist = dict(Counter(f.firm_type for f in save.firms))
                self.assertEqual(hist, EXPECTED_FIRM_TYPE_HIST)

    def test_every_firm_has_valid_base(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        from caplab_save import constants as C
        for f in save.firms:
            self.assertEqual(len(f.raw_base), C.SIZE_FIRM_BASE,
                             f"firm recno={f.recno} raw_base wrong length")
            # The prefix and base type fields must agree (sentinel check on
            # a different level than what the parser already validated).
            import struct
            base_type = struct.unpack_from("<H", f.raw_base, C.FIRM_OFF_TYPE)[0]
            self.assertEqual(base_type, f.firm_type,
                             f"firm recno={f.recno} base type mismatch")

    def test_firm_monthly_arrays_are_36_long(self) -> None:
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        for f in save.firms:
            self.assertEqual(len(f.monthly_net_profit), 36)
            self.assertEqual(len(f.monthly_revenue), 36)
            self.assertEqual(len(f.monthly_expense), 36)


class TestGroupFirmLinkage(unittest.TestCase):
    """Cross-reference: firm.group_recno ↔ group.firm_recnos."""

    def test_all_firms_point_to_a_valid_group(self) -> None:
        for path in EXPECTED:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                valid_grps = {g.recno for g in save.groups}
                for f in save.firms:
                    self.assertIn(f.group_recno, valid_grps,
                                  f"firm {f.recno} → unknown group {f.group_recno}")

    def test_group_firm_count_matches_linkage(self) -> None:
        """The engine's ``Group.firm_count`` should equal the number of
        firms whose ``Firm+0xF8`` points at that group."""
        for path in EXPECTED:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                for g in save.groups:
                    linked = save.firms_of(g.recno)
                    self.assertEqual(
                        len(linked), g.firm_count,
                        f"group {g.recno} (type={g.group_type}): "
                        f"declared firm_count={g.firm_count} but "
                        f"{len(linked)} firms link back"
                    )

    def test_firm_recnos_populated_on_groups(self) -> None:
        """After linkage, ``Group.firm_recnos`` should match ``firms_of``."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        for g in save.groups:
            linked = tuple(sorted(f.recno for f in save.firms_of(g.recno)))
            self.assertEqual(g.firm_recnos, linked,
                             f"group {g.recno}: firm_recnos mismatch")

    def test_government_owns_all_initial_firms(self) -> None:
        """In the fresh-start PLAY_* saves the synthetic Government group
        (recno=1, type=4) holds every pre-existing firm — this is how the
        engine seeds infrastructure before the AI has bought in."""
        for path, exp in EXPECTED.items():
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(path=os.path.basename(path)):
                gov = save.groups[0]
                self.assertEqual(gov.recno, 1)
                self.assertEqual(gov.group_type, 4)
                self.assertEqual(len(gov.firm_recnos), exp["firm_count"])
                # group_of() should round-trip for every firm.
                for f in save.firms:
                    self.assertIs(save.group_of(f.recno), gov)


class TestFirmSubclass(unittest.TestCase):
    """Typed decoding of the Firm subclass region (see ``firm_subclass.py``).

    Two layers of coverage live here:

    * **Empirical** — FirmPublic (types 0x20..0x24) is present in the
      fixtures: 4 firms of each of the five types, all with a zero-byte
      subclass payload.  We assert the dispatcher produces a
      :class:`FirmPublicSub` for every one.
    * **Synthetic** — the basic subclass decoders (types 0x01/0x02/0x04/
      0x05/0x07) are not exercised by any fixture firm, so we
      hand-build 8-byte payloads and assert the field maps.  This catches
      field-offset drift without needing a save that contains those
      firms.
    """

    def test_public_firms_have_zero_byte_subclass(self) -> None:
        """Every type-0x20..0x24 firm decodes to an empty FirmPublicSub."""
        from caplab_save.firm_subclass import FirmPublicSub
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        found = 0
        for f in save.firms:
            if 0x20 <= f.firm_type <= 0x24:
                self.assertEqual(
                    len(f.raw_subclass), 0,
                    f"firm {f.recno} type=0x{f.firm_type:02x}: "
                    f"FirmPublic should be base-only"
                )
                sub = f.subclass
                self.assertIsInstance(sub, FirmPublicSub)
                self.assertEqual(sub.raw, b"")
                found += 1
        # Fixture histogram: 4 firms × 5 types = 20.
        self.assertEqual(found, 20)

    def test_undecoded_types_return_none(self) -> None:
        """Types without a typed decoder should return ``None`` from .subclass."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        # Pick one firm of each undecoded type present in the fixture.
        undecoded_types = {0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x25, 0x28, 0x29}
        seen: set[int] = set()
        for f in save.firms:
            if f.firm_type in undecoded_types:
                self.assertIsNone(
                    f.subclass,
                    f"firm {f.recno} type=0x{f.firm_type:02x}: "
                    f"expected .subclass to be None (no decoder yet)"
                )
                seen.add(f.firm_type)
        self.assertEqual(
            seen, undecoded_types,
            f"missing coverage for types: {sorted(undecoded_types - seen)}"
        )

    def test_dispatcher_covers_basic_and_public_types(self) -> None:
        """All 5 basic + all 5 FirmPublic types have decoders registered."""
        from caplab_save.firm_subclass import supported_types
        covered = supported_types()
        for ft in (0x01, 0x02, 0x04, 0x05, 0x07):
            self.assertIn(ft, covered, f"missing decoder for type 0x{ft:02x}")
        for ft in range(0x20, 0x25):
            self.assertIn(ft, covered, f"missing decoder for type 0x{ft:02x}")

    def test_firm_rd_synthetic_payload(self) -> None:
        """FirmRD field map: char focus_flag @ +0x00, int product_recno @ +0x04."""
        import struct
        from caplab_save.firm_subclass import FirmRDSub, decode_firm_rd
        # focus_flag=1, pad=0, product_recno=42
        payload = bytes([0x01, 0x00, 0x00, 0x00]) + struct.pack("<i", 42)
        sub = decode_firm_rd(payload)
        self.assertIsInstance(sub, FirmRDSub)
        self.assertEqual(sub.focus_flag, 1)
        self.assertEqual(sub.product_recno, 42)
        self.assertEqual(sub.raw, payload)
        # Zero payload → both fields zero.
        sub2 = decode_firm_rd(b"\x00" * 8)
        self.assertEqual(sub2.focus_flag, 0)
        self.assertEqual(sub2.product_recno, 0)

    def test_basic_generic_word_decoders(self) -> None:
        """FirmHq/Retail/Factory/Warehouse expose the 8 bytes as 4 shorts."""
        import struct
        from caplab_save.firm_subclass import (
            FirmFactorySub, FirmHqSub, FirmRetailSub, FirmWarehouseSub,
            decode_firm_factory, decode_firm_hq, decode_firm_retail,
            decode_firm_warehouse,
        )
        payload = struct.pack("<HHHH", 0x1111, 0x2222, 0x3333, 0x4444)
        cases = (
            (decode_firm_hq, FirmHqSub),
            (decode_firm_retail, FirmRetailSub),
            (decode_firm_factory, FirmFactorySub),
            (decode_firm_warehouse, FirmWarehouseSub),
        )
        for decoder, cls in cases:
            with self.subTest(decoder=decoder.__name__):
                sub = decoder(payload)
                self.assertIsInstance(sub, cls)
                self.assertEqual(sub.words, (0x1111, 0x2222, 0x3333, 0x4444))
                self.assertEqual(sub.raw, payload)

    def test_size_mismatch_raises(self) -> None:
        """Decoders refuse payloads of the wrong length — catches drift fast."""
        from caplab_save.firm_subclass import (
            decode_firm_hq, decode_firm_public, decode_firm_rd,
        )
        with self.assertRaises(ValueError):
            decode_firm_rd(b"\x00" * 4)   # too short
        with self.assertRaises(ValueError):
            decode_firm_rd(b"\x00" * 16)  # too long
        with self.assertRaises(ValueError):
            decode_firm_hq(b"")           # basic expects 8 bytes
        with self.assertRaises(ValueError):
            decode_firm_public(b"\x00")   # public expects 0 bytes

    def test_dispatcher_roundtrips_via_firm_subclass(self) -> None:
        """``Firm.subclass.raw`` must equal ``Firm.raw_subclass`` for every
        firm whose type has a typed decoder."""
        _require_save(SAVE_PLAY_001)
        save = CapLabSave.load(SAVE_PLAY_001)
        from caplab_save.firm_subclass import supported_types
        covered = supported_types()
        checked = 0
        for f in save.firms:
            if f.firm_type in covered:
                sub = f.subclass
                self.assertIsNotNone(sub)
                self.assertEqual(sub.raw, f.raw_subclass,
                                 f"firm {f.recno} subclass raw mismatch")
                checked += 1
        # Sanity check: the fixture contains at least the 20 FirmPublic firms.
        self.assertGreaterEqual(checked, 20)


class TestBasicFirmFixtures(unittest.TestCase):
    """Real-save regression tests for the basic-type subclass decoders.

    These tests are the **empirical** complement to
    :class:`TestFirmSubclass` above.  Together they close the gap that was
    flagged in ``firm_subclass.py``'s docstring: the PLAY_* fixtures do
    not contain any type-0x01/0x02/0x04/0x05/0x07 firms, so the basic
    decoders previously had only structural (synthetic-payload) coverage.

    Each of these saves is the previous game state plus one newly bought
    player firm, with the type/subtype and raw_subclass tuple baked into
    :data:`BASIC_FIRM_STEPS` as the regression contract.
    """

    def test_basic_firm_saves_load(self) -> None:
        """All five cumulative saves load without error."""
        for path, _, meta in BASIC_FIRM_STEPS:
            _require_save(path)
            with self.subTest(save=os.path.basename(path)):
                save = CapLabSave.load(path)
                # Every save shares the same calendar and 22 groups
                # (scenario fingerprint).
                self.assertEqual(save.game_info.game_date,
                                 BASIC_FIRM_CALENDAR["game_date"])
                self.assertEqual(save.game_info.year,
                                 BASIC_FIRM_CALENDAR["year"])
                self.assertEqual(len(save.groups), 22)

    def test_player_firm_count_grows_one_per_save(self) -> None:
        """Player group holds exactly N firms in the Nth basic-firm save."""
        for path, n, meta in BASIC_FIRM_STEPS:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(save=os.path.basename(path)):
                player = next((g for g in save.groups if g.group_type == 1),
                              None)
                self.assertIsNotNone(player, "no player (type=1) group")
                self.assertEqual(
                    player.firm_count, n,
                    f"{meta['label']}: expected player.firm_count={n}, "
                    f"got {player.firm_count}"
                )
                self.assertEqual(len(player.firm_recnos), n)

    def test_new_firm_matches_expected_bytes(self) -> None:
        """The firm added in save N has the documented type / subtype /
        raw_subclass tuple, and decodes to the correct dataclass."""
        from caplab_save import firm_subclass as fs
        prev_player_recnos: set = set()
        for path, n, meta in BASIC_FIRM_STEPS:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(save=os.path.basename(path)):
                player = next(g for g in save.groups if g.group_type == 1)
                this_recnos = set(player.firm_recnos)
                added = this_recnos - prev_player_recnos
                self.assertEqual(
                    len(added), 1,
                    f"{meta['label']}: expected exactly 1 new player firm, "
                    f"got {len(added)}"
                )
                new_recno = added.pop()
                f = next(fi for fi in save.firms if fi.recno == new_recno)
                self.assertEqual(f.firm_type, meta["firm_type"],
                                 f"{meta['label']}: firm_type")
                self.assertEqual(f.firm_subtype, meta["firm_subtype"],
                                 f"{meta['label']}: firm_subtype")
                self.assertEqual(
                    f.raw_subclass, meta["raw_subclass"],
                    f"{meta['label']}: raw_subclass bytes"
                )
                self.assertEqual(f.group_recno, player.recno,
                                 f"{meta['label']}: group_recno link")

                # Typed dispatcher must return the right dataclass.
                sub = f.subclass
                expected_cls = getattr(fs, meta["typed_cls_name"])
                self.assertIsInstance(
                    sub, expected_cls,
                    f"{meta['label']}: expected {meta['typed_cls_name']}, "
                    f"got {type(sub).__name__}"
                )
                # Round-trip: typed view's .raw matches the firm's raw_subclass.
                self.assertEqual(sub.raw, f.raw_subclass)
                prev_player_recnos = this_recnos

    def test_firm_rd_fresh_place_has_zero_fields(self) -> None:
        """A freshly-placed R&D center is unfocused and has no dedicated
        product — ``FirmRDSub.focus_flag`` and ``.product_recno`` are 0."""
        from caplab_save.firm_subclass import FirmRDSub
        _require_save(SAVE_RD)
        save = CapLabSave.load(SAVE_RD)
        player = next(g for g in save.groups if g.group_type == 1)
        rd_firms = [f for f in save.firms
                    if f.group_recno == player.recno and f.firm_type == 0x05]
        self.assertEqual(len(rd_firms), 1, "expected exactly one R&D firm")
        sub = rd_firms[0].subclass
        self.assertIsInstance(sub, FirmRDSub)
        self.assertEqual(sub.focus_flag, 0)
        self.assertEqual(sub.product_recno, 0)

    def test_factory_and_warehouse_share_subclass_pattern(self) -> None:
        """Empirical RE observation: FirmFactory (0x04) and FirmWarehouse
        (0x07) serialize with the same first-byte-zero, next-three-bytes-
        all-0xFF, trailing-four-bytes-zero pattern right after placement.

        The leading 0xFF triplet is almost certainly a "no selection" /
        sentinel for a yet-unassigned inventory slot (product or item
        recno, or similar).  This test documents the pattern as a
        regression anchor; if a future engine change alters the default
        fill, we want to know."""
        _require_save(SAVE_FACTORY)
        _require_save(SAVE_WAREHOUSE)
        factory_save = CapLabSave.load(SAVE_FACTORY)
        warehouse_save = CapLabSave.load(SAVE_WAREHOUSE)

        def _player_firm(save, firm_type: int) -> bytes:
            p = next(g for g in save.groups if g.group_type == 1)
            fs = [f for f in save.firms
                  if f.group_recno == p.recno and f.firm_type == firm_type]
            self.assertEqual(len(fs), 1,
                             f"expected exactly one type=0x{firm_type:02x} "
                             f"player firm")
            return fs[0].raw_subclass

        factory_bytes = _player_firm(factory_save, 0x04)
        warehouse_bytes = _player_firm(warehouse_save, 0x07)
        self.assertEqual(factory_bytes, warehouse_bytes,
                         "factory/warehouse subclass patterns diverged")
        self.assertEqual(factory_bytes, bytes.fromhex("00ffffff00000000"))

    def test_hq_first_byte_is_active_marker(self) -> None:
        """Empirical RE observation: the first byte of the FirmHq subclass
        is 0x01 for a player-placed HQ.  We don't yet know whether this
        encodes "active", "primary HQ", or a small integer — but the
        value is non-zero, which distinguishes HQ from the zero-filled
        subclass of Retail / R&D."""
        _require_save(SAVE_HEADQUARTERS)
        save = CapLabSave.load(SAVE_HEADQUARTERS)
        player = next(g for g in save.groups if g.group_type == 1)
        hq_firms = [f for f in save.firms
                    if f.group_recno == player.recno and f.firm_type == 0x01]
        self.assertEqual(len(hq_firms), 1)
        self.assertEqual(hq_firms[0].raw_subclass[0], 0x01)
        # Remaining bytes are zero.
        self.assertEqual(hq_firms[0].raw_subclass[1:], b"\x00" * 7)

    def test_background_government_firms_unchanged(self) -> None:
        """Across all five basic-firm saves the Government-held firm
        histogram (everything *except* the new player firms) is constant —
        the scenario's pre-existing infrastructure doesn't shift when the
        player buys a building."""
        from collections import Counter
        for path, n, meta in BASIC_FIRM_STEPS:
            _require_save(path)
            save = CapLabSave.load(path)
            with self.subTest(save=os.path.basename(path)):
                gov = save.groups[0]  # recno=1, type=4
                self.assertEqual(gov.group_type, 4)
                gov_firms = [f for f in save.firms
                             if f.group_recno == gov.recno]
                hist = dict(Counter(f.firm_type for f in gov_firms))
                self.assertEqual(
                    hist, BASIC_FIRM_BACKGROUND_HIST,
                    f"{meta['label']}: government firm histogram drift"
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
