"""Regression tests for the ScenarioCfg (tag 0x1068) calendar parser.

Context
-------
Until 2026-04-21, :func:`caplab_save.parser.parse_gameinfo` guessed the
current game date by scanning the GameInfo payload (tag 0x1065) for
int32s in a plausible value range and picking the mode. That heuristic
silently matched a 15-element *stamp array* at GameInfo+0x610 stride
0x204 that happens to hold the value ``708407`` (= 1940-11-04 in the
engine's old "days since year 0" interpretation) across every save the
project had seen — so all PLAY_* / fixture saves reported Nov 4 1940,
even when the user was demonstrably on Jan 1 / Jan 2 1990.

Task #8 replaced the heuristic with a field-accurate read from
ScenarioCfg (tag 0x1068), whose layout was reverse-engineered from the
byte-diff of a consecutive-day save pair.  The tests in this file
anchor that layout to real bytes so any future regression is caught
immediately.
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caplab_save import CapLabSave  # noqa: E402
from caplab_save.parser import jdn_to_gregorian  # noqa: E402


SAVE_CONSEC_DAY_1 = os.path.join(
    _REPO_ROOT, "_____001_20260421_054902_000.SAV"
)
SAVE_CONSEC_DAY_2 = os.path.join(
    _REPO_ROOT, "_____002_20260421_054916_000.SAV"
)


def _require_save(path: str) -> None:
    if not os.path.exists(path):
        raise unittest.SkipTest(f"fixture save not present: {path}")


class TestConsecutiveDayCalendar(unittest.TestCase):
    """The _____001 / _____002 pair is the primary calendar regression.

    Both saves were captured by the user during a fresh Jan 1 1990
    scenario, 14 seconds apart in real time but exactly one game-day
    apart in the engine.  If the parser is correct:

    * day_counter (JDN) increments by exactly 1
    * day_of_month goes 1 → 2
    * days_since_start goes 0 → 1
    * year / month / start_day_counter / start_year stay constant
    * jdn_to_gregorian(day_counter) matches (year, month, day_of_month)
      independently of the parser's own fields
    """

    def test_day1_is_1990_01_01(self) -> None:
        _require_save(SAVE_CONSEC_DAY_1)
        save = CapLabSave.load(SAVE_CONSEC_DAY_1)
        gi = save.game_info
        self.assertEqual(gi.year, 1990)
        self.assertEqual(gi.month_of_year, 1)
        self.assertEqual(gi.day_of_month, 1)
        self.assertEqual(gi.day_of_year, 1)
        self.assertEqual(gi.days_since_start, 0)
        self.assertEqual(gi.start_day_counter, 2_447_893)
        self.assertEqual(gi.start_year, 1990)
        self.assertEqual(gi.game_date, 2_447_893)

    def test_day2_is_1990_01_02(self) -> None:
        _require_save(SAVE_CONSEC_DAY_2)
        save = CapLabSave.load(SAVE_CONSEC_DAY_2)
        gi = save.game_info
        self.assertEqual(gi.year, 1990)
        self.assertEqual(gi.month_of_year, 1)
        self.assertEqual(gi.day_of_month, 2)
        self.assertEqual(gi.day_of_year, 2)
        self.assertEqual(gi.days_since_start, 1)
        self.assertEqual(gi.start_day_counter, 2_447_893)
        self.assertEqual(gi.start_year, 1990)
        self.assertEqual(gi.game_date, 2_447_894)

    def test_exactly_one_day_advanced(self) -> None:
        _require_save(SAVE_CONSEC_DAY_1)
        _require_save(SAVE_CONSEC_DAY_2)
        a = CapLabSave.load(SAVE_CONSEC_DAY_1)
        b = CapLabSave.load(SAVE_CONSEC_DAY_2)
        # All four "day-like" counters should step by exactly 1.
        self.assertEqual(b.game_info.game_date - a.game_info.game_date, 1)
        self.assertEqual(b.game_info.day_of_year - a.game_info.day_of_year, 1)
        self.assertEqual(b.game_info.day_of_month - a.game_info.day_of_month, 1)
        self.assertEqual(
            b.game_info.days_since_start - a.game_info.days_since_start, 1
        )
        # Month / year / scenario-start invariants hold.
        self.assertEqual(a.game_info.month_of_year, b.game_info.month_of_year)
        self.assertEqual(a.game_info.year, b.game_info.year)
        self.assertEqual(a.game_info.start_day_counter,
                         b.game_info.start_day_counter)

    def test_jdn_roundtrip_agrees_with_gregorian_fields(self) -> None:
        for path in (SAVE_CONSEC_DAY_1, SAVE_CONSEC_DAY_2):
            _require_save(path)
            save = CapLabSave.load(path)
            gi = save.game_info
            y, m, d = jdn_to_gregorian(gi.game_date)
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual(
                    (y, m, d),
                    (gi.year, gi.month_of_year, gi.day_of_month),
                    msg=(
                        f"ScenarioCfg+0x18 decodes to {y}-{m:02d}-{d:02d} "
                        f"but +0x20/+0x24/+0x28 say "
                        f"{gi.year}-{gi.month_of_year:02d}-{gi.day_of_month:02d}"
                    ),
                )


class TestJdnEpoch(unittest.TestCase):
    """Lock down the JDN anchor independently of any save file."""

    def test_jdn_1990_01_01_is_2447893(self) -> None:
        self.assertEqual(jdn_to_gregorian(2_447_893), (1990, 1, 1))

    def test_jdn_2000_01_01_is_2451545(self) -> None:
        self.assertEqual(jdn_to_gregorian(2_451_545), (2000, 1, 1))

    def test_jdn_1582_10_15_is_gregorian_reform(self) -> None:
        # Gregorian reform day — canonical JDN anchor.
        self.assertEqual(jdn_to_gregorian(2_299_161), (1582, 10, 15))


if __name__ == "__main__":
    unittest.main()
