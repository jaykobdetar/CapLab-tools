"""Regression guard for the dynamic RNG-seed lookup in ``parse_rng_seed``.

The original implementation hardcoded ``RNG_SEED_OFFSET_BLOB0 = 0x1159A``
which happens to be correct for PLAY_001 and PLAY_002 but wrong for
every save where the GameInfo payload has grown (e.g. any save with
player-owned firms). This test fixes the contract that ``parse_rng_seed``
locates tag 0x1067 dynamically and reports the same seed across the
cumulative fixture saves — which are snapshots of the same game tick
differing only by one added player firm each, so their seeds must match
bit-exactly.

Run::

    python -m unittest caplab_save.tests.test_rng_seed_offset -v
"""

from __future__ import annotations

import os
import unittest

from caplab_save.decompress import decompress_save
from caplab_save.parser import parse_rng_seed


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "caplab_save", "saves")
CUMULATIVE_FIXTURES = [
    "Headquarters.SAV",
    "Department.SAV",
    "Factory.SAV",
    "R&D.SAV",
    "Warehouse.SAV",
]


class ParseRngSeedDynamicTests(unittest.TestCase):
    def test_cumulative_fixtures_report_identical_seed(self):
        seeds = []
        offsets = []
        for name in CUMULATIVE_FIXTURES:
            path = os.path.join(FIXTURE_DIR, name)
            if not os.path.exists(path):
                self.skipTest(f"fixture missing: {path}")
            r = decompress_save(path)
            rng = parse_rng_seed(r.blob0)
            seeds.append(rng.seed)
            offsets.append(rng.source_offset)
        self.assertEqual(len(set(seeds)), 1,
                         f"cumulative fixtures must share a seed, got {seeds}")
        # Each added player firm shifts the tag forward — offsets must
        # be strictly increasing. If they aren't, we'd likely be finding
        # a spurious earlier occurrence of the byte pattern.
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(set(offsets)), len(offsets),
                         f"offsets should be distinct across fixtures, "
                         f"got {list(zip(CUMULATIVE_FIXTURES, offsets))}")

    def test_tag_0x1067_not_found_raises(self):
        # Feed it a blob with no 0x1067 anywhere.
        import struct
        from caplab_save import constants as C
        # Build 10KB of non-0x1067 bytes; struct.pack('<I', 0x1067) is \x67\x10\x00\x00
        # so we avoid that pattern entirely by repeating 0xAA.
        bogus = bytes([0xAA]) * 10_000
        with self.assertRaisesRegex(ValueError, f"0x{C.RNG_SEED_TAG:04x}"):
            parse_rng_seed(bogus)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
