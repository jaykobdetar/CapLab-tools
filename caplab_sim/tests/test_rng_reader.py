"""Phase 2 stage 2 tests — GlobalGrpArray walker logic.

These tests do NOT require a running game.  They exercise the pure
logic around the GlobalGrpArray header parse and chunk-walk formula by
patching the module's private ``_read_u32`` / ``_read_mem`` helpers
with a dict-backed fake memory.  The live-process path is smoke-tested
separately by running ``python -m caplab_sim.rng_reader`` against a
real ``CapMain.exe``.
"""

from __future__ import annotations

import struct
import unittest
from typing import Dict
from unittest import mock

from caplab_sim import rng_reader
from caplab_sim.rng_reader import (
    BASE_GROUP_ARRAY_VA,
    ECONOMY_GROUP_ARRAY_VA,
    GGA_CHUNKS_BASE,
    GGA_CHUNK_STRIDE,
    GGA_ENTRY_SIZE,
    GGA_N_CHUNKS,
    GGA_N_GROUPS,
    GGA_STRIDE,
    GLOBAL_GROUP_ARRAY_VA,
    GROUP_ASSETS_BLOCK_OFFSET,
    GROUP_CORP_CASH_OFFSET,
    GROUP_FIRM_COUNT_OFFSET,
    KNOWN_GROUP_ARRAYS,
    GroupArrayInfo,
    GroupAssets,
    LiveGameReader,
    _iter_group_pointers,
    _read_group_array_header,
    _read_group_assets,
)


class FakeMem:
    """In-process replacement for ``/proc/<pid>/mem`` reads.

    Maps ``address -> bytes``.  Overlapping ranges are fine; the reader
    slices the shortest containing range.  ``_read_u32`` is served by
    unpacking the first 4 bytes of the range at ``address``.
    """

    def __init__(self):
        self.ranges: Dict[int, bytes] = {}

    def put(self, address: int, data: bytes) -> None:
        self.ranges[address] = bytes(data)

    def put_u32(self, address: int, value: int) -> None:
        self.put(address, struct.pack("<I", value & 0xFFFFFFFF))

    def read_u32(self, pid: int, address: int) -> int:
        if address in self.ranges:
            return struct.unpack("<I", self.ranges[address][:4])[0]
        # Synthesize from any range that contains this address.
        for base, data in self.ranges.items():
            if base <= address < base + len(data):
                off = address - base
                return struct.unpack("<I", data[off:off + 4])[0]
        raise OSError(f"fake read_u32: unmapped 0x{address:08x}")

    def read_mem(self, pid: int, address: int, size: int) -> bytes:
        for base, data in self.ranges.items():
            if base <= address < base + len(data):
                off = address - base
                return data[off:off + size]
        raise OSError(f"fake read_mem: unmapped 0x{address:08x}")


class GroupArrayInfoPlausibilityTests(unittest.TestCase):
    """``is_plausible`` should reject obviously bad headers."""

    def _info(self, **overrides) -> GroupArrayInfo:
        base = dict(
            stride=1024,
            n_groups=153,
            entry_size=4,
            n_chunks=1,
            chunk_stride=8,
            chunks_base=0x10000000,
        )
        base.update(overrides)
        return GroupArrayInfo(**base)

    def test_nominal_header_is_plausible(self):
        self.assertTrue(self._info().is_plausible())

    def test_zero_ngroups_rejected(self):
        self.assertFalse(self._info(n_groups=0).is_plausible())

    def test_huge_ngroups_rejected(self):
        self.assertFalse(self._info(n_groups=10_000_000).is_plausible())

    def test_non_pointer_entry_size_rejected(self):
        self.assertFalse(self._info(entry_size=8).is_plausible())

    def test_zero_stride_rejected(self):
        self.assertFalse(self._info(stride=0).is_plausible())

    def test_low_chunks_base_rejected(self):
        self.assertFalse(self._info(chunks_base=0x0000FFFF).is_plausible())


class ReadGroupArrayHeaderTests(unittest.TestCase):
    """Header reads must pull the six fields at the documented offsets."""

    def test_header_fields_decoded_from_canonical_offsets(self):
        mem = FakeMem()
        gga_addr = 0x009dd328
        # Use distinct sentinel values so a swapped offset would jump out.
        mem.put_u32(gga_addr + GGA_STRIDE,       0xA1A1A1A1)
        mem.put_u32(gga_addr + GGA_N_GROUPS,     0xA2A2A2A2)
        mem.put_u32(gga_addr + GGA_ENTRY_SIZE,   0xA3A3A3A3)
        mem.put_u32(gga_addr + GGA_N_CHUNKS,     0xA4A4A4A4)
        mem.put_u32(gga_addr + GGA_CHUNK_STRIDE, 0xA5A5A5A5)
        mem.put_u32(gga_addr + GGA_CHUNKS_BASE,  0xA6A6A6A6)

        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            info = _read_group_array_header(pid=0, gga_addr=gga_addr)

        self.assertEqual(info.stride,       0xA1A1A1A1)
        self.assertEqual(info.n_groups,     0xA2A2A2A2)
        self.assertEqual(info.entry_size,   0xA3A3A3A3)
        self.assertEqual(info.n_chunks,     0xA4A4A4A4)
        self.assertEqual(info.chunk_stride, 0xA5A5A5A5)
        self.assertEqual(info.chunks_base,  0xA6A6A6A6)


class IterGroupPointersTests(unittest.TestCase):
    """The walk formula must match the Ghidra decomp exactly.

    We mock a 5-group array spread across 2 chunks (stride=3) to make
    sure chunk-boundary math is right, and salt in one deleted slot
    (group_ptr == 0) to confirm the engine's skip behaviour.
    """

    def _setup_two_chunk_array(self, chunks_base: int, chunk_body_addrs,
                               group_ptrs_per_chunk):
        """Build a FakeMem that implements the engine's chunk layout.

        chunks_base: address of chunk-header table.
        chunk_body_addrs: list of (chunk_body_address) per chunk.
        group_ptrs_per_chunk: list-of-lists; group_ptrs_per_chunk[i][j]
                              is the pointer for slot j of chunk i.
        """
        mem = FakeMem()
        stride = max(len(row) for row in group_ptrs_per_chunk)
        chunk_stride = 0x20  # arbitrary, must match what we read
        for i, body_addr in enumerate(chunk_body_addrs):
            # Chunk header at chunks_base + chunk_stride*i: first u32 is body_addr.
            mem.put_u32(chunks_base + chunk_stride * i, body_addr)
            # Slot array: each slot is 4 bytes at body_addr + 4*j.
            for j, gp in enumerate(group_ptrs_per_chunk[i]):
                mem.put_u32(body_addr + 4 * j, gp)
        return mem, stride, chunk_stride

    def test_slot_ascending_walk_skips_deleted(self):
        chunks_base = 0x20000000
        chunk_body_addrs = [0x30000000, 0x30010000]
        # Chunk 0 slots: g1_ptr, g2_ptr, 0(deleted)
        # Chunk 1 slots: g4_ptr, g5_ptr, (unused — only 5 total)
        group_ptrs = [
            [0xAAAAAAA1, 0xAAAAAAA2, 0x00000000],
            [0xAAAAAAA4, 0xAAAAAAA5, 0x00000000],
        ]
        mem, stride, chunk_stride = self._setup_two_chunk_array(
            chunks_base, chunk_body_addrs, group_ptrs
        )
        info = GroupArrayInfo(
            stride=stride,
            n_groups=5,
            entry_size=4,
            n_chunks=2,
            chunk_stride=chunk_stride,
            chunks_base=chunks_base,
        )

        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            out = list(_iter_group_pointers(pid=0, info=info))

        # Slot 3 is deleted; slots 1,2,4,5 survive, slot-ascending.
        self.assertEqual(
            out,
            [(1, 0xAAAAAAA1), (2, 0xAAAAAAA2),
             (4, 0xAAAAAAA4), (5, 0xAAAAAAA5)],
        )

    def test_chunk_boundary_math(self):
        """With stride=3, slots 1-3 are in chunk 0, slots 4-6 in chunk 1.

        If the modulo/division is flipped, slot 4 would land at chunk 0
        slot 1 — i.e. the predicted group_ptr would be 0xAAAAAAA1 instead
        of 0xBBBBBBB1.  That's what this test guards against.
        """
        chunks_base = 0x20000000
        chunk_body_addrs = [0x30000000, 0x30010000]
        group_ptrs = [
            [0xAAAAAAA1, 0xAAAAAAA2, 0xAAAAAAA3],
            [0xBBBBBBB1, 0xBBBBBBB2, 0xBBBBBBB3],
        ]
        mem, stride, chunk_stride = self._setup_two_chunk_array(
            chunks_base, chunk_body_addrs, group_ptrs
        )
        info = GroupArrayInfo(
            stride=stride,
            n_groups=6,
            entry_size=4,
            n_chunks=2,
            chunk_stride=chunk_stride,
            chunks_base=chunks_base,
        )
        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            out = list(_iter_group_pointers(pid=0, info=info))
        self.assertEqual(
            [gp for _, gp in out],
            [0xAAAAAAA1, 0xAAAAAAA2, 0xAAAAAAA3,
             0xBBBBBBB1, 0xBBBBBBB2, 0xBBBBBBB3],
        )

    def test_empty_chunk_body_stops_slot(self):
        """If a chunk header points to NULL, slots in it are skipped.

        This shouldn't happen in a well-formed array but guards against
        the engine's own defensive ``iVar1 < 1 || ...`` check in the
        decomp, which we mirror via ``if chunk_body == 0: continue``.
        """
        chunks_base = 0x20000000
        chunk_stride = 0x20
        mem = FakeMem()
        # Chunk 0 body is NULL — slots 1 and 2 should be skipped.
        mem.put_u32(chunks_base + chunk_stride * 0, 0)
        # Chunk 1 body is valid with one pointer.
        mem.put_u32(chunks_base + chunk_stride * 1, 0x30010000)
        mem.put_u32(0x30010000, 0xBBBBBBB1)
        info = GroupArrayInfo(
            stride=2,
            n_groups=3,
            entry_size=4,
            n_chunks=2,
            chunk_stride=chunk_stride,
            chunks_base=chunks_base,
        )
        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            out = list(_iter_group_pointers(pid=0, info=info))
        self.assertEqual(out, [(3, 0xBBBBBBB1)])


class KnownGroupArraysTests(unittest.TestCase):
    """The known-array table must include both singleton VAs."""

    def test_backcompat_alias_points_at_economy_mode(self):
        # Callers that imported the old single-array name get the
        # Economy-Mode address — matches the original single-array
        # reader's behaviour.
        self.assertEqual(GLOBAL_GROUP_ARRAY_VA, ECONOMY_GROUP_ARRAY_VA)

    def test_base_game_array_comes_first(self):
        # Order matters: the base-game array is populated in every
        # scenario, the Economy-Mode one only when the DLC is on.
        # iter_groups() walks in this order, so the primary/first entry
        # should be the one that's always useful.
        self.assertEqual(KNOWN_GROUP_ARRAYS[0], ("base", BASE_GROUP_ARRAY_VA))
        self.assertEqual(
            KNOWN_GROUP_ARRAYS[1], ("economy", ECONOMY_GROUP_ARRAY_VA)
        )


class LiveGameReaderDualArrayTests(unittest.TestCase):
    """``iter_groups`` must merge populated arrays and skip empty ones.

    Uses a FakeMem to stand in for ``/proc/<pid>/mem`` — the real-process
    path is covered by the CLI smoke test described in
    ``RESEARCH_NOTES.md``.
    """

    def _build_array(self, mem, gga_addr, chunks_base, chunk_body,
                     group_ptrs):
        mem.put_u32(gga_addr + GGA_STRIDE,       len(group_ptrs))
        mem.put_u32(gga_addr + GGA_N_GROUPS,     len(group_ptrs))
        mem.put_u32(gga_addr + GGA_ENTRY_SIZE,   4)
        mem.put_u32(gga_addr + GGA_N_CHUNKS,     1)
        mem.put_u32(gga_addr + GGA_CHUNK_STRIDE, 0x20)
        mem.put_u32(gga_addr + GGA_CHUNKS_BASE,  chunks_base)
        mem.put_u32(chunks_base, chunk_body)
        for i, gp in enumerate(group_ptrs):
            mem.put_u32(chunk_body + 4 * i, gp)

    def test_iter_groups_walks_only_populated_arrays(self):
        mem = FakeMem()
        # Pretend process base == default image base so the VAs match.
        base_gga_addr    = 0x12340000
        econ_gga_addr    = 0x12350000
        self._build_array(mem, base_gga_addr,
                          chunks_base=0x12341000,
                          chunk_body=0x12342000,
                          group_ptrs=[0xAAAAAAA1, 0xAAAAAAA2])
        # Economy-Mode array: n_groups=0 and stride=0 → NOT plausible.
        mem.put_u32(econ_gga_addr + GGA_STRIDE,       0)
        mem.put_u32(econ_gga_addr + GGA_N_GROUPS,     0)
        mem.put_u32(econ_gga_addr + GGA_ENTRY_SIZE,   4)
        mem.put_u32(econ_gga_addr + GGA_N_CHUNKS,     0)
        mem.put_u32(econ_gga_addr + GGA_CHUNK_STRIDE, 0)
        mem.put_u32(econ_gga_addr + GGA_CHUNKS_BASE,  0)

        reader = LiveGameReader(
            pid=0, base=0, misc_addr=0,
            group_arrays=(("base", base_gga_addr),
                          ("economy", econ_gga_addr)),
        )
        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            walked = list(reader.iter_groups())

        self.assertEqual(
            walked,
            [("base", 1, 0xAAAAAAA1), ("base", 2, 0xAAAAAAA2)],
        )

    def test_iter_groups_merges_both_when_populated(self):
        mem = FakeMem()
        base_gga_addr = 0x12340000
        econ_gga_addr = 0x12350000
        self._build_array(mem, base_gga_addr,
                          chunks_base=0x12341000,
                          chunk_body=0x12342000,
                          group_ptrs=[0xAAAAAAA1])
        self._build_array(mem, econ_gga_addr,
                          chunks_base=0x12351000,
                          chunk_body=0x12352000,
                          group_ptrs=[0xBBBBBBB1, 0xBBBBBBB2])

        reader = LiveGameReader(
            pid=0, base=0, misc_addr=0,
            group_arrays=(("base", base_gga_addr),
                          ("economy", econ_gga_addr)),
        )
        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            walked = list(reader.iter_groups())

        # Base-first, Economy-second — matches KNOWN_GROUP_ARRAYS order.
        self.assertEqual(
            walked,
            [
                ("base", 1, 0xAAAAAAA1),
                ("economy", 1, 0xBBBBBBB1),
                ("economy", 2, 0xBBBBBBB2),
            ],
        )

    def test_gga_addr_prefers_plausible_array(self):
        """Backwards-compat property picks the first populated array."""
        mem = FakeMem()
        base_gga_addr = 0x12340000
        econ_gga_addr = 0x12350000
        # Base array EMPTY, Economy populated.
        for off in (GGA_STRIDE, GGA_N_GROUPS, GGA_ENTRY_SIZE,
                    GGA_N_CHUNKS, GGA_CHUNK_STRIDE, GGA_CHUNKS_BASE):
            mem.put_u32(base_gga_addr + off, 0)
        self._build_array(mem, econ_gga_addr,
                          chunks_base=0x12351000,
                          chunk_body=0x12352000,
                          group_ptrs=[0xBBBBBBB1])

        reader = LiveGameReader(
            pid=0, base=0, misc_addr=0,
            group_arrays=(("base", base_gga_addr),
                          ("economy", econ_gga_addr)),
        )
        with mock.patch.object(rng_reader, "_read_u32", mem.read_u32):
            primary = reader.gga_addr
        self.assertEqual(primary, econ_gga_addr)


class ReadGroupAssetsTests(unittest.TestCase):
    """``_read_group_assets`` pulls corp_cash + 10 doubles + firm_count.

    These tests don't depend on the walk formula — they feed a known
    memory layout to the reader and verify each offset decodes to the
    expected value.  A regression here means either (a) the offset
    constants have drifted from the Ghidra decomp, or (b) the reader
    got the byte ordering wrong.
    """

    def _mem_with_group(
        self,
        group_ptr: int,
        *,
        corp_cash: float = 0.0,
        bank_deposits: float = 0.0,
        inventory: float = 0.0,
        business_assets: float = 0.0,
        land: float = 0.0,
        technology: float = 0.0,
        stocks_held: float = 0.0,
        bank_net_assets: float = 0.0,
        insurance_net_assets: float = 0.0,
        loans: float = 0.0,
        bond_interest_liability: float = 0.0,
        earnings_metric_driver: float = 0.0,
        firm_count: int = 0,
        trailing_12m_net_profit: float = 0.0,
        trailing_12m_special_revenue: float = 0.0,
    ) -> FakeMem:
        mem = FakeMem()
        # corp_cash at Group+0x0CC0 — one isolated double.
        mem.put(
            group_ptr + GROUP_CORP_CASH_OFFSET,
            struct.pack("<d", corp_cash),
        )
        # Asset block at Group+0x39D8: 0x168 bytes, starting with
        # bank_deposits and ending with earnings_metric_driver far at
        # offset 0x160.  Build as a bytes-of-zeros buffer and patch the
        # known double slots.
        block = bytearray(rng_reader.GROUP_ASSETS_BLOCK_SIZE)
        struct.pack_into("<d", block, 0x00, bank_deposits)
        struct.pack_into("<d", block, 0x08, inventory)
        struct.pack_into("<d", block, 0x10, business_assets)
        struct.pack_into("<d", block, 0x18, land)
        struct.pack_into("<d", block, 0x20, technology)
        struct.pack_into("<d", block, 0x28, stocks_held)
        struct.pack_into("<d", block, 0x38, bank_net_assets)
        struct.pack_into("<d", block, 0x40, insurance_net_assets)
        struct.pack_into("<d", block, 0x48, loans)
        struct.pack_into("<d", block, 0x50, bond_interest_liability)
        struct.pack_into("<d", block, 0x160, earnings_metric_driver)
        mem.put(group_ptr + GROUP_ASSETS_BLOCK_OFFSET, bytes(block))
        # firm_count at Group+0xE4 (uint32).
        mem.put(
            group_ptr + GROUP_FIRM_COUNT_OFFSET,
            struct.pack("<I", firm_count & 0xFFFFFFFF),
        )
        # trailing-12m profit at Group+0xCE0/+0xCE8 (two contiguous doubles).
        # Feeds Group::compute_eps(1) via _read_group_assets; wiring must
        # not regress to an OSError-swallow zero-fill path.
        mem.put(
            group_ptr + rng_reader.GROUP_T12M_NET_PROFIT_OFFSET,
            struct.pack("<dd", trailing_12m_net_profit,
                        trailing_12m_special_revenue),
        )
        return mem

    def test_zero_group_yields_zero_assets(self):
        """A fresh-IPO group with zero everywhere must round-trip to zero."""
        group_ptr = 0x20000000
        mem = self._mem_with_group(group_ptr)
        with mock.patch.object(rng_reader, "_read_mem", mem.read_mem):
            a = _read_group_assets(pid=0, group_ptr=group_ptr)
        self.assertEqual(a, GroupAssets())  # all zero

    def test_all_eleven_doubles_plus_firm_count_decoded(self):
        """Each field decodes from its own offset — sentinel values make
        any swap or shift immediately visible."""
        group_ptr = 0x20000000
        mem = self._mem_with_group(
            group_ptr,
            corp_cash=100.0,
            bank_deposits=200.0,
            inventory=300.0,
            business_assets=400.0,
            land=500.0,
            technology=600.0,
            stocks_held=700.0,
            bank_net_assets=800.0,
            insurance_net_assets=900.0,
            loans=1000.0,
            bond_interest_liability=1100.0,
            earnings_metric_driver=1200.0,
            firm_count=7,
            trailing_12m_net_profit=1300.0,
            trailing_12m_special_revenue=1400.0,
        )
        with mock.patch.object(rng_reader, "_read_mem", mem.read_mem):
            a = _read_group_assets(pid=0, group_ptr=group_ptr)
        self.assertEqual(a.corp_cash, 100.0)
        self.assertEqual(a.bank_deposits, 200.0)
        self.assertEqual(a.inventory, 300.0)
        self.assertEqual(a.business_assets, 400.0)
        self.assertEqual(a.land, 500.0)
        self.assertEqual(a.technology, 600.0)
        self.assertEqual(a.stocks_held, 700.0)
        self.assertEqual(a.bank_net_assets, 800.0)
        self.assertEqual(a.insurance_net_assets, 900.0)
        self.assertEqual(a.loans, 1000.0)
        self.assertEqual(a.bond_interest_liability, 1100.0)
        self.assertEqual(a.earnings_metric_driver, 1200.0)
        self.assertEqual(a.firm_count, 7)
        self.assertEqual(a.trailing_12m_net_profit, 1300.0)
        self.assertEqual(a.trailing_12m_special_revenue, 1400.0)

    def test_matches_save_file_diagnostic_pair(self):
        """A concrete firm-owning group from the A.SAV/B.SAV diag output.

        Uses the numbers from SESSION_LOG "stock.py ↔ engine decomp
        diff" for g3 (firms=6, corp_cash=204.5M, biz_assets=82.8M).
        Regressing the save/live parity means this decodes to exactly
        the same values a ``stock_inputs_from_save(g3)`` call produces.
        """
        group_ptr = 0x30000000
        # Round numbers from the save diagnostic — exact doubles don't
        # matter as long as the bytes round-trip.
        mem = self._mem_with_group(
            group_ptr,
            corp_cash=204_498_021.0,
            business_assets=82_803_912.0,
            firm_count=6,
        )
        with mock.patch.object(rng_reader, "_read_mem", mem.read_mem):
            a = _read_group_assets(pid=0, group_ptr=group_ptr)
        self.assertAlmostEqual(a.corp_cash, 204_498_021.0)
        self.assertAlmostEqual(a.business_assets, 82_803_912.0)
        self.assertEqual(a.firm_count, 6)
        # Fields we didn't populate must stay zero.
        self.assertEqual(a.inventory, 0.0)
        self.assertEqual(a.land, 0.0)
        self.assertEqual(a.loans, 0.0)

    def test_unmapped_memory_yields_zero_filled(self):
        """If the Group pointer is stale (freed by engine mid-read), we
        return zero-filled GroupAssets rather than crashing — the
        monitor can then show a noticeably degraded match rate, which
        is the right failure mode."""
        group_ptr = 0x40000000  # Not inserted into FakeMem.
        mem = FakeMem()
        with mock.patch.object(rng_reader, "_read_mem", mem.read_mem):
            a = _read_group_assets(pid=0, group_ptr=group_ptr)
        self.assertEqual(a, GroupAssets())


if __name__ == "__main__":
    unittest.main()
