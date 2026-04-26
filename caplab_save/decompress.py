"""Decompress a Capitalism Lab .SAV file.

The SAV file layout:

    [0x0000 .. header_end]   ~28 KB plaintext header (MAX_PATH + metadata)
    [header_end .. offA]     zlib stream 0 (world state, ~5.8 MB decompressed)
    [offA .. EOF]            zlib stream 1 (~2.9 MB decompressed)

There can in principle be more streams; we scan the full file for zlib magic
(`0x78 XX` where XX in {0x01, 0x5E, 0x9C, 0xDA}) and attempt to decompress each
stream in order.

This module is deliberately stand-alone so it can be used from anywhere without
importing the rest of the parser.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from typing import List


ZLIB_MAGIC_BYTE_1 = 0x78
_VALID_SECOND_BYTES = (0x01, 0x5E, 0x9C, 0xDA)


@dataclass
class DecompressedStream:
    """One decompressed zlib stream from the SAV file."""

    file_offset: int        # byte offset of compressed stream within .SAV
    compressed_size: int    # number of compressed bytes consumed
    data: bytes             # decompressed payload

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DecompressedStream(file_offset=0x{self.file_offset:x}, "
            f"compressed_size={self.compressed_size:,}, "
            f"decompressed_size={len(self.data):,})"
        )


@dataclass
class DecompressResult:
    """Full decomposition of a .SAV file."""

    header: bytes                      # plaintext prefix before first zlib stream
    streams: List[DecompressedStream]  # one per zlib stream, in file order
    raw: bytes                         # original file bytes (for diffs / forensics)

    @property
    def blob0(self) -> bytes:
        """First zlib stream (world state)."""
        if not self.streams:
            raise RuntimeError("no zlib streams found in save file")
        return self.streams[0].data

    @property
    def blob1(self) -> bytes:
        """Second zlib stream (if present)."""
        if len(self.streams) < 2:
            raise RuntimeError("save file has fewer than 2 zlib streams")
        return self.streams[1].data


def decompress_save(path: str | os.PathLike) -> DecompressResult:
    """Read and decompress a .SAV file.

    Args:
        path: Filesystem path to the .SAV file.

    Returns:
        DecompressResult with plaintext header and all zlib streams.
    """
    with open(path, "rb") as f:
        data = f.read()

    streams: List[DecompressedStream] = []
    i = 0
    n = len(data)
    first_stream_offset: int | None = None

    while i < n - 1:
        if data[i] == ZLIB_MAGIC_BYTE_1 and data[i + 1] in _VALID_SECOND_BYTES:
            try:
                d = zlib.decompressobj()
                result = d.decompress(data[i:])
                # d.unused_data holds anything past the zlib stream end
                consumed = (n - i) - len(d.unused_data)
                streams.append(
                    DecompressedStream(
                        file_offset=i, compressed_size=consumed, data=result
                    )
                )
                if first_stream_offset is None:
                    first_stream_offset = i
                i += consumed
                continue
            except zlib.error:
                pass
        i += 1

    header_end = first_stream_offset if first_stream_offset is not None else len(data)
    return DecompressResult(
        header=data[:header_end],
        streams=streams,
        raw=data,
    )


# ---- CLI: python -m caplab_save.decompress path/to/PLAY.SAV [out_dir] --------

def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m caplab_save.decompress SAVE.SAV [out_dir]")
        return 2
    save_path = argv[1]
    base = os.path.splitext(os.path.basename(save_path))[0]
    outdir = argv[2] if len(argv) >= 3 else f"saves/decompressed/{base}"
    os.makedirs(outdir, exist_ok=True)

    result = decompress_save(save_path)
    with open(os.path.join(outdir, "header.bin"), "wb") as f:
        f.write(result.header)
    print(f"header: {len(result.header):,} bytes")
    for idx, s in enumerate(result.streams):
        path = os.path.join(outdir, f"blob{idx:02d}.bin")
        with open(path, "wb") as f:
            f.write(s.data)
        print(
            f"blob {idx}: compressed={s.compressed_size:,} at 0x{s.file_offset:x}, "
            f"decompressed={len(s.data):,}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv))
