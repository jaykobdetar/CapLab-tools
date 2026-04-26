#!/usr/bin/env python3
"""Capitalism Lab v11.1.2 — always-focused live-memory patch.

Stand-alone. No dependencies outside the Python stdlib. No imports from
``caplab_mcp``. Apply, revert, or inspect the single-byte patch that
neutralizes the game's pause-on-focus-loss behavior.

Background
----------
The game's main loops all contain a wait-for-focus spin on
``audio_muted_flag`` at ``0x0090D489``:

    while (audio_muted_flag != 0 || DAT_0090d488 == 0) {
        FUN_00655f40();   // pump SDL events
    }

The flag is set to 1 by exactly one instruction, inside the
SDL_WINDOWEVENT_FOCUS_LOST / MINIMIZED handler of ``FUN_00655f40`` at
``0x0065601E``:

    MOV byte ptr [0x0090D489], 0x01

Flipping the immediate byte (at file/memory offset ``0x00656024``) from
``0x01`` to ``0x00`` turns that instruction into a no-op write of 0,
the flag never transitions to 1, and the game keeps ticking regardless
of window focus.

The patch is live-memory only — it reverts automatically when the game
process exits. Runs via ``/proc/<pid>/mem`` which bypasses the .text
segment's page permissions for same-UID callers (no mprotect or ptrace
needed).

Usage
-----
    python focus_patch.py status          # show current patch + flag state
    python focus_patch.py apply           # set game to always-focused
    python focus_patch.py revert          # restore stock behavior
    python focus_patch.py apply --pid 12345   # pin a specific CapMain PID

The full research trail for why this one byte is sufficient is in the
block comment at the top of this file. Key evidence:

  * No Windows focus APIs are imported (CapMain has only SDL and
    the basic Kernel32/User32 set) — focus must route through SDL.
  * SDL_PollEvent has exactly one caller, the event pump at
    FUN_00655f40.
  * Decompiling that function's SDL_WINDOWEVENT switch shows cases
    7 (MINIMIZED) and 0xD (FOCUS_LOST) set `audio_muted_flag = 1`.
  * Xref-listing writes to the flag yields exactly two sites, both
    inside FUN_00655f40.
  * Decompiling multiple reader loops (CApp__MainMenuLoop,
    FUN_004e49d0 end-game loop, etc.) shows the canonical pattern
    `while (audio_muted_flag || DAT_0090d488 == 0) pump_events()`.
  * Flipping the immediate byte at 0x00656024 from 0x01 to 0x00 was
    verified live to stop the flag transitioning to 1 and to keep
    the game ticking while the window was unfocused.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Patch constants
# ---------------------------------------------------------------------------

IMAGE_BASE = 0x00400000           # PE default image base

FOCUS_PATCH_VA       = 0x00656024  # immediate byte of `MOV [flag], imm8`
FOCUS_PATCH_ORIGINAL = 0x01        # stock game: sets flag to 1 on FOCUS_LOST
FOCUS_PATCH_APPLIED  = 0x00        # patched:    sets flag to 0 (no-op)

AUDIO_MUTED_FLAG_VA  = 0x0090D489  # byte — 1=unfocused/paused, 0=focused


# ---------------------------------------------------------------------------
# Process discovery (lifted from caplab_mcp.reader, standalone)
# ---------------------------------------------------------------------------

_MAPS_LINE_RE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(.*)$"
)


def find_capmain_pid() -> Optional[int]:
    """Return the PID of a running `CapMain.exe` (Wine), else None."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                if "CapMain" in f.read():
                    return pid
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"CapMain" in f.read():
                    return pid
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return None


def find_capmain_base(pid: int) -> Optional[int]:
    """Return the base mapping address of `CapMain.exe` in the given PID."""
    try:
        with open(f"/proc/{pid}/maps", "r") as f:
            for line in f:
                m = _MAPS_LINE_RE.match(line.strip())
                if not m:
                    continue
                start_hex, _end_hex, perms, path = m.groups()
                if "CapMain" in path and ("r-xp" in perms or "r--p" in perms):
                    return int(start_hex, 16)
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        pass
    return None


# ---------------------------------------------------------------------------
# /proc/<pid>/mem primitives
# ---------------------------------------------------------------------------

def read_u8(pid: int, addr: int) -> int:
    with open(f"/proc/{pid}/mem", "rb") as f:
        f.seek(addr)
        return f.read(1)[0]


def write_u8(pid: int, addr: int, value: int) -> None:
    """Write one byte to the target process.

    `/proc/<pid>/mem` (opened in `wb`) bypasses page-level write
    protections on the kernel side for same-UID callers, so this works
    against the PE `.text` segment without any mprotect or ptrace dance.
    """
    with open(f"/proc/{pid}/mem", "wb") as f:
        f.seek(addr)
        f.write(bytes([value]))


# ---------------------------------------------------------------------------
# Patch ops
# ---------------------------------------------------------------------------

class PatchError(Exception):
    pass


def resolve_process(pid: Optional[int]) -> tuple[int, int]:
    """Return (pid, image_base). Raises PatchError with a human-readable
    message on any step that fails."""
    if pid is None:
        pid = find_capmain_pid()
        if pid is None:
            raise PatchError(
                "CapMain.exe not found in /proc. Is the game running under Wine?"
            )
    base = find_capmain_base(pid)
    if base is None:
        raise PatchError(
            f"Could not locate the CapMain.exe mapping for PID {pid}. "
            "Check /proc/{pid}/maps — is the process still alive?"
        )
    return pid, base


def read_patch_site(pid: int, base: int) -> int:
    """Read the current byte at the patch site, rebased."""
    return read_u8(pid, base + (FOCUS_PATCH_VA - IMAGE_BASE))


def read_focus_flag(pid: int, base: int) -> int:
    """Read the current live focus flag (1 = paused/unfocused, 0 = focused)."""
    return read_u8(pid, base + (AUDIO_MUTED_FLAG_VA - IMAGE_BASE))


def apply_patch(pid: int, base: int) -> dict:
    """Set the patch byte to APPLIED and force the live flag to 0."""
    current = read_patch_site(pid, base)
    if current not in (FOCUS_PATCH_ORIGINAL, FOCUS_PATCH_APPLIED):
        raise PatchError(
            f"unexpected byte 0x{current:02x} at patch site "
            f"0x{FOCUS_PATCH_VA:08x}; expected 0x01 or 0x00. "
            "Binary may have changed — refusing to write."
        )
    flag_before = read_focus_flag(pid, base)
    wrote_patch = False
    if current != FOCUS_PATCH_APPLIED:
        write_u8(pid, base + (FOCUS_PATCH_VA - IMAGE_BASE), FOCUS_PATCH_APPLIED)
        wrote_patch = True
    # Force any in-flight pause to end immediately.
    if flag_before == 1:
        write_u8(pid, base + (AUDIO_MUTED_FLAG_VA - IMAGE_BASE), 0)
    after = read_patch_site(pid, base)
    flag_after = read_focus_flag(pid, base)
    return {
        "wrote_patch": wrote_patch,
        "patch_byte_before": current,
        "patch_byte_after": after,
        "flag_before": flag_before,
        "flag_after": flag_after,
    }


def revert_patch(pid: int, base: int) -> dict:
    """Set the patch byte back to ORIGINAL. Leaves the live flag alone."""
    current = read_patch_site(pid, base)
    if current not in (FOCUS_PATCH_ORIGINAL, FOCUS_PATCH_APPLIED):
        raise PatchError(
            f"unexpected byte 0x{current:02x} at patch site "
            f"0x{FOCUS_PATCH_VA:08x}; expected 0x01 or 0x00."
        )
    wrote_patch = False
    if current != FOCUS_PATCH_ORIGINAL:
        write_u8(pid, base + (FOCUS_PATCH_VA - IMAGE_BASE), FOCUS_PATCH_ORIGINAL)
        wrote_patch = True
    after = read_patch_site(pid, base)
    return {
        "wrote_patch": wrote_patch,
        "patch_byte_before": current,
        "patch_byte_after": after,
    }


def status(pid: int, base: int) -> dict:
    """Return the current on-disk (i.e. in-memory) patch + flag state."""
    byte = read_patch_site(pid, base)
    flag = read_focus_flag(pid, base)
    return {
        "patch_byte": byte,
        "patch_applied": byte == FOCUS_PATCH_APPLIED,
        "focus_flag": flag,
        "game_state": "running" if flag == 0 else "paused-on-focus-loss",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_state(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, int) and k.startswith("patch_byte"):
            lines.append(f"  {k:18s} = 0x{v:02x}")
        elif isinstance(v, int) and k.startswith("flag"):
            lines.append(f"  {k:18s} = {v}  ({'paused' if v else 'running'})")
        else:
            lines.append(f"  {k:18s} = {v}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Capitalism Lab v11.1.2 — always-focused live-memory patch. "
            "Flips one byte at 0x00656024 so the game's FOCUS_LOST "
            "handler becomes a no-op. Reversible; auto-reverts on game "
            "exit."
        ),
    )
    p.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "apply", "revert"],
        help="what to do (default: status)",
    )
    p.add_argument(
        "--pid", type=int, default=None,
        help="pin a specific PID (default: auto-discover CapMain.exe)",
    )
    args = p.parse_args(argv)

    try:
        pid, base = resolve_process(args.pid)
    except PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[focus_patch] PID {pid}, image base 0x{base:08x}")

    try:
        if args.action == "status":
            print(_fmt_state(status(pid, base)))
        elif args.action == "apply":
            print(_fmt_state(apply_patch(pid, base)))
            print("-> patch applied (game is now always-focused)")
        elif args.action == "revert":
            print(_fmt_state(revert_patch(pid, base)))
            print("-> patch reverted (stock pause-on-focus-loss behavior restored)")
    except PermissionError as e:
        print(
            f"error: permission denied accessing /proc/{pid}/mem. "
            f"You must run this as the same user that started CapMain.exe, "
            f"or as root. Details: {e}",
            file=sys.stderr,
        )
        return 2
    except PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
