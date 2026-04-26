"""Deterministic Borland-C LCG replay for Capitalism Lab.

Ghidra decompile of ``FUN_006b0900`` (Misc::random) at 0x006b0900 in
CapMain.exe v11.1.2, confirmed 2026-04-21::

    // this + 0x7C is the uint32 seed field
    *(uint32 *)(this + 0x7C) = *(uint32 *)(this + 0x7C) * 0x015A4E35 + 1;
    ++DAT_00de5bd8;                              // global call counter
    if (n == 0) return 0;
    if (n > 0x7FFF) assert(0, "E:\\CLsrc\\Misc.cpp");
    return ((seed >> 16) & 0x7FFF) * n >> 15;    // uniform in [0, n-1]

This file owns the pure-Python implementation of that primitive plus a
``MiscRNG`` wrapper that records each call for divergence analysis.

Reference facts (from ``RNG_Call_Map.txt §1``)::

    multiplier  0x015A4E35  (22,695,477 — classic Borland C++ constant)
    increment   1
    modulus     2**32
    output      ((seed >> 16) & 0x7FFF) * n >> 15
    counter     global, unconditional increment (including n==0 path)

The counter (DAT_00de5bd8) is NOT persisted in the save file — we only
track the per-instance seed. See ``RNG_Call_Map.txt §6`` for the
counter vs seed discrepancy analysis (the counter also advances on
non-gameplay Misc instances and is not useful for gameplay replay).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Tuple


# LCG constants (public — part of the API so downstream code can reason
# about overflow behaviour if needed).
LCG_MULTIPLIER: int = 0x015A4E35
LCG_INCREMENT: int = 1
LCG_MODULUS: int = 1 << 32
LCG_MODULUS_MASK: int = LCG_MODULUS - 1

# The engine asserts n <= 0x7FFF inside Misc::random. Anything larger is a
# programming error in the engine and should not appear during legitimate
# replay; we surface it explicitly rather than silently wrapping.
MAX_RNG_N: int = 0x7FFF


def lcg_step(seed: int) -> int:
    """Advance the LCG by one tick, returning the next seed.

    Exactly equivalent to the first two lines of ``FUN_006b0900``::

        seed = (seed * 0x015A4E35 + 1) & 0xFFFFFFFF
    """
    return (seed * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MODULUS_MASK


def rng_output(post_step_seed: int, n: int) -> int:
    """Compute the ``random(n)`` output given a *post-step* seed.

    The engine advances the seed first, then reads the high 15 bits to
    produce the output. Callers are expected to have already stepped
    the seed before calling this. See ``MiscRNG.rng`` for the end-to-end
    behaviour that replay code should actually use.

    Raises :class:`ValueError` if ``n`` exceeds ``MAX_RNG_N`` — that would
    trip the engine's own assert, so refusing the call is the safest
    default (a legitimate replay never encounters this case).
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > MAX_RNG_N:
        raise ValueError(
            f"n={n} exceeds engine limit 0x{MAX_RNG_N:04x} — see "
            "FUN_006b0900 assertion at Misc.cpp"
        )
    if n == 0:
        return 0
    return (((post_step_seed >> 16) & 0x7FFF) * n) >> 15


# ---- Instrumented wrapper --------------------------------------------------


@dataclass
class RNGCall:
    """One recorded invocation of :meth:`MiscRNG.rng`.

    ``index``      — zero-based call number inside the trace (0 = first call).
    ``n``          — the ``n`` argument passed to ``random(n)``.
    ``out``        — the returned value, in ``[0, n-1]`` (or 0 if n==0).
    ``seed_after`` — the post-step seed state (= the new value of ``Misc+0x7C``).
    ``label``      — free-form string set by the caller just before the call
                      (see :meth:`MiscRNG.label`). Used for divergence reports.
    """

    index: int
    n: int
    out: int
    seed_after: int
    label: str = ""


@dataclass
class RNGTrace:
    """A sequence of :class:`RNGCall` records.

    Provides helpers for diffing two traces and pinpointing the first
    divergence. Traces are cheap to construct (one dict per call), so
    instrumenting the full daily replay has no meaningful cost for the
    ~500-call-per-day regime we target.
    """

    calls: List[RNGCall] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.calls)

    def __iter__(self) -> Iterator[RNGCall]:
        return iter(self.calls)

    def __getitem__(self, idx):  # type: ignore[override]
        return self.calls[idx]

    def append(self, call: RNGCall) -> None:
        self.calls.append(call)

    @property
    def last_seed(self) -> Optional[int]:
        return self.calls[-1].seed_after if self.calls else None

    def summary(self) -> str:
        if not self.calls:
            return "<empty RNG trace>"
        first, last = self.calls[0], self.calls[-1]
        return (
            f"RNGTrace: {len(self.calls)} calls, "
            f"seed 0x{first.seed_after:08x} -> 0x{last.seed_after:08x}"
        )

    def first_divergence(self, other: "RNGTrace") -> Optional[int]:
        """Return the index of the first mismatched ``seed_after`` (or n/out),
        or ``None`` if the overlapping prefix is identical.

        Length mismatch is also a divergence at ``min(len_self, len_other)``.
        """
        n = min(len(self), len(other))
        for i in range(n):
            a, b = self.calls[i], other.calls[i]
            if a.seed_after != b.seed_after or a.n != b.n or a.out != b.out:
                return i
        if len(self) != len(other):
            return n
        return None


class MiscRNG:
    """Instrumented in-process replica of the game's Misc RNG instance.

    Holds the seed state internally. Call :meth:`rng` with the same argument
    the engine would (``random(n)``), optionally tagging each call with a
    short label via :meth:`label` so divergences can be reported in
    human-readable form.

    Non-goals:
      * We do **not** model the DAT_00de5bd8 global counter — it isn't
        persisted in the save and advances on non-gameplay Misc instances.
      * We do **not** model the case where the engine reseeds the Misc
        (``FUN_006b0970`` / ``FUN_006b0e40``) mid-replay. Those calls only
        fire on new-game / scenario-start, which are outside the daily tick.

    Thread safety: none. The game is single-threaded for gameplay RNG; we
    follow suit and avoid locking overhead.
    """

    __slots__ = ("_seed", "_trace", "_pending_label", "_trace_enabled")

    def __init__(self, seed: int, *, trace: bool = True):
        if not (0 <= seed < LCG_MODULUS):
            raise ValueError(f"seed must be a uint32, got {seed!r}")
        self._seed: int = seed
        self._trace_enabled: bool = trace
        self._trace: RNGTrace = RNGTrace()
        self._pending_label: str = ""

    # -- state access --------------------------------------------------------

    @property
    def seed(self) -> int:
        """Current seed (= value at Misc+0x7C)."""
        return self._seed

    @property
    def trace(self) -> RNGTrace:
        return self._trace

    def set_seed(self, seed: int) -> None:
        """Explicit reseed. Mirrors ``FUN_006b0e40`` (Misc::set_seed).

        Note: the engine also zeroes the global counter on set_seed; we
        don't track that because we don't model the counter.
        """
        if not (0 <= seed < LCG_MODULUS):
            raise ValueError(f"seed must be a uint32, got {seed!r}")
        self._seed = seed

    # -- instrumentation helpers --------------------------------------------

    def label(self, label: str) -> None:
        """Attach a label to the next :meth:`rng` call.

        Labels are a debugging aid for divergence reports — e.g. call
        ``rng.label("firm#42 daily_update gate")`` before a single
        ``rng.rng(120)`` and the trace will carry that string.
        """
        self._pending_label = label

    # -- primary entry point -------------------------------------------------

    def rng(self, n: int) -> int:
        """Equivalent to ``Misc::random(n)``. Returns a value in ``[0, n-1]``.

        Follows ``FUN_006b0900`` exactly: step seed, increment (mocked)
        counter, then apply the output formula. Records the call into the
        trace if instrumentation is enabled.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n > MAX_RNG_N:
            # Engine would assert; mirror that as an exception so replay
            # bugs surface immediately rather than producing wrong output.
            raise ValueError(
                f"n={n} exceeds engine limit 0x{MAX_RNG_N:04x}; this would "
                "trip the assertion inside FUN_006b0900"
            )

        # Step seed — unconditional, even on n==0.
        self._seed = lcg_step(self._seed)

        if n == 0:
            out = 0
        else:
            out = (((self._seed >> 16) & 0x7FFF) * n) >> 15

        if self._trace_enabled:
            self._trace.append(
                RNGCall(
                    index=len(self._trace.calls),
                    n=n,
                    out=out,
                    seed_after=self._seed,
                    label=self._pending_label,
                )
            )
        # Always clear the pending label so it doesn't leak into the
        # next call by accident.
        self._pending_label = ""
        return out

    # -- convenience helpers -------------------------------------------------

    def advance(self, count: int) -> None:
        """Advance the seed by ``count`` LCG steps without tracing.

        Useful for skipping over known-deterministic call blocks when
        debugging. Does NOT record calls in the trace.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        s = self._seed
        for _ in range(count):
            s = lcg_step(s)
        self._seed = s

    def drive(self, ns: Iterable[int], labels: Optional[Iterable[str]] = None) -> List[int]:
        """Invoke ``rng(n)`` for each ``n`` in ``ns``, returning the outputs.

        Convenience for scripts that want to burn a fixed sequence of
        ``n`` values (e.g. test harnesses replaying a captured call list).
        """
        outs: List[int] = []
        if labels is None:
            for n in ns:
                outs.append(self.rng(int(n)))
        else:
            for n, lbl in zip(ns, labels):
                self.label(str(lbl))
                outs.append(self.rng(int(n)))
        return outs


# ---- pure math helpers (side-effect-free) ---------------------------------


def simulate_lcg(seed: int, count: int) -> int:
    """Return the seed after ``count`` LCG steps from ``seed``.

    No instrumentation, no output values — just the next-state calculation.
    Useful for "if the game made 449 calls, where does the seed land?"
    sanity checks.
    """
    s = seed & LCG_MODULUS_MASK
    for _ in range(count):
        s = (s * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MODULUS_MASK
    return s


def distance_if_reachable(start: int, target: int, max_steps: int) -> Optional[int]:
    """Return ``k`` such that ``simulate_lcg(start, k) == target``, or None.

    Bounded linear search up to ``max_steps``. Intended for validation:
    given an observed seed advance between two save snapshots, how many
    Misc::random() calls actually happened?

    The LCG has period 2**32, so there is always a unique ``k`` in
    ``[0, 2**32)``, but ``max_steps`` should be kept small (a few
    thousand) to fail fast on mismatches.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    s = start & LCG_MODULUS_MASK
    target &= LCG_MODULUS_MASK
    if s == target:
        return 0
    for i in range(1, max_steps + 1):
        s = (s * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MODULUS_MASK
        if s == target:
            return i
    return None
