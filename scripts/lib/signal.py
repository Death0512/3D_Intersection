"""Traffic signal phase timing (SPaT) — pure Python.

Fixed-cycle permissive-left plan:
  NS-green(30s) → NS-yellow(3s) → all-red(2s) → EW-green(30s) → EW-yellow(3s) → all-red(2s)

Left turns share the through-green phase (permissive).  Cycle = 70 s.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple

try:
    from . import geometry as G
except ImportError:
    import geometry as G


# Phase durations in seconds
_GREEN_S = 30
_YELLOW_S = 3
_ALL_RED_S = 2

_PHASE_DEFS: List[Tuple[str, int]] = [
    ("NS_GREEN",  _GREEN_S),
    ("NS_YELLOW", _YELLOW_S),
    ("ALL_RED",   _ALL_RED_S),
    ("EW_GREEN",  _GREEN_S),
    ("EW_YELLOW", _YELLOW_S),
    ("ALL_RED",   _ALL_RED_S),
]


class SignalPlan:
    """A fixed-cycle traffic signal plan.

    Args:
        fps: frames per second (must match scenario fps).
    """
    def __init__(self, fps: int = 30):
        self.fps = fps
        # phase boundaries in frames (cumulative)
        cumulative = 0
        self._boundaries: List[Tuple[int, str]] = []  # (start_frame, phase_name)
        for name, sec in _PHASE_DEFS:
            self._boundaries.append((cumulative, name))
            cumulative += int(round(sec * fps))
        self.cycle_frames = cumulative

    def _phase_at(self, frame: int) -> str:
        frame_mod = frame % self.cycle_frames
        for i in range(len(self._boundaries) - 1, -1, -1):
            if frame_mod >= self._boundaries[i][0]:
                return self._boundaries[i][1]
        return self._boundaries[0][1]

    def _next_boundary(self, frame: int) -> int:
        """Frame of the next phase boundary >= frame."""
        frame_mod = frame % self.cycle_frames
        for i in range(len(self._boundaries)):
            if self._boundaries[i][0] > frame_mod:
                return frame - frame_mod + self._boundaries[i][0]
        # wraps to next cycle
        return frame - frame_mod + self.cycle_frames

    def is_green(self, approach: G.Direction, turn: G.Turn, frame: int) -> bool:
        """True if the approach+turn has a green signal at *frame*."""
        phase = self._phase_at(frame)
        is_ns = approach in (G.Direction.N, G.Direction.S)
        if is_ns and phase == "NS_GREEN":
            return True
        if not is_ns and phase == "EW_GREEN":
            return True
        return False

    def next_green_frame(self, approach: G.Direction, turn: G.Turn,
                         frame: int) -> int:
        """Earliest frame >= *frame* when *is_green* becomes true."""
        f = frame
        for _ in range(100):
            if self.is_green(approach, turn, f):
                return f
            f = self._next_boundary(f)
        raise RuntimeError(
            f"no green found for {approach} {turn} in {100} attempts "
            f"(cycle={self.cycle_frames}, start={frame})")

    def next_red_frame(self, approach: G.Direction, turn: G.Turn,
                       frame: int) -> int:
        """Earliest frame >= *frame* when *is_green* becomes false."""
        f = frame
        for _ in range(100):
            if not self.is_green(approach, turn, f):
                return f
            f = self._next_boundary(f)
        return f
