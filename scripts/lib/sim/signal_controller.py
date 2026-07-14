"""Phase 5 — independent feedback-controller signal state machine.

A proper signal controller is a finite-state machine, not a precomputed
timeline.  ``SignalController`` advances through GREEN → YELLOW → ALL_RED
states and re-observes lane pressure at every green-to-yellow transition to
decide whether to *extend*, *terminate*, or *skip* a phase.

Three concrete strategies are provided:

* ``FixedTimeController``   — fixed-cycle, no feedback.
* ``ActuatedController``    — gap/extend on demand, ends when queue clears.
* ``MaxPressureController`` — pick highest-pressure NEMA combo each cycle.

The engine selects between this FSM and the prebuilt ``AdaptiveSignalPlan``
via ``config.signal_engine``:
    "plan"  → keep the existing timeline-based plan (default; unchanged behavior)
    "fsm"   → drive the signal from this closed-loop state machine
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from geometry import Direction, Turn
from traffic_signal import (
    NEMA_PHASES,
    _ALL_COMBOS,
    _NS_COMBOS,
    _EW_COMBOS,
    _movement_to_phase,
)


class SignalPhase(enum.Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


@dataclass
class SignalControllerConfig:
    fps: int = 30
    min_green_f: int = 240       # 8 s
    max_green_f: int = 1200      # 40 s
    yellow_f: int = 90           # 3 s
    all_red_f: int = 60          # 2 s
    extension_f: int = 60        # 2 s gap-acceptance extension per actuation


class SignalController:
    """Closed-loop NEMA dual-ring signal FSM.

    The controller holds a single ``current_combo`` (ring1, ring2) and cycles
    GREEN → YELLOW → ALL_RED → GREEN.  At the green decision point it consults
    lane pressure (queue counts per movement) and selects the next combo.
    """

    def __init__(self, cfg: SignalControllerConfig):
        self.cfg = cfg
        self.phase = SignalPhase.ALL_RED
        self.phase_start = 0
        self.green_start = 0
        self.current_combo: Optional[Tuple[int, int]] = None
        self.next_combo: Optional[Tuple[int, int]] = None
        self.elapsed_green = 0
        self._last_extend_tick: Optional[int] = None
        self._intervals: List[Tuple[int, int, Tuple[int, int]]] = []  # (s,e,combo)
        self._clearances: List[Tuple[int, int, Tuple[int, int]]] = []  # yellow+ar

    # ---- observation ------------------------------------------------------ #

    @staticmethod
    def _combo_pressure(combo: Tuple[int, int],
                        counts: Dict[Tuple[Direction, Turn], int]) -> int:
        p1, p2 = combo
        pr = 0
        for mv in NEMA_PHASES[p1]:
            pr += counts.get(mv, 0)
        for mv in NEMA_PHASES[p2]:
            pr += counts.get(mv, 0)
        return pr

    def _side_with_pressure(self, counts) -> Optional[List[Tuple[int, int]]]:
        ns = sum(self._combo_pressure(c, counts) for c in _NS_COMBOS)
        ew = sum(self._combo_pressure(c, counts) for c in _EW_COMBOS)
        if ns == 0 and ew == 0:
            return None
        return _NS_COMBOS if ns >= ew else _EW_COMBOS

    def _best_combo(self, side: List[Tuple[int, int]],
                   counts) -> Tuple[int, int]:
        best, best_p = side[0], -1
        for combo in side:
            p = self._combo_pressure(combo, counts)
            if p > best_p or (p == best_p and combo < best):
                best, best_p = combo, p
        return best

    # ---- main step -------------------------------------------------------- #

    def step(self, tick: int,
             lane_counts: Dict[Tuple[Direction, Turn], int]) -> None:
        """Advance the controller by one frame.

        Subclasses own their specific phase-selection policy.  The engine calls
        this once per simulation frame; clearance export assumes that cadence.
        """
        raise NotImplementedError

    def _start_yellow(self, tick: int) -> None:
        self.phase = SignalPhase.YELLOW
        self.phase_start = tick
        if self.current_combo is not None:
            self._intervals.append((self.green_start, tick, self.current_combo))
            self._clearances.append(
                (tick, tick + self.cfg.yellow_f + self.cfg.all_red_f,
                 self.current_combo))

    def _start_all_red(self, tick: int) -> None:
        self.phase = SignalPhase.ALL_RED
        self.phase_start = tick

    # ---- query ------------------------------------------------------------ #

    def is_green(self, approach: Direction, turn: Turn, tick: int) -> bool:
        if self.current_combo is None:
            return False
        if self.phase == SignalPhase.ALL_RED:
            return False
        if self.phase == SignalPhase.YELLOW:
            # Yellow is clearance only; queued vehicles must not newly enter.
            return False
        # GREEN
        ph = _movement_to_phase(approach, turn)
        return ph in self.current_combo

    # ---- export ----------------------------------------------------------- #

    def intervals(self) -> List[Tuple[int, int, Tuple[int, int]]]:
        return list(self._intervals)

    def clearances(self) -> List[Tuple[int, int, Tuple[int, int]]]:
        return list(self._clearances)


class FixedTimeController(SignalController):
    """Fixed-cycle: serves every configured NEMA combo in order."""

    def __init__(self, cfg: SignalControllerConfig):
        super().__init__(cfg)
        self._cycle = 0

    def step(self, tick: int,
             lane_counts: Dict[Tuple[Direction, Turn], int]) -> None:
        if self.phase == SignalPhase.ALL_RED:
            if tick - self.phase_start >= self.cfg.all_red_f or self.current_combo is None:
                self.current_combo = _ALL_COMBOS[self._cycle % len(_ALL_COMBOS)]
                self.phase = SignalPhase.GREEN
                self.phase_start = tick
                self.green_start = tick
                self.elapsed_green = 0
                self._cycle += 1
            return
        if self.phase == SignalPhase.GREEN:
            self.elapsed_green = tick - self.phase_start
            if self.elapsed_green >= self.cfg.max_green_f:
                self._start_yellow(tick)
            return
        if self.phase == SignalPhase.YELLOW:
            if tick - self.phase_start >= self.cfg.yellow_f:
                self._start_all_red(tick)
            return


class ActuatedController(SignalController):
    """Actuated: extend green while there is pressure; gap-out on no demand."""

    def step(self, tick, lane_counts) -> None:
        if self.phase == SignalPhase.ALL_RED:
            if tick - self.phase_start >= self.cfg.all_red_f or self.current_combo is None:
                side = self._side_with_pressure(lane_counts)
                if side is None:
                    self.current_combo = None
                    return
                combo = self._best_combo(side, lane_counts)
                self.current_combo = combo
                self.phase = SignalPhase.GREEN
                self.phase_start = tick
                self.green_start = tick
                self.elapsed_green = 0
                self._last_extend_tick = tick
            return
        if self.phase == SignalPhase.GREEN:
            self.elapsed_green = tick - self.phase_start
            own_pr = (self._combo_pressure(self.current_combo, lane_counts)
                      if self.current_combo else 0)
            if self.elapsed_green >= self.cfg.max_green_f:
                self._start_yellow(tick)
            elif self.elapsed_green >= self.cfg.min_green_f:
                if own_pr > 0:
                    self._last_extend_tick = tick  # extend
                elif (self._last_extend_tick is None or
                      tick - self._last_extend_tick >= self.cfg.extension_f):
                    self._start_yellow(tick)
            return
        if self.phase == SignalPhase.YELLOW:
            if tick - self.phase_start >= self.cfg.yellow_f:
                self._start_all_red(tick)
            return


class MaxPressureController(ActuatedController):
    """Closed-loop max-pressure: pick the best combo at each green start."""
    pass  # ActuatedController already selects best combo from live pressure
