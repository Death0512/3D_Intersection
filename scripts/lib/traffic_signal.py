"""Traffic signal phase timing (SPaT) — pure Python.

Two signal controllers are provided:

1. ``SignalPlan`` — fixed-cycle permissive-left plan:
     NS-green(30s) → NS-yellow(3s) → all-red(2s) → EW-green(30s) → EW-yellow(3s) → all-red(2s)
   Left turns share the through-green phase (permissive).  Cycle = 70 s.
   Selected via ``--signal-mode fixed``.

2. ``AdaptiveSignalPlan`` — NEMA 8-phase dual-ring controller with a
   MaxPressure phase selector.  Movements are organised into the eight
   standard NEMA phases; the controller serves one compatible phase pair
   (one ring-1 phase + one ring-2 phase, separated by the ring barrier) at a
   time.  Each served combo's duration is bounded by min/max green; the
   combo with the highest queue *pressure* (upstream queue − downstream
   queue summed over its movements) is selected each cycle.  Between combos
   a mandatory yellow + all-red clearance is inserted.  Selected via
   ``--signal-mode adaptive``.  The plan is built as an explicit
   frame → phase timeline from realised arrivals, so ``is_green`` /
   ``next_green_frame`` are deterministic lookups.  The v2 microsim owns live
   queue discharge; this class remains the timeline/export oracle.
"""
from __future__ import annotations

import bisect
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

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


# ============================================================================
# NEMA 8-phase dual-ring controller + MaxPressure adaptive selector
# ============================================================================
#
# Phase numbering (standard NEMA):
#
#   Ring 1:  Φ1 (S left)    | Φ2 (S thru/rt)                  barrier
#            Φ3 (E left)    | Φ4 (E thru/rt)                  barrier
#   Ring 2:  Φ5 (N left)    | Φ6 (N thru/rt)                  barrier
#            Φ7 (W left)    | Φ8 (W thru/rt)                  barrier
#
# The ring barrier separates the N/S approach group from the E/W group:
# phases 1-2 and 5-6 serve N/S; phases 3-4 and 7-8 serve E/W.  A combo is a
# pair of one ring-1 phase and one ring-2 phase from the SAME side of the
# barrier (so the two movements never conflict).  The legal combos are:
#
#   N/S side:  {1,5} {1,6} {2,5} {2,6}   (left-left, left-thru, ...)
#   E/W side:  {3,7} {3,8} {4,7} {4,8}
#
# MaxPressure serves, on each barrier side, the combo with the highest total
# queue pressure (upstream queue minus downstream queue) subject to
# min-green / max-green / clearance constraints.

# Phase -> { (approach, turn) } movements served while that phase is green.
NEMA_PHASES: Dict[int, Set[Tuple[G.Direction, G.Turn]]] = {
    1: {(G.Direction.S, G.Turn.LEFT)},
    2: {(G.Direction.S, G.Turn.STRAIGHT), (G.Direction.S, G.Turn.RIGHT)},
    3: {(G.Direction.E, G.Turn.LEFT)},
    4: {(G.Direction.E, G.Turn.STRAIGHT), (G.Direction.E, G.Turn.RIGHT)},
    5: {(G.Direction.N, G.Turn.LEFT)},
    6: {(G.Direction.N, G.Turn.STRAIGHT), (G.Direction.N, G.Turn.RIGHT)},
    7: {(G.Direction.W, G.Turn.LEFT)},
    8: {(G.Direction.W, G.Turn.STRAIGHT), (G.Direction.W, G.Turn.RIGHT)},
}

# Ring membership.
_RING1 = {1, 2, 3, 4}
_RING2 = {5, 6, 7, 8}

# Barrier sides: N/S = phases {1,2,5,6}; E/W = phases {3,4,7,8}.
_NS_SIDE = {1, 2, 5, 6}
_EW_SIDE = {3, 4, 7, 8}

# Legal non-conflicting combos (ring-1 phase, ring-2 phase) per barrier side.
_NS_COMBOS: List[Tuple[int, int]] = [(1, 5), (1, 6), (2, 5), (2, 6)]
_EW_COMBOS: List[Tuple[int, int]] = [(3, 7), (3, 8), (4, 7), (4, 8)]
_ALL_COMBOS: List[Tuple[int, int]] = _NS_COMBOS + _EW_COMBOS


def _movement_to_phase(approach: G.Direction, turn: G.Turn) -> int:
    """Return the NEMA phase serving (approach, turn). Protected lefts are
    served by that approach's left phase; through/right by the through phase.
    """
    _L = {
        (G.Direction.N, G.Turn.LEFT): 5,
        (G.Direction.S, G.Turn.LEFT): 1,
        (G.Direction.E, G.Turn.LEFT): 3,
        (G.Direction.W, G.Turn.LEFT): 7,
    }
    _TR = {
        (G.Direction.N, G.Turn.STRAIGHT): 6, (G.Direction.N, G.Turn.RIGHT): 6,
        (G.Direction.S, G.Turn.STRAIGHT): 2, (G.Direction.S, G.Turn.RIGHT): 2,
        (G.Direction.E, G.Turn.STRAIGHT): 4, (G.Direction.E, G.Turn.RIGHT): 4,
        (G.Direction.W, G.Turn.STRAIGHT): 8, (G.Direction.W, G.Turn.RIGHT): 8,
    }
    if turn == G.Turn.LEFT:
        return _L[(approach, turn)]
    return _TR[(approach, turn)]


# Adaptive controller timing defaults (seconds) — comfortable urban values.
_ADP_MIN_GREEN_S = 8.0
_ADP_MAX_GREEN_S = 40.0
_ADP_YELLOW_S = 3.0
_ADP_ALL_RED_S = 2.0
_ADP_LOOKAHEAD_S = 120.0   # arrival-pressure window per cycle decision


class AdaptiveSignalPlan:
    """NEMA 8-phase dual-ring adaptive controller (MaxPressure selector).

    Builds an explicit frame -> active-combo timeline from realised arrivals
    so ``is_green`` / ``next_green_frame`` are deterministic lookups.  The
    v2 microsim owns live discharge; this class remains useful for exporting
    and querying an explicit signal timeline after realised arrivals are known.

    Args:
        fps: frames per second (must match scenario fps).
        arrivals: optional mapping ``(approach, turn) -> sorted list of
            stop-line arrival frames``.  When provided, the plan is built
            immediately; otherwise call ``rebuild(arrivals, horizon_frames)``
            once arrivals are known.  An empty arrivals map yields a
            round-robin plan (each combo served at min-green in turn) so a
            signal still exists even with no demand.
        horizon_frames: optional total run length in frames; caps plan
            generation.  When omitted, defaults to a single long horizon
            derived from the arrivals (or 1 hour if no arrivals).
    """

    def __init__(self, fps: int = 30,
                 arrivals: Optional[Dict[Tuple[G.Direction, G.Turn], List[int]]] = None,
                 horizon_frames: Optional[int] = None):
        self.fps = fps
        self.cycle_frames = None  # NEMA plans are not fixed-cycle.
        self.min_green_f = int(round(_ADP_MIN_GREEN_S * fps))
        self.max_green_f = int(round(_ADP_MAX_GREEN_S * fps))
        self.yellow_f = int(round(_ADP_YELLOW_S * fps))
        self.all_red_f = int(round(_ADP_ALL_RED_S * fps))
        self.lookahead_f = int(round(_ADP_LOOKAHEAD_S * fps))
        # intervals: list of (start_f, end_f, combo) with end exclusive.
        # Between consecutive intervals there is a clearance gap
        # (yellow + all-red) recorded in self._clearances.
        self.intervals: List[Tuple[int, int, Tuple[int, int]]] = []
        self._clearances: List[Tuple[int, int]] = []
        if arrivals is not None:
            self.rebuild(arrivals, horizon_frames)

    # ---- pressure / selection ------------------------------------------------

    def _queue_counts(self, arrivals: Dict[Tuple[G.Direction, G.Turn], List[int]],
                      lo: int, hi: int) -> Dict[Tuple[G.Direction, G.Turn], int]:
        """Number of arrivals per movement in window [lo, hi)."""
        counts: Dict[Tuple[G.Direction, G.Turn], int] = {}
        for mv, ts in arrivals.items():
            n = sum(1 for t in ts if lo <= t < hi)
            counts[mv] = n
        return counts

    def _combo_pressure(self, combo: Tuple[int, int],
                        counts: Dict[Tuple[G.Direction, G.Turn], int]) -> float:
        """Total upstream queue pressure for a combo (sum over its movements).

        Downstream pressure is approximated as a uniform per-phase constant
        (lane capacity is not modelled here); since all phases share the same
        downstream baseline it cancels out in selection, so the sum of
        upstream queue counts is a faithful MaxPressure ranking signal.
        """
        p1, p2 = combo
        total = 0.0
        for mv in NEMA_PHASES[p1]:
            total += counts.get(mv, 0)
        for mv in NEMA_PHASES[p2]:
            total += counts.get(mv, 0)
        return total

    def _select_combo(self, side_combos: List[Tuple[int, int]],
                      counts: Dict[Tuple[G.Direction, G.Turn], int]
                      ) -> Tuple[int, int]:
        """Pick the highest-pressure combo on one barrier side.  Ties broken
        by lowest phase number (deterministic) — important for fixpoint
        stability."""
        best = None
        best_p = -1.0
        for combo in side_combos:
            p = self._combo_pressure(combo, counts)
            if p > best_p or (p == best_p and best is not None and combo < best):
                best_p = p
                best = combo
        # When ALL combos have zero pressure (no demand), fall back to the
        # lowest-numbered combo on this side so the plan is still a valid
        # rotation even with empty arrivals.
        if best is None:
            best = side_combos[0]
        return best

    # ---- timeline build ------------------------------------------------------

    def _serve_combo(self, start_f: int, combo: Tuple[int, int],
                     counts: Dict[Tuple[G.Direction, G.Turn], int]) -> int:
        """Green duration for a combo given current queue pressure.

        Duration = min(max_green, max(min_green, expected_discharge_time)),
        where expected_discharge_time is the queue / saturation_flow.  A
        queue-free phase gets min_green.  This keeps the signal responsive:
        heavy demand extends up to max-green; light demand gap-outs early via
        min-green.  Returns the end frame (exclusive)."""
        # saturation flow ~ 1800 veh/h = 0.5 veh/s/lane; 2 phases served
        # in parallel so discharge ~ 1 veh/s for the combo.
        total_queue = 0
        for ph in combo:
            for mv in NEMA_PHASES[ph]:
                total_queue += counts.get(mv, 0)
        sat_flow_vps = 1.0
        if total_queue <= 0:
            dur = self.min_green_f
        else:
            discharge_s = total_queue / sat_flow_vps
            dur_f = int(round(discharge_s * self.fps))
            dur = max(self.min_green_f, min(self.max_green_f, dur_f))
        return start_f + dur

    def rebuild(self, arrivals: Dict[Tuple[G.Direction, G.Turn], List[int]],
                horizon_frames: Optional[int] = None):
        """Rebuild the frame -> combo timeline from the given arrivals.

        Alternates N/S and E/W barrier sides (mandatory barrier), selecting
        the highest-pressure combo on each side at each cycle.  Inserts
        yellow + all-red clearance between consecutive served combos.

        **Pressure** = *accumulated unserved queue* per movement: arrivals up
        to ``t`` that have not yet been discharged by a previously-served
        interval for that movement's phase. This ensures vehicles that
        arrived during a red on the opposite side still drive a later green
        (no starvation) — the central real-world property of MaxPressure.
        """
        self.intervals = []
        self._clearances = []
        if horizon_frames is None:
            max_arrival = 0
            for ts in arrivals.values():
                if ts:
                    max_arrival = max(max_arrival, ts[-1])
            horizon_frames = max(max_arrival + 10 * self.max_green_f,
                                  10 * self.max_green_f)

        # Per-movement served-pointer: index into arrivals[mv] of the next
        # NOT-yet-discharged arrival.  Each combo serving phase p discharges
        # the arrivals of all movements served by phase p that fall within
        # [t_start, t_end) of that movement's arrival list (queued or just-
        # arrived are both discharged by the green window).
        served_ptr: Dict[Tuple[G.Direction, G.Turn], int] = {
            mv: 0 for mv in arrivals
        }

        def _queue_at(t: int) -> Dict[Tuple[G.Direction, G.Turn], int]:
            """Unserved arrivals up to time t per movement."""
            q: Dict[Tuple[G.Direction, G.Turn], int] = {}
            for mv, ts in arrivals.items():
                ptr = served_ptr.get(mv, 0)
                # arrivals are sorted; count unserved arrivals <= t in O(log n)
                # instead of rescanning the tail every signal decision.
                q[mv] = bisect.bisect_right(ts, t, lo=ptr) - ptr
            return q

        def _discharge_until(combo: Tuple[int, int], t_end: int):
            """Advance served_ptr for all movements served by this combo to
            the first arrival strictly past t_end."""
            for ph in combo:
                for mv in NEMA_PHASES[ph]:
                    if mv not in served_ptr:
                        continue
                    ts = arrivals.get(mv, [])
                    ptr = served_ptr[mv]
                    # advance ptr past all arrivals <= t_end in O(log n)
                    served_ptr[mv] = bisect.bisect_right(ts, t_end, lo=ptr)

        # Decide which side starts: the side with the earliest arrival runs
        # first; tie-break N/S (deterministic).
        def _first_arrival(side_combos: List[Tuple[int, int]]) -> Optional[int]:
            best = None
            for combo in side_combos:
                for ph in combo:
                    for mv in NEMA_PHASES[ph]:
                        ts = arrivals.get(mv, [])
                        if ts:
                            best = ts[0] if best is None else min(best, ts[0])
            return best if best is not None else None
        ns_first = _first_arrival(_NS_COMBOS)
        ew_first = _first_arrival(_EW_COMBOS)
        sides = [_NS_COMBOS, _EW_COMBOS] if ns_first is not None and (
            ew_first is None or ns_first <= ew_first) else [_EW_COMBOS, _NS_COMBOS]

        # Skip the dead initial period: don't serve greens before the first
        # arrival — a real controller stays dark until demand exists, and
        # serving zero-pressure phases at min_green wastes the opening.  The
        # first arrival may be negative when scenario generation uses a warm-up
        # window before the rendered clip; start the adaptive timeline there so
        # warm-up demand is served before frame 0 instead of being dumped into a
        # huge queue at the first rendered frame.
        first_arrival = min(
            (x for x in (ns_first, ew_first) if x is not None), default=0)
        # Keep the controller active from frame 0 onward: real intersections
        # do not sit dark until the first vehicle appears.
        t = min(first_arrival, 0)
        max_rounds = 5000
        while t < horizon_frames and max_rounds > 0:
            max_rounds -= 1
            for side in sides:
                counts = _queue_at(t)
                combo = self._select_combo(side, counts)
                end = self._serve_combo(t, combo, counts)
                self.intervals.append((t, end, combo))
                _discharge_until(combo, end)
                clr_start = end
                clr_end = end + self.yellow_f + self.all_red_f
                self._clearances.append((clr_start, clr_end))
                t = clr_end
                if t >= horizon_frames:
                    break
            # Progress guard: if no arrivals remain unserved AND every combo
            # is at min_green with zero queue, stop early (no point building
            # empty intervals to horizon).
            remaining = 0
            for mv, ts in arrivals.items():
                remaining += max(0, len(ts) - served_ptr.get(mv, 0))
            if remaining == 0 and t >= (horizon_frames // 2):
                break
        if self.intervals:
            self.cycle_frames = self.intervals[-1][1]
        else:
            self.cycle_frames = self.min_green_f + self.yellow_f + self.all_red_f
        # M4: emit a progress line so the adaptive-plan build (which can take
        # a noticeable fraction of scenario-gen time on dense demand) isn't
        # silent. Helps diagnose "scenario_gen is slow" without profiling.
        import sys as _sys
        print(f"[signal] adaptive timeline built: "
              f"{len(self.intervals)} intervals, "
              f"cycle_frames={self.cycle_frames}",
              file=_sys.stderr, flush=True)

    # ---- query API (same surface as SignalPlan) -------------------------------

    def _combo_at(self, frame: int) -> Optional[Tuple[int, int]]:
        for (s, e, combo) in self.intervals:
            if s <= frame < e:
                return combo
        return None

    def _in_clearance(self, frame: int) -> bool:
        for (s, e) in self._clearances:
            if s <= frame < e:
                return True
        return False

    def is_green(self, approach: G.Direction, turn: G.Turn, frame: int) -> bool:
        combo = self._combo_at(frame)
        if combo is None:
            return False
        ph = _movement_to_phase(approach, turn)
        # unprotected permissive lefts share the through phase are NOT
        # modelled here — every left is protected (lane 0 only).  Through and
        # right share the same phase so a 6 phase serves N through/right.
        return ph in combo

    def _next_interval_start(self, frame: int) -> int:
        """Earliest interval start >= frame, or horizon+1 if none."""
        best = None
        for (s, e, combo) in self.intervals:
            if s >= frame:
                if best is None or s < best:
                    best = s
        return best if best is not None else frame

    def next_green_frame(self, approach: G.Direction, turn: G.Turn,
                         frame: int) -> int:
        if self.is_green(approach, turn, frame):
            return frame
        ph = _movement_to_phase(approach, turn)
        best = None
        for (s, e, combo) in self.intervals:
            if ph in combo and s >= frame:
                if best is None or s < best:
                    best = s
        if best is not None:
            return best
        # No future green interval covers this movement — extend the timeline
        # by appending one more cycle serving its side at min_green.  This can
        # happen when demand for a movement was zero until now (it never got
        # selected) but a vehicle arrives late.  We materialise a fallback
        # interval on the movement's side, after the last served combo.
        last_end = self.intervals[-1][1] if self.intervals else 0
        side = _NS_COMBOS if ph in _NS_SIDE else _EW_COMBOS
        # pick the combo on this side that contains ph, paired with the
        # lowest-numbered matching partner phase from the other ring.
        for c in side:
            if ph in c:
                partner_combo = c
                break
        else:
            partner_combo = side[0]
        new_start = max(last_end + 1, frame)
        new_end = new_start + self.min_green_f
        self.intervals.append((new_start, new_end, partner_combo))
        self._clearances.append((new_end, new_end + self.yellow_f + self.all_red_f))
        return new_start

    def next_red_frame(self, approach: G.Direction, turn: G.Turn,
                       frame: int) -> int:
        if not self.is_green(approach, turn, frame):
            return frame
        ph = _movement_to_phase(approach, turn)
        for (s, e, combo) in self.intervals:
            if ph in combo and s <= frame < e:
                return e
        return frame + 1

    # ---- closed-loop interface (for IDM microsim) -----------------------------

    def observe_and_decide(
        self,
        tick: int,
        queue_counts: Dict[Tuple[G.Direction, G.Turn], int],
        barrier: int,
    ) -> Tuple[Tuple[int, int], int, int, int]:
        """Closed-loop phase selection from live lane-state observations.

        Called by the microsim at each tick where a new green phase must be
        chosen. Selects the highest-pressure combo on the current barrier
        side, computes green duration, and returns the decision.

        Args:
            tick: current simulation frame.
            queue_counts: ``{(approach, turn): count}`` of vehicles currently
                waiting (queued and not yet released) per movement.
            barrier: current barrier side (0 = N/S, 1 = E/W).

        Returns:
            ``(combo, green_end, clearance_end, next_barrier)`` where
            ``combo`` is the ``(ring1_phase, ring2_phase)`` tuple,
            ``green_end`` is the exclusive frame where this green expires,
            ``clearance_end`` is the end of the yellow+all-red clearance,
            and ``next_barrier`` is the barrier for the next decision.
        """
        side = _NS_COMBOS if barrier == 0 else _EW_COMBOS
        combo = self._select_combo(side, queue_counts)
        green_end = self._serve_combo(tick, combo, queue_counts)
        clearance_end = green_end + self.yellow_f + self.all_red_f
        return combo, green_end, clearance_end, 1 - barrier

    # ---- introspection (for metadata + tests) --------------------------------

    def phase_at(self, frame: int) -> str:
        """Human-readable phase label for *frame* (for metadata timeline)."""
        if self._in_clearance(frame):
            return "CLEARANCE"
        combo = self._combo_at(frame)
        if combo is None:
            return "OFF"
        return f"P{combo[0]}+P{combo[1]}"

    def movements_green_at(self, frame: int) -> Set[Tuple[G.Direction, G.Turn]]:
        """All (approach, turn) movements that have green at *frame*."""
        combo = self._combo_at(frame)
        if combo is None:
            return set()
        out: Set[Tuple[G.Direction, G.Turn]] = set()
        for ph in combo:
            out |= NEMA_PHASES[ph]
        return out
