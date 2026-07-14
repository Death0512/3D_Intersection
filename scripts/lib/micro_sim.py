"""Time-stepped microscopic traffic simulation with IDM car-following.

Replaces the event-driven scheduler (intersection_sim.py) with a physics-based
model where vehicle trajectories *emerge* from continuous state evolution
rather than being planned in advance.

Key differences from the legacy v2 sim:
  - Speed is an **output** (result of IDM dynamics), not a fixed input.
  - Leader-following is continuous: vehicles adjust acceleration every frame.
  - Red lights are modelled as virtual stopped leaders (same IDM formula).
  - Queue formation / discharge emerges naturally from car-following.
  - The signal controller observes live lane states (closed-loop).

Output contract: identical to intersection_sim.simulate() — mutates each
vehicle dict with stop_frame, release_frame, queue_slot, wait_frames so the
downstream pipeline (render, metadata, validation) works unchanged.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from . import geometry as G
    from . import traffic_signal as SG
    from . import car_following as CF
except ImportError:
    import geometry as G
    import traffic_signal as SG
    import car_following as CF


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MicroSimConfig:
    fps: int = 30
    approach_visible_length: float = 40.0
    exit_buffer_frames: int = 5
    sat_flow_vps: float = 1.0
    # Adaptive timing (match traffic_signal / intersection_sim defaults)
    min_green_f: int = 240   # 8s @ 30fps
    max_green_f: int = 1200  # 40s
    yellow_f: int = 90       # 3s
    all_red_f: int = 60      # 2s


# ---------------------------------------------------------------------------
# NEMA phase tables (same as intersection_sim.py)
# ---------------------------------------------------------------------------
_NEMA_PHASES: Dict[int, Set[Tuple[G.Direction, G.Turn]]] = {
    1: {(G.Direction.S, G.Turn.LEFT)},
    2: {(G.Direction.S, G.Turn.STRAIGHT), (G.Direction.S, G.Turn.RIGHT)},
    3: {(G.Direction.E, G.Turn.LEFT)},
    4: {(G.Direction.E, G.Turn.STRAIGHT), (G.Direction.E, G.Turn.RIGHT)},
    5: {(G.Direction.N, G.Turn.LEFT)},
    6: {(G.Direction.N, G.Turn.STRAIGHT), (G.Direction.N, G.Turn.RIGHT)},
    7: {(G.Direction.W, G.Turn.LEFT)},
    8: {(G.Direction.W, G.Turn.STRAIGHT), (G.Direction.W, G.Turn.RIGHT)},
}
_NS_COMBOS = [(1, 5), (1, 6), (2, 5), (2, 6)]
_EW_COMBOS = [(3, 7), (3, 8), (4, 7), (4, 8)]

def _phase_for(approach: G.Direction, turn: G.Turn) -> int:
    _L = {
        (G.Direction.N, G.Turn.LEFT): 5, (G.Direction.S, G.Turn.LEFT): 1,
        (G.Direction.E, G.Turn.LEFT): 3, (G.Direction.W, G.Turn.LEFT): 7,
    }
    _TR = {
        (G.Direction.N, G.Turn.STRAIGHT): 6, (G.Direction.N, G.Turn.RIGHT): 6,
        (G.Direction.S, G.Turn.STRAIGHT): 2, (G.Direction.S, G.Turn.RIGHT): 2,
        (G.Direction.E, G.Turn.STRAIGHT): 4, (G.Direction.E, G.Turn.RIGHT): 4,
        (G.Direction.W, G.Turn.STRAIGHT): 8, (G.Direction.W, G.Turn.RIGHT): 8,
    }
    return _L[(approach, turn)] if turn == G.Turn.LEFT else _TR[(approach, turn)]


# ---------------------------------------------------------------------------
# Intersection conflict (same logic as intersection_sim.py)
# ---------------------------------------------------------------------------
def _movements_conflict(mv_a: Tuple[G.Direction, G.Turn],
                        mv_b: Tuple[G.Direction, G.Turn]) -> bool:
    if mv_a == mv_b:
        return False
    ap_a, _ = mv_a
    ap_b, _ = mv_b
    if ap_a == ap_b:
        return False
    opp = {G.Direction.N: G.Direction.S, G.Direction.S: G.Direction.N,
           G.Direction.E: G.Direction.W, G.Direction.W: G.Direction.E}
    if opp[ap_a] == ap_b:
        return False
    ph_a = _phase_for(*mv_a)
    ph_b = _phase_for(*mv_b)
    for p1, p2 in _NS_COMBOS + _EW_COMBOS:
        if {ph_a, ph_b} <= {p1, p2}:
            return False
    return True


@dataclass
class _Reservation:
    vehicle_id: str
    approach: G.Direction
    turn: G.Turn
    entry_frame: int
    clear_frame: int

    @property
    def movement(self) -> Tuple[G.Direction, G.Turn]:
        return (self.approach, self.turn)


def _res_conflicts(active: List[_Reservation], entry_f: int, clear_f: int,
                   mv: Tuple[G.Direction, G.Turn]) -> bool:
    for r in active:
        if r.clear_frame <= entry_f or r.entry_frame >= clear_f:
            continue
        if _movements_conflict(mv, r.movement):
            return True
    return False


# ---------------------------------------------------------------------------
# Vehicle state for time-stepped simulation
# ---------------------------------------------------------------------------
STAGE_APPROACH = "APPROACH"
STAGE_QUEUED   = "QUEUED"
STAGE_IN_BOX   = "IN_BOX"
STAGE_EXIT     = "EXIT"
STAGE_DONE     = "DONE"


@dataclass
class VehicleState:
    """Dynamic state of one vehicle during microscopic simulation."""
    vid: str
    vdict: dict                    # reference to the original vehicle dict

    approach: G.Direction
    lane: int
    turn: G.Turn
    length: float

    # IDM parameters (desired_speed comes from vehicle's speed_ms)
    desired_speed: float
    idm_params: CF.IDMParams

    # Dynamic state
    s: float = 0.0                 # longitudinal position along approach (0 = appear)
    speed: float = 0.0             # current speed (m/s)
    accel: float = 0.0             # current acceleration (m/s²)

    stage: str = STAGE_APPROACH
    depart_frame: int = 0          # frame when vehicle appears on approach

    # Intersection traversal
    box_frames: int = 0            # delta_t_frames for black-box
    exit_key: Tuple[str, int] = ("N", 0)

    # Timing records (filled during sim, exported to vehicle dict)
    _stop_frame: Optional[int] = None
    _release_frame: Optional[int] = None
    _queue_slot: int = -1
    _active_ahead_at_stop: int = 0

    # Exit tracking
    exit_s: float = 0.0            # position along exit lane
    exit_speed: float = 0.0
    exit_appear_frame: int = 0
    exit_leave_frame: int = 0


# ---------------------------------------------------------------------------
# Microscopic simulation
# ---------------------------------------------------------------------------

def simulate(vehicles: list,
             approach_visible_length: float,
             fps: int,
             signal_plan=None,
             seed: int = 42) -> Tuple[List[dict], Dict]:
    """Run IDM-based microscopic simulation.

    Same signature and output contract as intersection_sim.simulate():
    mutates each vehicle dict with stop_frame, release_frame, queue_slot,
    wait_frames. Returns (vehicles, meta_dict).
    """
    cfg = MicroSimConfig(
        fps=fps,
        approach_visible_length=approach_visible_length,
        min_green_f=int(round(8.0 * fps)),
        max_green_f=int(round(40.0 * fps)),
        yellow_f=int(round(3.0 * fps)),
        all_red_f=int(round(2.0 * fps)),
    )
    dt = 1.0 / fps
    stop_line_s = approach_visible_length  # position of stop line along approach

    # Clear stale scheduling fields
    for v in vehicles:
        for k in ("stop_frame", "release_frame", "queue_slot", "wait_frames",
                   "entry_frame", "clear_frame"):
            v.pop(k, None)

    # ---- Build vehicle states ------------------------------------------------
    states: List[VehicleState] = []
    for vi, v in enumerate(vehicles):
        ap = G.Direction(v["approach"])
        lane = v["lane"]
        turn = G.Turn(v["turn"])
        spd = v["speed_ms"]
        box_f = G.delta_t_frames(turn, spd, fps, lane_index=lane)
        out_dir, ex_lane = G.exit_lane_for_movement(ap, lane, turn)

        idm_p = CF.IDMParams(
            desired_speed=spd,
            max_accel=2.5,
            comfortable_decel=2.0,
            min_gap=2.0,
            time_headway=1.5,
            delta=4,
        )

        states.append(VehicleState(
            vid=v["id"],
            vdict=v,
            approach=ap,
            lane=lane,
            turn=turn,
            length=v["length"],
            desired_speed=spd,
            idm_params=idm_p,
            s=0.0,
            speed=spd,  # start at desired approach speed
            accel=0.0,
            stage=STAGE_APPROACH,
            depart_frame=v["depart_frame"],
            box_frames=box_f,
            exit_key=(out_dir.value, ex_lane),
        ))

    # ---- Partition into per-lane lists, sorted by depart_frame ---------------
    lane_states: Dict[Tuple[str, int], List[VehicleState]] = {}
    for st in states:
        key = (st.vdict["approach"], st.lane)
        lane_states.setdefault(key, []).append(st)
    for key in lane_states:
        lane_states[key].sort(key=lambda s: s.depart_frame)

    # ---- Signal state --------------------------------------------------------
    has_signal = signal_plan is not None
    adaptive = isinstance(signal_plan, SG.AdaptiveSignalPlan)
    adaptive_plan: Optional[SG.AdaptiveSignalPlan] = \
        signal_plan if adaptive else None
    fixed_plan = signal_plan if (has_signal and not adaptive) else None

    # Adaptive live-selector state
    barrier = 0  # 0=NS, 1=EW
    combo: Optional[Tuple[int, int]] = None
    green_end: int = 0
    clearance_end: int = 0
    adaptive_intervals: List[Tuple[int, int, Tuple[int, int]]] = []
    adaptive_clearances: List[Tuple[int, int]] = []

    # ---- Intersection reservation state --------------------------------------
    reservations: List[_Reservation] = []
    exit_last_leave: Dict[Tuple[str, int], int] = {}

    # ---- Time bounds ---------------------------------------------------------
    min_depart = min((s.depart_frame for s in states), default=0)
    max_depart = max((s.depart_frame for s in states), default=0)
    # Need enough time for last vehicle to traverse approach + box + exit
    horizon = max_depart + int(round((approach_visible_length * 2) / 5.0 * fps)) + \
              fps * 180  # generous 3-min buffer
    tick_start = min_depart
    if adaptive:
        green_end = tick_start
        clearance_end = tick_start

    # ---- Per-lane release tracking -------------------------------------------
    lane_last_rel: Dict[Tuple[str, int], int] = {}
    lane_history: Dict[Tuple[str, int], List[Tuple[int, int]]] = {
        k: [] for k in lane_states
    }
    reaction_frames = int(round(0.5 * fps))  # same-lane stagger

    # Stats
    wait_list: List[int] = []

    # ---- MAIN TIME-STEPPED LOOP ----------------------------------------------
    for tick in range(tick_start, horizon):
        # --- Adaptive combo selection (closed-loop: observes live queues) ------
        if adaptive and tick >= green_end:
            if tick < clearance_end:
                continue

            # Count vehicles currently queued (stopped at stop line, not released)
            counts: Dict[Tuple[G.Direction, G.Turn], int] = {}
            for key, st_list in lane_states.items():
                for st in st_list:
                    if st.stage in (STAGE_APPROACH, STAGE_QUEUED) and \
                       st._release_frame is None and \
                       st.depart_frame <= tick:
                        mv = (st.approach, st.turn)
                        counts[mv] = counts.get(mv, 0) + 1

            # Delegate to AdaptiveSignalPlan.observe_and_decide()
            assert adaptive_plan is not None
            combo, green_end, clearance_end, barrier = \
                adaptive_plan.observe_and_decide(tick, counts, barrier)
            adaptive_intervals.append((tick, green_end, combo))
            adaptive_clearances.append((green_end, clearance_end))

        # --- Check green for this tick ----------------------------------------
        def _is_green_now(approach: G.Direction, turn: G.Turn) -> bool:
            if not has_signal:
                return True
            if adaptive:
                if tick >= green_end:
                    return False
                ph = _phase_for(approach, turn)
                return combo is not None and ph in combo
            else:
                return bool(fixed_plan is not None and
                            fixed_plan.is_green(approach, turn, tick))

        # --- IDM state update for each active vehicle -------------------------
        for key, st_list in lane_states.items():
            # Identify active vehicles on this approach lane
            active = [st for st in st_list
                      if st.depart_frame <= tick and
                      st.stage in (STAGE_APPROACH, STAGE_QUEUED)]
            if not active:
                continue

            # Sort by position (furthest along = front of queue)
            active.sort(key=lambda s: -s.s)

            for i, st in enumerate(active):
                # --- Find physical leader (vehicle ahead in same lane) ---
                phys_leader = None
                if i > 0:
                    leader = active[i - 1]
                    # leader.s is the front bumper; gap = leader.s - leader.length - st.s
                    phys_leader = (leader.s, leader.speed, leader.length)

                # --- Virtual signal leader (red light at stop line) ---
                green = _is_green_now(st.approach, st.turn)
                sig_leader = CF.virtual_signal_leader(
                    st.s, stop_line_s, st.speed, green)

                # --- Effective leader (most constraining) ---
                gap, dv = CF.effective_leader(
                    st.s, st.speed, st.length,
                    phys_leader, sig_leader)

                # --- IDM acceleration ---
                st.accel = CF.idm_acceleration(
                    st.speed, st.desired_speed, gap, dv, st.idm_params)

        # --- Position/speed update (Euler integration) ------------------------
        for key, st_list in lane_states.items():
            # Get active vehicles sorted front to back for anti-overtake clamping
            active_for_update = [
                st for st in st_list
                if st.depart_frame <= tick and
                st.stage in (STAGE_APPROACH, STAGE_QUEUED)
            ]
            active_for_update.sort(key=lambda s: -s.s)

            for idx, st in enumerate(active_for_update):
                # Update speed (clamp to non-negative)
                new_speed = max(0.0, st.speed + st.accel * dt)
                # Update position
                new_s = st.s + st.speed * dt + 0.5 * st.accel * dt * dt
                st.speed = new_speed

                # Anti-overtake clamp: never pass the vehicle ahead
                if idx > 0:
                    leader = active_for_update[idx - 1]
                    max_s = leader.s - leader.length - 0.1  # min 0.1m gap
                    if new_s > max_s:
                        new_s = max_s
                        st.speed = min(st.speed, leader.speed)

                st.s = new_s

                # Clamp to stop line (can't overshoot into box without release)
                if st._release_frame is None and st.s >= stop_line_s:
                    st.s = stop_line_s
                    st.speed = 0.0

                # Detect arrival at stop line region.
                # IDM stops the vehicle with gap ≈ s0 (min_gap) before the
                # virtual leader at the stop line, so the vehicle rests at
                # approximately (stop_line_s - min_gap).  We detect "stopped"
                # when speed drops below threshold within that region.
                min_gap = st.idm_params.min_gap
                near_stop = st.s >= stop_line_s - min_gap - 1.0
                at_stop_line = st.s >= stop_line_s - 0.1

                if st._stop_frame is None and (at_stop_line or
                        (near_stop and st.speed < 0.3)):
                    st._stop_frame = tick
                    st.stage = STAGE_QUEUED

                # Detect natural queue (speed ≈ 0 behind another queued vehicle)
                if st.stage == STAGE_APPROACH and st.speed < 0.3 and \
                   st.s > stop_line_s * 0.3:
                    st._stop_frame = st._stop_frame or tick
                    st.stage = STAGE_QUEUED

        # --- Try to release front vehicle per lane into intersection ----------
        for key in sorted(lane_states.keys()):
            st_list = lane_states[key]
            # Find front unreleased vehicle that has reached stop line
            front = None
            for st in st_list:
                if st.stage == STAGE_QUEUED and st._release_frame is None:
                    if st._stop_frame is not None and st._stop_frame <= tick:
                        front = st
                        break

            if front is None:
                continue

            # Green check
            if not _is_green_now(front.approach, front.turn):
                continue

            # Same-lane headway stagger
            prev_rel = lane_last_rel.get(key)
            if prev_rel is not None and tick < prev_rel + reaction_frames:
                continue

            # Intersection conflict check
            entry_f = tick
            clear_f = entry_f + front.box_frames
            if _res_conflicts(reservations, entry_f, clear_f,
                              (front.approach, front.turn)):
                continue

            # Exit-lane occupancy check
            travel_f = int(round(approach_visible_length / front.desired_speed * fps))
            reappear_f = clear_f
            leave_f = reappear_f + travel_f
            last_exit = exit_last_leave.get(front.exit_key, -999)
            if reappear_f < last_exit + cfg.exit_buffer_frames:
                continue

            # ---- RELEASE vehicle ----
            front._release_frame = tick
            w = tick - (front._stop_frame or tick)
            front.vdict["wait_frames"] = max(w, 0)
            wait_list.append(max(w, 0))

            # Queue slot: count earlier same-lane vehicles still active
            sf = front._stop_frame or 0
            active_ahead = sum(
                1 for _, rel in lane_history[key] if rel > sf)
            front._queue_slot = active_ahead if (has_signal and (w > 0 or active_ahead)) else -1

            front.stage = STAGE_IN_BOX

            reservations.append(_Reservation(
                vehicle_id=front.vid, approach=front.approach,
                turn=front.turn, entry_frame=entry_f, clear_frame=clear_f))
            exit_last_leave[front.exit_key] = leave_f
            lane_last_rel[key] = tick
            lane_history[key].append((front._stop_frame or tick, tick))

        # --- Purge expired reservations ---
        reservations = [r for r in reservations if r.clear_frame > tick]

        # --- Early exit: all vehicles done ---
        if all(st._release_frame is not None for st in states):
            break

    # ---- Handle unreleased vehicles (same policy as intersection_sim) --------
    for st in states:
        if st._release_frame is None:
            if st._stop_frame is None:
                # Never reached stop line — compute arrival analytically
                travel_f = int(round(approach_visible_length / st.desired_speed * fps))
                st._stop_frame = st.depart_frame + travel_f
            if not has_signal:
                st._release_frame = st._stop_frame
                st._queue_slot = -1
                st.vdict["wait_frames"] = 0
            else:
                # Unreleased under signal → horizon overflow
                st.vdict.setdefault("wait_frames", 0)

    # ---- Write results back to vehicle dicts ---------------------------------
    for st in states:
        st.vdict["stop_frame"] = st._stop_frame
        if st._release_frame is not None:
            st.vdict["release_frame"] = st._release_frame
        st.vdict["queue_slot"] = st._queue_slot

    # ---- Build arrival events for adaptive timeline export -------------------
    arrival_events: Dict[Tuple[G.Direction, G.Turn], List[int]] = {}
    for st in states:
        sf = st._stop_frame
        if sf is not None:
            mv = (st.approach, st.turn)
            arrival_events.setdefault(mv, []).append(sf)
    for mv in arrival_events:
        arrival_events[mv].sort()

    meta: Dict = {"arrival_events": arrival_events}
    if adaptive:
        meta["adaptive_intervals"] = adaptive_intervals
        meta["adaptive_clearances"] = adaptive_clearances

    # ---- Stats ----
    if has_signal:
        n_tot = len(vehicles)
        n_q = sum(1 for v in vehicles if v.get("queue_slot", -1) >= 0)
        mean_w = sum(wait_list) / len(wait_list) if wait_list else 0.0
        max_w = max(wait_list) if wait_list else 0
        print(f"[micro_sim] {n_tot} veh, {n_q} queued, "
              f"mean_wait={mean_w:.0f}f max_wait={max_w}f",
              file=sys.stderr, flush=True)

    return vehicles, meta
