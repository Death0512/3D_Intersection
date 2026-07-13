"""Event-driven intersection microsim (v2).

Replaces the old post-hoc scheduler with finite-capacity FIFO queues, per-tick
signal gating, intersection conflict reservations, and downstream exit-lane
occupancy.

Imports signal_plan.is_green for fixed-plan gating. For AdaptiveSignalPlan,
runs a live per-cycle MaxPressure selector — the sim owns discharge capacity
so the rebuild's infinite-discharge assumption is out of the picture.

DO NOT import scenario_gen (circular).
"""
from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from . import geometry as G
    from . import kinematics as K
    from . import traffic_signal as SG
except ImportError:
    import geometry as G
    import kinematics as K
    import traffic_signal as SG

# ---------------------------------------------------------------------------
# ponytail: local copies of NEMA tables (private imports would break encapsulation)
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
_NS_COMBOS: List[Tuple[int, int]] = [(1, 5), (1, 6), (2, 5), (2, 6)]
_EW_COMBOS: List[Tuple[int, int]] = [(3, 7), (3, 8), (4, 7), (4, 8)]
_NS_SIDE = {1, 2, 5, 6}
_EW_SIDE = {3, 4, 7, 8}


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


# Adaptive timing defaults (match traffic_signal.py)
_ADP_MIN_GREEN_S = 8.0
_ADP_MAX_GREEN_S = 40.0
_ADP_YELLOW_S = 3.0
_ADP_ALL_RED_S = 2.0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    fps: int = 30
    approach_visible_length: float = 40.0
    exit_buffer_frames: int = 5
    reaction_per_slot_frames: int = 15   # ~0.5 s
    sat_flow_vps: float = 1.0            # veh/s/lane discharge
    min_green_f: int = int(round(_ADP_MIN_GREEN_S * 30))
    max_green_f: int = int(round(_ADP_MAX_GREEN_S * 30))
    yellow_f: int = int(round(_ADP_YELLOW_S * 30))
    all_red_f: int = int(round(_ADP_ALL_RED_S * 30))


# ---------------------------------------------------------------------------
# Conflict / compatibility
# ---------------------------------------------------------------------------
def _movements_conflict(mv_a: Tuple[G.Direction, G.Turn],
                        mv_b: Tuple[G.Direction, G.Turn]) -> bool:
    """True if two movements could collide inside the box.

    Same movement = safe (sequential). Legal NEMA combo members are
    non-conflicting. Opposite-approach movements don't cross.
    """
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
    # Perpendicular: check NEMA combo compatibility
    ph_a = _phase_for(*mv_a)
    ph_b = _phase_for(*mv_b)
    for p1, p2 in _NS_COMBOS + _EW_COMBOS:
        if {ph_a, ph_b} <= {p1, p2}:
            return False
    return True


# ---------------------------------------------------------------------------
# Reservation system
# ---------------------------------------------------------------------------
@dataclass
class Reservation:
    vehicle_id: str
    approach: G.Direction
    turn: G.Turn
    entry_frame: int
    clear_frame: int

    @property
    def movement(self) -> Tuple[G.Direction, G.Turn]:
        return (self.approach, self.turn)


def _res_conflicts(active: List[Reservation],
                   entry_f: int, clear_f: int,
                   mv: Tuple[G.Direction, G.Turn]) -> bool:
    for r in active:
        if r.clear_frame <= entry_f or r.entry_frame >= clear_f:
            continue
        if _movements_conflict(mv, r.movement):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-vehicle pre-computed data
# ---------------------------------------------------------------------------
@dataclass
class _VehRec:
    v: dict
    arrive_frame: int        # depart_frame + travel_frames (stop-line arrival)
    box_frames: int          # delta_t_frames (box traversal)
    travel_frames: int       # visible travel on approach / exit
    speed_ms: float
    length: float
    approach: G.Direction
    lane: int
    turn: G.Turn
    exit_key: Tuple[str, int]  # (dir.value, exit_lane_idx)
    idx: int                  # global sorted index for tie-breaking


# ---------------------------------------------------------------------------
# The sim: one tick = one frame, process front vehicle per lane
# ---------------------------------------------------------------------------
def simulate(vehicles: list,
             approach_visible_length: float,
             fps: int,
             signal_plan=None,
             seed: int = 42) -> Tuple[List[dict], Dict]:
    """Run event-driven microsim. Mutates each vehicle dict with stop_frame,
    release_frame, queue_slot, wait_frames.

    Returns (vehicles, arrival_events) where arrival_events maps
    (approach, turn) -> sorted list of stop-line arrival frames for
    post-sim adaptive timeline building.
    """
    cfg = SimConfig(
        fps=fps,
        approach_visible_length=approach_visible_length,
        min_green_f=int(round(_ADP_MIN_GREEN_S * fps)),
        max_green_f=int(round(_ADP_MAX_GREEN_S * fps)),
        yellow_f=int(round(_ADP_YELLOW_S * fps)),
        all_red_f=int(round(_ADP_ALL_RED_S * fps)),
        reaction_per_slot_frames=int(round(0.5 * fps)),
    )

    # simulate() may be called twice after clip filtering; don't let stale
    # scheduling fields make every vehicle look already served.
    for v in vehicles:
        for k in ("stop_frame", "release_frame", "queue_slot", "wait_frames",
                  "entry_frame", "clear_frame"):
            v.pop(k, None)

    # ---- build vehicle records ---------------------------------------------
    recs: List[_VehRec] = []
    for vi, v in enumerate(vehicles):
        ap = G.Direction(v["approach"])
        lane = v["lane"]
        turn = G.Turn(v["turn"])
        spd = v["speed_ms"]
        travel_f = int(round(cfg.approach_visible_length / spd * fps))
        arrive_f = v["depart_frame"] + travel_f
        box_f = G.delta_t_frames(turn, spd, fps, lane_index=lane)
        out_dir, ex_lane = G.exit_lane_for_movement(ap, lane, turn)
        recs.append(_VehRec(
            v=v, arrive_frame=arrive_f, box_frames=box_f, travel_frames=travel_f,
            speed_ms=spd, length=v["length"],
            approach=ap, lane=lane, turn=turn,
            exit_key=(out_dir.value, ex_lane),
            idx=vi,
        ))

    # ---- partition into per-lane FIFO lists, sorted by arrive --------------
    lane_vehicles: Dict[Tuple[str, int], List[_VehRec]] = {}
    for r in recs:
        key = (r.v["approach"], r.v["lane"])
        lane_vehicles.setdefault(key, []).append(r)
    for key in lane_vehicles:
        lane_vehicles[key].sort(key=lambda r: (r.arrive_frame, r.idx))

    # ---- sim state ----------------------------------------------------------
    reservations: List[Reservation] = []
    exit_last_leave: Dict[Tuple[str, int], int] = {}  # exit_key → last cleared
    # Per-lane: index into lane_vehicles[key] of the next candidate front
    lane_next: Dict[Tuple[str, int], int] = {k: 0 for k in lane_vehicles}
    # Per-lane last release frame (for same-lane stagger gap)
    lane_last_rel: Dict[Tuple[str, int], int] = {}
    lane_history: Dict[Tuple[str, int], List[Tuple[int, int]]] = {
        k: [] for k in lane_vehicles
    }  # stop-line arrival, release

    # Signal state
    has_signal = signal_plan is not None
    adaptive = isinstance(signal_plan, SG.AdaptiveSignalPlan)
    fixed_plan: Optional[SG.SignalPlan] = signal_plan if (has_signal and not adaptive) else None

    # Adaptive live-selector
    barrier = 0          # 0=NS, 1=EW
    combo: Optional[Tuple[int, int]] = None
    green_end: int = 0
    clearance_end: int = 0
    adaptive_intervals: List[Tuple[int, int, Tuple[int, int]]] = []
    adaptive_clearances: List[Tuple[int, int]] = []

    # Horizon: simulate until all vehicles released or generous bound
    max_arrive = max((r.arrive_frame for r in recs), default=0)
    min_arrive = min((r.arrive_frame for r in recs), default=0)
    # Warm-up arrivals are often negative — start there so queues discharge
    # before the clip opens instead of piling everyone at frame 0.
    tick = min_arrive
    horizon = max_arrive + max(fps * 180, 30 * cfg.max_green_f)
    if adaptive:
        # First green decision at the earliest stop-line arrival.
        green_end = tick
        clearance_end = tick

    # Stats
    wait_frames: List[int] = []

    # ---- tick loop ----------------------------------------------------------
    while tick < horizon:
        # --- adaptive combo selection ---
        if adaptive and tick >= green_end:
            # Stay in clearance until it ends
            if tick < clearance_end:
                tick += 1
                continue

            # Count unserved (no release_frame yet) vehicles per movement
            counts: Dict[Tuple[G.Direction, G.Turn], int] = {}
            for key, rec_list in lane_vehicles.items():
                for r in rec_list:
                    if "release_frame" not in r.v and r.arrive_frame <= tick:
                        mv = (r.approach, r.turn)
                        counts[mv] = counts.get(mv, 0) + 1

            side = _NS_COMBOS if barrier == 0 else _EW_COMBOS
            best: Tuple[int, int] = side[0]
            best_p = -1.0
            for c in side:
                p = 0.0
                for ph in c:
                    for mv in _NEMA_PHASES.get(ph, set()):
                        p += counts.get(mv, 0)
                if p > best_p or (p == best_p and c < best):
                    best_p = p
                    best = c

            total_q = int(best_p + 0.5)
            if total_q <= 0:
                dur = cfg.min_green_f
            else:
                dur_s = total_q / cfg.sat_flow_vps
                dur = max(cfg.min_green_f, min(cfg.max_green_f, int(round(dur_s * fps))))

            combo = best
            green_end = tick + dur
            clearance_end = green_end + cfg.yellow_f + cfg.all_red_f
            adaptive_intervals.append((tick, green_end, combo))
            adaptive_clearances.append((green_end, clearance_end))
            barrier = 1 - barrier

        # --- try release: one vehicle from each lane where front is ready --
        # Only the front unserved vehicle per lane can release.
        for key in sorted(lane_vehicles.keys()):
            idx = lane_next[key]
            rec_list = lane_vehicles[key]
            # Skip served indices
            while idx < len(rec_list) and "release_frame" in rec_list[idx].v:
                idx += 1
            if idx >= len(rec_list):
                lane_next[key] = idx
                continue
            r = rec_list[idx]
            lane_next[key] = idx

            # Must have arrived
            if r.arrive_frame > tick:
                continue

            # ---- green check ----
            green = True
            if has_signal:
                if adaptive:
                    ph = _phase_for(r.approach, r.turn)
                    green = combo is not None and ph in combo
                else:
                    green = bool(
                        fixed_plan is not None
                        and fixed_plan.is_green(r.approach, r.turn, tick)
                    )

            if not green:
                continue

            # ---- headway stagger ----
            prev = lane_last_rel.get(key)
            if prev is not None and tick < prev + cfg.reaction_per_slot_frames:
                continue

            # ---- intersection conflict ----
            entry_f = tick
            clear_f = entry_f + r.box_frames
            if _res_conflicts(reservations, entry_f, clear_f,
                              (r.approach, r.turn)):
                continue

            # ---- exit-lane occupancy ----
            reappear_f = clear_f
            leave_f = reappear_f + r.travel_frames
            last_exit = exit_last_leave.get(r.exit_key, -999)
            if reappear_f < last_exit + cfg.exit_buffer_frames:
                continue

            # ---- ALL conditions met → RELEASE ----
            r.v["release_frame"] = entry_f
            r.v["stop_frame"] = r.arrive_frame
            w = entry_f - r.arrive_frame
            r.v["wait_frames"] = max(w, 0)
            wait_frames.append(max(w, 0))

            # queue_slot: for signal-controlled flow, physical queue position.
            # Count earlier same-lane vehicles still stopped/launched when this
            # vehicle reached the stop line. Free-flow vehicles stay -1.
            active_ahead = sum(1 for _, rel in lane_history[key]
                               if rel > r.arrive_frame)
            r.v["queue_slot"] = active_ahead if (has_signal and (w > 0 or active_ahead)) else -1

            reservations.append(Reservation(
                vehicle_id=r.v["id"], approach=r.approach, turn=r.turn,
                entry_frame=entry_f, clear_frame=clear_f,
            ))
            exit_last_leave[r.exit_key] = leave_f
            lane_last_rel[key] = entry_f
            lane_history[key].append((r.arrive_frame, entry_f))
            lane_next[key] = idx + 1

        # --- purge expired reservations ---
        reservations = [x for x in reservations if x.clear_frame > tick]

        # --- advance ---
        tick += 1

        # --- early exit: all vehicles served ---
        if all("release_frame" in r.v for r in recs):
            break

    # ---- finalize unreleased vehicles ----
    for r in recs:
        if "release_frame" not in r.v:
            if not has_signal:
                # Free-flow: no gating, no conflicts blocked them
                r.v["stop_frame"] = r.arrive_frame
                r.v["release_frame"] = r.arrive_frame
                r.v["queue_slot"] = -1
                r.v["wait_frames"] = 0
            else:
                # Unreleased under signal → horizon overflow.
                # Assign stop_frame; no release_frame.
                r.v["stop_frame"] = r.arrive_frame

    # ---- arrival events for adaptive timeline ----
    arrival_events: Dict[Tuple[G.Direction, G.Turn], List[int]] = {}
    for r in recs:
        sf = r.v.get("stop_frame")
        if sf is not None:
            mv = (r.approach, r.turn)
            arrival_events.setdefault(mv, []).append(sf)
    for mv in arrival_events:
        arrival_events[mv].sort()

    meta = {"arrival_events": arrival_events}
    if adaptive:
        meta["adaptive_intervals"] = adaptive_intervals
        meta["adaptive_clearances"] = adaptive_clearances

    # ---- stats ----
    if has_signal:
        n_tot = len(vehicles)
        n_q = sum(1 for v in vehicles if v.get("queue_slot", -1) >= 0)
        mean_w = sum(wait_frames) / len(wait_frames) if wait_frames else 0.0
        max_w = max(wait_frames) if wait_frames else 0
        print(f"[sim] {n_tot} veh, {n_q} queued, "
              f"mean_wait={mean_w:.0f}f max_wait={max_w}f "
              f"tick={tick}",
              file=sys.stderr, flush=True)

    return vehicles, meta


# ---------------------------------------------------------------------------
# Post-sim: build adaptive signal timeline for metadata/scenario export
# ---------------------------------------------------------------------------
def build_adaptive_timeline(
    arrivals: Dict[Tuple[G.Direction, G.Turn], List[int]],
    horizon_frames: int,
    fps: int = 30,
) -> Tuple[List[Tuple[int, int, Tuple[int, int]]], List[Tuple[int, int]]]:
    """Build a NEMA MaxPressure timeline from realised stop-line arrivals.

    Returns (intervals, clearances) where intervals are
    (start_f, end_f, combo) and clearances are (start_f, end_f).

    The sim owns discharge — this timeline is a *prediction* of green windows
    for is_green lookups. It uses the same logic as AdaptiveSignalPlan.rebuild
    but their discharge pointers are per-movement, not per-vehicle.
    """
    cfg = SimConfig(fps=fps)
    served: Dict[Tuple[G.Direction, G.Turn], int] = {mv: 0 for mv in arrivals}
    intervals: List[Tuple[int, int, Tuple[int, int]]] = []
    clears: List[Tuple[int, int]] = []

    def _counts(t: int):
        q: Dict[Tuple[G.Direction, G.Turn], int] = {}
        for mv, ts in arrivals.items():
            ptr = served.get(mv, 0)
            q[mv] = bisect.bisect_right(ts, t, lo=ptr) - ptr
        return q

    def _discharge(combo: Tuple[int, int], t_end: int):
        for ph in combo:
            for mv in _NEMA_PHASES.get(ph, set()):
                if mv not in served:
                    continue
                ts = arrivals.get(mv, [])
                served[mv] = bisect.bisect_right(ts, t_end, lo=served[mv])

    def _earliest(side_combos):
        best = None
        for c in side_combos:
            for ph in c:
                for mv in _NEMA_PHASES.get(ph, set()):
                    ts = arrivals.get(mv, [])
                    if ts:
                        best = ts[0] if best is None else min(best, ts[0])
        return best

    ns_first = _earliest(_NS_COMBOS)
    ew_first = _earliest(_EW_COMBOS)
    sides = [_NS_COMBOS, _EW_COMBOS] if (ns_first is not None and (
        ew_first is None or ns_first <= ew_first)) else [_EW_COMBOS, _NS_COMBOS]
    first_a = min((x for x in (ns_first, ew_first) if x is not None), default=0)

    t = first_a
    max_r = 5000
    while t < horizon_frames and max_r > 0:
        max_r -= 1
        for side in sides:
            cnts = _counts(t)
            best_c = side[0]
            best_p_val = -1.0
            for c in side:
                p = 0.0
                for ph in c:
                    for mv in _NEMA_PHASES.get(ph, set()):
                        p += cnts.get(mv, 0)
                if p > best_p_val or (p == best_p_val and c < best_c):
                    best_p_val = p
                    best_c = c

            total = int(best_p_val + 0.5)
            dur = cfg.min_green_f
            if total > 0:
                dur_s = total / cfg.sat_flow_vps
                dur = max(cfg.min_green_f, min(cfg.max_green_f, int(round(dur_s * fps))))

            end = t + dur
            intervals.append((t, end, best_c))
            _discharge(best_c, end)
            clr_end = end + cfg.yellow_f + cfg.all_red_f
            clears.append((end, clr_end))
            t = clr_end
            if t >= horizon_frames:
                break

        remaining = sum(max(0, len(arrivals.get(mv, [])) - served.get(mv, 0))
                        for mv in arrivals)
        if remaining == 0 and t >= horizon_frames // 2:
            break

    return intervals, clears
