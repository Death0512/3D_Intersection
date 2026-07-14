"""State-driven microscopic simulation engine.

The engine is intentionally compatible with the existing ``micro_sim`` output
contract while moving the implementation to explicit state, lane, dynamics, and
recording objects.
"""
from __future__ import annotations

import random
import sys
from typing import Dict, List, Optional, Tuple

import geometry as G
import traffic_signal as SG

from .config import SimulationConfig
from .conflict import IntersectionModel
from .dynamics import (
    DEFAULT_DRIVER_MIX,
    DRIVER_PROFILES,
    IDMDynamics,
    IntegrationMethod,
    build_idm_params,
    integrate,
    pick_profile,
)
from .lane import LaneState
from .recorder import TrajectoryRecorder
from .signal_controller import (
    SignalController,
    SignalControllerConfig,
    MaxPressureController,
)
from .state import (
    STAGE_APPROACH,
    STAGE_IN_BOX,
    STAGE_QUEUED,
    SimulationState,
    VehicleState,
)

class SimulationEngine:
    """Frame-based microscopic traffic simulation engine."""

    def __init__(self, config: SimulationConfig, signal_plan=None,
                 record_trajectories: bool = False,
                 driver_mix: Optional[Dict[str, float]] = None,
                 integration: str = "semi_implicit",
                 apply_reaction: bool = False,
                 seed: int = 42):
        self.config = config
        self.signal_plan = signal_plan
        self.has_signal = signal_plan is not None
        self.adaptive = isinstance(signal_plan, SG.AdaptiveSignalPlan)
        self.adaptive_plan: Optional[SG.AdaptiveSignalPlan] = (
            signal_plan if self.adaptive else None
        )
        self.fixed_plan = signal_plan if (self.has_signal and not self.adaptive) else None
        self.dynamics = IDMDynamics(apply_reaction=apply_reaction)
        self.recorder = TrajectoryRecorder(enabled=record_trajectories)
        # Phase 5: optional closed-loop signal FSM (default: legacy plan path).
        self.fsm: Optional[SignalController] = None
        self.fsm_cfg = SignalControllerConfig(
            fps=self.config.fps,
            min_green_f=self.config.min_green_frames,
            max_green_f=self.config.max_green_frames,
            yellow_f=self.config.yellow_frames,
            all_red_f=self.config.all_red_frames,
        )
        if self.config.signal_engine == "fsm" and self.adaptive:
            self.fsm = MaxPressureController(self.fsm_cfg)
        self.driver_mix = driver_mix
        self.integration = IntegrationMethod(integration)
        self.apply_reaction = apply_reaction
        self.rng = random.Random(seed)

        self.barrier = 0
        self.combo: Optional[Tuple[int, int]] = None
        self.green_end = 0
        self.clearance_end = 0
        self.adaptive_intervals: List[Tuple[int, int, Tuple[int, int]]] = []
        self.adaptive_clearances: List[Tuple[int, int]] = []

        self.intersection = IntersectionModel()
        # ponytail: keep self.reservations as an alias to the model's list so
        # legacy introspection and tests still see live reservations.
        self.reservations = self.intersection.reservations
        self.exit_occupancy: Dict[Tuple[str, int], List[Tuple[int, int, float]]] = {}
        self.exit_last_leave: Dict[Tuple[str, int], int] = {}
        self.lane_last_release: Dict[Tuple[str, int], int] = {}
        self.lane_history: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        self.wait_list: List[int] = []

    def build_state(self, vehicles: list) -> SimulationState:
        for v in vehicles:
            for k in ("stop_frame", "release_frame", "queue_slot", "wait_frames",
                      "entry_frame", "clear_frame"):
                v.pop(k, None)

        states: List[VehicleState] = []
        lanes: Dict[Tuple[str, int], LaneState] = {}
        for v in vehicles:
            ap = G.Direction(v["approach"])
            lane = int(v["lane"])
            turn = G.Turn(v["turn"])
            spd = float(v["speed_ms"])
            box_f = G.delta_t_frames(turn, spd, self.config.fps, lane_index=lane)
            out_dir, ex_lane = G.exit_lane_for_movement(ap, lane, turn)
            profile = (
                pick_profile(self.driver_mix, self.rng)
                if self.driver_mix else DRIVER_PROFILES["normal"]
            )
            idm_params = build_idm_params(
                profile, spd, self.rng if self.driver_mix else None)
            reaction_frames = (
                profile.reaction_frames(self.config.fps)
                if self.apply_reaction else 0
            )
            st = VehicleState(
                vid=v["id"],
                vdict=v,
                approach=ap,
                lane=lane,
                turn=turn,
                length=float(v["length"]),
                desired_speed=idm_params.desired_speed,
                idm_params=idm_params,
                depart_frame=int(v["depart_frame"]),
                box_frames=box_f,
                exit_key=(out_dir.value, ex_lane),
                speed=spd,
                driver_profile=profile.name,
                reaction_frames=reaction_frames,
            )
            states.append(st)
            key = (ap.value, lane)
            lanes.setdefault(key, LaneState(ap.value, lane, self.config.approach_visible_length))
            lanes[key].add_vehicle(st)

        self.lane_history = {k: [] for k in lanes}
        start = min((s.depart_frame for s in states), default=0)
        return SimulationState(frame=start, vehicles=states, lanes=lanes)

    def horizon_for(self, state: SimulationState) -> int:
        max_depart = max((s.depart_frame for s in state.vehicles), default=0)
        traverse = int(round((self.config.approach_visible_length * 2) / 5.0 * self.config.fps))
        return max_depart + traverse + int(round(self.config.max_horizon_buffer_s * self.config.fps))

    def run(self, vehicles: list) -> Tuple[List[dict], Dict]:
        state = self.build_state(vehicles)
        if self.adaptive and self.fsm is None:
            self.green_end = state.frame
            self.clearance_end = state.frame

        horizon = self.horizon_for(state)
        for tick in range(state.frame, horizon):
            state.frame = tick
            if self.fsm is not None:
                self._step_signal_fsm(state, tick)
            elif self.adaptive:
                self._step_adaptive_if_needed(state, tick)
            self._record_arrivals(state, tick)
            self._compute_accelerations(state, tick)
            self._integrate_positions(state, tick)
            self._try_releases(state, tick)
            self.intersection.expire(tick)
            for lane in state.lanes.values():
                lane.update_metrics(tick, self.config.dt)
            self.recorder.record(tick, state.vehicles)
            if all(st.release_frame is not None for st in state.vehicles):
                break

        self._finalize_unreleased(state)
        return vehicles, self._export_meta(state)

    def _step_adaptive_if_needed(self, state: SimulationState, tick: int) -> None:
        """Evaluate adaptive plan transition at tick.
        
        Physics continues every frame; this function only decides whether to
        select a new combo when the current green expires.  During clearance
        (yellow/all-red) it does nothing.  ``_is_green_now`` returns False
        because ``tick >= green_end``, but acceleration/integration still run.
        """
        if not self.adaptive or tick < self.green_end:
            return
        if tick < self.clearance_end:
            # clearance in progress — keep combo=None, let physics run
            return
        # green expired and clearance finished → select next combo
        self.combo = None
        counts = self._lane_counts(state, tick)
        assert self.adaptive_plan is not None
        self.combo, self.green_end, self.clearance_end, self.barrier = \
            self.adaptive_plan.observe_and_decide(tick, counts, self.barrier)
        self.adaptive_intervals.append((tick, self.green_end, self.combo))
        self.adaptive_clearances.append((self.green_end, self.clearance_end))

    def _lane_counts(self, state: SimulationState, tick: int) -> Dict[Tuple[G.Direction, G.Turn], int]:
        counts: Dict[Tuple[G.Direction, G.Turn], int] = {}
        for st in state.vehicles:
            if st.stage in (STAGE_APPROACH, STAGE_QUEUED) and \
               st.release_frame is None and st.depart_frame <= tick:
                counts[st.movement] = counts.get(st.movement, 0) + 1
        return counts

    def _step_signal_fsm(self, state: SimulationState, tick: int) -> None:
        if self.fsm is not None:
            self.fsm.step(tick, self._lane_counts(state, tick))

    def _is_green_now(self, approach: G.Direction, turn: G.Turn, tick: int) -> bool:
        if not self.has_signal:
            return True
        if self.fsm is not None:
            return self.fsm.is_green(approach, turn, tick)
        if self.adaptive:
            if tick >= self.green_end:
                return False
            return self.combo is not None and SG._movement_to_phase(approach, turn) in self.combo
        return bool(self.fixed_plan is not None and self.fixed_plan.is_green(approach, turn, tick))

    def _record_arrivals(self, state: SimulationState, tick: int) -> None:
        """Record a lane arrival when a vehicle first becomes active."""
        for lane in state.lanes.values():
            for st in lane.vehicles:
                if not st._arrival_recorded and st.depart_frame == tick:
                    lane.record_arrival()
                    st._arrival_recorded = True

    def _compute_accelerations(self, state: SimulationState, tick: int) -> None:
        for lane in state.lanes.values():
            active = lane.active(tick)
            for i, st in enumerate(active):
                leader = active[i - 1] if i > 0 else None
                green = self._is_green_now(st.approach, st.turn, tick)
                target = self.dynamics.acceleration(st, leader, green, self.config)
                self.dynamics.apply(st, target)

    def _integrate_positions(self, state: SimulationState, tick: int) -> None:
        for lane in state.lanes.values():
            active = lane.active(tick)
            for idx, st in enumerate(active):
                result = integrate(st, self.config.dt, self.integration)
                new_s = result.position
                st.speed = result.speed
                if idx > 0:
                    leader = active[idx - 1]
                    max_s = leader.s - leader.length - 0.1
                    if new_s > max_s:
                        new_s = max_s
                        st.speed = min(st.speed, leader.speed)
                st.s = new_s
                if st.release_frame is None and st.s >= self.config.stop_line_s:
                    st.s = self.config.stop_line_s
                    st.speed = 0.0
                min_gap = st.idm_params.min_gap
                near_stop = st.s >= self.config.stop_line_s - min_gap - 1.0
                at_stop_line = st.s >= self.config.stop_line_s - 0.1
                if st.stop_frame is None and (at_stop_line or (near_stop and st.speed < 0.3)):
                    st.stop_frame = tick
                    st.stage = STAGE_QUEUED
                if st.stage == STAGE_APPROACH and st.speed < 0.3 and \
                   st.s > self.config.stop_line_s * 0.3:
                    if st.stop_frame is None:
                        st.stop_frame = tick
                    st.stage = STAGE_QUEUED

    def _try_releases(self, state: SimulationState, tick: int) -> None:
        for key in sorted(state.lanes.keys()):
            lane = state.lanes[key]
            front = None
            for st in lane.vehicles:
                if st.stage == STAGE_QUEUED and st.release_frame is None and \
                   st.stop_frame is not None and st.stop_frame <= tick:
                    front = st
                    break
            if front is None:
                continue
            if not self._is_green_now(front.approach, front.turn, tick):
                continue
            prev = self.lane_last_release.get(key)
            if prev is not None and tick < prev + self.config.reaction_frames:
                continue
            entry_f = tick
            clear_f = entry_f + front.box_frames
            # Phase 4: downstream-space gate (off by default for compat).
            downstream_space = None
            if self.config.downstream_blocking:
                travel_f = int(round(self.config.approach_visible_length / front.desired_speed * self.config.fps))
                candidate_reappear = clear_f
                candidate_leave = candidate_reappear + travel_f
                downstream_space = self._downstream_space(
                    front.exit_key, candidate_reappear, candidate_leave)
            if not self.intersection.can_enter(
                    front.movement, entry_f, clear_f,
                    front.length, downstream_space):
                continue
            travel_f = int(round(self.config.approach_visible_length / front.desired_speed * self.config.fps))
            reappear_f = clear_f
            leave_f = reappear_f + travel_f
            last_exit = self.exit_last_leave.get(front.exit_key, -999)
            if reappear_f < last_exit + self.config.exit_buffer_frames:
                continue

            front.release_frame = tick
            state.lanes[front.lane_key].record_discharge()
            if front.stop_frame is not None:
                w = tick - front.stop_frame
            else:
                w = 0
            front.vdict["wait_frames"] = max(w, 0)
            self.wait_list.append(max(w, 0))
            sf = front.stop_frame if front.stop_frame is not None else 0
            active_ahead = sum(1 for _, rel in self.lane_history[key] if rel > sf)
            front.queue_slot = active_ahead if (self.has_signal and (w > 0 or active_ahead)) else -1
            front.stage = STAGE_IN_BOX
            self.intersection.reserve(
                front.vid, front.approach, front.turn, entry_f, clear_f)
            if self.config.downstream_blocking:
                self._record_exit_occupancy(front.exit_key, reappear_f, leave_f, front.length)
            self.exit_last_leave[front.exit_key] = leave_f
            self.lane_last_release[key] = tick
            sf_hist = front.stop_frame if front.stop_frame is not None else tick
            self.lane_history[key].append((sf_hist, tick))

    def _downstream_space(self, exit_key: Tuple[str, int],
                          start_frame: int, end_frame: int) -> float:
        active = []
        occupied = 0.0
        for entry_f, clear_f, length in self.exit_occupancy.get(exit_key, []):
            if clear_f <= start_frame:
                continue
            active.append((entry_f, clear_f, length))
            if entry_f < end_frame and start_frame < clear_f:
                occupied += length
        self.exit_occupancy[exit_key] = active
        return max(0.0, self.config.downstream_capacity_m - occupied)

    def _record_exit_occupancy(self, exit_key: Tuple[str, int],
                               entry_frame: int, clear_frame: int,
                               length: float) -> None:
        self.exit_occupancy.setdefault(exit_key, []).append(
            (entry_frame, clear_frame, length))

    def _finalize_unreleased(self, state: SimulationState) -> None:
        for st in state.vehicles:
            if st.release_frame is None:
                if st.stop_frame is None:
                    travel_f = int(round(self.config.approach_visible_length / st.desired_speed * self.config.fps))
                    st.stop_frame = st.depart_frame + travel_f
                if not self.has_signal:
                    st.release_frame = st.stop_frame
                    st.queue_slot = -1
                    st.vdict["wait_frames"] = 0
                else:
                    st.vdict.setdefault("wait_frames", 0)
            st.vdict["stop_frame"] = st.stop_frame
            if st.release_frame is not None:
                st.vdict["release_frame"] = st.release_frame
            st.vdict["queue_slot"] = st.queue_slot

    def _export_meta(self, state: SimulationState) -> Dict:
        arrival_events: Dict[Tuple[G.Direction, G.Turn], List[int]] = {}
        for st in state.vehicles:
            if st.stop_frame is not None:
                arrival_events.setdefault(st.movement, []).append(st.stop_frame)
        for mv in arrival_events:
            arrival_events[mv].sort()
        meta: Dict = {
            "arrival_events": arrival_events,
            "trajectory_samples": self.recorder.to_jsonable(),
            "lane_metrics": {
                f"{k[0]}_{k[1]}": lane.metrics.__dict__.copy()
                for k, lane in state.lanes.items()
            },
        }
        if self.fsm is not None:
            meta["adaptive_intervals"] = self.fsm.intervals()
            # Preserve the existing scenario_gen contract: clearances are
            # exported as (start, end), while intervals retain their combo.
            meta["adaptive_clearances"] = [
                (s, e) for s, e, _combo in self.fsm.clearances()
            ]
        elif self.adaptive:
            meta["adaptive_intervals"] = self.adaptive_intervals
            meta["adaptive_clearances"] = self.adaptive_clearances
        if self.has_signal:
            n_tot = len(state.vehicles)
            n_q = sum(1 for v in state.vehicles if v.queue_slot >= 0)
            mean_w = sum(self.wait_list) / len(self.wait_list) if self.wait_list else 0.0
            max_w = max(self.wait_list) if self.wait_list else 0
            print(f"[sim.engine] {n_tot} veh, {n_q} queued, "
                  f"mean_wait={mean_w:.0f}f max_wait={max_w}f",
                  file=sys.stderr, flush=True)
        return meta


def simulate(vehicles: list,
             approach_visible_length: float,
             fps: int,
             signal_plan=None,
             seed: int = 42,
             record_trajectories: bool = False,
             driver_mix: Optional[Dict[str, float]] = None,
             integration: str = "semi_implicit",
             apply_reaction: bool = False,
             signal_engine: str = "plan") -> Tuple[List[dict], Dict]:
    """Compatibility entry point matching ``micro_sim.simulate``.

    The default options (no driver mix, semi-implicit integration, no reaction
    delay) preserve the original deterministic behavior.  Pass ``driver_mix``,
    ``integration="euler"`` or ``apply_reaction=True`` to enable the Phase 2
    formalized dynamics features.
    """
    cfg = SimulationConfig.from_runtime(fps=fps, approach_visible_length=approach_visible_length)
    cfg.signal_engine = signal_engine
    engine = SimulationEngine(
        cfg,
        signal_plan=signal_plan,
        record_trajectories=record_trajectories,
        driver_mix=driver_mix,
        integration=integration,
        apply_reaction=apply_reaction,
        seed=seed,
    )
    return engine.run(vehicles)
