"""Lane-level traffic state and metrics.

Phase 3 promotes lanes from passive geometry to active traffic entities.  Each
lane continuously tracks occupancy, density, flow, arrival rate, discharge
rate, cumulative delay, and the downstream space available for the intersection
box.  Rates are computed over a sliding one-second window so they reflect local
traffic state that the adaptive signal and intersection controller can observe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .state import STAGE_APPROACH, STAGE_QUEUED, VehicleState


@dataclass
class LaneMetrics:
    vehicle_count: int = 0
    moving_count: int = 0
    stopped_count: int = 0
    queue_length: int = 0
    average_speed: float = 0.0
    occupancy: float = 0.0
    density: float = 0.0
    cumulative_delay_s: float = 0.0
    # Phase 3 flow measurements
    flow_vps: float = 0.0              # discharge flow (veh/s)
    arrival_rate_vps: float = 0.0      # arrivals into the lane (veh/s)
    discharge_rate_vps: float = 0.0    # alias of flow_vps for clarity
    downstream_space_m: float = 0.0    # free space beyond the stop line


@dataclass
class LaneState:
    """Dynamic lane entity that owns ordered vehicles and measurements."""

    approach: str
    index: int
    length_m: float
    vehicles: List[VehicleState] = field(default_factory=list)
    metrics: LaneMetrics = field(default_factory=LaneMetrics)

    # Phase 3 rate tracking (sliding 1-second window)
    downstream_capacity_m: float = 30.0
    window_frames: int = 30
    _arrivals_window: int = 0
    _discharges_window: int = 0
    _window_start: int = 0
    _total_arrivals: int = 0
    _total_discharges: int = 0

    @property
    def key(self):
        return (self.approach, self.index)

    def add_vehicle(self, vehicle: VehicleState) -> None:
        self.vehicles.append(vehicle)
        self.vehicles.sort(key=lambda v: (v.depart_frame, v.vid))

    def active(self, frame: int) -> List[VehicleState]:
        xs = [v for v in self.vehicles if v.is_active_on_approach(frame)]
        xs.sort(key=lambda v: -v.s)
        return xs

    def leader_for(self, vehicle: VehicleState, frame: int) -> Optional[VehicleState]:
        active = self.active(frame)
        for i, v in enumerate(active):
            if v is vehicle and i > 0:
                return active[i - 1]
        return None

    def record_arrival(self) -> None:
        self._arrivals_window += 1
        self._total_arrivals += 1

    def record_discharge(self) -> None:
        self._discharges_window += 1
        self._total_discharges += 1

    def downstream_space(self, tick: int = 0) -> float:
        """Free space beyond the stop line for box/exit occupancy.

        A released vehicle occupies the downstream segment only while it is
        traversing the intersection box (``box_frames`` after its release).
        Used by the intersection controller (Phase 4) to block entry when the
        exit is saturated.  Pass ``tick`` for time-aware occupancy.
        """
        occupied = 0.0
        for v in self.vehicles:
            if v.release_frame is not None:
                if tick and tick >= v.release_frame + v.box_frames:
                    continue
                occupied += v.length
        return max(0.0, self.downstream_capacity_m - occupied)

    def can_accept_downstream(self, vehicle_length: float, tick: int = 0) -> bool:
        return self.downstream_space(tick) >= vehicle_length

    def update_metrics(self, frame: int, dt: float) -> LaneMetrics:
        active = self.active(frame)
        count = len(active)
        stopped = sum(1 for v in active if v.speed < 0.3 or v.stage == STAGE_QUEUED)
        moving = count - stopped
        avg_speed = sum(v.speed for v in active) / count if count else 0.0
        occupied_m = sum(v.length for v in active)
        occupancy = min(1.0, occupied_m / self.length_m) if self.length_m > 0 else 0.0
        density = count / self.length_m if self.length_m > 0 else 0.0
        delay = sum(max(0.0, v.desired_speed - v.speed) * dt for v in active)

        # Sliding-window rate finalization (every window_frames ticks).
        elapsed = frame - self._window_start
        if elapsed >= self.window_frames:
            window_s = elapsed * dt
            self.metrics.flow_vps = self._discharges_window / window_s if window_s > 0 else 0.0
            self.metrics.arrival_rate_vps = self._arrivals_window / window_s if window_s > 0 else 0.0
            self.metrics.discharge_rate_vps = self.metrics.flow_vps
            self._arrivals_window = 0
            self._discharges_window = 0
            self._window_start = frame

        self.metrics.vehicle_count = count
        self.metrics.moving_count = moving
        self.metrics.stopped_count = stopped
        self.metrics.queue_length = stopped
        self.metrics.average_speed = avg_speed
        self.metrics.occupancy = occupancy
        self.metrics.density = density
        self.metrics.cumulative_delay_s += delay
        self.metrics.downstream_space_m = self.downstream_space(frame)
        return self.metrics
