"""Phase 9 — observation helpers for research interfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LaneObservation:
    lane_id: str
    queue_length: float = 0.0
    occupancy: float = 0.0
    average_speed: float = 0.0
    density: float = 0.0
    arrival_rate_vps: float = 0.0
    discharge_rate_vps: float = 0.0

    def as_vector(self) -> List[float]:
        return [
            self.queue_length,
            self.occupancy,
            self.average_speed,
            self.density,
            self.arrival_rate_vps,
            self.discharge_rate_vps,
        ]


@dataclass(frozen=True)
class SimulationObservation:
    frame: int
    lanes: Dict[str, LaneObservation] = field(default_factory=dict)
    active_vehicle_count: int = 0

    def as_vector(self) -> List[float]:
        out: List[float] = [float(self.active_vehicle_count)]
        for lane_id in sorted(self.lanes):
            out.extend(self.lanes[lane_id].as_vector())
        return out


def lane_observations_from_metrics(metrics: dict) -> Dict[str, LaneObservation]:
    schema = str(metrics.get("schema", "lane_metrics.v1"))
    lanes = metrics.get("lanes", {})
    out: Dict[str, LaneObservation] = {}
    for lane_id, values in lanes.items():
        if schema.startswith("lane_metrics.v2"):
            values = values.get("summary", {})
        if not isinstance(values, dict):
            values = {}
        out[str(lane_id)] = LaneObservation(
            lane_id=str(lane_id),
            queue_length=float(values.get("queue_length", 0.0)),
            occupancy=float(values.get("occupancy", 0.0)),
            average_speed=float(values.get("average_speed", 0.0)),
            density=float(values.get("density", 0.0)),
            arrival_rate_vps=float(values.get("arrival_rate_vps", 0.0)),
            discharge_rate_vps=float(values.get("discharge_rate_vps", 0.0)),
        )
    return out


__all__ = ["LaneObservation", "SimulationObservation", "lane_observations_from_metrics"]
