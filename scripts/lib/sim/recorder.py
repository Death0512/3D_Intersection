"""Recording/export helpers for simulation trajectories and metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .state import VehicleState


@dataclass
class FrameSample:
    frame: int
    vehicle_id: str
    approach: str
    lane: int
    turn: str
    stage: str
    s: float
    speed: float
    accel: float
    leader_id: Optional[str]
    gap: float
    release_frame: Optional[int]


class TrajectoryRecorder:
    """Collects per-frame vehicle states independent of Blender."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.samples: List[FrameSample] = []

    def record(self, frame: int, vehicles: List[VehicleState]) -> None:
        if not self.enabled:
            return
        for v in vehicles:
            # Keep exactly the release-tick sample so the rendered approach
            # track reaches the box edge; later in-box/exit motion uses the
            # legacy black-box exit track.
            if v.depart_frame <= frame and (
                    not v.has_released() or v.release_frame == frame):
                self.samples.append(FrameSample(
                    frame=frame,
                    vehicle_id=v.vid,
                    approach=v.approach.value,
                    lane=v.lane,
                    turn=v.turn.value,
                    stage=v.stage,
                    s=v.s,
                    speed=v.speed,
                    accel=v.accel,
                    leader_id=v.leader_id,
                    gap=v.gap,
                    release_frame=v.release_frame,
                ))

    def to_jsonable(self) -> List[Dict]:
        return [asdict(s) for s in self.samples]
