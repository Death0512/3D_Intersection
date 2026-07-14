"""State objects used by the microscopic simulation kernel."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import car_following as CF
import geometry as G


# Default history is ~2s at 30fps; __post_init__ expands it when a vehicle's
# configured reaction delay needs more frames.
_ACCEL_HISTORY_MINLEN = 64

STAGE_APPROACH = "APPROACH"
STAGE_QUEUED = "QUEUED"
STAGE_IN_BOX = "IN_BOX"
STAGE_EXIT = "EXIT"
STAGE_DONE = "DONE"


@dataclass
class VehicleState:
    """Dynamic state of one simulated vehicle.

    This is the source of truth during simulation.  Legacy fields such as
    ``stop_frame`` and ``release_frame`` are exported after the state evolution
    finishes so the Blender pipeline remains compatible.
    """

    vid: str
    vdict: dict
    approach: G.Direction
    lane: int
    turn: G.Turn
    length: float
    desired_speed: float
    idm_params: CF.IDMParams
    depart_frame: int
    box_frames: int
    exit_key: Tuple[str, int]

    s: float = 0.0
    speed: float = 0.0
    accel: float = 0.0
    stage: str = STAGE_APPROACH
    leader_id: Optional[str] = None
    gap: float = 1e6

    # Phase 2 dynamics metadata
    driver_profile: str = "normal"
    reaction_frames: int = 0
    _accel_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_ACCEL_HISTORY_MINLEN))

    # Phase 3 lane-entity bookkeeping
    _arrival_recorded: bool = False

    stop_frame: Optional[int] = None
    release_frame: Optional[int] = None
    queue_slot: int = -1
    active_ahead_at_stop: int = 0

    exit_s: float = 0.0
    exit_speed: float = 0.0
    exit_appear_frame: int = 0
    exit_leave_frame: int = 0

    def __post_init__(self) -> None:
        needed = max(_ACCEL_HISTORY_MINLEN, int(self.reaction_frames) + 1)
        if self._accel_history.maxlen != needed:
            self._accel_history = deque(self._accel_history, maxlen=needed)

    @property
    def movement(self) -> Tuple[G.Direction, G.Turn]:
        return (self.approach, self.turn)

    @property
    def lane_key(self) -> Tuple[str, int]:
        return (self.approach_value, self.lane)

    @property
    def approach_value(self) -> str:
        # Kept separate so tests can catch accidental enum/string drift.
        return self.approach.value

    def is_active_on_approach(self, frame: int) -> bool:
        return self.depart_frame <= frame and self.stage in (STAGE_APPROACH, STAGE_QUEUED)

    def has_released(self) -> bool:
        return self.release_frame is not None


@dataclass
class SimulationState:
    """Global state snapshot for the engine."""

    frame: int
    vehicles: List[VehicleState] = field(default_factory=list)
    lanes: Dict[Tuple[str, int], Any] = field(default_factory=dict)
    reservations: List[Any] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def active_vehicles(self, frame: Optional[int] = None) -> List[VehicleState]:
        f = self.frame if frame is None else frame
        return [v for v in self.vehicles if v.is_active_on_approach(f)]
