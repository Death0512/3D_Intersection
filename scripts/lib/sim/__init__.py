"""State-driven microscopic traffic simulation package.

This package is the formal research-grade simulation kernel.  It is designed
to evolve independently from Blender: traffic state is advanced here, then
recorded/exported for rendering and validation.
"""

from .config import SimulationConfig
from .state import (
    STAGE_APPROACH,
    STAGE_DONE,
    STAGE_EXIT,
    STAGE_IN_BOX,
    STAGE_QUEUED,
    SimulationState,
    VehicleState,
)
from .lane import LaneMetrics, LaneState
from .engine import SimulationEngine, simulate

__all__ = [
    "SimulationConfig",
    "VehicleState",
    "SimulationState",
    "LaneState",
    "LaneMetrics",
    "SimulationEngine",
    "simulate",
    "STAGE_APPROACH",
    "STAGE_QUEUED",
    "STAGE_IN_BOX",
    "STAGE_EXIT",
    "STAGE_DONE",
]
