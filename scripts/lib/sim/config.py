"""Configuration objects for the state-driven simulation engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Global parameters for one microscopic simulation run.

    Frames are used as the public clock because the renderer and scenario
    metadata are frame-based.  The integration timestep is derived from fps.
    """

    fps: int = 30
    approach_visible_length: float = 40.0
    exit_buffer_frames: int = 5
    saturation_flow_vps: float = 1.0
    min_green_frames: int = 240
    max_green_frames: int = 1200
    yellow_frames: int = 90
    all_red_frames: int = 60
    reaction_time_s: float = 0.5
    max_horizon_buffer_s: float = 180.0
    downstream_blocking: bool = False  # Phase 4: gate box entry on exit-lane space
    downstream_capacity_m: float = 30.0
    signal_engine: str = "plan"       # Phase 5: "plan" (legacy timeline) | "fsm"

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def stop_line_s(self) -> float:
        return self.approach_visible_length

    @property
    def reaction_frames(self) -> int:
        return int(round(self.reaction_time_s * self.fps))

    @classmethod
    def from_runtime(cls, fps: int, approach_visible_length: float) -> "SimulationConfig":
        """Build config matching the current pipeline's timing conventions."""
        return cls(
            fps=fps,
            approach_visible_length=approach_visible_length,
            min_green_frames=int(round(8.0 * fps)),
            max_green_frames=int(round(40.0 * fps)),
            yellow_frames=int(round(3.0 * fps)),
            all_red_frames=int(round(2.0 * fps)),
        )
