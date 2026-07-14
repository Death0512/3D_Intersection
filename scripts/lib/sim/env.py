"""Phase 9 — lightweight offline research environment.

This is a Gym-like reader over exported simulation artifacts.  It is intended
for offline RL/digital-twin experiments and dataset inspection; it does not yet
mutate the live simulator in response to actions.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .observation import SimulationObservation, lane_observations_from_metrics
from .reward import observation_reward


class OfflineTrafficEnv:
    def __init__(self, scenario_path: str):
        self.scenario_path = os.path.abspath(scenario_path)
        self.base_dir = os.path.dirname(self.scenario_path)
        with open(self.scenario_path) as f:
            self.scenario = json.load(f)
        self.duration_frames = int(self.scenario.get("duration_frames", 0))
        self.frame = 0
        self._samples_by_frame = self._load_samples()
        self._lane_metrics = self._load_lane_metrics()

    def _artifact_path(self, key: str) -> Optional[str]:
        rel = self.scenario.get("simulation_artifacts", {}).get(key)
        return os.path.join(self.base_dir, rel) if rel else None

    def _load_samples(self) -> Dict[int, List[dict]]:
        path = self._artifact_path("trajectory")
        if not path or not os.path.exists(path):
            return {}
        with open(path) as f:
            payload = json.load(f)
        out: Dict[int, List[dict]] = {}
        for sample in payload.get("samples", []):
            out.setdefault(int(sample.get("frame", 0)), []).append(sample)
        return out

    def _load_lane_metrics(self) -> dict:
        path = self._artifact_path("lane_metrics")
        if not path or not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def reset(self) -> SimulationObservation:
        self.frame = 0
        return self.observe()

    def observe(self) -> SimulationObservation:
        lanes = lane_observations_from_metrics(self._lane_metrics)
        return SimulationObservation(
            frame=self.frame,
            lanes=lanes,
            active_vehicle_count=len(self._samples_by_frame.get(self.frame, [])),
        )

    def step(self, action: Any = None) -> Tuple[SimulationObservation, float, bool, Dict[str, Any]]:
        self.frame += 1
        obs = self.observe()
        done = self.frame >= max(0, self.duration_frames - 1)
        info = {"action": action, "frame": self.frame}
        return obs, observation_reward(obs), done, info


__all__ = ["OfflineTrafficEnv"]
