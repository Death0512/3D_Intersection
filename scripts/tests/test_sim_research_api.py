"""Tests for Phase 9 research API helpers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from sim.env import OfflineTrafficEnv
from sim.observation import LaneObservation, SimulationObservation, lane_observations_from_metrics
from sim.reward import delay_reward, observation_reward, queue_reward


class TestResearchAPI(unittest.TestCase):
    def test_observation_vector(self):
        obs = SimulationObservation(
            frame=0,
            lanes={"N_1": LaneObservation("N_1", queue_length=2, occupancy=0.5)},
            active_vehicle_count=3,
        )
        vec = obs.as_vector()
        # 1 (active_vehicle_count) + 6 (lane features) = 7
        self.assertEqual(len(vec), 7)
        self.assertEqual(vec[0], 3.0)  # active_vehicle_count
        self.assertEqual(vec[1], 2.0)  # queue_length
        self.assertEqual(vec[2], 0.5)  # occupancy
        self.assertEqual(vec[3], 0.0)  # average_speed (default)
        self.assertEqual(vec[4], 0.0)  # density (default)
        self.assertEqual(vec[5], 0.0)  # arrival_rate_vps (default)
        self.assertEqual(vec[6], 0.0)  # discharge_rate_vps (default)

    def test_lane_observations_from_metrics(self):
        lanes = lane_observations_from_metrics({
            "lanes": {"E_0": {"queue_length": 4, "average_speed": 1.5}}
        })
        self.assertEqual(lanes["E_0"].queue_length, 4.0)
        self.assertEqual(lanes["E_0"].average_speed, 1.5)

    def test_lane_observations_missing_lanes_does_not_create_fake_lanes(self):
        lanes = lane_observations_from_metrics({
            "schema": "lane_metrics.v2",
            "fps": 30,
            "duration_frames": 120,
        })
        self.assertEqual(lanes, {})

    def test_rewards(self):
        self.assertLess(delay_reward(10.0, throughput_vph=0.0), 0.0)
        self.assertGreater(queue_reward(0.0, average_speed=10.0), 0.0)
        obs = SimulationObservation(0, {"N_1": LaneObservation("N_1", queue_length=2)})
        self.assertLess(observation_reward(obs), 0.0)

    def test_offline_env_loads_artifacts_and_steps(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = {
                "duration_frames": 3,
                "simulation_artifacts": {
                    "trajectory": "trajectory.json",
                    "lane_metrics": "lane_metrics.json",
                },
            }
            with open(os.path.join(td, "scenario.json"), "w") as f:
                json.dump(scenario, f)
            with open(os.path.join(td, "trajectory.json"), "w") as f:
                json.dump({"samples": [
                    {"frame": 0, "vehicle_id": "V0"},
                    {"frame": 1, "vehicle_id": "V0"},
                ]}, f)
            with open(os.path.join(td, "lane_metrics.json"), "w") as f:
                json.dump({"lanes": {"N_1": {"queue_length": 1}}}, f)

            env = OfflineTrafficEnv(os.path.join(td, "scenario.json"))
            obs0 = env.reset()
            self.assertEqual(obs0.active_vehicle_count, 1)
            obs1, reward, done, info = env.step(action={"phase": "hold"})
            self.assertEqual(obs1.frame, 1)
            self.assertFalse(done)
            self.assertIn("action", info)
            self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
