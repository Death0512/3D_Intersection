"""Tests for Phase 6 standalone simulation artifact export."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

import geometry as G
from scenario_gen import (
    DemandModel,
    DEFAULT_TURN_SPLIT,
    generate,
    _remap_trajectory_samples,
)
from sim.exporter import write_simulation_artifacts


class TestSimulationExporter(unittest.TestCase):
    def test_write_simulation_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = {
                "fps": 30,
                "duration_frames": 60,
                "simulator": "research",
            }
            sim_meta = {
                "trajectory_samples": [{"frame": 0, "vehicle_id": "V000"}],
                "lane_metrics": {"N_1": {"queue_length": 1}},
                "arrival_events": {(G.Direction("N"), G.Turn("straight")): [12]},
                "adaptive_intervals": [(0, 30, (2, 6))],
                "adaptive_clearances": [(30, 45)],
            }

            paths = write_simulation_artifacts(td, scenario, sim_meta)

            self.assertEqual(set(paths), {"trajectory", "lane_metrics", "simulation_meta"})
            with open(os.path.join(td, paths["trajectory"])) as f:
                traj = json.load(f)
            with open(os.path.join(td, paths["lane_metrics"])) as f:
                lanes = json.load(f)
            with open(os.path.join(td, paths["simulation_meta"])) as f:
                meta = json.load(f)
            self.assertEqual(traj["schema"], "trajectory.v1")
            self.assertEqual(lanes["schema"], "lane_metrics.v1")
            self.assertEqual(meta["schema"], "simulation_meta.v1")
            self.assertIn("N_straight", meta["arrival_events"])

    def test_research_generate_writes_artifact_references(self):
        zero_demand = DemandModel(
            flows={d: 0.0 for d in G.Direction},
            turn_split=DEFAULT_TURN_SPLIT,
        )
        with tempfile.TemporaryDirectory() as td:
            scenario = generate(42, 1.0, td, demand=zero_demand,
                                simulator="research")

            self.assertEqual(scenario["simulator"], "research")
            self.assertIn("simulation_artifacts", scenario)
            for rel in scenario["simulation_artifacts"].values():
                self.assertTrue(os.path.exists(os.path.join(td, rel)))

    def test_remap_trajectory_samples_drops_orphan_vehicle_ids(self):
        sim_meta = {"trajectory_samples": [
            {"vehicle_id": "raw-a", "frame": 0},
            {"vehicle_id": "dropped", "frame": 1},
        ]}

        _remap_trajectory_samples(sim_meta, {"raw-a": "V000"})

        self.assertEqual(sim_meta["trajectory_samples"], [
            {"vehicle_id": "V000", "frame": 0},
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
