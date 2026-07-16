"""Tests for Phase 6 standalone simulation artifact export."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

import geometry as G
import render
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
            self.assertIn("vehicles", traj)
            self.assertEqual(lanes["schema"], "lane_metrics.v1")
            self.assertEqual(meta["schema"], "simulation_meta.v1")
            self.assertIn("N_straight", meta["arrival_events"])

    def test_write_lane_metrics_v2_with_timeseries(self):
        scenario = {"fps": 30, "duration_frames": 10, "simulator": "research"}
        sim_meta = {
            "lane_metrics": {"N_1": {"queue_length": 2}},
            "lane_metrics_timeseries": {"N_1": [
                {"frame": 0, "queue_length": 1},
                {"frame": 1, "queue_length": 2},
            ]},
        }
        with tempfile.TemporaryDirectory() as td:
            paths = write_simulation_artifacts(td, scenario, sim_meta)
            with open(os.path.join(td, paths["lane_metrics"])) as f:
                payload = json.load(f)

        self.assertEqual(payload["schema"], "lane_metrics.v2")
        self.assertEqual(payload["lanes"]["N_1"]["summary"]["queue_length"], 2)
        self.assertEqual(len(payload["lanes"]["N_1"]["timeseries"]), 2)

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
            with open(os.path.join(td, scenario["simulation_artifacts"]["trajectory"])) as f:
                traj = json.load(f)
            self.assertEqual(traj["schema"], "trajectory.v2")
            self.assertEqual(traj["coordinate"], "world+lane-longitudinal")

    def test_remap_trajectory_samples_drops_orphan_vehicle_ids(self):
        sim_meta = {"trajectory_samples": [
            {"vehicle_id": "raw-a", "frame": 0},
            {"vehicle_id": "dropped", "frame": 1},
        ]}

        _remap_trajectory_samples(sim_meta, {"raw-a": "V000"})

        self.assertEqual(sim_meta["trajectory_samples"], [
            {"vehicle_id": "V000", "frame": 0},
        ])

    def test_research_trajectory_v2_contains_world_fields(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = generate(7, 1.0, td, simulator="research")

            with open(os.path.join(td, scenario["simulation_artifacts"]["trajectory"])) as f:
                traj = json.load(f)

            self.assertEqual(traj["schema"], "trajectory.v2")
            self.assertTrue(traj["samples"])
            sample = traj["samples"][0]
            for key in ("world_x", "world_y", "world_z", "velocity_x", "velocity_y"):
                self.assertIn(key, sample)
            self.assertNotIn("heading", sample)
            self.assertNotIn("yaw", sample)
            self.assertIn(sample["vehicle_id"], traj["vehicles"])
            veh_trace = traj["vehicles"][sample["vehicle_id"]]
            self.assertIn("spawn_position", veh_trace)
            self.assertIn("exit_position", veh_trace)
            self.assertIn("heading", veh_trace)

    def test_metadata_pass_upgrades_trajectory_v3_with_exit_samples(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = generate(7, 2.0, td, simulator="research")

            render._write_metadata(scenario, td)

            with open(os.path.join(td, scenario["simulation_artifacts"]["trajectory"])) as f:
                traj = json.load(f)
            self.assertEqual(traj["schema"], "trajectory.v3")
            exit_samples = [s for s in traj["samples"] if s.get("stage") == "EXIT"]
            self.assertTrue(exit_samples)
            sample = exit_samples[0]
            for key in ("world_x", "world_y", "world_z", "velocity_x", "velocity_y"):
                self.assertIn(key, sample)
            veh_trace = traj["vehicles"][sample["vehicle_id"]]
            self.assertIn("reappear_position", veh_trace)
            self.assertIn("leave_position", veh_trace)
            self.assertEqual(veh_trace["exit_position"], veh_trace["leave_position"])

    def test_metadata_uses_trajectory_v2_world_pose(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = generate(7, 1.0, td, simulator="research")
            with open(os.path.join(td, scenario["simulation_artifacts"]["trajectory"])) as f:
                traj = json.load(f)

            sample = next(
                s for s in traj["samples"]
                if 0 <= int(s["frame"]) < scenario["duration_frames"])
            meta = render.compute_metadata(scenario, PROJECT_ROOT, run_dir=td)
            vm = next(v for v in meta["vehicles"] if v["id"] == sample["vehicle_id"])
            frame_meta = next(f for f in vm["frames"] if f["frame"] == sample["frame"])

            self.assertAlmostEqual(frame_meta["pose"]["x"], sample["world_x"], places=3)
            self.assertAlmostEqual(frame_meta["pose"]["y"], sample["world_y"], places=3)
            self.assertEqual(vm["appear_frame"], min(s["frame"] for s in traj["samples"]
                                                     if s["vehicle_id"] == sample["vehicle_id"]
                                                     and s["stage"] in {"APPROACH", "QUEUED"}))

    def test_research_metadata_preserves_disappear_timing_after_clip_window(self):
        with tempfile.TemporaryDirectory() as td:
            scenario = generate(7, 1.0, td, simulator="research")
            meta = render.compute_metadata(scenario, PROJECT_ROOT, run_dir=td)

            traj_path = os.path.join(td, scenario["simulation_artifacts"]["trajectory"])
            with open(traj_path) as f:
                traj = json.load(f)

            traj_by_vid = {}
            for sample in traj["samples"]:
                traj_by_vid.setdefault(sample["vehicle_id"], []).append(sample)

            for veh in meta["vehicles"]:
                samples = traj_by_vid.get(veh["id"], [])
                if not samples:
                    continue
                in_window = [s for s in samples if 0 <= int(s["frame"]) < scenario["duration_frames"]]
                if not in_window:
                    continue
                before = next(v for v in scenario["vehicles"] if v["id"] == veh["id"])
                if veh["disappear_frame"] < before["release_frame"]:
                    self.fail(f"{veh['id']} disappear_frame regressed from {before['release_frame']} to {veh['disappear_frame']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
