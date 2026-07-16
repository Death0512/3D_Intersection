"""Tests for Phase 6 trajectory artifact consumption utilities."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))

import geometry as G
from sim.engine import simulate
from sim.trajectory import (
    apply_samples_to_motion,
    load_trajectory_index,
    samples_to_track,
    _complete_trajectory_payload,
    _exit_sample_from_metadata,
)


class TestSimulationTrajectory(unittest.TestCase):
    def test_load_trajectory_index(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "trajectory.json"), "w") as f:
                json.dump({
                    "schema": "trajectory.v1",
                    "samples": [
                        {"vehicle_id": "V1", "frame": 5},
                        {"vehicle_id": "V1", "frame": 1},
                    ],
                }, f)
            idx = load_trajectory_index(
                {"simulation_artifacts": {"trajectory": "trajectory.json"}}, td)
            self.assertEqual([s["frame"] for s in idx["V1"]], [1, 5])

    def test_samples_to_track_maps_s_to_world_position(self):
        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 40.0)
        samples = [
            {"frame": 0, "stage": "APPROACH", "s": 0.0},
            {"frame": 30, "stage": "APPROACH", "s": 40.0},
        ]

        track = samples_to_track(samples, approach, lane, anchor,
                                 road_meta={"approach_length": 40.0})

        self.assertEqual(len(track), 2)
        self.assertAlmostEqual(track[0].x, anchor[0], places=6)
        self.assertAlmostEqual(track[0].y, anchor[1], places=6)
        self.assertAlmostEqual(track[-1].x, stop[0], places=6)
        self.assertAlmostEqual(track[-1].y, stop[1], places=6)

    def test_samples_to_track_prefers_v2_world_coordinates(self):
        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 40.0)
        samples = [
            {
                "frame": 0,
                "stage": "APPROACH",
                "s": 0.0,
                "world_x": 123.456,
                "world_y": -7.89,
            },
        ]

        track = samples_to_track(samples, approach, lane, anchor,
                                 road_meta={"approach_length": 40.0})

        self.assertEqual(len(track), 1)
        self.assertAlmostEqual(track[0].x, 123.456, places=6)
        self.assertAlmostEqual(track[0].y, -7.89, places=6)

    def test_apply_samples_replaces_motion_track_in_only_when_usable(self):
        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 40.0)
        motion = G.compute_motion(
            "V1", approach, lane, G.Turn("straight"),
            speed_ms=10.0, depart_frame=0,
            appear_anchor=anchor, road_meta={"approach_length": 40.0})
        old_len = len(motion.track_in)

        updated = apply_samples_to_motion(
            motion,
            [{"frame": 0, "stage": "APPROACH", "s": 0.0},
             {"frame": 15, "stage": "APPROACH", "s": 20.0},
             {"frame": 30, "stage": "APPROACH", "s": 40.0}],
            anchor,
            road_meta={"approach_length": 40.0},
        )

        self.assertNotEqual(len(updated.track_in), old_len)
        self.assertEqual([p.frame for p in updated.track_in], [0, 15, 30])

    def test_apply_samples_replaces_exit_track_when_v3_exit_samples_exist(self):
        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 40.0)
        motion = G.compute_motion(
            "V1", approach, lane, G.Turn("straight"),
            speed_ms=10.0, depart_frame=0,
            appear_anchor=anchor, road_meta={"approach_length": 40.0})

        updated = apply_samples_to_motion(
            motion,
            [{"frame": 0, "stage": "APPROACH", "s": 0.0},
             {"frame": 30, "stage": "IN_BOX", "release_frame": 30,
              "s": 40.0},
             {"frame": 40, "stage": "EXIT", "world_x": 10.0, "world_y": 20.0},
             {"frame": 50, "stage": "EXIT", "world_x": 10.0, "world_y": 30.0}],
            anchor,
            road_meta={"approach_length": 40.0},
        )

        self.assertEqual([p.frame for p in updated.track_out], [40, 50])
        self.assertEqual(updated.reappear_frame, 40)
        self.assertEqual(updated.leave_frame, 50)
        self.assertAlmostEqual(updated.leave_pos[1], 30.0)

    def test_release_frame_in_box_sample_reaches_stop_line(self):
        vehicles = [{
            "id": "V0", "approach": "N", "lane": 1, "turn": "straight",
            "speed_ms": 30.0, "length": 4.5, "depart_frame": 0,
        }]
        out, meta = simulate(
            vehicles, approach_visible_length=1.0, fps=30,
            signal_plan=None, record_trajectories=True)
        release = out[0]["release_frame"]
        samples = [s for s in meta["trajectory_samples"] if s["vehicle_id"] == "V0"]
        self.assertTrue(any(
            s["frame"] == release and s["stage"] == "IN_BOX"
            and s["release_frame"] == release for s in samples))

        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 1.0)
        track = samples_to_track(samples, approach, lane, anchor,
                                 road_meta={"approach_length": 1.0})
        self.assertAlmostEqual(track[-1].x, stop[0], places=6)
        self.assertAlmostEqual(track[-1].y, stop[1], places=6)

    def test_samples_to_track_ignores_post_release_samples(self):
        approach = G.Direction("N")
        lane = 1
        stop = G.lane_entry_box_edge(approach, lane)
        anchor = (stop[0], stop[1] - 40.0)
        samples = [
            {"frame": 30, "stage": "IN_BOX", "release_frame": 20, "s": 40.0},
            {"frame": 40, "stage": "EXIT", "release_frame": 20, "s": 40.0},
        ]

        track = samples_to_track(samples, approach, lane, anchor,
                                 road_meta={"approach_length": 40.0})

        self.assertEqual(track, [])

    def test_exit_sample_from_metadata_basic(self):
        veh_meta = {
            "id": "V",
            "approach": "N",
            "lane": 1,
            "turn": "straight",
            "speed_ms": 8.0,
            "exit_direction": "S",
            "release_frame": 25,
        }
        frame = {"frame": 40, "pose": {"x": 1.0, "y": 2.0, "z": 0.0}}

        sample = _exit_sample_from_metadata(veh_meta, frame, 5.0)

        self.assertEqual(sample["stage"], "EXIT")
        self.assertEqual(sample["release_frame"], 25)
        self.assertEqual(sample["s"], 5.0)
        self.assertEqual(sample["world_x"], 1.0)
        self.assertEqual(sample["velocity_y"], -8.0)

    def test_complete_trajectory_preserves_samples_without_exit_frames(self):
        payload = {
            "schema": "trajectory.v2",
            "samples": [{"vehicle_id": "V1", "frame": 0, "stage": "APPROACH"}],
            "vehicles": {},
        }

        changed = _complete_trajectory_payload(payload, {"vehicles": []})

        self.assertFalse(changed)
        self.assertEqual(payload["schema"], "trajectory.v2")
        self.assertEqual(payload["samples"], [
            {"vehicle_id": "V1", "frame": 0, "stage": "APPROACH"}
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
