"""Tests for the formal state-based simulation package."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import traffic_signal as SG
from sim.config import SimulationConfig
from sim.engine import SimulationEngine, simulate
from sim.lane import LaneState
from sim.state import VehicleState


def _make_vehicle(vid: str, approach: str = "N", lane: int = 1,
                  turn: str = "straight", speed_kmh: float = 54.0,
                  depart_frame: int = 0, length: float = 4.5) -> dict:
    speed_ms = speed_kmh / 3.6
    return {
        "id": vid,
        "approach": approach,
        "lane": lane,
        "turn": turn,
        "speed_kmh": speed_kmh,
        "speed_ms": speed_ms,
        "length": length,
        "depart_frame": depart_frame,
        "class": "car",
        "color": [0.5, 0.5, 0.5],
        "color_name": "gray",
        "plate": f"ABC-{vid}",
    }


class TestStateBasedEngine(unittest.TestCase):
    def test_engine_builds_vehicle_and_lane_state(self):
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        engine = SimulationEngine(cfg)
        state = engine.build_state([_make_vehicle("V001")])
        self.assertEqual(len(state.vehicles), 1)
        self.assertIn(("N", 1), state.lanes)
        self.assertIsInstance(state.lanes[("N", 1)], LaneState)
        self.assertEqual(state.vehicles[0].stage, "APPROACH")
        self.assertEqual(state.vehicles[0].desired_speed, 15.0)

    def test_compatibility_fields_are_exported(self):
        """No-signal vehicle releases immediately with zero wait."""
        v = _make_vehicle("V002")
        vehicles, meta = simulate([v], 40.0, 30)
        self.assertIs(vehicles[0], v)
        self.assertIsInstance(v["stop_frame"], int)
        self.assertIsInstance(v["release_frame"], int)
        self.assertEqual(v["queue_slot"], -1, "no signal → queue_slot -1")
        self.assertEqual(v["wait_frames"], 0, "no signal → zero wait")
        self.assertIn("arrival_events", meta)
        self.assertIn("lane_metrics", meta)

    def test_recorder_exports_state_samples_when_enabled(self):
        v = _make_vehicle("V003", approach="E")
        signal = SG.SignalPlan(fps=30)
        _, meta = simulate([v], 40.0, 30, signal_plan=signal,
                           record_trajectories=True)
        samples = meta["trajectory_samples"]
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        self.assertIn("vehicle_id", sample)
        self.assertIn("speed", sample)
        self.assertIn("accel", sample)
        self.assertIn("stage", sample)

    def test_red_light_waiting_works_with_state_engine(self):
        """E-approach vehicle waits at red, then releases during green."""
        signal = SG.SignalPlan(fps=30)
        v = _make_vehicle("V004", approach="E", depart_frame=0)
        simulate([v], 40.0, 30, signal_plan=signal)
        self.assertIn("release_frame", v, "vehicle must release during a green phase")
        self.assertIsInstance(v["release_frame"], int)
        self.assertGreater(v["wait_frames"], 0,
                           "E-approach vehicle must wait at red before release")

    def test_same_lane_follower_does_not_overtake(self):
        v1 = _make_vehicle("V005", depart_frame=0, speed_kmh=36.0, lane=1)
        v2 = _make_vehicle("V006", depart_frame=15, speed_kmh=72.0, lane=1)
        simulate([v1, v2], 40.0, 30)
        self.assertGreater(v2["stop_frame"], v1["stop_frame"])

    def test_adaptive_signal_exports_intervals(self):
        signal = SG.AdaptiveSignalPlan(fps=30)
        v = _make_vehicle("V007", approach="N", depart_frame=0)
        _, meta = simulate([v], 40.0, 30, signal_plan=signal)
        self.assertIn("adaptive_intervals", meta)
        self.assertIn("adaptive_clearances", meta)

    def test_adaptive_clearance_does_not_freeze_physics(self):
        """Vehicle longitudinal position advances during clearance.
        
        With old behavior ``_step_adaptive_if_needed`` returned True during
        clearance and caused ``continue``, skipping all physics.  This test
        forces an adaptive plan-mode clearance interval and asserts that a
        moving vehicle's ``s`` changes across ticks inside that window.
        """
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=80.0)
        signal = SG.AdaptiveSignalPlan(fps=30)
        engine = SimulationEngine(cfg, signal_plan=signal, seed=42)
        # Build state with one vehicle on N approach, departing at tick 0
        state = engine.build_state([_make_vehicle("V100", approach="N",
                                                  depart_frame=0,
                                                  speed_kmh=54.0)])
        # Force the engine into a clearance interval:
        # green_end = state.frame (so green expired immediately),
        # clearance_end = state.frame + 150 (75 yellow + 75 all-red at 30 fps)
        engine.green_end = state.frame
        engine.clearance_end = state.frame + int(round(3.0 * cfg.fps)) + int(round(2.0 * cfg.fps))
        engine.adaptive = True
        engine.adaptive_plan = signal
        # combo stays None so _is_green_now returns False (all-red behavior)

        veh = state.vehicles[0]
        s_before = veh.s
        # Tick one frame inside clearance
        state.frame = state.frame + 1
        engine._step_adaptive_if_needed(state, state.frame)
        engine._compute_accelerations(state, state.frame)
        engine._integrate_positions(state, state.frame)
        s_after = veh.s

        self.assertGreater(s_after, s_before,
                           "vehicle position must advance during clearance")
        # Safety: shouldn't overshoot the stop line
        self.assertLessEqual(s_after, cfg.stop_line_s + 0.5)

    def test_stop_frame_zero_is_preserved(self):
        """``stop_frame=0`` is a legitimate value and must survive
        ``stop_frame or tick`` / ``stop_frame or 0`` shortcuts."""
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        engine = SimulationEngine(cfg, signal_plan=None, seed=42)
        state = engine.build_state([_make_vehicle("V101", depart_frame=0)])
        veh = state.vehicles[0]
        # Simulate a vehicle that stopped at the line at frame 0
        veh.stop_frame = 0
        veh.stage = "QUEUED"
        veh.s = cfg.stop_line_s
        veh.speed = 0.0

        # Run _try_releases manually (unsignaled — no green check)
        state.frame = 5
        engine.lane_history = {veh.lane_key: []}
        engine._try_releases(state, 5)

        # stop_frame must still be 0, not replaced by tick=5
        self.assertEqual(veh.stop_frame, 0,
                         "stop_frame=0 must survive, not be shadowed by or-tick")
        self.assertEqual(veh.release_frame, 5)
        # wait_frames = release_frame(5) - stop_frame(0) = 5
        self.assertEqual(veh.vdict["wait_frames"], 5)
        # lane_history stop_frame should be 0, not release tick
        self.assertEqual(engine.lane_history[veh.lane_key],
                         [(0, 5)])


if __name__ == "__main__":
    unittest.main()
