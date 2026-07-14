"""Tests for the IDM-based microscopic simulation."""
from __future__ import annotations

import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import geometry as G
import traffic_signal as SG
import micro_sim as MS

import unittest


def _make_vehicle(vid: str, approach: str = "N", lane: int = 1,
                  turn: str = "straight", speed_kmh: float = 54.0,
                  depart_frame: int = 0, length: float = 4.5) -> dict:
    """Create a minimal vehicle dict matching scenario_gen output format."""
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


class TestSingleVehicleFreeFlow(unittest.TestCase):
    """Single vehicle on an empty road with no signal."""

    def test_single_vehicle_reaches_stop_line(self):
        """Vehicle should reach the stop line and be released (no signal)."""
        v = _make_vehicle("V001", depart_frame=0, speed_kmh=54.0)
        vehicles = [v]
        result, meta = MS.simulate(vehicles, 40.0, 30, signal_plan=None)
        self.assertIn("stop_frame", v)
        self.assertIn("release_frame", v)
        self.assertEqual(v["queue_slot"], -1)
        self.assertEqual(v["wait_frames"], 0)

    def test_release_frame_equals_stop_frame_no_signal(self):
        """Without a signal, release should happen immediately on arrival."""
        v = _make_vehicle("V002", depart_frame=0, speed_kmh=54.0)
        vehicles = [v]
        MS.simulate(vehicles, 40.0, 30, signal_plan=None)
        # IDM may not release at exact analytical arrival, but release == stop
        self.assertEqual(v["release_frame"], v["stop_frame"])

    def test_stop_frame_is_realistic(self):
        """Stop frame should be approximately approach_length / speed * fps."""
        v = _make_vehicle("V003", depart_frame=0, speed_kmh=54.0)
        MS.simulate([v], 40.0, 30, signal_plan=None)
        expected = int(round(40.0 / (54.0 / 3.6) * 30))
        # Allow ±30% tolerance (IDM dynamics vs constant-speed assumption)
        self.assertAlmostEqual(v["stop_frame"], expected, delta=expected * 0.3)


class TestTwoVehiclesCarFollowing(unittest.TestCase):
    """Two vehicles in the same lane — leader-following behavior."""

    def test_no_collision(self):
        """Second vehicle should not collide with the first."""
        v1 = _make_vehicle("V010", depart_frame=0, speed_kmh=54.0, lane=1)
        v2 = _make_vehicle("V011", depart_frame=30, speed_kmh=54.0, lane=1)
        MS.simulate([v1, v2], 40.0, 30, signal_plan=None)
        # Both should have stop_frame and release_frame
        self.assertIn("stop_frame", v1)
        self.assertIn("stop_frame", v2)
        self.assertIn("release_frame", v1)
        self.assertIn("release_frame", v2)
        # Second vehicle should arrive later
        self.assertGreater(v2["stop_frame"], v1["stop_frame"])

    def test_follower_respects_headway(self):
        """Follower release should be staggered from leader release."""
        v1 = _make_vehicle("V020", depart_frame=0, speed_kmh=54.0, lane=1)
        v2 = _make_vehicle("V021", depart_frame=30, speed_kmh=54.0, lane=1)
        MS.simulate([v1, v2], 40.0, 30, signal_plan=None)
        # Release frames should be at least reaction_frames apart
        min_gap = int(round(0.5 * 30))  # 15 frames
        self.assertGreaterEqual(
            v2["release_frame"] - v1["release_frame"], min_gap)


class TestSignalStopAndGo(unittest.TestCase):
    """Vehicles with a fixed traffic signal."""

    def test_vehicle_waits_at_red(self):
        """Vehicle on a red phase should wait."""
        signal = SG.SignalPlan(fps=30)
        # E/W approach is red during the first 30s (NS green)
        # Depart at frame 0, approach = E
        v = _make_vehicle("V030", approach="E", depart_frame=0, speed_kmh=54.0)
        MS.simulate([v], 40.0, 30, signal_plan=signal)
        self.assertIn("stop_frame", v)
        if "release_frame" in v:
            # Should have waited (wait_frames > 0)
            self.assertGreater(v["wait_frames"], 0)

    def test_vehicle_passes_on_green(self):
        """Vehicle on a green phase should pass without waiting."""
        signal = SG.SignalPlan(fps=30)
        # N/S approach is green during the first 30s
        v = _make_vehicle("V031", approach="N", depart_frame=0, speed_kmh=54.0)
        MS.simulate([v], 40.0, 30, signal_plan=signal)
        self.assertIn("stop_frame", v)
        self.assertIn("release_frame", v)
        self.assertEqual(v["wait_frames"], 0)

    def test_queue_slot_assigned_for_waiting(self):
        """Queued vehicles should get queue_slot >= 0."""
        signal = SG.SignalPlan(fps=30)
        # Multiple E vehicles arriving on red
        vehicles = [
            _make_vehicle(f"V04{i}", approach="E", depart_frame=i * 30,
                         speed_kmh=54.0, lane=1)
            for i in range(3)
        ]
        MS.simulate(vehicles, 40.0, 30, signal_plan=signal)
        # At least some should have been queued
        queued = [v for v in vehicles if v.get("queue_slot", -1) >= 0]
        self.assertGreater(len(queued), 0)


class TestMultipleLanes(unittest.TestCase):
    """Vehicles in different lanes should not interfere."""

    def test_different_lanes_independent(self):
        """Vehicles in different lanes should have independent timing."""
        v1 = _make_vehicle("V050", lane=0, depart_frame=0, speed_kmh=54.0)
        v2 = _make_vehicle("V051", lane=2, depart_frame=0, speed_kmh=54.0)
        MS.simulate([v1, v2], 40.0, 30, signal_plan=None)
        # Both should arrive at similar times (same speed, no interaction)
        self.assertAlmostEqual(v1["stop_frame"], v2["stop_frame"], delta=5)


class TestOutputFormatCompatibility(unittest.TestCase):
    """Ensure output format matches intersection_sim.simulate() contract."""

    def test_required_fields_present(self):
        """All required scheduling fields should be set."""
        v = _make_vehicle("V060", depart_frame=0)
        MS.simulate([v], 40.0, 30, signal_plan=None)
        self.assertIn("stop_frame", v)
        self.assertIn("release_frame", v)
        self.assertIn("queue_slot", v)
        self.assertIn("wait_frames", v)

    def test_meta_has_arrival_events(self):
        """Meta dict should have arrival_events."""
        v = _make_vehicle("V061", depart_frame=0)
        _, meta = MS.simulate([v], 40.0, 30, signal_plan=None)
        self.assertIn("arrival_events", meta)
        events = meta["arrival_events"]
        self.assertIsInstance(events, dict)

    def test_adaptive_meta_has_intervals(self):
        """With adaptive signal, meta should have intervals and clearances."""
        signal = SG.AdaptiveSignalPlan(fps=30)
        v = _make_vehicle("V062", approach="N", depart_frame=0)
        _, meta = MS.simulate([v], 40.0, 30, signal_plan=signal)
        self.assertIn("adaptive_intervals", meta)
        self.assertIn("adaptive_clearances", meta)


class TestDeterminism(unittest.TestCase):
    """Same inputs should produce identical outputs."""

    def test_deterministic_results(self):
        """Two runs with the same input should produce identical stop/release frames."""
        def make_scenario():
            return [
                _make_vehicle("V070", depart_frame=0, speed_kmh=54.0, lane=1),
                _make_vehicle("V071", depart_frame=45, speed_kmh=60.0, lane=1),
                _make_vehicle("V072", depart_frame=0, speed_kmh=54.0, lane=2),
            ]

        v1 = make_scenario()
        MS.simulate(v1, 40.0, 30, signal_plan=None, seed=42)

        v2 = make_scenario()
        MS.simulate(v2, 40.0, 30, signal_plan=None, seed=42)

        for a, b in zip(v1, v2):
            self.assertEqual(a["stop_frame"], b["stop_frame"])
            self.assertEqual(a.get("release_frame"), b.get("release_frame"))
            self.assertEqual(a["wait_frames"], b["wait_frames"])


class TestSpeedIsOutput(unittest.TestCase):
    """Key thesis claim: speed emerges from IDM dynamics, not fixed input."""

    def test_follower_slows_behind_leader(self):
        """A fast follower should slow down behind a slow leader — speed
        is an *output* of IDM interaction, not a fixed input."""
        # Slow leader
        v1 = _make_vehicle("V080", depart_frame=0, speed_kmh=36.0, lane=1)
        # Fast follower close behind
        v2 = _make_vehicle("V081", depart_frame=15, speed_kmh=72.0, lane=1)
        MS.simulate([v1, v2], 40.0, 30, signal_plan=None)
        # Both should eventually reach stop line — the follower should not
        # pass through the leader. Follower arrives after leader.
        self.assertGreater(v2["stop_frame"], v1["stop_frame"])

    def test_idm_with_signal_changes_speed(self):
        """Vehicle approaching red should decelerate — actual speed differs
        from the assigned desired_speed."""
        signal = SG.SignalPlan(fps=30)
        # E approach is red for first 30s
        v = _make_vehicle("V082", approach="E", depart_frame=0, speed_kmh=54.0)
        MS.simulate([v], 40.0, 30, signal_plan=signal)
        # Vehicle was forced to stop → wait_frames > 0 proves speed changed
        if "release_frame" in v:
            self.assertGreater(v["wait_frames"], 0)


class TestQueueFormation(unittest.TestCase):
    """Multiple vehicles queuing at a red light."""

    def test_queue_fifo(self):
        """Vehicles should be released in FIFO order."""
        signal = SG.SignalPlan(fps=30)
        # N approach vehicles during green — should pass in order
        vehicles = [
            _make_vehicle(f"V09{i}", approach="N", depart_frame=i * 40,
                         speed_kmh=54.0, lane=1)
            for i in range(4)
        ]
        MS.simulate(vehicles, 40.0, 30, signal_plan=signal)
        released = [v for v in vehicles if "release_frame" in v]
        # Release frames should be monotonically increasing
        for i in range(len(released) - 1):
            self.assertLessEqual(
                released[i]["release_frame"],
                released[i + 1]["release_frame"])


class TestStaleFieldsCleared(unittest.TestCase):
    """Ensure stale scheduling fields are cleared on re-simulation."""

    def test_clears_stale_fields(self):
        """Running simulate twice should not carry over stale data."""
        v = _make_vehicle("V100", depart_frame=0)
        v["stop_frame"] = 999
        v["release_frame"] = 999
        v["queue_slot"] = 99
        v["wait_frames"] = 99
        MS.simulate([v], 40.0, 30, signal_plan=None)
        self.assertNotEqual(v["stop_frame"], 999)


if __name__ == "__main__":
    unittest.main()
