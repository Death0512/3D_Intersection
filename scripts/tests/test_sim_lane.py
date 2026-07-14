"""Phase 3: lanes as dynamic traffic entities (metrics, flow, downstream)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from sim.lane import LaneMetrics, LaneState
from sim.state import VehicleState, STAGE_APPROACH, STAGE_QUEUED
from sim.dynamics import build_idm_params, DRIVER_PROFILES
from car_following import IDMParams

from geometry import Direction, Turn


class TestLaneMetrics(unittest.TestCase):
    def _make_vehicle(self, vid, s, speed, stage=STAGE_APPROACH, release_frame=None,
                      box_frames=10, desired_speed=15.0):
        vdict = {
            "id": vid, "approach": "N", "lane": 1, "turn": "straight",
            "speed_ms": desired_speed, "length": 4.5, "depart_frame": 0,
        }
        return VehicleState(
            vid=vid, vdict=vdict, approach=Direction("N"), lane=1, turn=Turn("straight"),
            length=4.5, desired_speed=desired_speed,
            idm_params=build_idm_params(DRIVER_PROFILES["normal"], desired_speed),
            depart_frame=0, box_frames=box_frames, exit_key=("S", 1),
            s=s, speed=speed, stage=stage, release_frame=release_frame,
        )

    def test_metrics_count_moving_and_stopped(self):
        lane = LaneState("N", 1, 40.0)
        lane.add_vehicle(self._make_vehicle("V1", 30.0, 12.0))
        lane.add_vehicle(self._make_vehicle("V2", 10.0, 0.0, stage=STAGE_QUEUED))
        m = lane.update_metrics(100, 1.0 / 30.0)
        self.assertEqual(m.vehicle_count, 2)
        self.assertEqual(m.moving_count, 1)
        self.assertEqual(m.stopped_count, 1)
        self.assertEqual(m.queue_length, 1)
        self.assertGreater(m.occupancy, 0.0)
        self.assertGreater(m.density, 0.0)

    def test_flow_and_arrival_rates_windowed(self):
        lane = LaneState("N", 1, 40.0)
        lane.window_frames = 30
        for _ in range(3):
            lane.record_arrival()
        for _ in range(2):
            lane.record_discharge()
        # window elapses exactly at frame 30 -> 1 second at 30 fps
        m = lane.update_metrics(30, 1.0 / 30.0)
        self.assertAlmostEqual(m.arrival_rate_vps, 3.0, places=5)
        self.assertAlmostEqual(m.flow_vps, 2.0, places=5)
        self.assertAlmostEqual(m.discharge_rate_vps, 2.0, places=5)

    def test_downstream_space_frees_after_box(self):
        lane = LaneState("N", 1, 40.0)
        lane.downstream_capacity_m = 30.0
        v = self._make_vehicle("V1", 0.0, 0.0, release_frame=0, box_frames=10)
        lane.add_vehicle(v)
        occ = lane.downstream_space(5)        # inside box window
        self.assertAlmostEqual(occ, 25.5, places=5)
        freed = lane.downstream_space(20)     # after box traversal
        self.assertAlmostEqual(freed, 30.0, places=5)

    def test_can_accept_downstream(self):
        lane = LaneState("N", 1, 40.0)
        lane.downstream_capacity_m = 5.0
        v = self._make_vehicle("V1", 0.0, 0.0, release_frame=0, box_frames=10)
        lane.add_vehicle(v)  # 4.5m occupies, 0.5m free during box
        self.assertFalse(lane.can_accept_downstream(4.5, tick=5))
        self.assertTrue(lane.can_accept_downstream(0.5, tick=5))
        self.assertTrue(lane.can_accept_downstream(4.5, tick=20))


# Reuse the same engine run used by Phase 1 to verify meta contains lane metrics.
class TestEngineLaneMetricsInMeta(unittest.TestCase):
    def test_meta_contains_flow_metrics(self):
        from sim.engine import simulate

        vehicles = [{
            "id": f"V{i}", "approach": "N", "lane": 1, "turn": "straight",
            "speed_ms": 13.9, "length": 4.5, "depart_frame": 0,
        } for i in range(3)]
        _, meta = simulate(vehicles, 40.0, 30, signal_plan=None, seed=42)
        self.assertIn("lane_metrics", meta)
        key = "N_1"
        self.assertIn(key, meta["lane_metrics"])
        lm = meta["lane_metrics"][key]
        self.assertIn("flow_vps", lm)
        self.assertIn("arrival_rate_vps", lm)
        self.assertIn("downstream_space_m", lm)
        self.assertIn("cumulative_delay_s", lm)
        # structure/type checks (count is 0 at end-of-run by design: every
        # vehicle has left the approach; box occupancy lives in downstream_space)
        self.assertIsInstance(lm["flow_vps"], (int, float))
        self.assertIsInstance(lm["arrival_rate_vps"], (int, float))
        self.assertGreater(lm["cumulative_delay_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
