"""Tests for Phase 8 extension modules."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from sim.lane_changing import LaneChangeContext, MOBILParams, mobil_incentive, mobil_safe, should_change_lane
from sim.participants import AgentType, participant_spec
from sim.routing import route_from_od, shortest_path, single_intersection_route


class TestMOBIL(unittest.TestCase):
    def test_incentive_positive_for_better_target_lane(self):
        ctx = LaneChangeContext(current_accel=0.2, target_accel=1.0)
        self.assertGreater(mobil_incentive(ctx), 0.0)
        self.assertTrue(should_change_lane(ctx, MOBILParams(threshold=0.1)))

    def test_safety_blocks_hard_braking_new_follower(self):
        ctx = LaneChangeContext(current_accel=0.0, target_accel=2.0,
                                new_follower_accel_after=-6.0)
        self.assertFalse(mobil_safe(ctx, MOBILParams(safe_decel=4.0)))
        self.assertFalse(should_change_lane(ctx, MOBILParams(safe_decel=4.0)))


class TestRouting(unittest.TestCase):
    def test_shortest_path(self):
        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        self.assertEqual(shortest_path(graph, "A", "D"), ["A", "B", "D"])

    def test_route_from_od(self):
        route = route_from_od({"A": ["B"], "B": []}, "A", "B")
        self.assertEqual(route.movements, ["A->B"])

    def test_single_intersection_route(self):
        route = single_intersection_route("N", "left")
        self.assertEqual(route.origin, "N_in")
        self.assertEqual(route.destination, "W_out")
        self.assertEqual(route.movements, ["N_left"])


class TestParticipants(unittest.TestCase):
    def test_participant_specs(self):
        ped = participant_spec("pedestrian")
        emergency = participant_spec(AgentType.EMERGENCY)
        self.assertLess(ped.desired_speed_mps, emergency.desired_speed_mps)
        self.assertGreater(emergency.priority, ped.priority)


if __name__ == "__main__":
    unittest.main(verbosity=2)
