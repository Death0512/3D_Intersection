"""Tests for the IDM car-following module."""
from __future__ import annotations

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from car_following import (
    IDMParams, idm_acceleration, free_road_acceleration,
    virtual_signal_leader, effective_leader,
    DEFAULT_DESIRED_SPEED, DEFAULT_MAX_ACCEL, DEFAULT_COMFORT_DECEL,
    DEFAULT_MIN_GAP, DEFAULT_TIME_HEADWAY, DEFAULT_DELTA,
)

import unittest


class TestIDMParams(unittest.TestCase):
    def test_defaults(self):
        p = IDMParams()
        self.assertAlmostEqual(p.desired_speed, DEFAULT_DESIRED_SPEED)
        self.assertAlmostEqual(p.max_accel, DEFAULT_MAX_ACCEL)
        self.assertEqual(p.delta, DEFAULT_DELTA)

    def test_invalid_desired_speed(self):
        with self.assertRaises(ValueError):
            IDMParams(desired_speed=0)
        with self.assertRaises(ValueError):
            IDMParams(desired_speed=-1)

    def test_invalid_max_accel(self):
        with self.assertRaises(ValueError):
            IDMParams(max_accel=0)

    def test_invalid_comfort_decel(self):
        with self.assertRaises(ValueError):
            IDMParams(comfortable_decel=-1)

    def test_invalid_min_gap(self):
        with self.assertRaises(ValueError):
            IDMParams(min_gap=-1)


class TestIDMAcceleration(unittest.TestCase):
    def setUp(self):
        self.params = IDMParams()

    def test_free_road_accelerates_from_rest(self):
        """Vehicle starting from rest on an empty road should accelerate."""
        a = idm_acceleration(
            speed=0.0, desired_speed=15.0, gap=1e6, delta_v=0.0,
            params=self.params)
        self.assertGreater(a, 0.0)
        # Should be close to max_accel when starting from rest
        self.assertAlmostEqual(a, self.params.max_accel, places=1)

    def test_free_road_at_desired_speed_no_accel(self):
        """Vehicle at desired speed on free road: acceleration ≈ 0."""
        a = idm_acceleration(
            speed=15.0, desired_speed=15.0, gap=1e6, delta_v=0.0,
            params=self.params)
        self.assertAlmostEqual(a, 0.0, places=1)

    def test_close_leader_decelerates(self):
        """Vehicle close to a slower leader should decelerate."""
        a = idm_acceleration(
            speed=15.0, desired_speed=15.0, gap=5.0, delta_v=5.0,
            params=self.params)
        self.assertLess(a, 0.0)

    def test_stopped_leader_produces_full_stop(self):
        """Vehicle approaching a stopped leader at close range decelerates hard."""
        a = idm_acceleration(
            speed=10.0, desired_speed=15.0, gap=3.0, delta_v=10.0,
            params=self.params)
        self.assertLess(a, -1.0)

    def test_zero_gap_extreme_braking(self):
        """Zero gap should produce extreme braking."""
        a = idm_acceleration(
            speed=10.0, desired_speed=15.0, gap=0.0, delta_v=5.0,
            params=self.params)
        self.assertLess(a, -2.0)

    def test_speed_never_negative_integration(self):
        """Simulate a vehicle approaching a wall; speed should never go negative."""
        params = IDMParams()
        v = 15.0
        s = 50.0  # initial gap to wall
        dt = 0.1
        for _ in range(1000):
            a = idm_acceleration(
                speed=v, desired_speed=15.0, gap=s, delta_v=v,
                params=params)
            v = max(0.0, v + a * dt)
            s = max(0.0, s - v * dt)
            self.assertGreaterEqual(v, 0.0)
        # Should have stopped before the wall
        self.assertAlmostEqual(v, 0.0, places=1)

    def test_idm_reproduces_known_free_road_accel(self):
        """At v=0 on free road: a = a_max * [1 - 0 - 0] = a_max."""
        p = IDMParams(desired_speed=30.0, max_accel=1.5, delta=4)
        a = idm_acceleration(speed=0.0, desired_speed=30.0, gap=1e6,
                             delta_v=0.0, params=p)
        self.assertAlmostEqual(a, 1.5, places=3)

    def test_converges_to_desired_speed(self):
        """Vehicle on free road should converge to desired speed."""
        params = IDMParams(desired_speed=20.0)
        v = 0.0
        dt = 0.1
        for _ in range(5000):
            a = free_road_acceleration(v, params)
            v = max(0.0, v + a * dt)
        self.assertAlmostEqual(v, 20.0, places=0)

    def test_leader_following_equilibrium(self):
        """Two vehicles at same speed with large gap: acceleration ≈ 0 (at v0)."""
        params = IDMParams(desired_speed=15.0)
        # At v = v0 with delta_v = 0, the desired gap s* = s0 + v*T.
        # The free-road term (v/v0)^4 = 1.0, so a = a_max*(1 - 1 - (s*/s)^2).
        # For a ≈ 0 we need the gap term to vanish → very large gap.
        # With a large gap and v == v0, accel should be near zero.
        a = idm_acceleration(
            speed=15.0, desired_speed=15.0, gap=500.0, delta_v=0.0,
            params=params)
        self.assertAlmostEqual(a, 0.0, delta=0.1)

    def test_faster_than_desired_speed_decelerates(self):
        """Vehicle above desired speed on free road decelerates."""
        params = IDMParams(desired_speed=15.0)
        a = free_road_acceleration(20.0, params)
        self.assertLess(a, 0.0)


class TestFreeRoadAcceleration(unittest.TestCase):
    def test_matches_idm_with_infinite_gap(self):
        params = IDMParams()
        a1 = free_road_acceleration(5.0, params)
        a2 = idm_acceleration(5.0, params.desired_speed, 1e6, 0.0, params)
        self.assertAlmostEqual(a1, a2, places=6)


class TestVirtualSignalLeader(unittest.TestCase):
    def test_green_returns_none(self):
        result = virtual_signal_leader(
            vehicle_s=10.0, stop_line_s=40.0,
            vehicle_speed=10.0, is_green=True)
        self.assertIsNone(result)

    def test_red_returns_stop_line(self):
        result = virtual_signal_leader(
            vehicle_s=10.0, stop_line_s=40.0,
            vehicle_speed=10.0, is_green=False)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 40.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_past_stop_line_returns_none(self):
        """Vehicle already past stop line should not be blocked by virtual leader."""
        result = virtual_signal_leader(
            vehicle_s=41.0, stop_line_s=40.0,
            vehicle_speed=10.0, is_green=False)
        self.assertIsNone(result)


class TestEffectiveLeader(unittest.TestCase):
    def test_no_leaders_free_road(self):
        gap, dv = effective_leader(
            vehicle_s=10.0, vehicle_speed=15.0, vehicle_length=4.5,
            physical_leader=None, signal_leader=None)
        self.assertGreater(gap, 1e5)
        self.assertAlmostEqual(dv, 0.0)

    def test_physical_leader_only(self):
        # Leader at s=50, length=4.5, speed=10
        gap, dv = effective_leader(
            vehicle_s=10.0, vehicle_speed=15.0, vehicle_length=4.5,
            physical_leader=(50.0, 10.0, 4.5), signal_leader=None)
        self.assertAlmostEqual(gap, 50.0 - 4.5 - 10.0)  # 35.5
        self.assertAlmostEqual(dv, 5.0)  # 15 - 10

    def test_signal_leader_only(self):
        gap, dv = effective_leader(
            vehicle_s=10.0, vehicle_speed=15.0, vehicle_length=4.5,
            physical_leader=None, signal_leader=(40.0, 0.0))
        self.assertAlmostEqual(gap, 30.0)  # 40 - 10
        self.assertAlmostEqual(dv, 15.0)  # 15 - 0

    def test_signal_closer_than_physical(self):
        """Signal leader at stop line is closer than physical leader."""
        gap, dv = effective_leader(
            vehicle_s=10.0, vehicle_speed=15.0, vehicle_length=4.5,
            physical_leader=(100.0, 15.0, 4.5),  # far away
            signal_leader=(20.0, 0.0))             # close
        self.assertAlmostEqual(gap, 10.0)  # 20 - 10
        self.assertAlmostEqual(dv, 15.0)

    def test_physical_closer_than_signal(self):
        """Physical leader is closer than stop line."""
        gap, dv = effective_leader(
            vehicle_s=10.0, vehicle_speed=15.0, vehicle_length=4.5,
            physical_leader=(20.0, 5.0, 4.5),  # gap = 20 - 4.5 - 10 = 5.5
            signal_leader=(40.0, 0.0))           # gap = 30
        self.assertAlmostEqual(gap, 5.5)
        self.assertAlmostEqual(dv, 10.0)

    def test_red_light_causes_stop_via_idm(self):
        """Integration test: vehicle approaching red light should stop."""
        params = IDMParams(desired_speed=15.0)
        v = 15.0
        s = 0.0  # position along approach
        stop_line = 40.0
        dt = 1.0 / 30.0  # 30 fps
        length = 4.5

        for _ in range(600):  # 20 seconds
            sig = virtual_signal_leader(s, stop_line, v, is_green=False)
            gap, dv = effective_leader(s, v, length,
                                       physical_leader=None,
                                       signal_leader=sig)
            a = idm_acceleration(v, params.desired_speed, gap, dv, params)
            v = max(0.0, v + a * dt)
            s += v * dt

        # Vehicle should have stopped before the stop line
        self.assertLess(s, stop_line + 0.5)
        self.assertAlmostEqual(v, 0.0, places=1)

    def test_green_light_no_stop(self):
        """Vehicle at green light should pass through."""
        params = IDMParams(desired_speed=15.0)
        v = 15.0
        s = 0.0
        stop_line = 40.0
        dt = 1.0 / 30.0
        length = 4.5

        for _ in range(200):
            sig = virtual_signal_leader(s, stop_line, v, is_green=True)
            gap, dv = effective_leader(s, v, length,
                                       physical_leader=None,
                                       signal_leader=sig)
            a = idm_acceleration(v, params.desired_speed, gap, dv, params)
            v = max(0.0, v + a * dt)
            s += v * dt

        # Vehicle should have passed the stop line
        self.assertGreater(s, stop_line)
        self.assertGreater(v, 10.0)


if __name__ == "__main__":
    unittest.main()
