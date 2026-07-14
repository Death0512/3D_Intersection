"""Tests for Phase 2 formalized vehicle dynamics."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import geometry as G
import traffic_signal as SG
from sim.config import SimulationConfig
from sim.dynamics import (
    DEFAULT_DRIVER_MIX,
    DRIVER_PROFILES,
    DriverProfile,
    IDMDynamics,
    IntegrationMethod,
    build_idm_params,
    integrate,
    pick_profile,
)
from sim.engine import SimulationEngine, simulate
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


class TestDriverProfiles(unittest.TestCase):
    def test_profiles_are_distinct(self):
        a = DRIVER_PROFILES["aggressive"]
        n = DRIVER_PROFILES["normal"]
        c = DRIVER_PROFILES["cautious"]
        self.assertGreater(a.max_accel, n.max_accel)
        self.assertGreater(n.max_accel, c.max_accel)
        self.assertLess(a.min_gap, n.min_gap)
        self.assertLess(n.min_gap, c.min_gap)
        self.assertLess(a.time_headway, c.time_headway)
        self.assertLess(a.reaction_time_s, c.reaction_time_s)

    def test_build_idm_params_applies_speed_factor(self):
        spd = 15.0
        aggressive = build_idm_params(DRIVER_PROFILES["aggressive"], spd)
        cautious = build_idm_params(DRIVER_PROFILES["cautious"], spd)
        self.assertGreater(aggressive.desired_speed, spd)
        self.assertLess(cautious.desired_speed, spd)

    def test_build_idm_params_noise_is_deterministic(self):
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        p = DRIVER_PROFILES["normal"]
        a1 = build_idm_params(p, 15.0, rng_a).desired_speed
        a2 = build_idm_params(p, 15.0, rng_b).desired_speed
        self.assertAlmostEqual(a1, a2, places=9)

    def test_pick_profile_respects_weights(self):
        rng = random.Random(1)
        counts = {"aggressive": 0, "normal": 0, "cautious": 0}
        for _ in range(2000):
            counts[pick_profile(DEFAULT_DRIVER_MIX, rng).name] += 1
        # Normal should dominate; aggressive > cautious.
        self.assertGreater(counts["normal"], counts["cautious"])
        self.assertGreater(counts["aggressive"], 0)


class TestIntegrationMethods(unittest.TestCase):
    def _vehicle(self, speed=10.0, accel=-2.0):
        return VehicleState(
            vid="V",
            vdict={},
            approach=G.Direction.N,
            lane=1,
            turn=G.Turn.STRAIGHT,
            length=4.5,
            desired_speed=speed,
            idm_params=build_idm_params(DRIVER_PROFILES["normal"], speed),
            depart_frame=0,
            box_frames=10,
            exit_key=("N", 1),
            speed=speed,
            accel=accel,
        )

    def test_euler_uses_pre_step_speed(self):
        v = self._vehicle()
        result = integrate(v, 0.1, IntegrationMethod.EULER)
        self.assertAlmostEqual(result.position, v.s + 10.0 * 0.1)
        self.assertAlmostEqual(result.speed, 9.8)

    def test_semi_implicit_uses_updated_speed(self):
        v = self._vehicle()
        result = integrate(v, 0.1, IntegrationMethod.SEMI_IMPLICIT)
        self.assertAlmostEqual(result.position, v.s + 9.8 * 0.1)
        self.assertAlmostEqual(result.speed, 9.8)

    def test_speed_never_negative(self):
        v = self._vehicle(speed=1.0, accel=-50.0)
        result = integrate(v, 0.1, IntegrationMethod.SEMI_IMPLICIT)
        self.assertGreaterEqual(result.speed, 0.0)


class TestReactionDelay(unittest.TestCase):
    def test_reaction_applies_delayed_command(self):
        dyn = IDMDynamics(apply_reaction=True)
        v = VehicleState(
            vid="V",
            vdict={},
            approach=G.Direction.N,
            lane=1,
            turn=G.Turn.STRAIGHT,
            length=4.5,
            desired_speed=15.0,
            idm_params=build_idm_params(DRIVER_PROFILES["normal"], 15.0),
            depart_frame=0,
            box_frames=10,
            exit_key=("N", 1),
            reaction_frames=3,
        )
        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            dyn.apply(v, t)
        # At the 5th command, accel should be the command from 3 frames ago = 2.0
        self.assertAlmostEqual(v.accel, 2.0)

    def test_reaction_history_cap_covers_reaction_frames(self):
        v = VehicleState(
            vid="V",
            vdict={},
            approach=G.Direction.N,
            lane=1,
            turn=G.Turn.STRAIGHT,
            length=4.5,
            desired_speed=15.0,
            idm_params=build_idm_params(DRIVER_PROFILES["normal"], 15.0),
            depart_frame=0,
            box_frames=10,
            exit_key=("N", 1),
            reaction_frames=120,
        )

        self.assertGreaterEqual(v._accel_history.maxlen, 121)

    def test_no_reaction_uses_latest_command(self):
        dyn = IDMDynamics(apply_reaction=False)
        v = VehicleState(
            vid="V",
            vdict={},
            approach=G.Direction.N,
            lane=1,
            turn=G.Turn.STRAIGHT,
            length=4.5,
            desired_speed=15.0,
            idm_params=build_idm_params(DRIVER_PROFILES["normal"], 15.0),
            depart_frame=0,
            box_frames=10,
            exit_key=("N", 1),
        )
        dyn.apply(v, 9.9)
        self.assertAlmostEqual(v.accel, 9.9)


class TestEngineDynamicsFeatures(unittest.TestCase):
    def test_driver_mix_changes_desired_speeds(self):
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        engine = SimulationEngine(
            cfg,
            driver_mix=DEFAULT_DRIVER_MIX,
            seed=123,
        )
        vehicles = [_make_vehicle(f"V{i}") for i in range(20)]
        state = engine.build_state(vehicles)
        speeds = {st.desired_speed for st in state.vehicles}
        profiles = {st.driver_profile for st in state.vehicles}
        # Heterogeneity introduces variety in desired speeds and profiles.
        self.assertGreater(len(profiles), 1)
        self.assertGreater(len(speeds), 1)

    def test_engine_default_is_plain_normal(self):
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        engine = SimulationEngine(cfg)
        state = engine.build_state([_make_vehicle("V1")])
        st = state.vehicles[0]
        self.assertEqual(st.driver_profile, "normal")
        self.assertEqual(st.reaction_frames, 0)
        self.assertAlmostEqual(st.desired_speed, 15.0)

    def test_apply_reaction_engine_runs_and_exports_compat_fields(self):
        v = _make_vehicle("V1", depart_frame=0)
        vehicles, meta = simulate(
            [v], 40.0, 30, apply_reaction=True, seed=5)
        self.assertIn("stop_frame", v)
        self.assertIn("release_frame", v)
        self.assertIn("wait_frames", v)
        self.assertIn("arrival_events", meta)

    def test_determinism_with_seed(self):
        def scenario():
            return [_make_vehicle(f"V{i}", depart_frame=i * 20) for i in range(5)]

        a = scenario()
        simulate(a, 40.0, 30, driver_mix=DEFAULT_DRIVER_MIX, seed=42)
        b = scenario()
        simulate(b, 40.0, 30, driver_mix=DEFAULT_DRIVER_MIX, seed=42)
        for x, y in zip(a, b):
            self.assertEqual(x["stop_frame"], y["stop_frame"])
            self.assertEqual(x.get("release_frame"), y.get("release_frame"))


if __name__ == "__main__":
    unittest.main()
