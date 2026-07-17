"""Tests for Phase 5 signal-controller state machine."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import geometry as G
import traffic_signal as SG
from sim.config import SimulationConfig
from sim.engine import SimulationEngine, simulate
from sim.state import STAGE_QUEUED
from sim.signal_controller import (
    ActuatedController,
    FixedTimeController,
    MaxPressureController,
    SignalControllerConfig,
    SignalPhase,
)


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


class TestSignalControllerFSM(unittest.TestCase):
    def test_fsm_starts_with_green_at_frame_zero(self):
        ctrl = MaxPressureController(SignalControllerConfig())

        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        self.assertEqual(ctrl.current_combo, SG._NS_COMBOS[3])
        greens = [
            (d, t)
            for d in (G.Direction("N"), G.Direction("S"), G.Direction("E"), G.Direction("W"))
            for t in (G.Turn("left"), G.Turn("straight"), G.Turn("right"))
            if ctrl.is_green(d, t, 0)
        ]
        self.assertGreater(len(greens), 0)

    def test_max_pressure_starts_green_from_live_counts(self):
        cfg = SignalControllerConfig(min_green_f=2, max_green_f=5,
                                     yellow_f=1, all_red_f=1)
        ctrl = MaxPressureController(cfg)
        counts = {(G.Direction("N"), G.Turn("straight")): 3}

        ctrl.step(0, counts)

        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        self.assertTrue(ctrl.is_green(G.Direction("N"), G.Turn("straight"), 0))
        self.assertFalse(ctrl.is_green(G.Direction("E"), G.Turn("straight"), 0))

    def test_fsm_gap_out_yellow_all_red_cycle(self):
        cfg = SignalControllerConfig(min_green_f=2, max_green_f=10,
                                     yellow_f=1, all_red_f=1,
                                     extension_f=0)
        ctrl = MaxPressureController(cfg)
        demand = {(G.Direction("N"), G.Turn("straight")): 1}
        ctrl.step(0, demand)
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)

        ctrl.step(1, {})
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        ctrl.step(2, {})
        self.assertEqual(ctrl.phase, SignalPhase.YELLOW)
        self.assertFalse(ctrl.is_green(G.Direction("N"), G.Turn("straight"), 2))
        ctrl.step(3, {})
        self.assertEqual(ctrl.phase, SignalPhase.ALL_RED)
        self.assertEqual(len(ctrl.clearances()), 1)
        self.assertEqual(len(ctrl.intervals()), 1)
        self.assertEqual(ctrl.intervals()[0][1], 2)
        self.assertEqual(ctrl.clearances()[0][0], 2)
        self.assertEqual(ctrl.clearances()[0][1], 4)

    def test_fsm_max_green_forces_termination(self):
        cfg = SignalControllerConfig(min_green_f=1, max_green_f=3,
                                     yellow_f=1, all_red_f=1)
        ctrl = MaxPressureController(cfg)
        demand = {(G.Direction("N"), G.Turn("straight")): 2}
        ctrl.step(0, demand)
        ctrl.step(1, demand)
        ctrl.step(2, demand)
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        ctrl.step(3, demand)
        self.assertEqual(ctrl.phase, SignalPhase.YELLOW)

    def test_fsm_no_demand_does_not_stay_all_red(self):
        cfg = SignalControllerConfig(min_green_f=1, max_green_f=2,
                                     yellow_f=1, all_red_f=1)
        ctrl = MaxPressureController(cfg)
        phases = set()
        for tick in range(8):
            ctrl.step(tick, {})
            phases.add(ctrl.phase)
        self.assertIn(SignalPhase.GREEN, phases)
        self.assertIn(SignalPhase.YELLOW, phases)
        self.assertNotEqual(ctrl.phase, SignalPhase.ALL_RED)

    def test_engine_does_not_release_during_fsm_yellow(self):
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        cfg.signal_engine = "fsm"
        plan = SG.AdaptiveSignalPlan(fps=30)
        engine = SimulationEngine(cfg, signal_plan=plan)
        state = engine.build_state([_make_vehicle("V001")])
        st = state.vehicles[0]
        st.s = cfg.stop_line_s
        st.speed = 0.0
        st.stage = STAGE_QUEUED
        st.stop_frame = 0
        assert engine.fsm is not None
        engine.fsm.current_combo = (1, 6)  # phase 6 serves N through/right
        engine.fsm.phase = SignalPhase.YELLOW

        engine._try_releases(state, tick=10)

        self.assertIsNone(st.release_frame)


class TestEngineSignalFSMIntegration(unittest.TestCase):
    def test_engine_uses_signal_fsm_when_configured(self):
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        cfg.signal_engine = "fsm"
        plan = SG.AdaptiveSignalPlan(fps=30)
        engine = SimulationEngine(cfg, signal_plan=plan)

        self.assertIsNotNone(engine.fsm)

    def test_simulate_accepts_signal_engine_fsm(self):
        vehicles = [_make_vehicle("V001")]
        plan = SG.AdaptiveSignalPlan(fps=30)

        out, meta = simulate(
            vehicles,
            approach_visible_length=40.0,
            fps=30,
            signal_plan=plan,
            signal_engine="fsm",
        )

        self.assertIn("release_frame", out[0])
        self.assertIn("adaptive_intervals", meta)
        self.assertIn("adaptive_clearances", meta)


class TestFixedTimeController(unittest.TestCase):
    def test_cycles_through_configured_combos_on_max_green(self):
        """Fixed-cycle controller serves each configured NEMA combo in order."""
        cfg = SignalControllerConfig(min_green_f=2, max_green_f=3,
                                     yellow_f=1, all_red_f=1)
        ctrl = FixedTimeController(cfg)
        ctrl.step(0, {})
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        self.assertEqual(ctrl.current_combo, SG._NS_COMBOS[3])
        ctrl.step(1, {})
        ctrl.step(2, {})
        ctrl.step(3, {})
        self.assertEqual(ctrl.phase, SignalPhase.YELLOW)
        ctrl.step(4, {})
        self.assertEqual(ctrl.phase, SignalPhase.ALL_RED)
        ctrl.step(5, {})
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        expected_next = SG._ALL_COMBOS[(SG._ALL_COMBOS.index(SG._NS_COMBOS[3]) + 1)
                                      % len(SG._ALL_COMBOS)]
        self.assertEqual(ctrl.current_combo, expected_next)

    def test_full_cycle_produces_intervals_and_clearances(self):
        cfg = SignalControllerConfig(min_green_f=1, max_green_f=2,
                                     yellow_f=1, all_red_f=1)
        ctrl = FixedTimeController(cfg)
        # Run one complete cycle
        for t in range(5):  # 0: green, 1: green, 2: yellow, 3: all_red, 4: next green
            ctrl.step(t, {})
        self.assertEqual(len(ctrl.intervals()), 1)
        self.assertEqual(len(ctrl.clearances()), 1)


class TestActuatedController(unittest.TestCase):
    def test_gap_out_when_demand_drops(self):
        """Actuated controller terminates green when pressure drops to 0
        after min_green and extension window expires.
        _last_extend_tick is set to tick on green start and on each
        tick where own_pr > 0.  Gap-out fires when
        tick - _last_extend_tick >= extension_f AND own_pr == 0."""
        cfg = SignalControllerConfig(min_green_f=2, max_green_f=20,
                                     yellow_f=1, all_red_f=1,
                                     extension_f=3)
        ctrl = ActuatedController(cfg)
        demand = {(G.Direction("N"), G.Turn("straight")): 3}
        ctrl.step(0, demand)  # GREEN start, _last_extend=0
        self.assertEqual(ctrl.phase, SignalPhase.GREEN)
        ctrl.step(1, demand)  # _last_extend=1 (demand present)
        ctrl.step(2, demand)  # _last_extend=2, min_green reached
        # Demand drops — extension window starts from _last_extend=2
        ctrl.step(3, {})  # elapsed=3 >= min_green=2, own_pr=0, tick-last=1 < 3
        self.assertEqual(ctrl.phase, SignalPhase.GREEN, "within extension window")
        ctrl.step(4, {})  # tick-last=2 < 3
        self.assertEqual(ctrl.phase, SignalPhase.GREEN, "still in extension")
        ctrl.step(5, {})  # tick-last=3 >= 3 → gap-out → yellow
        self.assertEqual(ctrl.phase, SignalPhase.YELLOW, "gap-out after extension")

    def test_extension_on_continued_pressure(self):
        """Green extends while there is continued demand."""
        cfg = SignalControllerConfig(min_green_f=2, max_green_f=20,
                                     yellow_f=1, all_red_f=1,
                                     extension_f=2)
        ctrl = ActuatedController(cfg)
        demand = {(G.Direction("N"), G.Turn("straight")): 1}
        for t in range(10):
            ctrl.step(t, demand)
        self.assertEqual(ctrl.phase, SignalPhase.GREEN,
                         "green continues while demand exists")

    def test_max_green_forces_termination_with_demand(self):
        """Even with continued demand, max_green forces yellow."""
        cfg = SignalControllerConfig(min_green_f=1, max_green_f=4,
                                     yellow_f=1, all_red_f=1)
        ctrl = ActuatedController(cfg)
        demand = {(G.Direction("N"), G.Turn("straight")): 5}
        for t in range(5):
            ctrl.step(t, demand)
        # max_green=4, so at tick 4 it should be yellow
        self.assertEqual(ctrl.phase, SignalPhase.YELLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
