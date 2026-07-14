"""Phase 4: intersection conflict-resource model and downstream blocking."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import geometry as G
from geometry import Direction, Turn
from sim.conflict import (
    IntersectionModel,
    Reservation,
    _phase_for,
    movements_conflict,
)
import traffic_signal as SG


class TestMovementConflict(unittest.TestCase):
    def test_same_movement_and_same_approach_do_not_conflict(self):
        mv = (Direction.N, Turn.STRAIGHT)
        self.assertFalse(movements_conflict(mv, mv))
        self.assertFalse(movements_conflict(
            (Direction.N, Turn.STRAIGHT), (Direction.N, Turn.LEFT)))

    def test_opposing_approaches_do_not_conflict(self):
        self.assertFalse(movements_conflict(
            (Direction.N, Turn.STRAIGHT), (Direction.S, Turn.STRAIGHT)))

    def test_crossing_movements_conflict(self):
        self.assertTrue(movements_conflict(
            (Direction.N, Turn.STRAIGHT), (Direction.E, Turn.STRAIGHT)))

    def test_compatible_nema_combo_does_not_conflict(self):
        # phases 2 and 6 are a compatible NS combo
        self.assertFalse(movements_conflict(
            (Direction.S, Turn.STRAIGHT), (Direction.N, Turn.STRAIGHT)))


class TestIntersectionModel(unittest.TestCase):
    def test_can_enter_no_conflict(self):
        im = IntersectionModel()
        ok = im.can_enter((Direction.N, Turn.STRAIGHT), 0, 30, 4.5)
        self.assertTrue(ok)

    def test_conflicting_reservation_blocks_entry(self):
        im = IntersectionModel()
        im.reserve("V0", Direction.E, Turn.STRAIGHT, 0, 30)
        # N straight conflicts with E straight (crossing) during overlap
        blocked = im.can_enter((Direction.N, Turn.STRAIGHT), 5, 25, 4.5)
        self.assertFalse(blocked)

    def test_non_overlapping_reservation_allows_entry(self):
        im = IntersectionModel()
        im.reserve("V0", Direction.E, Turn.STRAIGHT, 0, 30)
        # entry after the E reservation clears
        ok = im.can_enter((Direction.N, Turn.STRAIGHT), 35, 60, 4.5)
        self.assertTrue(ok)

    def test_compatible_movements_coexist(self):
        im = IntersectionModel()
        im.reserve("V0", Direction.N, Turn.STRAIGHT, 0, 30)
        # S straight is opposing -> compatible (no conflict)
        ok = im.can_enter((Direction.S, Turn.STRAIGHT), 5, 25, 4.5)
        self.assertTrue(ok)

    def test_downstream_blocking_when_space_too_small(self):
        im = IntersectionModel()
        ok = im.can_enter((Direction.N, Turn.STRAIGHT), 0, 30, 4.5,
                          downstream_space=3.0)
        self.assertFalse(ok)

    def test_downstream_ok_when_space_sufficient(self):
        im = IntersectionModel()
        ok = im.can_enter((Direction.N, Turn.STRAIGHT), 0, 30, 4.5,
                          downstream_space=10.0)
        self.assertTrue(ok)

    def test_expire_drops_cleared_reservations(self):
        im = IntersectionModel()
        im.reserve("V0", Direction.N, Turn.STRAIGHT, 0, 30)
        im.expire(31)
        self.assertEqual(im.occupancy_count(31), 0)

    def test_expire_preserves_reservation_list_alias(self):
        im = IntersectionModel()
        alias = im.reservations
        im.reserve("V0", Direction.N, Turn.STRAIGHT, 0, 30)

        im.expire(31)

        self.assertIs(im.reservations, alias)
        self.assertEqual(alias, [])

    def test_active_zones_track_occupancy(self):
        im = IntersectionModel()
        im.reserve("V0", Direction.N, Turn.STRAIGHT, 0, 30)
        self.assertIn("center", im.active_zones(15))
        self.assertEqual(im.active_zones(40), set())


class TestPhaseMapping(unittest.TestCase):
    def test_conflict_phase_for_matches_traffic_signal(self):
        """Every approach × turn mapping matches the canonical source."""
        for approach in (Direction.N, Direction.S, Direction.E, Direction.W):
            for turn in (Turn.LEFT, Turn.STRAIGHT, Turn.RIGHT):
                self.assertEqual(
                    _phase_for(approach, turn),
                    SG._movement_to_phase(approach, turn),
                    f"mismatch for {approach} {turn}")


class TestEngineDownstreamBlocking(unittest.TestCase):
    """End-to-end: enabling downstream_blocking prevents overfilling the exit
    lane, while the default (off) preserves original behavior."""
    def _make_vehicles(self, n=6):
        return [{
            "id": f"V{i}", "approach": "N", "lane": 1, "turn": "straight",
            "speed_ms": 13.9, "length": 4.5, "depart_frame": i * 3,
        } for i in range(n)]

    def test_downstream_blocking_off_by_default(self):
        from sim.engine import SimulationEngine, SimulationConfig
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        self.assertFalse(cfg.downstream_blocking)
        eng = SimulationEngine(cfg, signal_plan=None, seed=42)
        vehicles, _ = eng.run(self._make_vehicles())
        # no signal => all vehicles eventually released (no downstream gate)
        self.assertTrue(all(v.get("release_frame") is not None
                            for v in vehicles))
        self.assertIsInstance(eng.intersection, IntersectionModel)

    def test_downstream_blocking_flag_does_not_break_release(self):
        """Smoke: enabling downstream_blocking doesn't crash and all vehicles
        still complete.  Actual blocking gate is tested by can_enter() unit
        tests above (test_downstream_blocking_when_space_too_small, etc.)."""
        from sim.engine import SimulationEngine, SimulationConfig
        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=40.0)
        cfg.downstream_blocking = True
        eng = SimulationEngine(cfg, signal_plan=None, seed=42)
        vehicles, _ = eng.run(self._make_vehicles(6))
        self.assertTrue(all(v.get("release_frame") is not None
                            for v in vehicles),
                        "all vehicles should release even with flag on")

    def test_downstream_blocking_uses_receiving_occupancy_window(self):
        """Engine gate uses receiving-lane occupancy windows, not origin lanes."""
        from sim.engine import SimulationEngine, SimulationConfig

        cfg = SimulationConfig.from_runtime(fps=30, approach_visible_length=1.0)
        cfg.downstream_blocking = True
        cfg.downstream_capacity_m = 4.5
        cfg.reaction_time_s = 0.0
        eng = SimulationEngine(cfg, signal_plan=None, seed=42)
        state = eng.build_state([{
            "id": "V0", "approach": "N", "lane": 3, "turn": "right",
            "speed_ms": 30.0, "length": 4.5, "depart_frame": 0,
        }])
        st = state.vehicles[0]
        st.stage = "QUEUED"
        st.stop_frame = 0
        st.s = cfg.stop_line_s
        # Candidate release at tick 0 would reappear after box traversal.
        box_f = G.delta_t_frames(G.Turn.RIGHT, 30.0, fps=30, lane_index=3)
        eng.exit_occupancy[st.exit_key] = [(box_f, box_f + 30, 4.5)]

        eng._try_releases(state, tick=0)
        self.assertIsNone(st.release_frame)

        eng.exit_occupancy[st.exit_key] = []
        eng._try_releases(state, tick=0)
        self.assertEqual(st.release_frame, 0)


if __name__ == "__main__":
    unittest.main()
