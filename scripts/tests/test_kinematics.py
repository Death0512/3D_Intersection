"""Phase 1 — unit tests for geometry & kinematics (pure python, no bpy).

Run:
    cd /home/death/Documents/3D_Intersection_Video
    python3 -m pytest scripts/tests/test_kinematics.py -v
  or simply:
    python3 scripts/tests/test_kinematics.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import geometry as G
from lib import kinematics as K


def test_box_and_lane_dims():
    s = G.summary()
    assert s["box_size"] == 30.0
    assert s["arm_width"] == 14.0
    assert s["axis_width"] == 30.0
    assert s["lane_centerlines"] == [-5.25, -1.75, 1.75, 5.25]
    assert s["num_lanes"] == 4


def test_routing_table_has_12_movements_per_approach():
    # 4 approaches x 4 lanes x 3 turns = 48 rows actually (lane-indexed).
    assert len(G.ROUTING_TABLE) == 4 * 4 * 3
    # 8 cameras
    assert len(G.camera_names()) == 8


def test_straight_exit_keeps_heading_same_lane():
    # straight: outbound direction == approach (keeps going same way),
    # exit lane index == entry lane index.
    for ap in G.Direction:
        ob, ex_lane = G.exit_lane_for_movement(ap, 2, G.Turn.STRAIGHT)
        assert ob == ap
        assert ex_lane == 2


def test_right_exit_turns_right():
    right_map = {G.Direction.N: G.Direction.E, G.Direction.E: G.Direction.S,
                 G.Direction.S: G.Direction.W, G.Direction.W: G.Direction.N}
    for ap in G.Direction:
        ob, ex_lane = G.exit_lane_for_movement(ap, 1, G.Turn.RIGHT)
        assert ob == right_map[ap]
        assert ex_lane == 0


def test_left_exit_turns_left():
    left_map = {G.Direction.N: G.Direction.W, G.Direction.W: G.Direction.S,
                G.Direction.S: G.Direction.E, G.Direction.E: G.Direction.N}
    for ap in G.Direction:
        ob, ex_lane = G.exit_lane_for_movement(ap, 1, G.Turn.LEFT)
        assert ob == left_map[ap]
        assert ex_lane == G.NUM_LANES - 1


def test_straight_n_exits_at_top_heading_north():
    # N-straight: disappears at box bottom (y=-15), reappears at box TOP (y=+15)
    # and continues north (+Y). leave must have larger Y than reappear.
    m = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert abs(m.disappear_pos[1] - (-G.BOX_SIZE / 2)) < 1e-6  # y = -15
    assert abs(m.reappear_pos[1] - (+G.BOX_SIZE / 2)) < 1e-6   # y = +15
    assert m.leave_pos[1] > m.reappear_pos[1]                  # goes +Y
    assert m.exit_direction == G.Direction.N


def test_straight_s_exits_at_bottom_heading_south():
    m = K.plan_motion("v", G.Direction.S, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert abs(m.disappear_pos[1] - (+G.BOX_SIZE / 2)) < 1e-6  # y = +15 (S entry at top)
    assert abs(m.reappear_pos[1] - (-G.BOX_SIZE / 2)) < 1e-6   # y = -15
    assert m.leave_pos[1] < m.reappear_pos[1]                  # goes -Y
    assert m.exit_direction == G.Direction.S


def test_right_turn_n_to_e_exits_east():
    # N-right: outbound = E. Reappear on +X edge, leave toward +X.
    m = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert m.exit_direction == G.Direction.E
    assert abs(m.reappear_pos[0] - (+G.BOX_SIZE / 2)) < 1e-6   # x = +15
    assert m.leave_pos[0] > m.reappear_pos[0]                  # goes +X


def test_compute_motion_positions_advance():
    m = K.plan_motion("v1", G.Direction.N, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    # N approach: forward = +Y. appear is behind (smaller Y) than disappear.
    assert m.appear_pos[1] < m.disappear_pos[1]
    # exit is also N (straight): reappear at top, leave further +Y.
    assert m.reappear_pos[1] < m.leave_pos[1]


def test_delta_t_straight_40kmh():
    v = K.speed_kmh_to_ms(40)  # 11.1111... m/s
    dt = G.delta_t_seconds(G.Turn.STRAIGHT, v)
    assert abs(dt - G.BOX_SIZE / v) < 1e-6
    frames = G.delta_t_frames(G.Turn.STRAIGHT, v, 30)
    assert frames == round(G.BOX_SIZE / v * 30)  # ~81


def test_delta_t_turns_longer_or_shorter():
    v = K.speed_kmh_to_ms(40)
    dt_straight = G.delta_t_seconds(G.Turn.STRAIGHT, v)
    dt_right = G.delta_t_seconds(G.Turn.RIGHT, v)
    dt_left = G.delta_t_seconds(G.Turn.LEFT, v)
    # right turn arc (radius 6) is shorter than straight; left (radius 12) shorter too
    assert dt_right < dt_straight
    assert dt_left < dt_straight
    assert dt_left > dt_right


def test_compute_motion_frame_ordering():
    m = K.plan_motion("v1", G.Direction.N, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=10)
    assert m.appear_frame < m.disappear_frame < m.reappear_frame < m.leave_frame
    # reappear - disappear == delta_t frames
    assert m.reappear_frame - m.disappear_frame == G.delta_t_frames(
        G.Turn.STRAIGHT, K.speed_kmh_to_ms(40))


def test_headway_conflict_detection():
    # departures in same lane: (frame, length, speed)
    # two cars 4.5m long at 11.1 m/s; min headway = (4.5+2)/11.1 * 30 ~ 17.6 -> 18 frames
    needed = K.min_headway_frames(4.5, 11.111)
    ok = [(0, 4.5, 11.111), (needed, 4.5, 11.111)]
    bad = [(0, 4.5, 11.111), (needed - 5, 4.5, 11.111)]
    assert K.conflict_free(ok) is True
    assert K.conflict_free(bad) is False


def test_approach_rotation():
    # arm forward is +Y (0,1). For N approach (forward +Y) rotation = 0.
    assert abs(G.approach_rotation(G.Direction.N)) < 1e-6
    # E approach forward = +X = (1,0): atan2(1,0) = pi/2
    assert abs(G.approach_rotation(G.Direction.E) - math.pi / 2) < 1e-6
    # S: forward -Y = (0,-1): atan2(0,-1) = pi
    assert abs(G.approach_rotation(G.Direction.S) - math.pi) < 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
