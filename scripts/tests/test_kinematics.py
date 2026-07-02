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
from lib import envfile as ENV


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
    # right-hand driving: RIGHT turn -> curb-side exit lane (NUM_LANES-1).
    right_map = {G.Direction.N: G.Direction.E, G.Direction.E: G.Direction.S,
                 G.Direction.S: G.Direction.W, G.Direction.W: G.Direction.N}
    for ap in G.Direction:
        ob, ex_lane = G.exit_lane_for_movement(ap, 1, G.Turn.RIGHT)
        assert ob == right_map[ap]
        assert ex_lane == G.NUM_LANES - 1


def test_left_exit_turns_left():
    # right-hand driving: LEFT turn -> median-side exit lane (index 0).
    left_map = {G.Direction.N: G.Direction.W, G.Direction.W: G.Direction.S,
                G.Direction.S: G.Direction.E, G.Direction.E: G.Direction.N}
    for ap in G.Direction:
        ob, ex_lane = G.exit_lane_for_movement(ap, 1, G.Turn.LEFT)
        assert ob == left_map[ap]
        assert ex_lane == 0


def test_straight_n_exits_at_top_heading_north():
    # N-straight: disappears at box bottom (y=-15), reappears at box TOP (y=+15)
    # and continues north (+Y). leave must have larger Y than reappear.
    # Per-shot frame: lanes centred on road axis; lane 0 (median-side) at
    # x = LANE_CENTERLINES[0] = -5.25.
    m = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert abs(m.disappear_pos[1] - (-G.BOX_SIZE / 2)) < 1e-6  # y = -15
    assert abs(m.reappear_pos[1] - (+G.BOX_SIZE / 2)) < 1e-6   # y = +15
    assert m.leave_pos[1] > m.reappear_pos[1]                  # goes +Y
    assert m.exit_direction == G.Direction.N
    # axis-centred: entry & exit lane 0 at x = LANE_CENTERLINES[0] = -5.25
    assert abs(m.disappear_pos[0] - G.LANE_CENTERLINES[0]) < 1e-6
    assert abs(m.reappear_pos[0] - G.LANE_CENTERLINES[0]) < 1e-6


def test_straight_s_exits_at_bottom_heading_south():
    # S approach: forward = -Y; lane 0 (median-side) at x = +5.25
    # (right of -Y heading is +X; off = +5.25 rotated by S-right = +X -> x=+5.25... wait:
    # S forward = (0,-1), S right = (-1,0)*(-1) = actually approach_right(S):
    # fx,fy=(0,-1) -> right=(fy,-fx)=(-1,0) so x = (-1)*(-5.25)=+5.25).
    m = K.plan_motion("v", G.Direction.S, 0, G.Turn.STRAIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert abs(m.disappear_pos[1] - (+G.BOX_SIZE / 2)) < 1e-6  # y = +15 (S entry at top)
    assert abs(m.reappear_pos[1] - (-G.BOX_SIZE / 2)) < 1e-6   # y = -15
    assert m.leave_pos[1] < m.reappear_pos[1]                  # goes -Y
    assert m.exit_direction == G.Direction.S
    # S right = (-1,0); lane 0 offset = (-1)*(-5.25) = +5.25
    expected_x = (-1) * G.LANE_CENTERLINES[0]   # = +5.25
    assert abs(m.disappear_pos[0] - expected_x) < 1e-6
    assert abs(m.reappear_pos[0] - expected_x) < 1e-6


def test_n_and_s_per_shot_lanes_are_axis_centred():
    # Per-shot frame: each .blend is independent. N lane 0 x = -5.25;
    # S lane 0 x = +5.25 (S right = (-1,0)); they are symmetric about 0.
    # They CAN share the same |x| value because they are in separate scenes.
    for lane in range(G.NUM_LANES):
        nx, _ = G.lane_entry_box_edge(G.Direction.N, lane)
        sx, _ = G.lane_entry_box_edge(G.Direction.S, lane)
        # N right = (+1,0) -> x = LANE_CENTERLINES[lane]
        assert abs(nx - G.LANE_CENTERLINES[lane]) < 1e-6
        # S right = (-1,0) -> x = -LANE_CENTERLINES[lane]
        assert abs(sx - (-G.LANE_CENTERLINES[lane])) < 1e-6
    # All lanes stay within the arm width
    for lane in range(G.NUM_LANES):
        nx, _ = G.lane_entry_box_edge(G.Direction.N, lane)
        assert abs(nx) <= G.ARM_WIDTH / 2 + 1e-6


def test_right_turn_n_to_e_exits_east():
    # N-right: outbound = E, exit lane = curb-side (NUM_LANES-1 = 3).
    # Reappear on +X edge (x=+15), leave toward +X.
    # E approach right = (0,-1)*... approach_right(E): fx,fy=(1,0)->right=(0,-1).
    # Lane 3 (curb-side) offset: right_vec*(LANE_CENTERLINES[3]) = (0,-1)*5.25 -> y=-5.25
    m = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT,
                      speed_ms=K.speed_kmh_to_ms(40), depart_frame=0)
    assert m.exit_direction == G.Direction.E
    assert abs(m.reappear_pos[0] - (+G.BOX_SIZE / 2)) < 1e-6   # x = +15
    assert m.leave_pos[0] > m.reappear_pos[0]                  # goes +X
    # E right = (0,-1); lane 3 y = (0,-1)*5.25 = -5.25
    expected_y = (-1) * G.LANE_CENTERLINES[G.NUM_LANES - 1]    # = -5.25
    assert abs(m.reappear_pos[1] - expected_y) < 1e-6


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


def test_compute_motion_with_appear_anchor_overrides_start():
    # An explicit appear_anchor sets appear_pos; disappear_pos is derived
    # forward from it by approach_visible_length. Timing is unchanged.
    v = K.speed_kmh_to_ms(40)
    base = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0, fps=30)
    anchored = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0, fps=30,
                             appear_anchor=(10.0, -100.0))
    # appear_pos is the anchor
    assert anchored.appear_pos == (10.0, -100.0)
    # disappear is 40m forward (N = +Y) from the anchor
    assert abs(anchored.disappear_pos[0] - 10.0) < 1e-9
    assert abs(anchored.disappear_pos[1] - (-100.0 + 40.0)) < 1e-9
    # frames unchanged (anchor only moves position, not timing)
    assert anchored.appear_frame == base.appear_frame
    assert anchored.disappear_frame == base.disappear_frame
    # the out segment is untouched when only appear_anchor is given
    assert anchored.reappear_pos == base.reappear_pos


def test_compute_motion_with_reappear_anchor_overrides_out_segment():
    v = K.speed_kmh_to_ms(40)
    base = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT, v, 0, fps=30)
    anchored = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT, v, 0, fps=30,
                             reappear_anchor=(50.0, 20.0))
    assert anchored.reappear_pos == (50.0, 20.0)
    # exit direction for N->right is E (+X): leave is 40m further along +X
    assert abs(anchored.leave_pos[0] - (50.0 + 40.0)) < 1e-9
    assert abs(anchored.leave_pos[1] - 20.0) < 1e-9
    # in segment untouched
    assert anchored.appear_pos == base.appear_pos


def test_anchor_equivalent_to_default_when_matching_geometry():
    # When the anchor equals the geometry-derived position, anchored motion
    # is identical to the default motion (proves env files that match geometry
    # reproduce the procedural output exactly).
    v = K.speed_kmh_to_ms(45)
    base = K.plan_motion("V0", G.Direction.N, 1, G.Turn.RIGHT, v, 5, fps=30)
    anchored = K.plan_motion("V0", G.Direction.N, 1, G.Turn.RIGHT, v, 5, fps=30,
                             appear_anchor=base.appear_pos,
                             reappear_anchor=base.reappear_pos)
    assert anchored.appear_pos == base.appear_pos
    assert anchored.disappear_pos == base.disappear_pos
    assert anchored.reappear_pos == base.reappear_pos
    assert anchored.leave_pos == base.leave_pos


def test_road_meta_in_anchor_drives_to_box_edge():
    # When road_meta + appear_anchor are given, disappear_pos = lane_entry_box_edge
    # (the box near edge), not anchor + fixed visible_length.
    v = K.speed_kmh_to_ms(40)
    road = {"approach_length": 54.751}
    anchor = (10.0, -100.0)
    anchored = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                             v, 0, fps=30,
                             appear_anchor=anchor, road_meta=road)
    # disappear is the box near edge for N lane 0
    expected_disappear = G.lane_entry_box_edge(G.Direction.N, 0)
    assert abs(anchored.disappear_pos[0] - expected_disappear[0]) < 1e-9
    assert abs(anchored.disappear_pos[1] - expected_disappear[1]) < 1e-9
    # appear is the anchor itself
    assert anchored.appear_pos == anchor
    # out segment (no reappear_anchor) uses geometry default (unchanged)
    assert anchored.reappear_pos == G.lane_exit_box_edge(
        *G.exit_lane_for_movement(G.Direction.N, 0, G.Turn.STRAIGHT))


def test_road_meta_out_anchor_drives_to_road_end():
    # When road_meta + reappear_anchor are given, leave_pos = anchor + outbound *
    # road_meta["approach_length"] (the far end of the exit road).
    v = K.speed_kmh_to_ms(40)
    road = {"approach_length": 54.751}
    # N turn right -> outbound = E (+X)
    anchor = (15.0, -5.25)  # typical out_E lane 3 reappear anchor
    anchored = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT,
                             v, 0, fps=30,
                             reappear_anchor=anchor, road_meta=road)
    assert anchored.reappear_pos == anchor
    # leave is 54.751 m east of the reappear anchor
    assert abs(anchored.leave_pos[0] - (15.0 + 54.751)) < 1e-9
    assert abs(anchored.leave_pos[1] - (-5.25)) < 1e-9
    # In segment (no appear_anchor) uses geometry default
    base = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT, v, 0, fps=30)
    assert anchored.appear_pos == base.appear_pos


def test_road_meta_timing_scales_with_full_distance():
    # Frame counts should increase proportionally when travel distance grows.
    v = K.speed_kmh_to_ms(10)  # slow vehicle to amplify frame differences
    road = {"approach_length": 54.751}
    short = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                          v, 0, fps=30,
                          appear_anchor=(-5.25, -55.0))  # 40m travel
    full = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                         v, 0, fps=30,
                         appear_anchor=(-5.25, -55.0), road_meta=road)
    # full travel (55→15=40m) same as short, so frames are equal
    # When anchor is at -75 (60m to box edge), frames should be ~1.5x
    further = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                            v, 0, fps=30,
                            appear_anchor=(-5.25, -75.0), road_meta=road)
    # 60m / 40m = 1.5 → 1.5 * short frames (rounding may give slight diff)
    short_frames = short.disappear_frame - short.appear_frame
    further_frames = further.disappear_frame - further.appear_frame
    assert further_frames > short_frames, f"{further_frames} <= {short_frames}"
    # 60m at 10km/h (2.778 m/s) = 21.6 s = 648 frames @30fps
    # 40m = 14.4 s = 432 frames
    assert abs(further_frames - 648) <= 1


def test_headway_conflict_detection():
    # departures in same lane: (frame, length, speed)
    # two cars 4.5m long at 11.1 m/s; min headway = (4.5+2)/11.1 * 30 ~ 17.6 -> 18 frames
    needed = K.min_headway_frames(4.5, 11.111)
    ok = [(0, 4.5, 11.111), (needed, 4.5, 11.111)]
    bad = [(0, 4.5, 11.111), (needed - 5, 4.5, 11.111)]
    assert K.conflict_free(ok) is True
    assert K.conflict_free(bad) is False


def test_catchup_faster_follower_unsafe_at_same_frame():
    # Error 5: a 60 km/h follower released at the same frame as a 30 km/h
    # leader in the same lane WILL catch up on the 40 m approach -> unsafe.
    ls = K.speed_kmh_to_ms(30)
    fs = K.speed_kmh_to_ms(60)
    assert K.catchup_safe(0, ls, 4.5, 0, fs) is False


def test_catchup_faster_follower_safe_when_delayed():
    # Pushing the faster follower's depart_frame to the computed minimum makes
    # the schedule catch-up-safe while preserving both speeds.
    ls = K.speed_kmh_to_ms(30)
    fs = K.speed_kmh_to_ms(60)
    mf = K.min_follow_depart_frame(0, ls, 4.5, fs)
    assert K.catchup_safe(0, ls, 4.5, mf, fs) is True
    # one frame earlier is unsafe (boundary)
    assert K.catchup_safe(0, ls, 4.5, mf - 1, fs) is False


def test_catchup_slower_follower_always_safe():
    # A slower follower can never catch the leader -> safe regardless of gap.
    ls = K.speed_kmh_to_ms(60)
    fs = K.speed_kmh_to_ms(30)
    assert K.catchup_safe(0, ls, 4.5, 5, fs) is True


def test_schedule_departures_is_catchup_safe():
    # End-to-end: a generated scenario must have no start-gap OR catch-up
    # conflicts in any (approach, lane). For each adjacent pair sorted by
    # depart_frame, the later vehicle (follower) must not catch the earlier
    # one (leader) before the leader enters the Black Box.
    import random
    import scenario_gen as S
    rng = random.Random(7)
    veh = [S.make_vehicle(f"V{i:03d}", rng) for i in range(40)]
    S.schedule_departures(veh, 300, rng)
    lanes = {}
    for v in veh:
        lanes.setdefault((v["approach"], v["lane"]), []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda d: d["depart_frame"])
        for a, b in zip(vs, vs[1:]):
            # a is the leader (earlier), b is the follower (later)
            assert K.catchup_safe(a["depart_frame"], a["speed_ms"], a["length"],
                                  b["depart_frame"], b["speed_ms"]), (key, a["id"], b["id"])


def test_approach_rotation():
    # arm forward is +Y (0,1). For N approach (forward +Y) rotation = 0.
    assert abs(G.approach_rotation(G.Direction.N)) < 1e-6
    # E approach forward = +X = (1,0): atan2(1,0) = pi/2
    assert abs(G.approach_rotation(G.Direction.E) - math.pi / 2) < 1e-6
    # S: forward -Y = (0,-1): atan2(0,-1) = pi
    assert abs(G.approach_rotation(G.Direction.S) - math.pi) < 1e-6


def test_camera_pose_axis_centred_and_road_json_driven():
    # camera_pose must be a pure function of road_meta (no hardcoded road dims)
    # and must sit on the road axis (lateral x = 0 for N approach).
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    cam_loc, look = G.camera_pose(G.Direction.N, True, road_meta)
    # N approach: camera on axis -> x = 0
    assert abs(cam_loc[0] - 0.0) < 1e-6
    assert cam_loc[2] == G.CAM_HEIGHT
    # camera behind the box (y < -BOX/2 for N in-shot)
    assert cam_loc[1] < -(G.BOX_SIZE / 2)
    # changing road_meta moves the camera (proves it's not hardcoded)
    road_meta2 = {"crosswalk_y": 30.0, "approach_length": 60.0}
    cam_loc2, _ = G.camera_pose(G.Direction.N, True, road_meta2)
    assert abs(cam_loc[1] - cam_loc2[1]) > 1e-6


def test_metadata_pose_matches_motion_plan():
    # Consistency: render.compute_metadata per-frame poses must equal the
    # linear interpolation of the kinematics motion plan (the same plan
    # build_scene keyframes). Guards against metadata/render drift.
    # The reference plan_motion uses the SAME anchors + road_meta that
    # compute_metadata loads from env files and road.json.
    import render
    v = K.speed_kmh_to_ms(45)
    scn = {"seed": 1, "fps": 30, "duration_frames": 400, "box_size": G.BOX_SIZE,
           "vehicles": [{"id": "V0", "class": "car", "color": [0.1, 0.2, 0.8, 1.0],
                          "color_name": "blue", "plate": "59X-1234", "approach": "N",
                          "lane": 1, "turn": "right", "speed_ms": v, "speed_kmh": 45,
                          "length": 4.47, "depart_frame": 5}]}
    meta = render.compute_metadata(scn, ROOT)
    vm = meta["vehicles"][0]
    # Replicate the same plan inputs that compute_metadata uses internally.
    env_in = ENV.load_env("in_N", ROOT)
    env_out = ENV.load_env("out_E", ROOT)
    in_anchor, _ = ENV.lane_default_anchor(env_in, 1)
    out_anchor, _ = ENV.lane_default_anchor(env_out, 3)
    road_meta_ref = {"crosswalk_y": 27.846, "approach_length": 54.751}
    motion = K.plan_motion("V0", G.Direction.N, 1, G.Turn.RIGHT, v, 5, fps=30,
                           appear_anchor=in_anchor[:2],
                           reappear_anchor=out_anchor[:2],
                           road_meta=road_meta_ref)
    for fr in vm["frames"]:
        if fr["frame"] <= motion.disappear_frame:
            t = (fr["frame"] - motion.appear_frame) / max(1, motion.disappear_frame - motion.appear_frame)
            ex = motion.appear_pos[0] + (motion.disappear_pos[0] - motion.appear_pos[0]) * t
            ey = motion.appear_pos[1] + (motion.disappear_pos[1] - motion.appear_pos[1]) * t
            assert abs(fr["pose"]["x"] - round(ex, 3)) < 1e-3
            assert abs(fr["pose"]["y"] - round(ey, 3)) < 1e-3
            assert fr["camera"] == "in_N"
        else:
            t = (fr["frame"] - motion.reappear_frame) / max(1, motion.leave_frame - motion.reappear_frame)
            ex = motion.reappear_pos[0] + (motion.leave_pos[0] - motion.reappear_pos[0]) * t
            ey = motion.reappear_pos[1] + (motion.leave_pos[1] - motion.reappear_pos[1]) * t
            assert abs(fr["pose"]["x"] - round(ex, 3)) < 1e-3
            assert abs(fr["pose"]["y"] - round(ey, 3)) < 1e-3
            assert fr["camera"] == "out_E"
    # no bbox field anywhere
    assert all("bbox" not in fr for fr in vm["frames"])


def test_out_camera_aligned_with_exit_vehicles():
    # Bug A guard: every out_<D> camera must sit at the box far edge, looking
    # outward along the outbound heading, with the Black Box behind the camera.
    # Per-shot frame: camera lateral position = 0 (on road axis).
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    half = G.BOX_SIZE / 2
    for d in G.Direction:
        m = K.plan_motion("v", d, 0, G.Turn.STRAIGHT,
                          K.speed_kmh_to_ms(40), depart_frame=0)
        cam_loc, look = G.camera_pose(d, False, road_meta)
        ofx, ofy = d.vec
        # 1. looks outward along travel (rear plate filmed as car drives away)
        dx, dy = look[0] - cam_loc[0], look[1] - cam_loc[1]
        assert dx * ofx + dy * ofy > 0, f"out_{d.value} not looking outward"
        # 2. camera on the outbound side of the box (not filming across it)
        along = cam_loc[0] * ofx + cam_loc[1] * ofy
        assert along >= half - 1e-6, f"out_{d.value} camera on wrong side of box"
        # 3. car reappears at the box far edge
        car_along = m.reappear_pos[0] * ofx + m.reappear_pos[1] * ofy
        assert abs(car_along - half) < 1e-6, f"out_{d.value} car not at box edge"
        # 4. camera on road axis (lateral = 0)
        perp_axis = (-ofy, ofx)
        cam_perp = cam_loc[0] * perp_axis[0] + cam_loc[1] * perp_axis[1]
        assert abs(cam_perp) < 1e-6, f"out_{d.value} camera not on road axis"


def test_in_camera_aligned_with_entry_vehicles():
    # Mirror guard for in_<D>: camera behind the box entry edge, looking toward
    # the box. Per-shot frame: camera lateral position = 0 (on road axis).
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    half = G.BOX_SIZE / 2
    for d in G.Direction:
        m = K.plan_motion("v", d, 0, G.Turn.STRAIGHT,
                          K.speed_kmh_to_ms(40), depart_frame=0)
        cam_loc, look = G.camera_pose(d, True, road_meta)
        fx, fy = d.vec
        # 1. looks toward box
        dx, dy = look[0] - cam_loc[0], look[1] - cam_loc[1]
        assert dx * fx + dy * fy > 0, f"in_{d.value} not looking toward box"
        # 2. camera behind the entry edge
        along = cam_loc[0] * fx + cam_loc[1] * fy
        assert along <= -half + 1e-6, f"in_{d.value} camera not behind box"
        # 3. camera on road axis (lateral = 0)
        perp_axis = (-fy, fx)
        cam_perp = cam_loc[0] * perp_axis[0] + cam_loc[1] * perp_axis[1]
        assert abs(cam_perp) < 1e-6, f"in_{d.value} camera not on road axis"


def test_visible_heading_applies_forward_offset_both_shots():
    # Bug B guard: forward_offset_deg must be applied on BOTH in and out shots.
    # Without it, a sideways model (offset ±90) would face the wrong way after
    # keyframe_motion overrides the static rotation.
    m = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                      K.speed_kmh_to_ms(40), depart_frame=0)
    off = -90.0
    h_in = G.visible_heading(m, is_in_camera=True, forward_offset_deg=off)
    h_out = G.visible_heading(m, is_in_camera=False, forward_offset_deg=off)
    base_in = G.approach_rotation(G.Direction.N)
    base_out = G.approach_rotation(m.exit_direction)   # N straight -> N
    assert abs(h_in - (base_in + math.radians(off))) < 1e-9
    assert abs(h_out - (base_out + math.radians(off))) < 1e-9
    # offset 0 -> heading equals the base rotation (no correction)
    assert abs(G.visible_heading(m, True, 0.0) - base_in) < 1e-9
    # out shot of a turning movement uses the EXIT direction's heading + offset
    mr = K.plan_motion("v", G.Direction.N, 1, G.Turn.RIGHT,
                       K.speed_kmh_to_ms(40), depart_frame=0)
    h_r = G.visible_heading(mr, is_in_camera=False, forward_offset_deg=off)
    assert abs(h_r - (G.approach_rotation(G.Direction.E) + math.radians(off))) < 1e-9


def test_validate_length_axis_respects_forward_offset():
    # validate_assets picks the length axis from forward_offset_deg. Mirror the
    # pure-python decision so the validator stays coherent with sideways models.
    def length_axis_idx(fwd_off):
        return 0 if abs((fwd_off % 180) - 90.0) < 1e-3 else 1
    assert length_axis_idx(0) == 1      # nose +Y -> length on Y (index 1)
    assert length_axis_idx(90) == 0     # nose +X -> length on X (index 0)
    assert length_axis_idx(-90) == 0    # nose -X -> length on X (index 0)
    assert length_axis_idx(180) == 1    # nose -Y -> length on Y (index 1)


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
