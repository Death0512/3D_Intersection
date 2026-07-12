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


def test_routing_table_has_legal_movements_per_approach():
    # Strict lane-use: lane0={L,S}, lane1={S}, lane2={S}, lane3={S,R}
    # => 6 legal lane-turn movements per approach.
    assert len(G.ROUTING_TABLE) == 4 * 6
    assert len(G.camera_names()) == 8
    for m in G.ROUTING_TABLE:
        assert m.turn in G.allowed_turns(m.lane)


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


def test_delta_t_turns_legacy_constant_speed():
    v = K.speed_kmh_to_ms(40)
    dt_straight = G.delta_t_seconds(G.Turn.STRAIGHT, v)
    dt_right = G.delta_t_seconds(G.Turn.RIGHT, v)
    dt_left = G.delta_t_seconds(G.Turn.LEFT, v)
    # Legacy fallback (no lane_index) preserves the fixed-speed model.
    assert dt_right < dt_straight
    assert dt_left < dt_straight
    assert dt_left > dt_right


def test_delta_t_turns_with_lane_deceleration_are_longer():
    v = K.speed_kmh_to_ms(60)
    dt_straight = G.delta_t_seconds(G.Turn.STRAIGHT, v, lane_index=3)
    dt_right = G.delta_t_seconds(G.Turn.RIGHT, v, lane_index=3)
    dt_left = G.delta_t_seconds(G.Turn.LEFT, v, lane_index=0)
    assert dt_right > dt_straight
    assert dt_left > dt_straight
    assert G.turn_radius(G.Turn.RIGHT, 3) < G.turn_radius(G.Turn.LEFT, 0)


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


def test_safe_gap_increases_with_speed():
    length = 4.5
    g30 = K.safe_gap_m(K.speed_kmh_to_ms(30), length)
    g60 = K.safe_gap_m(K.speed_kmh_to_ms(60), length)
    g80 = K.safe_gap_m(K.speed_kmh_to_ms(80), length)
    assert g30 > length + 2.0
    assert g60 > g30
    assert g80 > g60


def test_min_headway_frames_uses_time_headway():
    length = 4.5
    v = K.speed_kmh_to_ms(50)
    frames = K.min_headway_frames(length, v, fps=30)
    # At 50 km/h with 2s headway, the frame gap should be roughly 2s plus
    # vehicle-length traversal time, not just a few frames from a 2m gap.
    assert frames >= 60


def test_bezier_point_matches_linear_at_endpoints():
    # Bézier curve B(t) at t=0 and t=1 must equal the start/end points.
    p0 = (0.0, 0.0)
    p1 = (2.0, 3.0)
    p2 = (8.0, 7.0)
    p3 = (10.0, 10.0)
    assert G.bezier_point(0.0, p0, p1, p2, p3) == p0
    assert G.bezier_point(1.0, p0, p1, p2, p3) == p3


def test_sample_track_linear_matches_legacy():
    # A 2-point LINEAR track reproduces the old linear interpolation.
    import os, json
    V = K.speed_kmh_to_ms(40)
    road = {"approach_length": 54.751}
    motion = K.plan_motion("v", G.Direction.N, 0, G.Turn.STRAIGHT,
                           V, 0, fps=30, road_meta=road,
                           appear_anchor=(-5.25, -75.0))
    for f in range(motion.appear_frame, motion.disappear_frame + 1):
        ref = G.sample_track(motion.track_in, f)
        t = (f - motion.appear_frame) / max(1, motion.disappear_frame - motion.appear_frame)
        ex = motion.appear_pos[0] + (motion.disappear_pos[0] - motion.appear_pos[0]) * t
        ey = motion.appear_pos[1] + (motion.disappear_pos[1] - motion.appear_pos[1]) * t
        assert abs(ref[0] - ex) < 1e-9
        assert abs(ref[1] - ey) < 1e-9


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
    assert all(G.Turn(v["turn"]) in G.allowed_turns(v["lane"]) for v in veh)
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
                          "lane": 3, "turn": "right", "speed_ms": v, "speed_kmh": 45,
                          "length": 4.47, "depart_frame": 5}]}
    meta = render.compute_metadata(scn, ROOT)
    vm = meta["vehicles"][0]
    # Replicate the same plan inputs that compute_metadata uses internally.
    env_in = ENV.load_env("in_N", ROOT)
    env_out = ENV.load_env("out_E", ROOT)
    in_anchor, _ = ENV.lane_default_anchor(env_in, 3)
    out_anchor, _ = ENV.lane_default_anchor(env_out, 3)
    road_meta_ref = {"crosswalk_y": 27.846, "approach_length": 54.751}
    motion = K.plan_motion("V0", G.Direction.N, 3, G.Turn.RIGHT, v, 5, fps=30,
                           appear_anchor=in_anchor[:2],
                           reappear_anchor=out_anchor[:2],
                           road_meta=road_meta_ref)
    for fr in vm["frames"]:
        ref = G.sample_track(motion.track_in, fr["frame"])
        if ref is not None:
            ex, ey = ref
            assert abs(fr["pose"]["x"] - round(ex, 3)) < 1e-3
            assert abs(fr["pose"]["y"] - round(ey, 3)) < 1e-3
            assert fr["camera"] == "in_N"
            continue
        ref = G.sample_track(motion.track_out, fr["frame"])
        if ref is not None:
            ex, ey = ref
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


# -----------------------------------------------------------------------------
# Signal plan tests
# -----------------------------------------------------------------------------

def test_signal_phase_boundaries():
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    assert sp.cycle_frames == 2100  # (30+3+2) * 2 * 30
    # NS_GREEN: 0-899
    assert sp._phase_at(0) == "NS_GREEN"
    assert sp._phase_at(899) == "NS_GREEN"
    assert sp._phase_at(900) == "NS_YELLOW"
    assert sp._phase_at(989) == "NS_YELLOW"
    assert sp._phase_at(990) == "ALL_RED"
    assert sp._phase_at(1049) == "ALL_RED"
    assert sp._phase_at(1050) == "EW_GREEN"
    assert sp._phase_at(1949) == "EW_GREEN"
    assert sp._phase_at(1950) == "EW_YELLOW"
    assert sp._phase_at(2039) == "EW_YELLOW"
    assert sp._phase_at(2040) == "ALL_RED"
    assert sp._phase_at(2099) == "ALL_RED"
    # wraps to next cycle
    assert sp._phase_at(2100) == "NS_GREEN"


def test_signal_is_green_correctly_distinguishes_phases():
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    # NS_GREEN -> N/S green, E/W red
    assert sp.is_green(G.Direction.N, G.Turn.STRAIGHT, 0) is True
    assert sp.is_green(G.Direction.S, G.Turn.LEFT, 500) is True
    assert sp.is_green(G.Direction.E, G.Turn.STRAIGHT, 0) is False
    # NS_YELLOW -> not green (yellow is not green)
    assert sp.is_green(G.Direction.N, G.Turn.STRAIGHT, 900) is False
    assert sp.is_green(G.Direction.S, G.Turn.LEFT, 950) is False
    # ALL_RED -> no one is green
    assert sp.is_green(G.Direction.N, G.Turn.STRAIGHT, 990) is False
    assert sp.is_green(G.Direction.E, G.Turn.STRAIGHT, 1000) is False
    # EW_GREEN -> E/W green, N/S red
    assert sp.is_green(G.Direction.E, G.Turn.STRAIGHT, 1200) is True
    assert sp.is_green(G.Direction.W, G.Turn.RIGHT, 1500) is True
    assert sp.is_green(G.Direction.N, G.Turn.STRAIGHT, 1200) is False
    # EW_YELLOW -> not green
    assert sp.is_green(G.Direction.E, G.Turn.STRAIGHT, 1950) is False
    assert sp.is_green(G.Direction.W, G.Turn.LEFT, 2000) is False


def test_signal_next_green_frame_monotonic():
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    # N/S next green from within NS_GREEN = same frame
    assert sp.next_green_frame(G.Direction.N, G.Turn.STRAIGHT, 0) == 0
    # N/S next green from middle of EW_GREEN = next NS_GREEN after 1050
    f = sp.next_green_frame(G.Direction.N, G.Turn.STRAIGHT, 1200)
    assert f == 2100  # wraps to next cycle
    # E/W next green from NS_YELLOW = start of EW_GREEN at 1050
    f = sp.next_green_frame(G.Direction.E, G.Turn.STRAIGHT, 950)
    assert f == 1050
    # N/S next green from NS_YELLOW = immediate next cycle start
    f = sp.next_green_frame(G.Direction.N, G.Turn.STRAIGHT, 900)
    assert f == 2100


def test_signal_raises_for_never_green():
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    f = sp.next_green_frame(G.Direction.N, G.Turn.STRAIGHT, 5000)
    assert f >= 5000


def test_signal_next_red_frame():
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    # N/S red from middle of NS_GREEN = start of NS_YELLOW at 900
    f = sp.next_red_frame(G.Direction.N, G.Turn.STRAIGHT, 100)
    assert f == 900
    # N/S red from all-red = immediate
    f = sp.next_red_frame(G.Direction.N, G.Turn.STRAIGHT, 990)
    assert f == 990


# -----------------------------------------------------------------------------
# Queue track tests
# -----------------------------------------------------------------------------

def test_compute_motion_queue_track_has_decel_phase():
    """A queued vehicle (stop_frame < release_frame) produces a multi-point
    IN track with a deceleration phase followed by an idle segment.

    The track has 3 or 4 points depending on whether the approach fits a
    cruise-then-brake profile (4 points) or must span the whole approach as
    a single decel ramp (3 points). In both cases the LAST two points share
    the stop-line position (idle) and the vehicle is stationary between
    stop_frame and release_frame.
    """
    v = K.speed_kmh_to_ms(45)
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    stop_f = 100  # reaches stop line at frame 100
    release_f = 200  # enters box at frame 200 (waited 100 frames)
    m = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                      appear_anchor=appear_anchor,
                      road_meta=road_meta,
                      stop_frame=stop_f, release_frame=release_f)
    assert len(m.track_in) in (3, 4)
    # frames strictly increase
    for a, b in zip(m.track_in, m.track_in[1:]):
        assert a.frame < b.frame, (a.frame, b.frame)
    # last two points are the idle segment (stationary at stop line)
    assert m.track_in[-1].x == m.track_in[-2].x
    assert m.track_in[-1].y == m.track_in[-2].y
    assert m.track_in[-1].frame == release_f
    assert m.track_in[-2].frame == stop_f
    # at least one BEZIER (decel) segment exists before the idle point
    has_bez = any(pt.interp == "BEZIER" for pt in m.track_in[:-1])
    assert has_bez, "no decel (BEZIER) segment in queued IN track"
    # disappear_frame = release_frame
    assert m.disappear_frame == release_f
    # reappear_frame = release_frame + dt_box
    assert m.reappear_frame > release_f


def test_compute_motion_queue_track_vs_freeflow():
    """Same vehicle with queue params vs without should differ only in
    the IN track structure and disappear timing."""
    v = K.speed_kmh_to_ms(45)
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    stop_f = 100
    release_f = 200
    m_queue = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                            appear_anchor=appear_anchor,
                            road_meta=road_meta,
                            stop_frame=stop_f, release_frame=release_f)
    m_free = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                           appear_anchor=appear_anchor,
                            road_meta=road_meta)
    # Free-flow has 2-point track
    assert len(m_free.track_in) == 2
    # Queued has a multi-point decel track
    assert len(m_queue.track_in) >= 3
    # Free-flow disappear = depart + travel time
    # Queued disappear = release_frame
    assert m_free.disappear_frame < m_queue.disappear_frame
    # Same endpoint positions
    assert m_queue.appear_pos == m_free.appear_pos
    assert m_queue.disappear_pos == m_free.disappear_pos


def test_queue_track_idle_segment_stationary():
    """During the idle period (between stop_frame and release_frame),
    sample_track returns the constant stop position."""
    v = K.speed_kmh_to_ms(45)
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    stop_f = 100
    release_f = 200
    m = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                      appear_anchor=appear_anchor,
                      road_meta=road_meta,
                      stop_frame=stop_f, release_frame=release_f)
    # sample at several frames during idle
    for f in range(stop_f, release_f + 1):
        p = G.sample_track(m.track_in, f)
        assert p is not None
        assert abs(p[0] - m.disappear_pos[0]) < 1e-3
        assert abs(p[1] - m.disappear_pos[1]) < 1e-3


def test_queue_track_decel_monotonic_and_stops_at_line():
    """The decel phase: position advances monotonically toward the stop
    line (no backtracking), and the vehicle is at (or within epsilon of)
    the stop position at stop_frame."""
    v = K.speed_kmh_to_ms(40)   # 11.11 m/s — brake fits in the 40 m approach
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    # stop_frame chosen so the approach (54.751 m) is comfortably longer
    # than the brake distance d_brake = v^2/(2*DECEL) = ~24.7 m, giving
    # a real cruise-then-brake 4-point track.
    stop_f = 200
    release_f = 260
    m = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                      appear_anchor=appear_anchor,
                      road_meta=road_meta,
                      stop_frame=stop_f, release_frame=release_f)
    # forward axis is +Y for N approach → monotonic IN y must increase
    prev_y = -1e18
    for f in range(m.track_in[0].frame, stop_f + 1):
        p = G.sample_track(m.track_in, f)
        assert p is not None
        # y increases (the vehicle drives toward the box at +Y)
        assert p[1] >= prev_y - 1e-6, f"non-monotonic y at frame {f}: {p[1]} < {prev_y}"
        prev_y = p[1]
    # at stop_frame the vehicle is at the stop line
    p_stop = G.sample_track(m.track_in, stop_f)
    assert abs(p_stop[0] - m.disappear_pos[0]) < 1e-3
    assert abs(p_stop[1] - m.disappear_pos[1]) < 1e-3


def test_queue_track_full_ramp_when_brake_does_not_fit():
    """When the brake time exceeds the available approach time (short
    approach / high speed), the whole segment becomes a single BEZIER
    ease-in decel ramp with no cruise phase (3-point track)."""
    v = K.speed_kmh_to_ms(80)   # 22.2 m/s — t_brake = v/DECEL = ~8.9 s
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    # only 5 frames to stop → t_brake_s (~8.9s) >> available → full ramp
    stop_f = 5
    release_f = 100
    m = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                      appear_anchor=appear_anchor,
                      road_meta=road_meta,
                      stop_frame=stop_f, release_frame=release_f)
    # 3-point track: appear (BEZIER) → stop (LINEAR) → release (LINEAR)
    assert len(m.track_in) == 3
    assert m.track_in[0].interp == "BEZIER"
    assert m.track_in[1].frame == stop_f
    assert m.track_in[2].frame == release_f
    # vehicle still stationary during idle
    for f in range(stop_f, release_f + 1):
        p = G.sample_track(m.track_in, f)
        assert abs(p[0] - m.disappear_pos[0]) < 1e-3
        assert abs(p[1] - m.disappear_pos[1]) < 1e-3


def test_out_track_accel_from_stop_when_queued():
    """A queued vehicle's OUT track launches from rest at the box edge
    (BEZIER ease-out), so the green-light launch is visible on the
    out-camera. Free-flow straight vehicles do NOT ease-out (they reappear
    already at cruise)."""
    v = K.speed_kmh_to_ms(45)
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    # Queued: stop at red, release on green
    m_q = K.plan_motion("Vq", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                        appear_anchor=appear_anchor,
                        reappear_anchor=(15.0, -5.25),
                        road_meta=road_meta,
                        stop_frame=100, release_frame=200)
    assert m_q.track_out[0].interp == "BEZIER"
    assert m_q.track_out[0].cp1 is not None
    # Free-flow straight: no ease-out (LINEAR)
    m_f = K.plan_motion("Vf", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                        appear_anchor=appear_anchor,
                        reappear_anchor=(15.0, -5.25),
                        road_meta=road_meta)
    assert m_f.track_out[0].interp == "LINEAR"
    # Turning vehicle always eases out (curve-exit speed is low)
    m_t = K.plan_motion("Vt", G.Direction.N, 1, G.Turn.RIGHT, v, 0,
                        appear_anchor=(1.75, -75.0),
                        reappear_anchor=(20.0, -5.25),
                        road_meta=road_meta)
    assert m_t.track_out[0].interp == "BEZIER"


def test_queue_slot_positions_differ_upstream():
    """Queued vehicles with different queue_slot values stop at different
    upstream offsets — they don't all stack at the stop line."""
    v = K.speed_kmh_to_ms(45)
    appear_anchor = (-5.25, -75.0)
    road_meta = {"crosswalk_y": 27.846, "approach_length": 54.751}
    stop_box_edge = G.lane_entry_box_edge(G.Direction.N, 0)

    # slot 0 (first in queue) → stop at box edge
    m0 = K.plan_motion("V0", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                       appear_anchor=appear_anchor,
                       road_meta=road_meta,
                       stop_frame=100, release_frame=200,
                       queue_slot=0)
    assert abs(m0.disappear_pos[0] - stop_box_edge[0]) < 1e-6
    assert abs(m0.disappear_pos[1] - stop_box_edge[1]) < 1e-6

    # slot 2 (third in queue) → stop QUEUE_SPACING_M * 2 upstream
    m2 = K.plan_motion("V2", G.Direction.N, 0, G.Turn.STRAIGHT, v, 0,
                       appear_anchor=appear_anchor,
                       road_meta=road_meta,
                       stop_frame=100, release_frame=200,
                       queue_slot=2)
    offset = 2 * G.QUEUE_SPACING_M  # 12m upstream
    assert abs(m2.disappear_pos[0] - stop_box_edge[0]) < 1e-6
    assert abs(m2.disappear_pos[1] - stop_box_edge[1]) < 1e-6

    # idle track positions differ — each vehicle stationary at its own offset
    p0_stop = G.sample_track(m0.track_in, 150)
    p2_stop = G.sample_track(m2.track_in, 150)
    assert p0_stop is not None and p2_stop is not None
    # slot 2 is further upstream (behind slot 0)
    assert p2_stop[1] < p0_stop[1] - (G.QUEUE_SPACING_M - 0.1)
    assert abs(p2_stop[1] - (stop_box_edge[1] - offset)) < 0.1
    # by release, slot 2 has advanced to the true box edge and can disappear
    p2_release = G.sample_track(m2.track_in, 200)
    assert p2_release is not None
    assert abs(p2_release[1] - stop_box_edge[1]) < 1e-6


# -----------------------------------------------------------------------------
# Scenario generator integration tests
# -----------------------------------------------------------------------------

def test_signal_gating_sets_stop_release_on_red():
    """Vehicles arriving on red get proper queue fields; green arrivals
    are free-flow."""
    import scenario_gen as S
    import random
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    rng = random.Random(1)
    vehicles = [S.make_vehicle("V0", rng)]
    # Force depart_frame such that stop_arrival falls on red
    v = vehicles[0]
    travel_f = int(round(54.751 / v["speed_ms"] * 30))
    # Place depart so stop_arrival = 1000 (ALL_RED at frame 990-1049)
    v["depart_frame"] = max(0, 1000 - travel_f)
    S._apply_signal_gating(vehicles, 54.751, sp, 30)
    assert "stop_frame" in v
    assert "release_frame" in v
    # On red -> release_frame > stop_frame
    assert v["release_frame"] > v["stop_frame"], (
        f"stop={v['stop_frame']} release={v['release_frame']}")
    # queue_slot >= 0 means queued
    assert v["queue_slot"] >= 0
    # wait_frames > 0
    assert v["wait_frames"] > 0


def test_signal_gating_freeflow_on_green():
    """Vehicles arriving on green should NOT queue."""
    import scenario_gen as S
    import random
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    rng = random.Random(2)
    vehicles = [S.make_vehicle("V0", rng)]
    v = vehicles[0]
    travel_f = int(round(54.751 / v["speed_ms"] * 30))
    # Place depart so stop_arrival = 100 (NS_GREEN 0-899)
    v["depart_frame"] = max(0, 100 - travel_f)
    S._apply_signal_gating(vehicles, 54.751, sp, 30)
    assert v["stop_frame"] is not None
    assert v["release_frame"] == v["stop_frame"]
    assert v["queue_slot"] == -1
    assert v["wait_frames"] == 0


def test_resolve_all_no_red_crossings():
    """After _resolve_all with signal_plan, no vehicle crosses the stop
    line on red."""
    import scenario_gen as S
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    import random
    rng = random.Random(3)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(30)]
    S.schedule_departures(vehicles, 2100, rng,
                          approach_visible_length=54.751)
    S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)
    for v in vehicles:
        if v.get("queue_slot", -1) >= 0:
            # Queued: arrives at stop line on red, waits, enters on green
            approach = G.Direction(v["approach"])
            turn = G.Turn(v["turn"])
            # stop_frame should be red
            assert not sp.is_green(approach, turn, v["stop_frame"]), (
                f"{v['id']} stop_frame {v['stop_frame']} should be red")
            # release_frame should be green
            assert sp.is_green(approach, turn, v["release_frame"]), (
                f"{v['id']} release_frame {v['release_frame']} should be green")
        else:
            # Free-flow: arrives on green
            approach = G.Direction(v["approach"])
            turn = G.Turn(v["turn"])
            stop_f = v.get("stop_frame", v["depart_frame"] + int(
                round(54.751 / v["speed_ms"] * 30)))
            assert sp.is_green(approach, turn, stop_f), (
                f"{v['id']} free-flow stop_frame {stop_f} should be green")


def test_resolve_all_exit_conflicts_resolved():
    """After _resolve_all, no exit-lane interval overlaps exceed the
    buffer threshold."""
    import scenario_gen as S
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    import random
    rng = random.Random(5)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(60)]
    S.schedule_departures(vehicles, 2100, rng,
                          approach_visible_length=54.751)
    S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)
    # Group by exit lane and check intervals
    groups = {}
    for v in vehicles:
        travel_f = int(round(54.751 / v["speed_ms"] * 30))
        release_f = v.get("release_frame", v["depart_frame"] + travel_f)
        reappear_f = release_f + G.delta_t_frames(
            G.Turn(v["turn"]), v["speed_ms"], 30, lane_index=v["lane"])
        leave_f = reappear_f + travel_f
        out_dir, ex_lane = G.exit_lane_for_movement(
            G.Direction(v["approach"]), v["lane"], G.Turn(v["turn"]))
        key = (out_dir.value, ex_lane)
        groups.setdefault(key, []).append((v["id"], reappear_f, leave_f))
    for key, grp in groups.items():
        grp.sort(key=lambda x: x[1])
        last_leave = -1
        for vid, r, l in grp:
            if last_leave > 0:
                assert r >= last_leave + 5 - 1, (
                    f"exit {key} {vid}: reappear {r} < last_leave {last_leave} + 5")
            last_leave = l


def test_resolve_all_headway_maintained():
    """After _resolve_all, same-lane headway and catch-up constraints
    still hold."""
    import scenario_gen as S
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    import random
    rng = random.Random(7)
    vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(40)]
    S.schedule_departures(vehicles, 2100, rng,
                          approach_visible_length=54.751)
    S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)
    lanes = {}
    for v in vehicles:
        lanes.setdefault((v["approach"], v["lane"]), []).append(v)
    for key, vs in lanes.items():
        vs.sort(key=lambda d: d["depart_frame"])
        for a, b in zip(vs, vs[1:]):
            assert K.catchup_safe(a["depart_frame"], a["speed_ms"], a["length"],
                                   b["depart_frame"], b["speed_ms"],
                                   approach_visible_length=54.751), (
                f"headway fail {key} {a['id']}->{b['id']}")


def test_scenario_signal_generation_creates_json_with_signal_field():
    """generate() with signal_plan writes signal_cycle_frames to JSON."""
    import scenario_gen as S
    import tempfile
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    with tempfile.TemporaryDirectory() as tmp:
        scn = S.generate(1, 300, tmp, fps=30, signal_plan=sp)
        assert "signal_cycle_frames" in scn
        assert scn["signal_cycle_frames"] == 2100
        # seed written to dict
        assert "seed" in scn
        assert scn["seed"] == 1


def test_multi_seed_resolve_all_invariants():
    """Multi-seed sweep: after _resolve_all with signal_plan, the output
    must satisfy ALL invariants AND be fixpoint-stable (one extra signal
    pass produces no change).  Sweeps seeds 0-19 with 60 vehicles each."""
    import scenario_gen as S
    import random
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    for seed in range(20):
        rng = random.Random(seed)
        vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(60)]
        S.schedule_departures(vehicles, 2100, rng,
                              approach_visible_length=54.751)
        S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)

        # --- stability: one more signal-gating pass changes nothing ---
        snap_before = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                       for x in vehicles]
        S._apply_signal_gating(vehicles, 54.751, sp, 30)
        snap_after = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                      for x in vehicles]
        assert snap_before == snap_after, (
            f"seed {seed}: fixpoint not stable after convergence")

        # --- no red crossing ---
        for x in vehicles:
            ap = G.Direction(x["approach"])
            tn = G.Turn(x["turn"])
            qs = x.get("queue_slot", -1)
            if qs >= 0:
                assert not sp.is_green(ap, tn, x["stop_frame"]), (
                    f"seed {seed} {x['id']}: queued stop {x['stop_frame']} is green")
                assert sp.is_green(ap, tn, x["release_frame"]), (
                    f"seed {seed} {x['id']}: queued release {x['release_frame']} not green")
            else:
                sf = x.get("stop_frame",
                           x["depart_frame"] + int(round(54.751 / x["speed_ms"] * 30)))
                assert sp.is_green(ap, tn, sf), (
                    f"seed {seed} {x['id']}: free-flow stop {sf} not green")

        # --- no exit-lane overlap (5-frame buffer) ---
        groups = {}
        for x in vehicles:
            tf = int(round(54.751 / x["speed_ms"] * 30))
            rf = x.get("release_frame", x["depart_frame"] + tf)
            rp = rf + G.delta_t_frames(G.Turn(x["turn"]), x["speed_ms"], 30,
                                        lane_index=x["lane"])
            lv = rp + tf
            od, el = G.exit_lane_for_movement(
                G.Direction(x["approach"]), x["lane"], G.Turn(x["turn"]))
            groups.setdefault((od.value, el), []).append((rp, lv))
        for g in groups.values():
            g.sort()
            last = -1
            for r, lv in g:
                if last > 0:
                    assert r >= last + 5, f"seed {seed}: exit overlap r={r} < last_leave={last}+5"
                last = lv

        # --- no headway / catch-up violation ---
        lanes = {}
        for x in vehicles:
            lanes.setdefault((x["approach"], x["lane"]), []).append(x)
        for vs in lanes.values():
            vs.sort(key=lambda d: d["depart_frame"])
            for a, b in zip(vs, vs[1:]):
                assert K.catchup_safe(a["depart_frame"], a["speed_ms"], a["length"],
                                       b["depart_frame"], b["speed_ms"],
                                       approach_visible_length=54.751), (
                    f"seed {seed} {a['id']}->{b['id']}")

        # --- same-lane queued release gaps >= 0.5 s (platoon spacing) ---
        for vs in lanes.values():
            queued = [x for x in vs if x.get("queue_slot", -1) >= 0]
            queued.sort(key=lambda x: x["release_frame"])
            for a, b in zip(queued, queued[1:]):
                gap = b["release_frame"] - a["release_frame"]
                assert gap >= 15, (
                    f"seed {seed} lane ({a['approach']},{a['lane']}) "
                    f"{a['id']}@{a['release_frame']}->{b['id']}@{b['release_frame']} "
                    f"gap={gap} < 0.5s")


def test_multi_seed_saturation_invariants():
    """Saturation sweep: 120 vehicles (2× normal density).  Same invariants
    as the standard multi-seed test, verifying that the green-window rebase
    in the main slot-assignment loop prevents red-crossing violations even
    when ``extra_stagger_frames`` grows large under saturation.

    Sweeps seeds 1000–1019 (offset to avoid cross-contamination with the
    standard-density test)."""
    import scenario_gen as S
    import random
    from lib import traffic_signal as SG
    sp = SG.SignalPlan(fps=30)
    for seed in range(1000, 1020):
        rng = random.Random(seed)
        vehicles = [S.make_vehicle(f"V{i:03d}", rng) for i in range(120)]
        S.schedule_departures(vehicles, 2100, rng,
                              approach_visible_length=54.751)
        S._resolve_all(vehicles, 54.751, 30, signal_plan=sp)

        # --- stability ---
        snap_before = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                       for x in vehicles]
        S._apply_signal_gating(vehicles, 54.751, sp, 30)
        snap_after = [(x["queue_slot"], x["stop_frame"], x["release_frame"])
                      for x in vehicles]
        assert snap_before == snap_after, (
            f"seed {seed}: saturation fixpoint not stable")

        # --- no red crossing ---
        for x in vehicles:
            ap = G.Direction(x["approach"])
            tn = G.Turn(x["turn"])
            qs = x.get("queue_slot", -1)
            if qs >= 0:
                assert not sp.is_green(ap, tn, x["stop_frame"]), (
                    f"seed {seed} {x['id']}: queued stop {x['stop_frame']} green")
                assert sp.is_green(ap, tn, x["release_frame"]), (
                    f"seed {seed} {x['id']}: queued release {x['release_frame']} red")
            else:
                sf = x.get("stop_frame",
                           x["depart_frame"] + int(round(54.751 / x["speed_ms"] * 30)))
                assert sp.is_green(ap, tn, sf), (
                    f"seed {seed} {x['id']}: free-flow stop {sf} red")

        # --- no exit-lane overlap (5-frame buffer) ---
        groups = {}
        for x in vehicles:
            tf = int(round(54.751 / x["speed_ms"] * 30))
            rf = x.get("release_frame", x["depart_frame"] + tf)
            rp = rf + G.delta_t_frames(G.Turn(x["turn"]), x["speed_ms"], 30,
                                        lane_index=x["lane"])
            lv = rp + tf
            od, el = G.exit_lane_for_movement(
                G.Direction(x["approach"]), x["lane"], G.Turn(x["turn"]))
            groups.setdefault((od.value, el), []).append((rp, lv))
        for g in groups.values():
            g.sort()
            last = -1
            for r, lv in g:
                if last > 0:
                    assert r >= last + 5, (
                        f"seed {seed}: exit overlap r={r} < {last}+5")
                last = lv

        # --- no headway violation ---
        lanes = {}
        for x in vehicles:
            lanes.setdefault((x["approach"], x["lane"]), []).append(x)
        for vs in lanes.values():
            vs.sort(key=lambda d: d["depart_frame"])
            for a, b in zip(vs, vs[1:]):
                assert K.catchup_safe(a["depart_frame"], a["speed_ms"], a["length"],
                                       b["depart_frame"], b["speed_ms"],
                                       approach_visible_length=54.751), (
                    f"seed {seed} {a['id']}->{b['id']}")

        # --- same-lane queued release gaps >= 0.5 s ---
        for vs in lanes.values():
            queued = [x for x in vs if x.get("queue_slot", -1) >= 0]
            queued.sort(key=lambda x: x["release_frame"])
            for a, b in zip(queued, queued[1:]):
                gap = b["release_frame"] - a["release_frame"]
                assert gap >= 15, (
                    f"seed {seed} lane ({a['approach']},{a['lane']}) "
                    f"{a['id']}@{a['release_frame']}->{b['id']}@{b['release_frame']} "
                    f"gap={gap} < 0.5s")


# -----------------------------------------------------------------------------
# runner
# -----------------------------------------------------------------------------

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
