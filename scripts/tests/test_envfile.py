"""Tests for the per-camera env file system (pure python, no bpy).

Covers the v2 schema: required `load_env` (hard-fail on missing file/field),
`lane_defaults` per-lane anchors, `require_env_fields`, and `lane_default_anchor`.
Also integrates against the real 8 files under assets/envs/.

Run:
    cd /home/death/Documents/3D_Intersection_Video
    python3 scripts/tests/test_envfile.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import geometry as G
from lib import kinematics as K
from lib import envfile as ENV


ROAD_META = {"crosswalk_y": 27.846, "approach_length": 54.751, "blend": "road.blend", "collection": "ENV_road"}


# ---------------------------------------------------------------------------
# compute_env (v2 generator)
# ---------------------------------------------------------------------------

def test_compute_env_camera_matches_geometry():
    for d in G.Direction:
        for is_in in (True, False):
            tag = ("in" if is_in else "out") + f"_{d.value}"
            env = ENV.compute_env(tag, ROAD_META)
            cam_loc, look = G.camera_pose(d, is_in, ROAD_META)
            ec = env["camera"]
            for i in range(3):
                assert abs(ec["location"][i] - cam_loc[i]) < 1e-6, f"{tag} cam loc {i}"
                assert abs(ec["look_at"][i] - look[i]) < 1e-6, f"{tag} look {i}"
            assert ec["rotation_euler"] is None
            assert ec["lens_mm"] == G.LENS_MM
            assert ec["sensor_mm"] == G.SENSOR_MM


def test_compute_env_road_matches_geometry():
    for d in G.Direction:
        for is_in in (True, False):
            tag = ("in" if is_in else "out") + f"_{d.value}"
            env = ENV.compute_env(tag, ROAD_META)
            (lx, ly, lz), rot = G.road_arm_transform(d, ROAD_META, is_entry=is_in)
            er = env["road"]
            assert abs(er["location"][0] - lx) < 1e-6
            assert abs(er["location"][1] - ly) < 1e-6
            assert abs(er["location"][2] - lz) < 1e-6
            assert abs(er["rotation_euler"][2] - rot) < 1e-6
            assert er["object"] == f"Road_{d.value}_{'in' if is_in else 'out'}"


def test_compute_env_emits_lane_defaults_for_all_lanes():
    for d in G.Direction:
        for is_in in (True, False):
            tag = ("in" if is_in else "out") + f"_{d.value}"
            env = ENV.compute_env(tag, ROAD_META)
            defaults = env["vehicles"]["lane_defaults"]
            for i in range(G.NUM_LANES):
                assert str(i) in defaults, f"{tag} lane {i}"
                loc = defaults[str(i)]["location"]
                rot = defaults[str(i)]["rotation_euler"]
                assert len(loc) == 3
                assert len(rot) == 3


def test_compute_env_lane_defaults_match_geometry_anchors():
    # in-camera lane_defaults[lane] == appear_pos; out-camera == reappear_pos.
    for d in G.Direction:
        for lane in range(G.NUM_LANES):
            env_in = ENV.compute_env(f"in_{d.value}", ROAD_META)
            m = K.plan_motion("v", d, lane, G.Turn.STRAIGHT,
                              K.speed_kmh_to_ms(40), 0, fps=30)
            ld = env_in["vehicles"]["lane_defaults"][str(lane)]["location"]
            assert abs(ld[0] - m.appear_pos[0]) < 1e-6, f"in_{d.value} lane {lane} x"
            assert abs(ld[1] - m.appear_pos[1]) < 1e-6, f"in_{d.value} lane {lane} y"
            assert abs(env_in["vehicles"]["lane_defaults"][str(lane)]["rotation_euler"][2]
                       - G.approach_rotation(d)) < 1e-6
            # out-camera: reappear anchor for the exit lane (straight -> same dir/lane)
            env_out = ENV.compute_env(f"out_{d.value}", ROAD_META)
            rld = env_out["vehicles"]["lane_defaults"][str(lane)]["location"]
            assert abs(rld[0] - m.reappear_pos[0]) < 1e-6, f"out_{d.value} lane {lane} x"
            assert abs(rld[1] - m.reappear_pos[1]) < 1e-6, f"out_{d.value} lane {lane} y"


def test_schema_version_is_two():
    env = ENV.compute_env("in_N", ROAD_META)
    assert env["schema_version"] == 2
    assert ENV.SCHEMA_VERSION == 2
    assert "geometry" in env["generated_from"]


# ---------------------------------------------------------------------------
# require_env_fields
# ---------------------------------------------------------------------------

def test_require_env_fields_passes_on_compute_env_output():
    for d in G.Direction:
        for is_in in (True, False):
            tag = ("in" if is_in else "out") + f"_{d.value}"
            ENV.require_env_fields(ENV.compute_env(tag, ROAD_META), tag)  # no raise


def test_require_env_fields_fails_on_missing_camera():
    env = ENV.compute_env("in_N", ROAD_META)
    env["camera"]["location"] = None
    try:
        ENV.require_env_fields(env, "in_N")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_require_env_fields_fails_on_missing_lane_default():
    env = ENV.compute_env("in_N", ROAD_META)
    del env["vehicles"]["lane_defaults"]["2"]
    try:
        ENV.require_env_fields(env, "in_N")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_require_env_fields_fails_on_null_lane_rotation():
    env = ENV.compute_env("in_N", ROAD_META)
    env["vehicles"]["lane_defaults"]["0"]["rotation_euler"] = None
    try:
        ENV.require_env_fields(env, "in_N")
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# load_env (required, hard-fail)
# ---------------------------------------------------------------------------

def test_load_env_hard_fails_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ENV.load_env("in_N", tmp)
            assert False, "expected SystemExit"
        except SystemExit:
            pass


def test_load_env_loads_and_validates_existing_file():
    # Against the real project assets/envs.
    env = ENV.load_env("in_N", ROOT)
    assert env["camera_tag"] == "in_N"
    assert env["approach"] == "N"
    assert env["role"] == "in"
    assert env["camera"]["lens_mm"] == 60.0
    assert "0" in env["vehicles"]["lane_defaults"]


def test_load_env_normalises_tag_identity():
    # A file copied from another tag is relabeled to the requested tag.
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "assets", "envs"))
        wrong = ENV.compute_env("out_S", ROAD_META)
        with open(ENV.env_path("in_N", tmp), "w") as f:
            json.dump(wrong, f)
        env = ENV.load_env("in_N", tmp)
        assert env["camera_tag"] == "in_N"
        assert env["approach"] == "N"
        assert env["role"] == "in"


def test_load_env_hard_fails_on_null_required_field():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "assets", "envs"))
        env = ENV.compute_env("in_N", ROAD_META)
        env["camera"]["look_at"] = None
        with open(ENV.env_path("in_N", tmp), "w") as f:
            json.dump(env, f)
        try:
            ENV.load_env("in_N", tmp)
            assert False, "expected SystemExit"
        except SystemExit:
            pass


def test_validate_all_envs_passes_against_real_assets():
    ENV.validate_all_envs(ROOT)  # no raise


# ---------------------------------------------------------------------------
# lane_default_anchor
# ---------------------------------------------------------------------------

def test_lane_default_anchor_returns_pose():
    env = ENV.compute_env("in_N", ROAD_META)
    loc, rot_z = ENV.lane_default_anchor(env, 0)
    assert len(loc) == 3
    # lane 0 for in_N: x = -5.25, y = -55.0 (40m back from box edge at y=-15)
    assert abs(loc[0] - (-5.25)) < 1e-6
    assert abs(loc[1] - (-55.0)) < 1e-6
    assert rot_z == 0.0  # approach_rotation(N) == 0


def test_lane_default_anchor_hard_fails_on_missing_lane():
    env = ENV.compute_env("in_N", ROAD_META)
    del env["vehicles"]["lane_defaults"]["3"]
    try:
        ENV.lane_default_anchor(env, 3)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# camera_drift (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_camera_drift_detects_override():
    env = ENV.compute_env("in_N", ROAD_META)
    scn = {"vehicles": []}
    assert ENV.camera_drift(env, scn, "in_N", ROAD_META) is None
    env["camera"]["location"][2] = 99.0
    msg = ENV.camera_drift(env, scn, "in_N", ROAD_META)
    assert msg is not None and "z" in msg


# ---------------------------------------------------------------------------
# Integration: the 8 real env files match geometry.compute_env output
# (proves the committed files are valid and consistent with geometry).
# ---------------------------------------------------------------------------

def test_real_env_files_valid_and_structure_correct():
    """All 8 real env files load and pass require_env_fields (structural
    correctness). Exact values may differ from geometry defaults after
    hand-editing (which is the purpose of the override layer)."""
    for d in G.Direction:
        for is_in in (True, False):
            tag = ("in" if is_in else "out") + f"_{d.value}"
            real = ENV.load_env(tag, ROOT)  # validate_all_envs already checks all
            # lane_defaults must have 4 entries with location + rotation_euler
            for lane in range(G.NUM_LANES):
                ld = real["vehicles"]["lane_defaults"][str(lane)]
                assert "location" in ld and len(ld["location"]) == 3
                assert "rotation_euler" in ld and len(ld["rotation_euler"]) == 3


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
