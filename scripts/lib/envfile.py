"""Per-camera environment files (pure Python, no bpy) — REQUIRED input layer.

Each of the 8 camera shots has a JSON env file at
``assets/envs/<tag>.json`` (e.g. ``in_N.json``) that stores the location /
rotation of every static object in the shot (camera, road, sun light) plus the
per-lane vehicle spawn *anchors* (``vehicles.lane_defaults``).

These files are MANDATORY inputs to blend/video generation. The pipeline
hard-fails if any file is missing or any required field is null — there is no
auto-generation and no geometry fallback at render time. ``geometry.py`` remains
the single source of truth for ``metadata.json`` timing/heading, but the vehicle
START positions (and camera/road/sun) now come from these files.

Schema (v2):
  * ``camera.location``, ``camera.look_at``, ``camera.lens_mm``,
    ``camera.sensor_mm`` — required.
  * ``road.location``, ``road.rotation_euler`` — required.
  * ``lights.Sun.rotation_euler``, ``lights.Sun.energy`` — required.
  * ``vehicles.lane_defaults["0".."3"].location`` /
    ``.rotation_euler`` — required. Each lane entry is the frame-0 spawn pose
    for a vehicle in that lane on this camera (appear anchor for in-cameras,
    reappear anchor for out-cameras). The kinematics motion then advances the
    vehicle forward from this anchor by speed/turn, so ``metadata.json`` and the
    render share the same JSON-derived start point.

``compute_env`` (below) reproduces this schema from ``geometry.py`` and is used
by tests / a regeneration tool; it is NOT called at render time. ``load_env`` is
the render-time entry point.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

try:                                   # imported as lib.envfile (tests)
    from . import geometry as G
    from . import kinematics as K
except ImportError:                    # imported as top-level envfile (build_scene)
    import geometry as G
    import kinematics as K


SCHEMA_VERSION = 2
ENVS_DIRNAME = "envs"

# Visible segment lengths — must match geometry.compute_motion defaults so the
# env generator's appear/reappear anchors equal what the kinematics would use.
APPROACH_VISIBLE_LENGTH = 40.0
EXIT_VISIBLE_LENGTH = 40.0


# Sun light defaults — must match build_scene.setup_render so the env file
# reflects what is actually placed.
SUN_ENERGY = 4.0
SUN_ROT_EULER = (math.radians(55), 0.0, math.radians(30))


def envs_dir(root: str) -> str:
    return os.path.join(root, "assets", ENVS_DIRNAME)


def env_path(tag: str, root: str) -> str:
    return os.path.join(envs_dir(root), f"{tag}.json")


def parse_tag(tag: str) -> Tuple[G.Direction, bool]:
    role, d = tag.split("_")
    return G.Direction(d), (role == "in")


def vehicles_on_camera(scenario: dict, approach: G.Direction,
                       is_in: bool) -> List[dict]:
    """Vehicles visible on the given camera shot (same filter as build_shot)."""
    out = []
    for veh in scenario["vehicles"]:
        if is_in:
            if veh["approach"] != approach.value:
                continue
        else:
            ex_dir, _ = G.exit_lane_for_movement(
                G.Direction(veh["approach"]), veh["lane"],
                G.Turn(veh["turn"]))
            if ex_dir != approach:
                continue
        out.append(veh)
    return out


# ---------------------------------------------------------------------------
# Generator (geometry-derived v2 schema; used by tests / regeneration tool)
# ---------------------------------------------------------------------------

def compute_env(tag: str, road_meta: dict) -> dict:
    """Build the v2 env dict for one camera tag from geometry.py.

    The lane_defaults are per-lane spawn anchors: for an in-camera this is the
    appear position (box near edge - approach_visible_length along forward);
    for an out-camera this is the reappear position (box far edge on the exit
    lane). These equal what geometry.compute_motion would derive, so an
    unedited env file reproduces the procedural output exactly.
    """
    approach, is_in = parse_tag(tag)
    cam_loc, look_at = G.camera_pose(approach, is_in, road_meta)
    road_loc, road_rot = G.road_arm_transform(approach, road_meta, is_entry=is_in)

    lane_defaults: Dict[str, dict] = {}
    for lane in range(G.NUM_LANES):
        if is_in:
            disp = G.lane_entry_box_edge(approach, lane)
            fx, fy = G.approach_forward(approach)
            ax = disp[0] - fx * APPROACH_VISIBLE_LENGTH
            ay = disp[1] - fy * APPROACH_VISIBLE_LENGTH
            rot_z = G.approach_rotation(approach)
        else:
            edge = G.lane_exit_box_edge(approach, lane)
            ax, ay = edge[0], edge[1]
            rot_z = G.approach_rotation(approach)
        lane_defaults[str(lane)] = {
            "location": [round(ax, 6), round(ay, 6), 0.0],
            "rotation_euler": [0.0, 0.0, round(rot_z, 6)],
        }

    return {
        "camera_tag": tag,
        "approach": approach.value,
        "role": "in" if is_in else "out",
        "schema_version": SCHEMA_VERSION,
        "generated_from": "geometry.camera_pose + geometry.road_arm_transform",
        "camera": {
            "location": [round(cam_loc[0], 6), round(cam_loc[1], 6),
                         round(cam_loc[2], 6)],
            "look_at": [round(look_at[0], 6), round(look_at[1], 6),
                        round(look_at[2], 6)],
            "rotation_euler": None,   # null => derive from look_at
            "lens_mm": G.LENS_MM,
            "sensor_mm": G.SENSOR_MM,
        },
        "road": {
            "object": f"Road_{approach.value}_{'in' if is_in else 'out'}",
            "location": [round(road_loc[0], 6), round(road_loc[1], 6),
                         round(road_loc[2], 6)],
            "rotation_euler": [0.0, 0.0, round(road_rot, 6)],
        },
        "lights": {
            "Sun": {
                "rotation_euler": list(SUN_ROT_EULER),
                "energy": SUN_ENERGY,
            }
        },
        "vehicles": {
            "lane_defaults": lane_defaults,
        },
    }


def dump_env(env: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(env, f, indent=2)


# ---------------------------------------------------------------------------
# Required load + validation (render-time entry point)
# ---------------------------------------------------------------------------

def _fail(msg: str):
    raise SystemExit(f"FAIL: {msg}")


def require_env_fields(env: dict, tag: str) -> None:
    """Hard-fail if any required field is missing or null in ``env``."""
    def need(cond, msg):
        if not cond:
            _fail(f"env {tag}: {msg}")

    cam = env.get("camera") or {}
    need(cam.get("location") is not None and len(cam["location"]) >= 3,
         "camera.location missing")
    need(cam.get("look_at") is not None and len(cam["look_at"]) >= 3,
         "camera.look_at missing")
    need(cam.get("lens_mm") is not None, "camera.lens_mm missing")
    need(cam.get("sensor_mm") is not None, "camera.sensor_mm missing")

    road = env.get("road") or {}
    need(road.get("location") is not None and len(road["location"]) >= 3,
         "road.location missing")
    need(road.get("rotation_euler") is not None and len(road["rotation_euler"]) >= 3,
         "road.rotation_euler missing")

    sun = (env.get("lights") or {}).get("Sun") or {}
    need(sun.get("rotation_euler") is not None and len(sun["rotation_euler"]) >= 3,
         "lights.Sun.rotation_euler missing")
    need(sun.get("energy") is not None, "lights.Sun.energy missing")

    defaults = (env.get("vehicles") or {}).get("lane_defaults") or {}
    for i in range(G.NUM_LANES):
        need(str(i) in defaults, f"vehicles.lane_defaults[{i}] missing")
        e = defaults[str(i)] or {}
        need(e.get("location") is not None and len(e["location"]) >= 3,
             f"lane_defaults[{i}].location missing")
        need(e.get("rotation_euler") is not None and len(e["rotation_euler"]) >= 3,
             f"lane_defaults[{i}].rotation_euler missing")


def load_env(tag: str, root: str) -> dict:
    """Load + validate the env file for ``tag``. Hard-fails if missing/invalid.

    This is the render-time entry point: build_scene and render.compute_metadata
    call this instead of computing from geometry, so the JSON file is the
    required input for all placed model objects.
    """
    path = env_path(tag, root)
    if not os.path.exists(path):
        _fail(f"required env file missing: {path}")
    with open(path) as f:
        env = json.load(f)
    # normalise identity keys to the requested tag (a copied file can't
    # silently mislabel a shot).
    approach, is_in = parse_tag(tag)
    env["camera_tag"] = tag
    env["approach"] = approach.value
    env["role"] = "in" if is_in else "out"
    require_env_fields(env, tag)
    return env


def validate_all_envs(root: str) -> None:
    """Load + validate all 8 env files. Hard-fails on the first problem."""
    for tag in G.camera_names():
        load_env(tag, root)
    print("[env] all 8 env files present and valid")


def lane_default_anchor(env: dict, lane: int) -> Tuple[Tuple[float, float, float], float]:
    """Return ((x, y, z), rot_z_rad) spawn anchor for ``lane`` from the env.

    Hard-fails if the lane entry is missing (required input).
    """
    defaults = (env.get("vehicles") or {}).get("lane_defaults") or {}
    entry = defaults.get(str(lane))
    if entry is None:
        _fail(f"env {env.get('camera_tag')}: lane_defaults[{lane}] missing")
    loc = entry.get("location")
    rot_e = entry.get("rotation_euler")
    if loc is None or rot_e is None or len(loc) < 3 or len(rot_e) < 3:
        _fail(f"env {env.get('camera_tag')}: lane_defaults[{lane}] missing location/rotation_euler")
    return (tuple(loc), rot_e[2])


# ---------------------------------------------------------------------------
# Camera resolution — the single source of truth for the camera pose used by
# both build_scene.place_camera (Blender) and render.compute_metadata (pure
# python).  Applies the env-override-else-geometry precedence in one place so
# the Blender pixels and the metadata ground-truth agree exactly.
# ---------------------------------------------------------------------------

def resolve_camera(env: dict, road_meta: dict) -> dict:
    """Return the resolved camera spec for the env's tag.

    Mirrors ``build_scene.place_camera`` precedence exactly:
      * ``location`` / ``look_at`` — env ``camera`` block if non-null, else
        ``geometry.camera_pose`` (the geometry default for this tag).
      * ``lens_mm`` / ``sensor_mm`` — env value if present, else
        ``G.LENS_MM`` / ``G.SENSOR_MM``.
      * ``rotation_euler`` — passthrough of the env value (may be ``None``,
        meaning "derive from look_at via the track-quat in Blender").  Pure-
        python consumers (metadata) record it as-is; the Blender side derives
        the actual object rotation from look_at when it is null.

    The returned dict is the complete, self-contained camera descriptor that
    ``compute_metadata`` writes verbatim into ``metadata.json["cameras"][tag]``
    and that ``place_camera`` applies to the Blender camera object — so the
    render and the metadata share one resolved value per tag.
    """
    approach, is_in = parse_tag(env["camera_tag"])
    cam_loc, look_at = G.camera_pose(approach, is_in, road_meta)
    ec = env.get("camera") or {}
    location = ec.get("location")
    if location is None:
        location = [round(cam_loc[0], 6), round(cam_loc[1], 6),
                    round(cam_loc[2], 6)]
    look = ec.get("look_at")
    if look is None:
        look = [round(look_at[0], 6), round(look_at[1], 6),
                round(look_at[2], 6)]
    return {
        "location": list(location),
        "look_at": list(look),
        "rotation_euler": ec.get("rotation_euler"),  # None => derive in Blender
        "lens_mm": ec.get("lens_mm", G.LENS_MM),
        "sensor_mm": ec.get("sensor_mm", G.SENSOR_MM),
    }


# ---------------------------------------------------------------------------
# Drift check (non-fatal warning: env camera vs geometry-derived camera)
# ---------------------------------------------------------------------------

def camera_drift(env: dict, scenario: dict, tag: str,
                 road_meta: dict, tol: float = 1e-3) -> Optional[str]:
    """If the env camera location/look_at differs from the geometry-derived
    camera_pose beyond ``tol`` metres, return a human-readable warning string;
    else None.  Used by build_shot to flag that the render may diverge from
    metadata.json (which is always geometry-derived)."""
    approach, is_in = parse_tag(tag)
    cam_loc, look_at = G.camera_pose(approach, is_in, road_meta)
    ec = env.get("camera", {})
    for i, name in enumerate(("x", "y", "z")):
        if abs(ec.get("location", [0, 0, 0])[i] - cam_loc[i]) > tol:
            return (f"camera.location[{name}] env={ec['location'][i]} "
                    f"vs geometry={cam_loc[i]:.6f}")
        if abs(ec.get("look_at", [0, 0, 0])[i] - look_at[i]) > tol:
            return (f"camera.look_at[{name}] env={ec['look_at'][i]} "
                    f"vs geometry={look_at[i]:.6f}")
    return None
