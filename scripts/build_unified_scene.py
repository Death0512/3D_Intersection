#!/usr/bin/env python3
"""Build one large Blender scene from a SUMO-backed trajectory scenario."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)

import bpy

import blender_utils as bu
import envfile as ENV
import geometry as G
import build_scene as BS


ROAD_JSON = os.path.join(ROOT, "assets", "road.json")


def _vehicle_mesh_children(root):
    return [root] + list(root.children_recursive)


def _keyframe_visibility(root, start: int, end: int):
    start = max(0, int(start))
    end = max(start, int(end))
    for obj in _vehicle_mesh_children(root):
        obj.hide_render = start > 0
        obj.hide_viewport = start > 0
        obj.keyframe_insert(data_path="hide_render", frame=0)
        obj.keyframe_insert(data_path="hide_viewport", frame=0)
        obj.hide_render = False
        obj.hide_viewport = False
        obj.keyframe_insert(data_path="hide_render", frame=start)
        obj.keyframe_insert(data_path="hide_viewport", frame=start)
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=end + 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=end + 1)
        BS._set_step_interpolation(obj, "hide_render")
        BS._set_step_interpolation(obj, "hide_viewport")


def _set_linear(obj, data_path: str):
    for fc in BS._fcurves_for(obj, data_path):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _make_vehicle_root(veh: dict, veh_manifest: dict, plates_dir: str):
    cls = veh.get("class", "car")
    meta = veh_manifest[cls]
    coll = BS.append_collection_from_blend(
        os.path.join(ROOT, meta["blend"]), meta["collection"],
        new_name=f"VEH_{veh['id']}_{cls}")
    tex_dir = os.path.join(ROOT, "models", cls, "textures")
    if os.path.isdir(tex_dir):
        bu.remap_textures_to_local(tex_dir)
    BS.assign_plate_and_color(coll, veh.get("plate", veh["id"]), plates_dir,
                              rgba=veh.get("color"))

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    root = bpy.context.view_layer.objects.active
    root.name = f"VEH_{veh['id']}"
    root["forward_offset_deg"] = float(meta.get("forward_offset_deg", 0.0))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in coll.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    return root


def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute angular difference in radians."""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _keyframe_sumo_trajectory(root, veh: dict, frame_end: int,
                              keyframe_stride: int = 6,
                              heading_threshold_deg: float = 1.0,
                              speed_threshold: float = 0.8):
    pts = list(veh.get("trajectory") or [])
    if not pts:
        return False
    fwd_off = math.radians(float(root.get("forward_offset_deg", 0.0)))
    # Keep turn/accel detail, but do not keyframe every SUMO sample.  Blender
    # interpolates linearly between keyframes, so for long 300s renders a 5 FPS
    # animation track (stride=6 at 30 FPS) is visually smooth enough while
    # cutting F-curves, .blend size, build RAM and save time roughly in half
    # versus the old every-3-frame fallback.
    keyframe_stride = max(1, int(keyframe_stride))
    heading_threshold = math.radians(float(heading_threshold_deg))
    speed_threshold = float(speed_threshold)
    selected = []
    prev = None
    for i, p in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            selected.append(p); prev = p; continue
        if prev is None:
            selected.append(p); prev = p; continue
        heading_delta = _angle_delta(float(p.get("rot_z", 0.0)), float(prev.get("rot_z", 0.0)))
        speed_delta = abs(float(p.get("speed", 0.0)) - float(prev.get("speed", 0.0)))
        if heading_delta > heading_threshold or speed_delta > speed_threshold or int(p["frame"]) % keyframe_stride == 0:
            selected.append(p); prev = p

    for p in selected:
        frame = int(p["frame"])
        root.location = (float(p["x"]), float(p["y"]), float(p.get("z", 0.0)))
        root.rotation_euler = (0.0, 0.0, float(p.get("rot_z", 0.0)) + fwd_off)
        root.keyframe_insert(data_path="location", frame=frame)
        root.keyframe_insert(data_path="rotation_euler", frame=frame)
    _set_linear(root, "location")
    _set_linear(root, "rotation_euler")
    _keyframe_visibility(root, pts[0]["frame"], min(int(pts[-1]["frame"]), frame_end))
    return True


def _add_center_plane():
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.03))
    obj = bpy.context.object
    obj.name = "Intersection_Center_Plane"
    obj.dimensions = (G.BOX_SIZE, G.BOX_SIZE, 0.02)
    mat = bpy.data.materials.new("Center_Asphalt_Gray")
    mat.diffuse_color = (0.22, 0.22, 0.22, 1.0)
    obj.data.materials.append(mat)


def _camera_visible_vehicle(veh: dict, tag: str) -> bool:
    direction, is_in = BS.parse_camera_tag(tag)
    if is_in:
        return veh.get("approach") == direction.value
    ex_dir, _ex_lane = G.exit_lane_for_movement(
        G.Direction(veh["approach"]), veh["lane"], G.Turn(veh["turn"]))
    return ex_dir == direction


def _filter_visible_vehicles(scenario: dict, camera_tags: list[str]) -> list[dict]:
    if not camera_tags:
        return list(scenario.get("vehicles", []))
    visible = []
    for veh in scenario.get("vehicles", []):
        for tag in camera_tags:
            if _camera_visible_vehicle(veh, tag):
                visible.append(veh)
                break
    return visible


def _selected_camera_tags(only: str | None) -> list[str]:
    if not only:
        return G.camera_names()
    tags = [t.strip() for t in only.split(",") if t.strip()]
    valid = set(G.camera_names())
    bad = [t for t in tags if t not in valid]
    if bad:
        raise SystemExit(f"FAIL: invalid --only camera tag(s): {', '.join(bad)}")
    if not tags:
        raise SystemExit("FAIL: --only did not contain any camera tag")
    return tags


def build_unified_scene(scenario: dict, out_blend: str, only: str | None = None,
                        keyframe_stride: int = 6,
                        heading_threshold_deg: float = 1.0,
                        speed_threshold: float = 0.8):
    bu.reset_scene()
    with open(ROAD_JSON) as f:
        road_meta = json.load(f)
    veh_manifest = BS.load_vehicle_manifest()
    plates_dir = os.path.join(os.path.dirname(out_blend), "plates")
    os.makedirs(plates_dir, exist_ok=True)

    selected_tags = _selected_camera_tags(only)
    selected_vehicles = _filter_visible_vehicles(scenario, selected_tags)
    total_vehicles = len(scenario.get("vehicles", []))
    # Drop non-visible vehicles from the decoded JSON object before appending
    # Blender assets. Dense 300s scenarios can decode to >2GB of Python objects;
    # keeping all four directions in RAM while rendering only in_N/out_N is a
    # direct waste on 16GB VMs.
    scenario["vehicles"] = selected_vehicles
    gc.collect()

    for d in G.Direction:
        BS.place_road(d, road_meta, is_entry=True, unified=True)
        BS.place_road(d, road_meta, is_entry=False, unified=True)
    _add_center_plane()

    for tag in selected_tags:
        env = ENV.load_env(tag, ROOT)
        direction, is_in = BS.parse_camera_tag(tag)
        cam = BS.place_camera(direction, is_in, road_meta, env=env, unified=True)
        cam.name = f"Camera_{tag}"

    frame_end = max(0, int(scenario.get("duration_frames", 1)) - 1)
    print(f"[unified] placing {len(selected_vehicles)}/{total_vehicles} SUMO vehicles for {len(selected_tags)} camera(s)", flush=True)
    n = 0
    for veh in selected_vehicles:
        root = _make_vehicle_root(veh, veh_manifest, plates_dir)
        if _keyframe_sumo_trajectory(root, veh, frame_end,
                                     keyframe_stride=keyframe_stride,
                                     heading_threshold_deg=heading_threshold_deg,
                                     speed_threshold=speed_threshold):
            n += 1
            if n == 1 or n % 10 == 0:
                print(f"  [unified] keyframed {n} vehicles", flush=True)

    BS.configure_gpu()
    BS.setup_render(samples=48)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = frame_end
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"[unified] saved {out_blend} ({n} vehicles, {len(selected_tags)} cameras)", flush=True)


def main():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender -b --python build_unified_scene.py -- --scenario scenario.json --out unified_scene.blend")
    post = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated camera tags to include (e.g. in_N,out_N)")
    ap.add_argument("--keyframe-stride", type=int, default=6,
                    help="fallback frame spacing for straight/steady SUMO tracks (default: 6 = 5 FPS at 30 FPS)")
    ap.add_argument("--heading-threshold-deg", type=float, default=1.0,
                    help="always keep a pose when heading changed by more than this many degrees")
    ap.add_argument("--speed-threshold", type=float, default=0.8,
                    help="always keep a pose when speed changed by more than this many m/s")
    ns = ap.parse_args(post)
    with open(ns.scenario) as f:
        scenario = json.load(f)
    build_unified_scene(scenario, ns.out, only=ns.only,
                        keyframe_stride=ns.keyframe_stride,
                        heading_threshold_deg=ns.heading_threshold_deg,
                        speed_threshold=ns.speed_threshold)


if __name__ == "__main__":
    main()
