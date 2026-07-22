#!/usr/bin/env python3
"""Build one large Blender scene from a SUMO-backed trajectory scenario."""
from __future__ import annotations

import argparse
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


def _keyframe_sumo_trajectory(root, veh: dict, frame_end: int):
    pts = list(veh.get("trajectory") or [])
    if not pts:
        return False
    fwd_off = math.radians(float(root.get("forward_offset_deg", 0.0)))
    # Keep all turning/accel detail, but reduce long straight constant-speed runs.
    selected = []
    prev = None
    for i, p in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            selected.append(p); prev = p; continue
        if prev is None:
            selected.append(p); prev = p; continue
        heading_delta = abs(float(p.get("rot_z", 0.0)) - float(prev.get("rot_z", 0.0)))
        speed_delta = abs(float(p.get("speed", 0.0)) - float(prev.get("speed", 0.0)))
        if heading_delta > math.radians(0.5) or speed_delta > 0.4 or int(p["frame"]) % 3 == 0:
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


def build_unified_scene(scenario: dict, out_blend: str):
    bu.reset_scene()
    with open(ROAD_JSON) as f:
        road_meta = json.load(f)
    veh_manifest = BS.load_vehicle_manifest()
    plates_dir = os.path.join(os.path.dirname(out_blend), "plates")
    os.makedirs(plates_dir, exist_ok=True)

    for d in G.Direction:
        BS.place_road(d, road_meta, is_entry=True)
        BS.place_road(d, road_meta, is_entry=False)
    _add_center_plane()

    for tag in G.camera_names():
        env = ENV.load_env(tag, ROOT)
        direction, is_in = BS.parse_camera_tag(tag)
        cam = BS.place_camera(direction, is_in, road_meta, env=env)
        cam.name = f"Camera_{tag}"

    frame_end = max(0, int(scenario.get("duration_frames", 1)) - 1)
    print(f"[unified] placing {len(scenario.get('vehicles', []))} SUMO vehicles", flush=True)
    n = 0
    for veh in scenario.get("vehicles", []):
        root = _make_vehicle_root(veh, veh_manifest, plates_dir)
        if _keyframe_sumo_trajectory(root, veh, frame_end):
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
    print(f"[unified] saved {out_blend} ({n} vehicles, 8 cameras)", flush=True)


def main():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender -b --python build_unified_scene.py -- --scenario scenario.json --out unified_scene.blend")
    post = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(post)
    with open(ns.scenario) as f:
        scenario = json.load(f)
    build_unified_scene(scenario, ns.out)


if __name__ == "__main__":
    main()
