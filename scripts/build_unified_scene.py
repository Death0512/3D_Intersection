#!/usr/bin/env python3
"""Build one large Blender scene from a SUMO-backed trajectory scenario."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import site as _site_module
_user_site = _site_module.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import ijson

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

# ponytail: template cache per class — append each vehicle .blend once and clone
# mesh objects + plate/body materials per vehicle.  Cleared after reset_scene.
_TEMPLATE_CACHE: dict[str, bpy.types.Collection] = {}


def _template_for_class(cls: str, veh_manifest: dict,
                        out_blend_dir: str | None = None) -> bpy.types.Collection:
    """Return the master appended collection for a vehicle class, appending once."""
    if cls in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cls]
    meta = veh_manifest[cls]
    tmpl = BS.append_collection_from_blend(
        os.path.join(ROOT, meta["blend"]), meta["collection"],
        new_name=f"TEMPLATE_{cls}")
    # Remap textures once per class, relative to the output .blend dir
    tex_dir = os.path.join(ROOT, "models", cls, "textures")
    if os.path.isdir(tex_dir):
        bu.remap_textures_to_local(tex_dir, relative_to_dir=out_blend_dir)
    # Hide template objects and unlink from scene so they don't render
    for obj in tmpl.objects:
        obj.hide_render = True
        obj.hide_viewport = True
    _TEMPLATE_CACHE[cls] = tmpl
    return tmpl


def _reset_template_cache():
    """Clear template cache after reset_scene invalidates Blender data."""
    _TEMPLATE_CACHE.clear()


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
        for dp in ("hide_render", "hide_viewport"):
            for fc in BS._fcurves_for(obj, dp):
                fc.keyframe_points.foreach_set("interpolation", [0] * len(fc.keyframe_points))


def _make_vehicle_root(veh: dict, veh_manifest: dict, plates_dir: str,
                       out_blend_dir: str | None = None):
    """Clone vehicle mesh objects from the per-class template, isolating
    plate/body materials per vehicle.  All other mesh data stays shared."""
    cls = veh.get("class", "car")
    meta = veh_manifest[cls]
    tmpl = _template_for_class(cls, veh_manifest, out_blend_dir=out_blend_dir)

    # Find plate and body materials on the template BEFORE cloning so we know
    # which material slots to duplicate per-vehicle.
    tmpl_plate_mat, _tmpl_plate_obj = BS._find_plate_material_in_collection(tmpl)
    tmpl_body_mat, _tmpl_body_obj = BS._find_body_material_in_collection(tmpl)

    # Build a per-vehicle map: template-material → cloned copy (or None = shared)
    mat_copies: dict[bpy.types.Material, bpy.types.Material | None] = {}
    if tmpl_plate_mat is not None:
        mat_copies[tmpl_plate_mat] = tmpl_plate_mat.copy()
    if tmpl_body_mat is not None:
        mat_copies[tmpl_body_mat] = tmpl_body_mat.copy()

    # Clone every object from the template.
    new_objs = []
    for obj in tmpl.objects:
        new_obj = obj.copy()
        new_obj.hide_render = False
        new_obj.hide_viewport = False
        new_obj.animation_data_clear()
        if obj.type == "MESH":
            needs_copy = any(
                slot.material is not None and slot.material in mat_copies
                for slot in obj.material_slots)
            if needs_copy:
                # Only copy mesh data that carries plate/body materials so
                # per-vehicle material assignments don't bleed into siblings.
                new_obj.data = obj.data.copy()
            # Replace plate and body materials with per-vehicle copies
            for slot in new_obj.material_slots:
                if slot.material is None:
                    continue
                orig = slot.material
                for tmpl_mat, copy_mat in mat_copies.items():
                    if orig == tmpl_mat:
                        slot.material = copy_mat
                        break
        bpy.context.scene.collection.objects.link(new_obj)
        new_objs.append(new_obj)

    # Create the per-vehicle collection
    coll_name = f"VEH_{veh['id']}_{cls}"
    coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(coll)
    for obj in new_objs:
        coll.objects.link(obj)
        bpy.context.scene.collection.objects.unlink(obj)

    # Now apply plate + color to the per-vehicle collection (the mat copies
    # were already swapped, so assign_plate_and_color finds the copies)
    # ponytail: assign_plate_and_color re-finds materials by name prefix;
    # the copies have the same base name so the finders still work.
    BS.assign_plate_and_color(coll, veh.get("plate", veh["id"]), plates_dir,
                              rgba=veh.get("color"),
                              out_blend_dir=out_blend_dir)

    root = bpy.data.objects.new(f"VEH_{veh['id']}", None)
    root.empty_display_type = "PLAIN_AXES"
    root["forward_offset_deg"] = float(meta.get("forward_offset_deg", 0.0))
    bpy.context.scene.collection.objects.link(root)
    for obj in new_objs:
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
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

    # Bulk-populate transform F-curves via the Blender 5.2 slotted-action API.
    # Bootstrap: one keyframe on each data_path creates the action/slot/layer/channelbag.
    root.location = (float(pts[0]["x"]), float(pts[0]["y"]), float(pts[0].get("z", 0.0)))
    root.rotation_euler = (0.0, 0.0, float(pts[0].get("rot_z", 0.0)) + fwd_off)
    root.keyframe_insert(data_path="location", frame=0)
    root.keyframe_insert(data_path="rotation_euler", frame=0)

    # Locate all six fcurves (3 loc + 3 rot) via the shared API.
    loc_fcs = sorted(BS._fcurves_for(root, "location"), key=lambda fc: fc.array_index)
    rot_fcs = sorted(BS._fcurves_for(root, "rotation_euler"), key=lambda fc: fc.array_index)
    all_fcs = loc_fcs + rot_fcs
    if len(all_fcs) != 6:
        raise RuntimeError(
            f"Expected 6 FCurves for {root.name}, got {len(all_fcs)}. "
            f"Check Blender 5.2 slotted-action bootstrap."
        )

    # Clear the dummy keyframe points we created during bootstrap.
    for fc in all_fcs:
        n = len(fc.keyframe_points)
        for _ in range(n):
            fc.keyframe_points.remove(fc.keyframe_points[0])

    # Build per-channel flat arrays for foreach_set('co', ...).
    # co order: [frame0, value0, frame1, value1, ...]
    n_sel = len(selected)
    frames_flat: list[float] = []
    loc_x: list[float] = []; loc_y: list[float] = []; loc_z: list[float] = []
    rot_x: list[float] = []; rot_y: list[float] = []; rot_z: list[float] = []
    for p in selected:
        f = float(p["frame"])
        frames_flat.append(f)
        loc_x.append(float(p["x"]))
        loc_y.append(float(p["y"]))
        loc_z.append(float(p.get("z", 0.0)))
        rot_x.append(0.0)
        rot_y.append(0.0)
        rot_z.append(float(p.get("rot_z", 0.0)) + fwd_off)

    channel_values = [loc_x, loc_y, loc_z, rot_x, rot_y, rot_z]

    for fc, vals in zip(all_fcs, channel_values):
        fc.keyframe_points.add(n_sel)
        flat = []
        for i in range(n_sel):
            flat.append(frames_flat[i])
            flat.append(vals[i])
        fc.keyframe_points.foreach_set("co", flat)
        fc.keyframe_points.foreach_set("interpolation", [1] * n_sel)
        fc.update()

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


def _vehicle_visible(veh: dict, camera_tags: list[str]) -> bool:
    """Check visibility using top-level vehicle fields only — no trajectory needed."""
    if not camera_tags:
        return True
    for tag in camera_tags:
        if _camera_visible_vehicle(veh, tag):
            return True
    return False


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


def _veh_trajectory_frames(veh: dict) -> tuple[int, int] | None:
    """Return (first_global_frame, last_global_frame) or None if no trajectory."""
    pts = veh.get("trajectory") or []
    if not pts:
        return None
    return (int(pts[0]["frame"]), int(pts[-1]["frame"]))


def _veh_overlaps_chunk(veh: dict, chunk_start: int, chunk_end: int) -> bool:
    tf = _veh_trajectory_frames(veh)
    if tf is None:
        return False
    return tf[0] <= chunk_end and tf[1] >= chunk_start


def build_unified_scene(scenario_path: str, out_blend: str, only: str | None = None,
                        keyframe_stride: int = 6,
                        heading_threshold_deg: float = 1.0,
                        speed_threshold: float = 0.8,
                        chunk_start: int | None = None,
                        chunk_end: int | None = None):
    selected_tags = _selected_camera_tags(only)

    # Stream-read top-level metadata (duration_frames), then stream vehicles.
    duration_frames = 30 * 10  # default fallback
    with open(scenario_path, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        for prefix, event, value in parser:
            if prefix == "fps" and event == "number":
                pass
            elif prefix == "duration_frames" and event == "number":
                duration_frames = int(value)
            elif prefix.startswith("vehicles."):
                break

    full_frame_end = max(0, duration_frames - 1)

    # Chunk mode: use chunk bounds for scene frame range
    if chunk_start is not None and chunk_end is not None:
        c_start = int(chunk_start)
        c_end = int(chunk_end)
    else:
        c_start = 0
        c_end = full_frame_end

    bu.reset_scene()
    _reset_template_cache()
    with open(ROAD_JSON) as f:
        road_meta = json.load(f)
    veh_manifest = BS.load_vehicle_manifest()
    # Plates are pre-generated once per scenario by step_plates(). Chunks must
    # share that cache; a per-chunk cache would regenerate PNGs in Blender.
    plates_dir = os.path.join(os.path.dirname(os.path.abspath(scenario_path)), "plates")
    os.makedirs(plates_dir, exist_ok=True)

    for d in G.Direction:
        BS.place_road(d, road_meta, is_entry=True, unified=True)
        BS.place_road(d, road_meta, is_entry=False, unified=True)
    _add_center_plane()

    for tag in selected_tags:
        env = ENV.load_env(tag, ROOT)
        direction, is_in = BS.parse_camera_tag(tag)
        cam = BS.place_camera(direction, is_in, road_meta, env=env, unified=True)
        cam.name = f"Camera_{tag}"

    # Stream vehicles: only instantiate if trajectory overlaps chunk window.
    n = 0
    total_count = 0
    skipped_outside = 0
    with open(scenario_path, "rb") as f:
        for veh in ijson.items(f, "vehicles.item", use_float=True):
            total_count += 1
            if not _vehicle_visible(veh, selected_tags):
                continue
            if not _veh_overlaps_chunk(veh, c_start, c_end):
                skipped_outside += 1
                continue
            root = _make_vehicle_root(veh, veh_manifest, plates_dir,
                                      out_blend_dir=os.path.dirname(out_blend))
            # Keyframe ALL trajectory points (full global frames, never rebase)
            # so vehicles that straddle the boundary interpolate correctly.
            if _keyframe_sumo_trajectory(root, veh, full_frame_end,
                                         keyframe_stride=keyframe_stride,
                                         heading_threshold_deg=heading_threshold_deg,
                                         speed_threshold=speed_threshold):
                n += 1
                veh.clear()
                if n % 10 == 1 or n == 1:
                    print(f"  [unified] keyframed {n} vehicles", flush=True)

    print(f"[unified] chunk [{c_start},{c_end}] placing {n}/{total_count} "
          f"vehicles for {len(selected_tags)} camera(s) "
          f"(skipped {skipped_outside} outside chunk)", flush=True)

    BS.setup_render(samples=48)
    scene = bpy.context.scene
    scene.frame_start = c_start
    scene.frame_end = c_end
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
    ap.add_argument("--chunk-start", type=int, default=None,
                    help="global frame start for a time chunk (default: 0)")
    ap.add_argument("--chunk-end", type=int, default=None,
                    help="global frame end for a time chunk (default: duration_frames-1)")
    ns = ap.parse_args(post)
    build_unified_scene(ns.scenario, ns.out, only=ns.only,
                        keyframe_stride=ns.keyframe_stride,
                        heading_threshold_deg=ns.heading_threshold_deg,
                        speed_threshold=ns.speed_threshold,
                        chunk_start=ns.chunk_start,
                        chunk_end=ns.chunk_end)


if __name__ == "__main__":
    main()
