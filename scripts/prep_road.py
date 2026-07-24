"""Phase 0 — Road arm asset preparation.

Run headless:
    blender -b models/road/road.blend --python scripts/prep_road.py

Produces:
    assets/road.blend        (collection ENV_road, width normalized to 14m)
    assets/road.json         (arm geometry metadata for later phases)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

import bpy
from mathutils import Vector

import blender_utils as bu


# ---- locked geometry -------------------------------------------------------
TARGET_WIDTH = 14.0          # 4 lanes x 3.5 m  (one travel direction)
LANE_WIDTH = 3.5
NUM_LANES = 4
# lane centre lines, centered on arm centre (entry side; +X = right)
LANE_CENTERLINES = [-5.25, -1.75, 1.75, 5.25]

ROAD_SOURCE = os.path.join(HERE, "..", "models", "road", "road.blend")
OUT_BLEND = os.path.join(HERE, "..", "assets", "road.blend")
OUT_JSON = os.path.join(HERE, "..", "assets", "road.json")
COLLECTION = "ENV_road"

SEMANTIC_DEFAULTS = {
    "approach_length": 54.751,
    "crosswalk_y": 27.846,
    "stop_line_y": 27.846,
}


def bbox_world(obj):
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for v in obj.bound_box:
        wc = obj.matrix_world @ Vector(v)
        for i in range(3):
            mn[i] = min(mn[i], wc[i])
            mx[i] = max(mx[i], wc[i])
    return mn, mx


def apply_transforms(obj, loc=True, rot=True, sca=True):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=sca)


def main() -> None:
    previous_meta = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            previous_meta = json.load(f)
    if os.path.abspath(bpy.data.filepath) != os.path.abspath(ROAD_SOURCE):
        bpy.ops.wm.open_mainfile(filepath=ROAD_SOURCE)

    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise SystemExit("No mesh objects found in road.blend")

    # 1. Join all road meshes into one.
    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    road = bpy.context.view_layer.objects.active
    road.name = "RoadArm"

    # 2a. The source object may carry a non-uniform/rotated object transform
    #     (the mesh data itself can be tiny while matrix_world scales it up).
    #     Apply ALL transforms first so the mesh lives in true world units.
    apply_transforms(road, loc=True, rot=True, sca=True)

    # 2b. Now measure the true width and uniform-scale to TARGET_WIDTH.
    mn, mx = bbox_world(road)
    cur_w = mx.x - mn.x
    if cur_w <= 0:
        raise SystemExit(f"Road has zero width? bbox mn={mn} mx={mx}")
    s = TARGET_WIDTH / cur_w
    road.scale = (s, s, s)
    apply_transforms(road, loc=False, rot=False, sca=True)

    # 3. Set origin to centre of bbox (X centre, Y centre, Z = surface=0).
    mn, mx = bbox_world(road)
    centre = Vector(((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, mn.z))
    bpy.context.scene.cursor.location = centre
    bpy.ops.object.select_all(action="DESELECT")
    road.select_set(True)
    bpy.context.view_layer.objects.active = road
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    apply_transforms(road, loc=True, rot=False, sca=False)

    # 4. Now origin is at (0,0,0) = ground centre. Measure final.
    mn, mx = bbox_world(road)
    width = mx.x - mn.x
    length = mx.y - mn.y
    mesh_y_min = mn.y
    mesh_y_max = mx.y

    # 5. Drop any stray cameras/lights.
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o, do_unlink=True)

    # 6. Move into ENV_road collection.
    coll = bu.ensure_collection(COLLECTION)
    bu.move_objects_to_collection([road], coll)
    bu.purge_orphans()

    # 7. Save.
    os.makedirs(os.path.dirname(OUT_BLEND), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

    # 8. Manifest.
    meta = {
        "blend": "assets/road.blend",
        "collection": COLLECTION,
        "object": road.name,
        "arm_width": round(width, 3),
        "lane_width": LANE_WIDTH,
        "num_lanes": NUM_LANES,
        "lane_centerlines_x": LANE_CENTERLINES,
        # Visual mesh placement metadata. Semantic camera/sim fields below are
        # intentionally preserved instead of being re-derived from mesh bbox.
        "mesh_y_min": round(mesh_y_min, 6),
        "mesh_y_max": round(mesh_y_max, 6),
        "approach_length": previous_meta.get("approach_length", SEMANTIC_DEFAULTS["approach_length"]),
        "crosswalk_y": previous_meta.get("crosswalk_y", SEMANTIC_DEFAULTS["crosswalk_y"]),
        "stop_line_y": previous_meta.get("stop_line_y", SEMANTIC_DEFAULTS["stop_line_y"]),
        "forward_axis": "+Y",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("ROAD PREP OK")
    print(f"  source width : {cur_w:.3f} m  ->  {width:.3f} m  (x{s:.4f})")
    print(f"  length       : {length:.3f} m   mesh_y=[{mesh_y_min:.3f}, {mesh_y_max:.3f}]")
    print(f"  lane X       : {LANE_CENTERLINES}")
    print(f"  saved        : {OUT_BLEND}")
    print("=" * 60)


if __name__ == "__main__":
    main()
