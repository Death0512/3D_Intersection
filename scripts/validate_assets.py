"""Phase 0 — Asset validation (car + road).

Re-opens each prepared asset .blend and asserts the prep invariants hold.
Prints a PASS/FAIL report. Exits non-zero if any asset fails.

Run:
    blender -b --python scripts/validate_assets.py
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

ASSETS_DIR = os.path.join(HERE, "..", "assets")
VEHICLES_JSON = os.path.join(ASSETS_DIR, "vehicles.json")
ROAD_JSON = os.path.join(ASSETS_DIR, "road.json")

LICENSE_MAT = "LicensePlate_Mat"
TOLERANCE = 0.05  # 5 % length tolerance


def bbox_world(objs):
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for o in objs:
        if o.type != "MESH":
            continue
        for v in o.bound_box:
            wc = o.matrix_world @ Vector(v)
            for i in range(3):
                mn[i] = min(mn[i], wc[i]); mx[i] = max(mx[i], wc[i])
    return mn, mx


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def validate_road() -> bool:
    print("=" * 60)
    print("VALIDATE: road")
    print("=" * 60)
    ok = True
    if not os.path.exists(ROAD_JSON):
        check("road.json exists", False)
        return False
    with open(ROAD_JSON) as f:
        meta = json.load(f)
    blend_path = os.path.join(HERE, "..", meta["blend"])
    if not os.path.exists(blend_path):
        check("road.blend exists", False)
        return False
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    coll = bpy.data.collections.get(meta["collection"])
    ok &= check("ENV_road collection present", coll is not None)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    ok &= check("has mesh objects", len(meshes) >= 1, f"{len(meshes)} mesh(es)")
    mn, mx = bbox_world(meshes)
    width = mx.x - mn.x
    length = mx.y - mn.y
    ok &= check(f"width ~ {meta['arm_width']} m", abs(width - meta["arm_width"]) < 0.1, f"got {width:.3f}")
    ok &= check(f"length > 30 m", length > 30.0, f"got {length:.3f}")
    ok &= check("min Z ~ 0", abs(mn.z) < 0.1, f"got {mn.z:.3f}")
    ok &= check(f"no cameras", not any(o.type == "CAMERA" for o in bpy.data.objects))
    ok &= check(f"no lights", not any(o.type == "LIGHT" for o in bpy.data.objects))
    print("-" * 60)
    return ok


def validate_vehicle(veh_class: str, meta: dict) -> bool:
    print("=" * 60)
    print(f"VALIDATE: {veh_class}")
    print("=" * 60)
    ok = True
    blend_path = os.path.join(HERE, "..", meta["blend"])
    if not os.path.exists(blend_path):
        check("blend exists", False, blend_path)
        return False
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    coll = bpy.data.collections.get(meta["collection"])
    ok &= check(f"collection {meta['collection']} present", coll is not None)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    ok &= check("has mesh objects", len(meshes) >= 1, f"{len(meshes)}")
    mn, mx = bbox_world(meshes)
    dims = [round(mx.x - mn.x, 3), round(mx.y - mn.y, 3), round(mx.z - mn.z, 3)]
    # The model's length axis depends on forward_offset_deg: 0/180 -> length
    # along Y; +/-90 -> length along X. Compare target_length against the
    # axis the vehicle will actually travel after the offset is applied.
    fwd_off = meta.get("forward_offset_deg", 0.0) % 180
    length_axis_idx = 0 if abs(fwd_off - 90.0) < 1e-3 else 1   # 0=X, 1=Y
    length_axis_name = "X" if length_axis_idx == 0 else "Y"
    ok &= check(f"length ~ {meta['target_length']} m on {length_axis_name} (±{TOLERANCE*100:.0f}%)",
                abs(dims[length_axis_idx] - meta["target_length"]) / meta["target_length"] <= TOLERANCE,
                f"got dims={dims}")
    ok &= check("min Z ~ 0 (grounded)", abs(mn.z) < 0.05, f"got {mn.z:.3f}")

    # license plate material
    mat = bpy.data.materials.get(LICENSE_MAT)
    ok &= check(f"{LICENSE_MAT} present", mat is not None)
    if mat:
        node = bu.find_image_texture_node(mat)
        ok &= check(f"{LICENSE_MAT} has Image Texture node", node is not None)
        if node:
            ok &= check("plate image assigned", node.image is not None,
                        node.image.name if node.image else "None")

    # no stray cameras / lights / armatures
    ok &= check("no cameras", not any(o.type == "CAMERA" for o in bpy.data.objects))
    ok &= check("no lights", not any(o.type == "LIGHT" for o in bpy.data.objects))
    ok &= check("no armatures", not any(o.type == "ARMATURE" for o in bpy.data.objects))

    if meta.get("missing_textures"):
        print(f"  [INFO] missing textures: {len(meta['missing_textures'])}")

    print(f"  dims: {dims}")
    print("-" * 60)
    return ok


def main():
    all_ok = True
    all_ok &= validate_road()
    if not os.path.exists(VEHICLES_JSON):
        print("FAIL: vehicles.json not found")
        sys.exit(1)
    with open(VEHICLES_JSON) as f:
        vdata = json.load(f)
    for veh_class, meta in vdata.get("vehicles", {}).items():
        all_ok &= validate_vehicle(veh_class, meta)

    print("=" * 60)
    print("OVERALL:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
