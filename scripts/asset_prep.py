"""Phase 0 — Vehicle asset preparation (per model).

Normalizes scale, origin, orientation, textures, and license-plate
infrastructure for each vehicle class, producing a linkable collection.

Usage (one model at a time):
    blender -b <source.blend> --python scripts/asset_prep.py -- <class>

where <class> is one of: car, van, truck, bus.

Produces:
    assets/<class>.blend     (collection VEH_<class>, LicensePlate_Mat)
    appends to assets/vehicles.json
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

import bpy
from mathutils import Vector

import blender_utils as bu


# ---- per-class configuration ------------------------------------------------
# target_length : uniform-scale so the vehicle's Y length matches this (m)
# plate_offsets : (front, rear) Y offsets from origin where a plate plane is
#                 added for van/truck/bus. car keeps its existing Plate mesh.
TARGET_LENGTH = {
    "car": 4.47,
    "van": 5.0,
    "truck": 7.0,
    "bus": 8.33,
}
# standard EU plate size
PLATE_W = 0.52
PLATE_H = 0.11
PLATE_Z = 0.55  # approx bumper/plate height above ground for built plates

ASSETS_DIR = os.path.join(HERE, "..", "assets")
OUT_JSON = os.path.join(ASSETS_DIR, "vehicles.json")
COLLECTION_PREFIX = "VEH_"
LICENSE_MAT = "LicensePlate_Mat"


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


def apply_transforms(obj, loc=True, rot=True, sca=True):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=sca)


def all_mesh_objs():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def keep_only_vehicle_meshes(veh_class: str):
    """Remove non-vehicle meshes (rigs, lamps, sensors, stray env geometry).

    Heuristic: drop objects whose name matches rig/sensor/light patterns or
    that have no materials and tiny vertex counts (often helper widgets).
    """
    drop_patterns = ("wgt-", "wgt-carrig", "drifthandle", "groundsensor",
                     "groundsensor.axle", "steering", "suspension",
                     "wheelbrake", "wheeldamper", "carrig.wheel")
    removed = []
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        nm = o.name.lower()
        if any(p in nm for p in drop_patterns):
            bpy.data.objects.remove(o, do_unlink=True); removed.append(o.name); continue
        # widget meshes have no materials and very few verts
        if not o.data.materials and len(o.data.vertices) < 200:
            bpy.data.objects.remove(o, do_unlink=True); removed.append(o.name); continue
    if removed:
        print(f"  stripped {len(removed)} non-vehicle meshes: {removed[:8]}{'...' if len(removed)>8 else ''}")


def join_all_vehicle_meshes():
    """Join every mesh object into one (so transform_apply / origin ops are
    unambiguous). Returns the joined object."""
    objs = all_mesh_objs()
    if not objs:
        raise SystemExit("No meshes to join")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_scale(target_len: float):
    """Uniform-scale the single joined vehicle so its Y length = target_len."""
    obj = join_all_vehicle_meshes()
    # apply any pre-existing object transforms first so we measure true mesh size
    apply_transforms(obj, loc=True, rot=True, sca=True)
    mn, mx = bbox_world([obj])
    cur_len = mx.y - mn.y
    if cur_len <= 0:
        raise SystemExit(f"Vehicle has zero length? bbox mn={mn} mx={mx}")
    s = target_len / cur_len
    obj.scale = (s, s, s)
    apply_transforms(obj, loc=False, rot=False, sca=True)
    return s, cur_len


def set_origin_ground_center():
    """Set the joined vehicle's origin so it sits at ground-center
    (X/Y center, min Z = 0)."""
    obj = all_mesh_objs()[0]
    mn, mx = bbox_world([obj])
    center = Vector(((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, mn.z))
    bpy.context.scene.cursor.location = center
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    apply_transforms(obj, loc=True, rot=False, sca=False)
    # drop to Z=0
    mn, _ = bbox_world([obj])
    if mn.z != 0.0:
        obj.location.z -= mn.z
        apply_transforms(obj, loc=True, rot=False, sca=False)


def make_license_plate_material(plate_image_path: str):
    """Create/standardize LicensePlate_Mat as Image Texture -> Principled -> Output."""
    mat = bu.ensure_material(LICENSE_MAT)
    nt = mat.node_tree
    # clear existing nodes
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(plate_image_path) if os.path.exists(plate_image_path) else None
    if img is None:
        img = bpy.data.images.new("plate_default", 64, 32)
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    # mark as non-emissive flat plate
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat, tex


def add_plate_plane(name: str, y_offset: float, mat):
    """Add a standard plate plane at +Y (front) or -Y (rear) of the vehicle."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, y_offset, PLATE_Z))
    plane = bpy.context.view_layer.objects.active
    plane.name = name
    plane.scale = (PLATE_W, PLATE_H, 1.0)
    apply_transforms(plane, loc=False, rot=False, sca=True)
    # rotate to face outward along Y (plane default faces +Z)
    plane.rotation_euler = (math.radians(90), 0.0, 0.0)
    apply_transforms(plane, loc=False, rot=True, sca=False)
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    return plane


def prep_car_plate(plate_image_path: str):
    """Car already has a Plate mesh + Plate/Plateb materials. Standardize:
    rename material to LicensePlate_Mat, repoint its image texture to the local
    plate.png, return the image-texture node datapath.
    """
    plate_obj = bpy.data.objects.get("Plate")
    if plate_obj is None:
        # fallback: build one like other vehicles
        mat, tex = make_license_plate_material(plate_image_path)
        add_plate_plane("Plate_Front", +TARGET_LENGTH["car"] / 2.0 - 0.05, mat)
        add_plate_plane("Plate_Rear", -TARGET_LENGTH["car"] / 2.0 + 0.05, mat)
        return mat, tex

    # Use the existing Plate material: rename & repoint.
    mat = plate_obj.data.materials.get("Plate") or plate_obj.data.materials[0]
    mat.name = LICENSE_MAT
    mat.use_nodes = True
    nt = mat.node_tree
    tex = bu.find_image_texture_node(mat)
    if tex is None:
        # rebuild node tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        tex = nt.nodes.new("ShaderNodeTexImage")
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if os.path.exists(plate_image_path):
        img = bpy.data.images.load(plate_image_path)
        tex.image = img
    return mat, tex


def prep_vehicle(veh_class: str):
    print("=" * 60)
    print(f"VEHICLE PREP: {veh_class}")
    print("=" * 60)

    target_len = TARGET_LENGTH[veh_class]

    # 1. Strip non-vehicle helper meshes.
    keep_only_vehicle_meshes(veh_class)

    # 2. Normalize scale.
    s, cur_len = normalize_scale(target_len)
    print(f"  scale: {cur_len:.3f} m -> {target_len} m  (x{s:.4f})")

    # 3. Origin -> ground center, forward = +Y.
    set_origin_ground_center()
    mn, mx = bbox_world(all_mesh_objs())
    print(f"  final dims: X={mx.x-mn.x:.3f} Y={mx.y-mn.y:.3f} Z={mx.z-mn.z:.3f}  minZ={mn.z:.3f}")

    # 4. Texture remap to local textures/ dir.
    tex_dir = os.path.join(HERE, "..", "models", veh_class, "textures")
    missing = []
    n = bu.remap_textures_to_local(tex_dir, missing)
    print(f"  textures: remapped {n}, missing {len(missing)}")

    # 5. Plate setup.
    plate_img = os.path.join(HERE, "..", "models", "car", "textures", "plate.png")
    if veh_class == "car":
        mat, tex = prep_car_plate(plate_img)
        plate_node = f"materials['{LICENSE_MAT}'].node_tree.nodes['{tex.name}']"
    else:
        mat, tex = make_license_plate_material(plate_img)
        half = target_len / 2.0
        add_plate_plane(f"Plate_Front", +half - 0.08, mat)
        add_plate_plane(f"Plate_Rear", -half + 0.08, mat)
        plate_node = f"materials['{LICENSE_MAT}'].node_tree.nodes['{tex.name}']"

    # 6. Strip stray cameras / lamps / armatures (rigs not needed for Black-Box motion).
    for o in list(bpy.data.objects):
        if o.type in {"CAMERA", "LIGHT", "ARMATURE"}:
            bpy.data.objects.remove(o, do_unlink=True)

    # 7. Move all vehicle meshes into VEH_<class> collection.
    coll_name = COLLECTION_PREFIX + veh_class
    coll = bu.ensure_collection(coll_name)
    bu.move_objects_to_collection(all_mesh_objs(), coll)
    bu.purge_orphans()

    # 8. Re-measure final.
    mn, mx = bbox_world(all_mesh_objs())
    dims = [round(mx.x - mn.x, 3), round(mx.y - mn.y, 3), round(mx.z - mn.z, 3)]

    # 9. Save .blend. Disable image packing so a missing source texture
    #    doesn't abort the save (we accept flat materials for missing tex).
    out_blend = os.path.join(ASSETS_DIR, f"{veh_class}.blend")
    os.makedirs(ASSETS_DIR, exist_ok=True)
    # unpack any packed images so save doesn't try to re-pack missing files
    try:
        bpy.ops.file.unpack_all(method="USE_LOCAL")
    except Exception:
        pass
    try:
        bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=False)
    except RuntimeError as e:
        # save usually succeeds even if it warns about packing
        print(f"  (save warning: {e})")

    # 10. Update manifest.
    entry = {
        "class": veh_class,
        "blend": f"assets/{veh_class}.blend",
        "collection": coll_name,
        "dims": dims,
        "plate_material": LICENSE_MAT,
        "plate_node": plate_node,
        "plate_default_image": "models/car/textures/plate.png",
        "target_length": target_len,
        "missing_textures": missing,
    }
    update_manifest(entry, out_blend)
    print(f"  saved: {out_blend}")
    print(f"  dims:  {dims}")
    print(f"  plate: {LICENSE_MAT}  node={tex.name}")
    if missing:
        print(f"  MISSING TEXTURES ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    print("-" * 60)


def update_manifest(entry, out_blend):
    data = {}
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as f:
                data = json.load(f)
        except Exception:
            data = {}
    vehicles = data.get("vehicles", {})
    vehicles[entry["class"]] = entry
    data["vehicles"] = vehicles
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = sys.argv
    if "--" not in args:
        raise SystemExit("Usage: blender -b <src.blend> --python scripts/asset_prep.py -- <class>")
    veh_class = args[args.index("--") + 1].strip().lower()
    if veh_class not in TARGET_LENGTH:
        raise SystemExit(f"Unknown class '{veh_class}'. One of: {list(TARGET_LENGTH)}")
    prep_vehicle(veh_class)


if __name__ == "__main__":
    main()
