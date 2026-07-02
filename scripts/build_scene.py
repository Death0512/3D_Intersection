"""Phase 3 — Scene builder (headless bpy).

Builds a renderable Blender scene for ONE camera shot from a scenario + the
prepped asset library. The full dataset is 8 shots (4 approaches x {In, Out}).
The pipeline driver calls this once per camera.

Run (standalone, for a single shot):
    blender -b --python scripts/build_scene.py -- \\
        --scenario output/run1/scenario.json --camera in_N --out output/run1/scene_in_N.blend

Workflow:
  1. Reset scene.
  2. Link ENV_road from assets/road.blend, orient + position for the approach
     so the road lanes coincide with the world lane centerlines.
  3. For each vehicle visible on this camera, link the VEH_<class> collection,
     make a library override on the instance, assign body color + plate texture.
  4. Keyframe linear motion on the visible segment; hide during Black-Box.
  5. Place one telephoto camera, pitch so frame bottom aligns to stop line
     (In) or crosswalk (Out) -> the box interior is below frame (Black-Box).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)  # for gen_plate

import bpy
from mathutils import Vector, Matrix

import blender_utils as bu
import envfile as ENV
import geometry as G
import kinematics as K
from gen_plate import render_plate


ASSETS_DIR = os.path.join(HERE, "..", "assets")
VEHICLES_JSON = os.path.join(ASSETS_DIR, "vehicles.json")
ROAD_JSON = os.path.join(ASSETS_DIR, "road.json")

LENS_MM = G.LENS_MM          # telephoto (shared single source of truth)
RES_X = G.RES_X
RES_Y = G.RES_Y
FPS = G.FPS
CAM_HEIGHT = G.CAM_HEIGHT    # camera elevation (m)
CAM_BACK_DIST = G.CAM_BACK_DIST  # how far behind the stop line the camera sits
                         # (must be > approach_visible_length=40 so vehicles
                         #  appear IN FRONT of the camera, not behind it)


# ---------------------------------------------------------------------------
# Asset linking
# ---------------------------------------------------------------------------

def link_collection_from_blend(blend_path: str, collection_name: str, link=True):
    """Link or append a collection from an external .blend into the current scene.
    Returns the collection. When link=False (append), the collection and its
    objects become LOCAL (editable)."""
    blend_path = os.path.abspath(blend_path)
    with bpy.data.libraries.load(blend_path, link=link) as (src, dst):
        if collection_name not in src.collections:
            raise RuntimeError(f"Collection '{collection_name}' not in {blend_path}")
        dst.collections = [collection_name]
    linked = None
    for c in bpy.data.collections:
        if c.name == collection_name:
            linked = c
            break
    if linked is None:
        linked = bpy.data.collections.get(collection_name)
    return linked


def append_collection_from_blend(blend_path: str, collection_name: str,
                                 new_name: str = None):
    """Append a collection (make local) so its objects/materials are editable.
    Links it into the current scene's collection tree. Returns the collection."""
    coll = link_collection_from_blend(blend_path, collection_name, link=False)
    if new_name and coll.name != new_name:
        coll.name = new_name
    # ensure it's in the scene's collection tree so its objects are in the view layer
    scene_coll = bpy.context.scene.collection
    if coll.name not in [c.name for c in scene_coll.children]:
        scene_coll.children.link(coll)
    return coll


def instantiate_linked_collection(linked_coll, instance_name: str,
                                  location, rotation_z=0.0):
    """Create an Empty instance of the linked collection at the given location."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=location)
    empty = bpy.context.view_layer.objects.active
    empty.name = instance_name
    empty.instance_type = "COLLECTION"
    empty.instance_collection = linked_coll
    empty.rotation_euler = (0.0, 0.0, rotation_z)
    return empty


# ---------------------------------------------------------------------------
# Road placement (aligned to lane centerlines)
# ---------------------------------------------------------------------------

def place_road(approach: G.Direction, road_meta: dict, is_entry: bool = True,
               env_road: dict = None):
    """Link the road arm and orient + position it for the given approach.

    Each road arm is a self-contained 14 m-wide, ~54.75 m-long asset whose
    local +Y end carries the crosswalk / stop-line (crosswalk_y ≈ +27.85) and
    whose local −Y end is the far/back end (≈ −26.9).

    Per-shot frame: road axis centred on (0,0,0); no carriageway lateral offset.
    Lane centrelines sit at x = LANE_CENTERLINES[k] in the arm's local frame,
    which aligns with world x after rotation by approach_rotation(approach).

    Entry road (is_entry=True):
        The arm's local +Y (crosswalk/stop-line) is placed at the box near-edge
        (−approach × BOX/2) and extends outward.
        origin = near_edge − forward × crosswalk_y.

    Exit road (is_entry=False):
        The arm's local −Y (back) end is placed at the box far edge
        (+approach × BOX/2) and extends outward to approach_length.
        origin = far_edge + forward × arm_back.

    If ``env_road`` (the env file's ``road`` block) is given, its location and
    rotation_euler OVERRIDE the computed transform (override layer).
    """
    coll = link_collection_from_blend(
        os.path.join(HERE, "..", road_meta["blend"]),
        road_meta["collection"])

    (ox, oy, oz), rot = G.road_arm_transform(approach, road_meta, is_entry=is_entry)
    if env_road is not None:
        loc = env_road.get("location")
        rot_e = env_road.get("rotation_euler")
        if loc is not None:
            ox, oy, oz = loc
        if rot_e is not None and len(rot_e) >= 3:
            rot = rot_e[2]

    label = "in" if is_entry else "out"
    empty = instantiate_linked_collection(coll, f"Road_{approach.value}_{label}",
                                          (ox, oy, oz), rotation_z=rot)
    return empty, coll


BOX_HALF = G.BOX_SIZE / 2


# ---------------------------------------------------------------------------
# Vehicle placement, plate swap, body color
# ---------------------------------------------------------------------------

def load_vehicle_manifest():
    with open(VEHICLES_JSON) as f:
        return json.load(f)["vehicles"]


def _find_plate_material_in_collection(coll):
    """Find a LicensePlate_Mat* material among the meshes of a collection.
    Matches any material whose name starts with 'LicensePlate_Mat' (appended
    duplicates get suffixes like .001)."""
    for obj in coll.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material.name.startswith("LicensePlate_Mat"):
                return slot.material, obj
    return None, None


def _find_body_material_in_collection(coll):
    """Find a likely body-paint material in the collection."""
    for obj in coll.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if not slot.material:
                continue
            n = slot.material.name.lower()
            if any(c in n for c in ("carpaint", "car paint", "bodycolour")):
                return slot.material, obj
    return None, None


def assign_plate_and_color(coll, plate_str: str, plates_dir: str, rgba=None):
    """Edit the (local/appended) collection's materials:
      * load the unique plate PNG into LicensePlate_Mat's Image Texture node
      * set the body-paint material base color to rgba (if given)
    """
    safe = "".join(c if c.isalnum() else "_" for c in plate_str) + ".png"
    plate_path = os.path.abspath(os.path.join(plates_dir, safe))
    render_plate(plate_str, plate_path)

    plate_mat, _ = _find_plate_material_in_collection(coll)
    if plate_mat is not None:
        try:
            plate_mat.use_nodes = True
            nt = plate_mat.node_tree
            tex_node = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
            if tex_node is None:
                tex_node = nt.nodes.new("ShaderNodeTexImage")
                bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
                out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
                if bsdf and out:
                    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            img = bpy.data.images.load(plate_path)
            tex_node.image = img
        except Exception as e:
            print(f"  (plate assign warn: {e})")

    if rgba is not None:
        body_mat, _ = _find_body_material_in_collection(coll)
        if body_mat is None:
            body_mat = bpy.data.materials.get("CarPaint") or bpy.data.materials.get("Car Paint")
        if body_mat is not None:
            try:
                body_mat.use_nodes = True
                bsdf = next((n for n in body_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = rgba
            except Exception as e:
                print(f"  (body color warn: {e})")


def make_vehicle_instance(veh: dict, veh_manifest: dict, plates_dir: str,
                          anchor_loc, anchor_rot_z: float, is_in_camera: bool,
                          road_meta: dict):
    """Append + duplicate one vehicle at its env-JSON spawn anchor, assign
    plate+color. Returns (root_object, motion).

    ``anchor_loc`` / ``anchor_rot_z`` come from the required env file's
    ``lane_defaults[lane]`` (the frame-0 spawn pose for this camera view). The
    full kinematics motion is then keyframed on top so the vehicle still drives
    forward by speed/turn from that anchor (see keyframe_motion).

    ``road_meta`` (from road.json) is passed through so the motion plan drives
    the full road length (anchor → box edge for in, box edge → road end for out).
    """
    cls = veh["class"]
    meta = veh_manifest[cls]
    # Append the vehicle collection (local) so we can edit materials per-vehicle.
    # Use a unique collection name per vehicle so each has its own materials.
    veh_coll_name = f"VEH_{veh['id']}_{cls}"
    coll = append_collection_from_blend(
        os.path.join(HERE, "..", meta["blend"]),
        meta["collection"], new_name=veh_coll_name)

    # Assign plate + color to this vehicle's (now local) materials.
    assign_plate_and_color(coll, veh["plate"], plates_dir, rgba=veh.get("color"))

    approach = G.Direction(veh["approach"])
    # The motion is anchored at the env-JSON spawn point for THIS camera view:
    # appear_anchor for in-cameras, reappear_anchor for out-cameras. The other
    # segment uses the geometry default (not filmed on this camera).
    # road_meta lets compute_motion drive the full anchor → box edge / road end.
    ax, ay = anchor_loc[0], anchor_loc[1]
    anchor_xy = (ax, ay)
    motion = K.plan_motion(
        veh["id"], approach, veh["lane"], G.Turn(veh["turn"]),
        veh["speed_ms"], veh["depart_frame"], fps=FPS,
        appear_anchor=anchor_xy if is_in_camera else None,
        reappear_anchor=anchor_xy if not is_in_camera else None,
        road_meta=road_meta)
    fwd_off = meta.get("forward_offset_deg", 0.0)

    # Create the parent Empty at the WORLD ORIGIN with no rotation first, then
    # parent the meshes before moving it. This ensures matrix_parent_inverse is
    # identity so the meshes are rigidly attached to the empty's local frame and
    # will follow every location/rotation keyframe exactly.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    root = bpy.context.view_layer.objects.active
    root.name = f"VEH_{veh['id']}"
    root["forward_offset_deg"] = fwd_off
    root["anchor_rot_z"] = float(anchor_rot_z)

    # Parent meshes while empty is at origin (matrix_parent_inverse = identity).
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objs = [o for o in coll.objects if o.type == "MESH"]
    for o in mesh_objs:
        o.select_set(True)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    # Now place the empty at the spawn anchor and apply heading (anchor + fwd
    # offset). Meshes follow rigidly because matrix_parent_inverse is identity.
    rot = anchor_rot_z + math.radians(fwd_off)
    root.location = (ax, ay, 0.0)
    root.rotation_euler = (0.0, 0.0, rot)
    return root, motion


# ---------------------------------------------------------------------------
# Keyframing
# ---------------------------------------------------------------------------

def keyframe_motion(empty, motion: G.VehicleMotion, is_in_camera: bool):
    """Keyframe location across the visible segment, hide outside it.
    Rotation is also keyframed: the heading is the env-JSON anchor rotation
    (stashed on the empty by make_vehicle_instance) plus the per-class
    forward_offset_deg, so a sideways model stays corrected on both in and out
    shots. The rotation change (entry->exit heading) happens while hidden.

    The motion plan is already anchored at the env-JSON spawn point, so the
    keyframed locations are the motion positions directly (no extra offset).
    """
    obj = empty
    fwd_off = obj.get("forward_offset_deg", 0.0) if hasattr(obj, "get") else 0.0
    anchor_rot_z = float(obj.get("anchor_rot_z", 0.0)) if hasattr(obj, "get") else 0.0
    if is_in_camera:
        frame_start, frame_end = motion.appear_frame, motion.disappear_frame
        pos_start, pos_end = motion.appear_pos, motion.disappear_pos
    else:
        frame_start, frame_end = motion.reappear_frame, motion.leave_frame
        pos_start, pos_end = motion.reappear_pos, motion.leave_pos
    heading = anchor_rot_z + math.radians(fwd_off)

    obj.location = (pos_start[0], pos_start[1], 0.0)
    obj.keyframe_insert(data_path="location", frame=frame_start)
    obj.location = (pos_end[0], pos_end[1], 0.0)
    obj.keyframe_insert(data_path="location", frame=frame_end)

    # keyframe rotation (constant across the visible segment, incl. fwd offset)
    obj.rotation_euler.z = heading
    obj.keyframe_insert(data_path="rotation_euler", frame=frame_start)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame_end)

    _set_linear_interpolation(obj)

    # hide outside visible window (stepped interpolation for visibility)
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=0)
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_render", frame=max(0, frame_start))
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=frame_end + 1)
    _set_step_interpolation(obj, "hide_render")


def _set_linear_interpolation(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    a = ad.action
    try:
        if getattr(a, "is_action_layered", False) and a.layers and a.slots:
            strip = a.layers[0].strips[0]
            cb = strip.channelbag(a.slots[0])
            for fcu in cb.fcurves:
                for kp in fcu.keyframe_points:
                    kp.interpolation = "LINEAR"
        elif hasattr(a, "fcurves"):
            for fcu in a.fcurves:
                for kp in fcu.keyframe_points:
                    kp.interpolation = "LINEAR"
    except Exception:
        pass


def _set_step_interpolation(obj, data_path):
    """Set interpolation to CONSTANT (step) for a given data_path (for hide flags)."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    a = ad.action
    try:
        if getattr(a, "is_action_layered", False) and a.layers and a.slots:
            strip = a.layers[0].strips[0]
            cb = strip.channelbag(a.slots[0])
            for fcu in cb.fcurves:
                if fcu.data_path == data_path:
                    for kp in fcu.keyframe_points:
                        kp.interpolation = "CONSTANT"
        elif hasattr(a, "fcurves"):
            for fcu in a.fcurves:
                if fcu.data_path == data_path:
                    for kp in fcu.keyframe_points:
                        kp.interpolation = "CONSTANT"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Camera (telephoto, all cameras film the REAR plate)
# ---------------------------------------------------------------------------

def place_camera(approach: G.Direction, is_in: bool, road_meta: dict,
                 env_camera: dict = None):
    """Place a telephoto CCTV camera that films the REAR license plate.

    The camera sits at the NEAR end of the road arm (the end closest to the
    camera operator / far from the box for entry, at the box edge for exit),
    elevated CAM_HEIGHT metres, looking down the road in the direction cars
    drive away.  Cars always drive away from the camera → rear plate is filmed.

    The default camera/look-at positions come from G.camera_pose (single source
    of truth shared with the per-frame pose ground truth in render.compute_metadata).
    If ``env_camera`` (the env file's ``camera`` block) is given, its ``location``
    and ``look_at`` OVERRIDE the computed values (override layer). When
    ``rotation_euler`` is non-null in the env it is applied directly; otherwise
    the rotation is derived from look_at via the track-quat (same as default).
    """
    cam_loc, look_ground = G.camera_pose(approach, is_in, road_meta,
                                         cam_height=CAM_HEIGHT,
                                         cam_back_dist=CAM_BACK_DIST)
    if env_camera is not None:
        if env_camera.get("location") is not None:
            cam_loc = tuple(env_camera["location"])
        if env_camera.get("look_at") is not None:
            look_ground = tuple(env_camera["look_at"])
    label = "in" if is_in else "out"
    cam_data = bpy.data.cameras.new(f"CamData_{approach.value}_{label}")
    cam_data.lens = (env_camera or {}).get("lens_mm", LENS_MM)
    # Sensor fit must match the metadata convention (SENSOR_MM width, horizontal fit)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = (env_camera or {}).get("sensor_mm", G.SENSOR_MM)
    cam_obj = bpy.data.objects.new(f"Camera_{approach.value}_{label}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc

    rot_override = (env_camera or {}).get("rotation_euler")
    if rot_override is not None:
        cam_obj.rotation_euler = tuple(rot_override)
    else:
        direction = Vector(look_ground) - Vector(cam_loc)
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam_obj
    return cam_obj


# ---------------------------------------------------------------------------
# GPU configuration (Cycles + OPTIX, headless-safe)
# ---------------------------------------------------------------------------

CYCLES_SAMPLES = 48


def configure_gpu():
    """Enable Cycles GPU rendering via OPTIX. Hard-fails if no OPTIX device.

    Must be called after the Cycles addon is available (it is, by default).
    """
    import bpy
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        raise SystemExit("FAIL: Cycles addon not found — cannot use GPU rendering.")

    # Try OPTIX first (uses RT cores on the 3050 Ti), then CUDA as a fallback
    # within the GPU family; but per the chosen policy we HARD-FAIL if no
    # OPTIX device is available.
    prefs.compute_device_type = "OPTIX"
    try:
        prefs.refresh_devices()
    except Exception:
        pass

    optix_devs = [d for d in prefs.devices if d.type == "OPTIX"]
    if not optix_devs:
        raise SystemExit(
            "FAIL: no OPTIX GPU device found. An NVIDIA GPU with driver + "
            "OptiX support is required (headless Cycles cannot use EEVEE's "
            "GL path). Found devices: "
            + ", ".join(f"{d.name}({d.type})" for d in prefs.devices))

    # Enable only OPTIX devices; disable CPU to avoid VRAM contention on 4 GB.
    for d in prefs.devices:
        d.use = (d.type == "OPTIX")

    names = ", ".join(d.name for d in optix_devs if d.use)
    print(f"[GPU] Cycles OPTIX enabled: {names}")
    return optix_devs


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

def setup_render(env_lights: dict = None):
    """Configure Cycles + OPTIX GPU rendering at 1080p, 30 fps, PNG sequence.

    If ``env_lights`` (the env file's ``lights`` block) is given, the Sun
    light's rotation/energy are overridden from it.
    """
    scene = bpy.context.scene
    # Engine: Cycles (EEVEE cannot use the GPU in headless -b mode).
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = True
    # OptiX denoiser (uses RT cores); guard in case the enum populates late.
    try:
        scene.cycles.denoiser = "OPTIX"
    except Exception:
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
    # Resolution / fps / output format
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    # world light
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = world.node_tree.nodes.get("Background") if world.use_nodes else None
    if bg:
        bg.inputs["Color"].default_value = (0.5, 0.6, 0.8, 1.0)
        bg.inputs["Strength"].default_value = 2.0
    if not any(o.type == "LIGHT" and o.data.type == "SUN" for o in bpy.data.objects):
        sun = bpy.data.lights.new("Sun", type="SUN")
        sun_obj = bpy.data.objects.new("Sun", sun)
        bpy.context.scene.collection.objects.link(sun_obj)
        sun_env = (env_lights or {}).get("Sun", {})
        sun.energy = sun_env.get("energy", 4.0)
        rot_e = sun_env.get("rotation_euler", [math.radians(55), 0.0, math.radians(30)])
        sun_obj.rotation_euler = tuple(rot_e)


# ---------------------------------------------------------------------------
# Build one shot
# ---------------------------------------------------------------------------

def parse_camera_tag(tag: str):
    role, direction = tag.split("_")
    return G.Direction(direction), (role == "in")


def build_shot(scenario: dict, camera_tag: str, out_blend: str):
    bu.reset_scene()
    approach, is_in = parse_camera_tag(camera_tag)

    with open(ROAD_JSON) as f:
        road_meta = json.load(f)
    veh_manifest = load_vehicle_manifest()
    plates_dir = os.path.join(os.path.dirname(out_blend), "plates")
    os.makedirs(plates_dir, exist_ok=True)

    # Load the REQUIRED per-camera env file (hard-fails if missing/invalid).
    env = ENV.load_env(camera_tag, ROOT)

    # 1. Road. In-camera shows the entry carriageway of the approach; Out-camera
    #    shows the exit carriageway (outbound direction = approach for out_<D>).
    place_road(approach, road_meta, is_entry=is_in, env_road=env.get("road"))

    # 2. Vehicles visible on this camera. Each spawns at its env-JSON
    #    lane_defaults[lane] anchor, then drives forward by the kinematics.
    scene_objs = []
    motions = []
    for veh in scenario["vehicles"]:
        if is_in:
            if veh["approach"] != approach.value:
                continue
            anchor_lane = veh["lane"]
        else:
            ex_dir, ex_lane = G.exit_lane_for_movement(
                G.Direction(veh["approach"]), veh["lane"], G.Turn(veh["turn"]))
            if ex_dir != approach:
                continue
            anchor_lane = ex_lane
        anchor_loc, anchor_rot_z = ENV.lane_default_anchor(env, anchor_lane)
        empty, motion = make_vehicle_instance(
            veh, veh_manifest, plates_dir,
            anchor_loc=anchor_loc, anchor_rot_z=anchor_rot_z,
            is_in_camera=is_in, road_meta=road_meta)
        keyframe_motion(empty, motion, is_in_camera=is_in)
        scene_objs.append(empty)
        motions.append((veh, motion))

    # 3. Camera + render setup (GPU: Cycles + OPTIX)
    place_camera(approach, is_in, road_meta, env_camera=env.get("camera"))
    # Warn if the env camera diverges from the geometry-derived one — metadata
    # is always geometry-derived, so a drift means render != metadata.
    drift = ENV.camera_drift(env, scenario, camera_tag, road_meta)
    if drift:
        print(f"  [env] WARNING camera drift: {drift} — render will not match metadata.json")
    configure_gpu()
    setup_render(env_lights=env.get("lights"))
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = scenario["duration_frames"]

    # 4. Save
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"Built {camera_tag}: {len(scene_objs)} vehicles -> {out_blend}")
    return motions


def main():
    args = sys.argv
    if "--" not in args:
        raise SystemExit("Usage: blender -b --python build_scene.py -- --scenario X --camera in_N --out Y")
    post = args[args.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--camera", required=True, help="e.g. in_N or out_S")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(post)
    with open(ns.scenario) as f:
        scenario = json.load(f)
    build_shot(scenario, ns.camera, ns.out)


if __name__ == "__main__":
    main()
