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
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, HERE)  # for gen_plate

import bpy
from mathutils import Vector, Matrix

import blender_utils as bu
import geometry as G
import kinematics as K
from gen_plate import render_plate


ASSETS_DIR = os.path.join(HERE, "..", "assets")
VEHICLES_JSON = os.path.join(ASSETS_DIR, "vehicles.json")
ROAD_JSON = os.path.join(ASSETS_DIR, "road.json")

LENS_MM = 60.0          # telephoto
RES_X = 1920
RES_Y = 1080
FPS = 30
CAM_HEIGHT = 7.0        # camera elevation (m)
CAM_BACK_DIST = 45.0    # how far behind the stop line the camera sits
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

def place_road(approach: G.Direction, road_meta: dict, is_entry: bool = True):
    """Link the road arm and orient + position it for the given approach.

    Each road arm is a self-contained 14 m-wide, ~54.75 m-long asset whose
    local +Y end carries the crosswalk / stop-line (crosswalk_y ≈ +27.85) and
    whose local −Y end is the far/back end (≈ −26.9).

    Entry road (is_entry=True):
        Cars drive TOWARD the box (+approach forward).  We orient the arm so
        its local +Y (crosswalk) sits exactly at the box near-edge, extending
        outward.  Arm +Y → approach forward; arm origin = near-edge − forward
        × crosswalk_y (no lateral offset — arm is centred on the branch axis).

    Exit road (is_entry=False):
        Cars emerge from the box and drive AWAY (outbound = approach direction
        for out_<D> shots — same direction label).  The crosswalk must sit at
        the box edge (the camera end), and the car drives outward.  We orient
        the arm so its local +Y (crosswalk) also sits at the box near-edge but
        points TOWARD the box (= −approach forward), so the arm extends outward
        in the +approach direction.  Rotation = approach_rotation(approach) + π.
        Cars travel in −local-Y of the arm = +approach forward (away from box).
    """
    coll = link_collection_from_blend(
        os.path.join(HERE, "..", road_meta["blend"]),
        road_meta["collection"])

    fx, fy = approach.vec
    crosswalk_y = road_meta.get("crosswalk_y", 0.0)
    # box near-edge for this approach (world coords, ground)
    near_edge = (-fx * BOX_HALF, -fy * BOX_HALF)

    if is_entry:
        # arm +Y → approach forward; crosswalk at box near-edge
        rot = G.approach_rotation(approach)
        origin_world = (near_edge[0] - fx * crosswalk_y,
                        near_edge[1] - fy * crosswalk_y)
    else:
        # arm +Y → toward box (−approach); crosswalk still at box near-edge
        rot = G.approach_rotation(approach) + math.pi
        origin_world = (near_edge[0] + fx * crosswalk_y,
                        near_edge[1] + fy * crosswalk_y)

    label = "in" if is_entry else "out"
    empty = instantiate_linked_collection(coll, f"Road_{approach.value}_{label}",
                                          (origin_world[0], origin_world[1], 0.0),
                                          rotation_z=rot)
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
            if any(c in n for c in ("carpaint", "car paint", "bodycolour",
                                     "paint2", "van_paint", "trailer paint")):
                return slot.material, obj
    return None, None


def assign_plate_and_color(coll, plate_str: str, plates_dir: str, rgba=None):
    """Edit the (local/appended) collection's materials:
      * load the unique plate PNG into LicensePlate_Mat's Image Texture node
      * set the body-paint material base color to rgba (if given)
    """
    safe = "".join(c if c.isalnum() else "_" for c in plate_str) + ".png"
    plate_path = os.path.join(plates_dir, safe)
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


def make_vehicle_instance(veh: dict, veh_manifest: dict, plates_dir: str):
    """Append + duplicate one vehicle at its appear position, assign plate+color.
    Returns (root_object, motion). The root object is a parent Empty whose
    children are the duplicated vehicle meshes (so we can animate the Empty)."""
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
    motion = K.plan_motion(
        veh["id"], approach, veh["lane"], G.Turn(veh["turn"]),
        veh["speed_ms"], veh["depart_frame"], fps=FPS)
    ax, ay = motion.appear_pos
    rot = G.approach_rotation(approach)

    # Create a parent Empty and parent all the collection's meshes to it,
    # then move the Empty to the appear position. This gives us a single
    # transform to keyframe.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(ax, ay, 0.0))
    root = bpy.context.view_layer.objects.active
    root.name = f"VEH_{veh['id']}"
    # Rotation: use the exit direction's heading so the vehicle faces the
    # correct way on both entry and exit segments. For straight movements
    # entry and exit headings are the same; for turns the vehicle is
    # invisible inside the Black Box so the rotation change isn't seen.
    root.rotation_euler = (0.0, 0.0, rot)
    # parent each mesh to root (keep transform)
    mesh_objs = [o for o in coll.objects if o.type == "MESH"]
    for o in mesh_objs:
        o.select_set(True)
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    return root, motion


# ---------------------------------------------------------------------------
# Keyframing
# ---------------------------------------------------------------------------

def keyframe_motion(empty, motion: G.VehicleMotion, is_in_camera: bool):
    """Keyframe location across the visible segment, hide outside it.
    Rotation is also keyframed: on entry use the approach heading, on exit
    use the exit heading. The rotation change happens while hidden."""
    obj = empty
    if is_in_camera:
        frame_start, frame_end = motion.appear_frame, motion.disappear_frame
        pos_start, pos_end = motion.appear_pos, motion.disappear_pos
        heading = G.approach_rotation(motion.approach)
    else:
        frame_start, frame_end = motion.reappear_frame, motion.leave_frame
        pos_start, pos_end = motion.reappear_pos, motion.leave_pos
        heading = G.approach_rotation(motion.exit_direction)

    obj.location = (pos_start[0], pos_start[1], 0.0)
    obj.keyframe_insert(data_path="location", frame=frame_start)
    obj.location = (pos_end[0], pos_end[1], 0.0)
    obj.keyframe_insert(data_path="location", frame=frame_end)

    # keyframe rotation (constant across the visible segment)
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

def place_camera(approach: G.Direction, is_in: bool, road_meta: dict):
    """Place a telephoto CCTV camera that films the REAR license plate.

    The camera sits at the NEAR end of the road arm (the end closest to the
    camera operator / far from the box for entry, at the box edge for exit),
    elevated CAM_HEIGHT metres, looking down the road in the direction cars
    drive away.  Cars always drive away from the camera → rear plate is filmed.

    Entry (in_<D>):
        Road arm extends from box near-edge outward ~54.75 m.  Camera at the
        outer/back end of the arm, looking TOWARD the box (+approach forward).
        Cars appear near the camera and drive toward the stop line (top of frame).

    Exit (out_<D>):
        Road arm extends from box near-edge outward.  Camera at the box-edge/
        crosswalk end, looking OUTWARD (+approach forward, away from box).
        Cars emerge from the box just ahead of the camera and drive away.

    No lateral offset — each arm is centred on its own branch axis.
    """
    fx, fy = approach.vec
    arm_back = road_meta.get("approach_length", 54.75) - road_meta.get("crosswalk_y", 27.85)
    # back-end local Y in arm coords ≈ −(approach_length − crosswalk_y) ≈ −26.9

    if is_in:
        # Camera at outer (back) end of entry arm, looking toward box.
        # Arm origin placed so crosswalk is at box near-edge; back end is
        # crosswalk_y + arm_back further out (away from box).
        dist_from_box = road_meta.get("crosswalk_y", 27.85) + arm_back
        cam_ground = (-fx * (BOX_HALF + dist_from_box),
                      -fy * (BOX_HALF + dist_from_box))
        # look toward box near-edge (crosswalk), slightly past it
        look_ground = (-fx * (BOX_HALF - 2.0),
                       -fy * (BOX_HALF - 2.0), 0.0)
    else:
        # Camera at box-edge/crosswalk end of exit arm, looking outward.
        cam_ground = (-fx * BOX_HALF,
                      -fy * BOX_HALF)
        look_ground = (-fx * BOX_HALF + fx * CAM_BACK_DIST,
                       -fy * BOX_HALF + fy * CAM_BACK_DIST, 0.0)

    cam_loc = (cam_ground[0], cam_ground[1], CAM_HEIGHT)
    label = "in" if is_in else "out"
    cam_data = bpy.data.cameras.new(f"CamData_{approach.value}_{label}")
    cam_data.lens = LENS_MM
    cam_obj = bpy.data.objects.new(f"Camera_{approach.value}_{label}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc

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

def setup_render():
    """Configure Cycles + OPTIX GPU rendering at 1080p, 30 fps, PNG sequence."""
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
        sun.energy = 4.0
        sun_obj = bpy.data.objects.new("Sun", sun)
        bpy.context.scene.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(55), 0.0, math.radians(30))


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

    # 1. Road. In-camera shows the entry carriageway of the approach; Out-camera
    #    shows the exit carriageway (outbound direction = approach for out_<D>).
    place_road(approach, road_meta, is_entry=is_in)

    # 2. Vehicles visible on this camera.
    scene_objs = []
    motions = []
    for veh in scenario["vehicles"]:
        if is_in:
            if veh["approach"] != approach.value:
                continue
        else:
            ex_dir, _ = G.exit_lane_for_movement(
                G.Direction(veh["approach"]), veh["lane"], G.Turn(veh["turn"]))
            if ex_dir != approach:
                continue
        empty, motion = make_vehicle_instance(veh, veh_manifest, plates_dir)
        keyframe_motion(empty, motion, is_in_camera=is_in)
        scene_objs.append(empty)
        motions.append((veh, motion))

    # 3. Camera + render setup (GPU: Cycles + OPTIX)
    place_camera(approach, is_in, road_meta)
    configure_gpu()
    setup_render()
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
