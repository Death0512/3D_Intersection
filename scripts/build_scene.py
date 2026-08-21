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
from sim.trajectory import apply_samples_to_motion, load_trajectory_index
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
               env_road: dict = None, unified: bool = False):
    """Link the road arm and orient + position it for the given approach.

    Each road arm is a self-contained 14 m-wide linked asset. Metadata records
    both semantic road markings and mesh placement anchors:
        mesh_y_max  -> local +Y mesh head/visual edge
        crosswalk_y -> local crosswalk / stop-line position
        mesh_y_min  -> local -Y mesh tail edge

    Per-shot frame: road axis centred on (0,0,0); no carriageway lateral offset.
    Lane centrelines sit at x = LANE_CENTERLINES[k] in the arm's local frame,
    which aligns with world x after rotation by approach_rotation(approach).

    Entry road (is_entry=True):
        The arm's local +Y mesh head is placed at the box near-edge
        (−approach × BOX/2) and the mesh extends outward via local -Y.

    Exit road (is_entry=False):
        The arm's local +Y mesh head is also placed at the box
        far edge (+approach × BOX/2), so entry and exit road heads both point
        inward toward the blind zone. The mesh extends outward from there using
        local −Y. Vehicles on exit roads still travel outward along ``approach``;
        do not infer vehicle heading from the road instance's local +Y.

    If ``env_road`` (the env file's ``road`` block) is given, its location and
    rotation_euler OVERRIDE the computed transform (override layer).
    """
    coll = link_collection_from_blend(
        os.path.join(HERE, "..", road_meta["blend"]),
        road_meta["collection"])

    (ox, oy, oz), rot = G.road_arm_transform(
        approach, road_meta, is_entry=is_entry, unified=unified)
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


def assign_plate_and_color(coll, plate_str: str, plates_dir: str, rgba=None,
                           out_blend_dir: str | None = None):
    """Edit the (local/appended) collection's materials:
      * load the unique plate PNG into LicensePlate_Mat's Image Texture node
      * set the body-paint material base color to rgba (if given)

    When out_blend_dir is provided and plates_dir is inside it, store the image
    path relative to the generated .blend (e.g. ``//plates/ABC.png``) so CPU1
    artifacts can be copied to a VM without baking the source machine's absolute
    output path into the scene.
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
            if out_blend_dir is not None:
                try:
                    rel = os.path.relpath(plate_path, os.path.abspath(out_blend_dir))
                    img.filepath = "//" + rel.replace(os.sep, "/")
                except Exception:
                    pass
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
                          road_meta: dict,
                          trajectory_samples=None,
                          out_blend_dir: str | None = None):
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

    # The appended assets may carry absolute texture filepaths baked in on a
    # different host (e.g. the dev machine). Remap every image-block to the
    # project-local textures dir so rendering works on any checkout (Kaggle,
    # CI, another machine) without re-running asset_prep.
    tex_dir = os.path.join(HERE, "..", "models", cls, "textures")
    if os.path.isdir(tex_dir):
        _remapped = bu.remap_textures_to_local(tex_dir,
                                               relative_to_dir=out_blend_dir)
        if _remapped:
            print(f"  [tex] {veh['id']}: remapped {_remapped} textures to {tex_dir}")

    # Assign plate + color to this vehicle's (now local) materials. Plate/
    # color failures are data-quality (vehicle renders without a plate / with
    # default color), NOT structural — so catch and warn here rather than
    # aborting the whole vehicle. assign_plate_and_color already swallows
    # material-assignment errors internally; this outer guard covers the
    # render_plate() call (line 201) which isn't in that internal try/except.
    try:
        assign_plate_and_color(coll, veh["plate"], plates_dir,
                               rgba=veh.get("color"),
                               out_blend_dir=out_blend_dir)
    except Exception as e:
        print(f"  [{veh['id']}] [WARN] plate/color generation failed: {e} "
              f"— vehicle will render without a plate", flush=True)

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
        road_meta=road_meta,
        stop_frame=veh.get("stop_frame"),
        release_frame=veh.get("release_frame"),
        queue_slot=veh.get("queue_slot", -1))
    if is_in_camera and trajectory_samples:
        motion = apply_samples_to_motion(
            motion, trajectory_samples, anchor_xy,
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

def keyframe_motion(empty, motion: G.VehicleMotion, is_in_camera: bool,
                   frame_end: int = None):
    """Keyframe location across the visible segment using the motion track.

    Each point in the track produces a keyframe. LINEAR interpolation for
    straight segments; BEZIER with explicit handles for turning segments so
    the easing matches metadata exactly (per-axis fcurves in the slotted-action
    layout of Blender 5.x).

    The motion track is already anchored at the env-JSON spawn point.

    If ``frame_end`` is given and the segment ends after it (vehicle cannot
    complete before the window), an extra ``hide_render=True`` keyframe is
    inserted at ``frame_end + 1`` so the vehicle remains visible through the
    final rendered frame, then vanishes instead of freezing
    mid-road.
    """
    obj = empty
    fwd_off = obj.get("forward_offset_deg", 0.0) if hasattr(obj, "get") else 0.0
    anchor_rot_z = float(obj.get("anchor_rot_z", 0.0)) if hasattr(obj, "get") else 0.0
    track = motion.track_in if is_in_camera else motion.track_out

    # Pass 1 — insert all location keyframes at LINEAR (per-axis post-pass
    # upgrades BEZIER segments so handles are set per fcurve axis).
    for i, pt in enumerate(track):
        obj.location = (pt.x, pt.y, 0.0)
        obj.keyframe_insert(data_path="location", frame=pt.frame)

    # Gather which keyframe frames belong to BEZIER segments.  Everything else
    # stays LINEAR (keyframe_insert defaults to BEZIER auto-handles — we force
    # LINEAR on non-BEZIER keyframes next, then override BEZIER per axis).
    coeff_per_segment = {}        # (frame_i, frame_i+1) -> (cp1[axis], cp2[axis])
    bez_frames = set()
    for i, pt in enumerate(track):
        if i < len(track) - 1 and pt.interp == "BEZIER" and pt.cp1 is not None:
            nxt = track[i + 1]
            dt = nxt.frame - pt.frame
            rh_frame = pt.frame + dt / 3.0
            lh_frame = nxt.frame - dt / 3.0
            coeff_per_segment[(pt.frame, nxt.frame)] = (pt.cp1, pt.cp2, rh_frame, lh_frame)
            bez_frames.add(pt.frame)
            bez_frames.add(nxt.frame)

    loc_fcs = _fcurves_for(obj, "location")
    # Force LINEAR on all keyframes first, then selectively override BEZIER ones.
    for fc in loc_fcs:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    # Per-axis BEZIER override: set handles so Blender evaluates the cubic
    # Bézier parametrised linearly in frame (frame handles at ±dt/3 and raw
    # cp1/cp2 coordinate values).
    for fc in loc_fcs:
        axis = fc.array_index          # 0 = x,  1 = y,  2 = z
        for kp in fc.keyframe_points:
            f = kp.co[0]
            for (f0, f1), (cp1, cp2, rh_f, lh_f) in coeff_per_segment.items():
                if f == f0:
                    kp.interpolation = "BEZIER"
                    kp.handle_right_type = "FREE"
                    kp.handle_left_type = "FREE"
                    cv = cp1[axis] if axis < 2 else 0.0
                    kp.handle_right = (rh_f, cv)
                elif f == f1:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = "FREE"
                    kp.handle_right_type = "AUTO_CLAMPED"
                    cv = cp2[axis] if axis < 2 else 0.0
                    kp.handle_left = (lh_f, cv)

    # rotation keyframes (constant across the visible segment, incl. fwd offset)
    heading = anchor_rot_z + math.radians(fwd_off)
    start_frame = track[0].frame
    end_frame = track[-1].frame
    obj.rotation_euler.z = heading
    obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)

    # hide outside visible window (stepped interpolation for visibility).
    # Applied to the Empty AND every descendant mesh: `hide_render` is
    # per-object in Blender and does NOT propagate parent→child, and an Empty
    # renders nothing anyway — so keyframing only the Empty left the child
    # meshes visible for the whole timeline, frozen at leave_pos after
    # leave_frame (and at the spawn anchor before appear_frame). Mirroring the
    # keyframes onto every descendant fixes both tails and restores
    # render==metadata (compute_metadata emits frames only in
    # [appear,disappear] ∪ [reappear,leave]). `hide_viewport` is set in
    # parallel so the saved .blend is also correct when opened interactively.
    vis_start = max(0, start_frame)
    hide_end = end_frame + 1
    if frame_end is not None and end_frame > frame_end:
        hide_end = frame_end + 1

    def _keyframe_visibility(o):
        initial_hidden = vis_start > 0
        o.hide_render = initial_hidden
        o.hide_viewport = initial_hidden
        o.keyframe_insert(data_path="hide_render", frame=0)
        o.keyframe_insert(data_path="hide_viewport", frame=0)
        o.hide_render = False
        o.hide_viewport = False
        if vis_start > 0:
            o.keyframe_insert(data_path="hide_render", frame=vis_start)
            o.keyframe_insert(data_path="hide_viewport", frame=vis_start)
        o.hide_render = True
        o.hide_viewport = True
        o.keyframe_insert(data_path="hide_render", frame=hide_end)
        o.keyframe_insert(data_path="hide_viewport", frame=hide_end)
        _set_step_interpolation(o, "hide_render")
        _set_step_interpolation(o, "hide_viewport")

    _keyframe_visibility(obj)
    for child in obj.children_recursive:
        _keyframe_visibility(child)


def _action_fcurves(obj):
    """Return all ActionFCurves from the object's active slotted action.

    Blender 5.x: actions are slot-based — fcurves live in the first layer/strip
    channelbag keyed to slot 0. No legacy fallback (``Action.fcurves`` removed).
    Returns an empty list if no animation is assigned.

    M14: wraps the layered-action traversal in a clearer error so a Blender
    API change (e.g. a new sub-version renames ``channelbag`` or reorders
    ``layers``/``slots``) surfaces a diagnosable message instead of a bare
    ``AttributeError: 'NoneType' object has no attribute 'fcurves'`` mid-render.
    """
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    a = ad.action
    if not (a.layers and a.slots):
        return []
    try:
        strip = a.layers[0].strips[0]
        cb = strip.channelbag(a.slots[0])
        return list(cb.fcurves) if cb else []
    except Exception as e:
        name = getattr(obj, "name", "<unknown>")
        raise RuntimeError(
            f"[scene] keyframe API error for object {name!r} "
            f"(action={getattr(a, 'name', '?')!r}): {type(e).__name__}: {e}. "
            f"This usually means the Blender 5.x slotted-action API changed — "
            f"check action.layers[0].strips[0].channelbag(slots[0]).fcurves."
        ) from e


def _fcurves_for(obj, data_path):
    """Return the ActionFCurves matching *data_path* on *obj*."""
    return [fc for fc in _action_fcurves(obj) if fc.data_path == data_path]


def _set_linear_interpolation(obj):
    """Set all keyframes on *obj* to LINEAR interpolation."""
    for fc in _action_fcurves(obj):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _set_step_interpolation(obj, data_path):
    """Set all keyframes of *data_path* on *obj* to CONSTANT (step)."""
    for fc in _fcurves_for(obj, data_path):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


# ---------------------------------------------------------------------------
# Camera (telephoto, all cameras film the REAR plate)
# ---------------------------------------------------------------------------

def place_camera(approach: G.Direction, is_in: bool, road_meta: dict,
                 env_camera: dict = None, env: dict = None,
                 unified: bool = False):
    """Place a telephoto CCTV camera that films the REAR license plate.

    The camera sits at the NEAR end of the road arm (the end closest to the
    camera operator / far from the box for entry, at the box edge for exit),
    elevated CAM_HEIGHT metres, looking down the road in the direction cars
    drive away.  Cars always drive away from the camera → rear plate is filmed.

    The resolved camera spec comes from ``ENV.resolve_camera`` (single source
    of truth shared with the per-frame pose ground truth in render.compute_metadata):
    env ``camera`` overrides the geometry default if present.  When
    ``rotation_euler`` is non-null in the env it is applied directly; otherwise
    the rotation is derived from look_at via the track-quat (same as before).

    Accepts either the legacy ``env_camera`` (the env ``camera`` block alone,
    kept for backward compat with direct callers) or the full ``env`` dict
    (preferred — lets us reuse the resolver verbatim).
    """
    # Build the resolved spec via the shared resolver. We accept a minimal env
    # constructed from env_camera for legacy callers, or a full env dict.
    if env is None:
        tag = f"{'in' if is_in else 'out'}_{approach.value}"
        env = {"camera_tag": tag, "camera": env_camera or {}}
    resolved = ENV.resolve_camera(env, road_meta, unified=unified)
    cam_loc = tuple(resolved["location"])
    look_ground = tuple(resolved["look_at"])
    lens_mm = resolved["lens_mm"]
    sensor_mm = resolved["sensor_mm"]
    rot_override = resolved["rotation_euler"]

    label = "in" if is_in else "out"
    cam_data = bpy.data.cameras.new(f"CamData_{approach.value}_{label}")
    cam_data.lens = lens_mm
    # Sensor fit must match the metadata convention (SENSOR_MM width, horizontal fit)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = sensor_mm
    cam_obj = bpy.data.objects.new(f"Camera_{approach.value}_{label}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc

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

CYCLES_SAMPLES = 24


# Track whether configure_gpu has already refreshed devices in this Blender
# process. refresh_devices() is an expensive (potentially multi-second) NVIDIA
# driver round-trip; render_one calls configure_gpu twice (once in build_shot
# and once after the .blend reload). The second call's refresh is redundant —
# the device list doesn't change within a process — so skip it.
_GPU_CONFIGURED = False


def configure_gpu():
    """Enable Cycles GPU rendering: OPTIX if available, else CUDA.

    Tries OPTIX first (uses RT cores where present), then gracefully degrades
    to CUDA. This keeps the pipeline working on GPUs / containers that expose
    no OptiX device — e.g. a Kaggle Tesla P100 (Pascal, no RT cores) or a T4
    whose headless container only surfaces the CUDA backend. Only hard-fails
    when NEITHER OPTIX nor CUDA devices exist.

    Must be called after the Cycles addon is available (it is, by default).
    Idempotent: the second call in a Blender process reuses the device list
    rather than re-querying the driver (the `prefs.refresh_devices()` driver
    probe is the most likely place for a multi-process GPU init deadlock on
    headless containers, so we want it run once per process, not twice).
    """
    import bpy
    global _GPU_CONFIGURED
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        raise SystemExit("FAIL: Cycles addon not found — cannot use GPU rendering.")

    if not _GPU_CONFIGURED:
        print("[GPU] refreshing devices...", flush=True)
        prefs.refresh_devices()
        # D10: echo what Blender saw — a silent "no devices" leaves the user
        # with an unhelpful "Found devices: " message on the next branch.
        names = [f"{d.name}({d.type})" for d in prefs.devices]
        print(f"[GPU] {len(prefs.devices)} device(s): {', '.join(names)}",
              flush=True)
        _GPU_CONFIGURED = True

    optix_devs = [d for d in prefs.devices if d.type == "OPTIX"]
    if optix_devs:
        backend = "OPTIX"
    else:
        cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
        if not cuda_devs:
            raise SystemExit(
                "FAIL: no OPTIX or CUDA GPU device found. An NVIDIA GPU with "
                "driver + CUDA (OptiX preferred) is required. Found devices: "
                + ", ".join(f"{d.name}({d.type})" for d in prefs.devices))
        backend = "CUDA"

    prefs.compute_device_type = backend
    # Enable only the chosen backend's devices; disable CPU to avoid VRAM
    # contention and keep the whole render on the GPU.
    for d in prefs.devices:
        d.use = (d.type == backend)

    gpu_devs = [d for d in prefs.devices if d.type == backend]
    names = ", ".join(d.name for d in gpu_devs if d.use)
    print(f"[GPU] Cycles {backend} enabled: {names}", flush=True)
    return gpu_devs


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

def setup_render(env_lights: dict = None, samples: int = None):
    """Configure Cycles GPU rendering at 1080p, 30 fps, JPEG sequence.

    Cycles denoising is GPU-only: use the probe denoiser when Blender
    selected the back end and the NVIDIA OptiX weights file exists. On
    CUDA-only devices denoising is disabled instead of falling back to
    OpenImageDenoise.

    ``samples`` (optional int) overrides the module-level CYCLES_SAMPLES so
    callers (render.py --samples) can trade quality for speed without editing
    this file.  None/0 keeps the default (24).
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    # M9: honor --samples override (None → default 24). Lower sample counts
    # (16-24) let users iterate faster; denoising remains GPU-only below.
    scene.cycles.samples = samples if samples else CYCLES_SAMPLES
    scene.cycles.use_denoising = False
    # GPU-only denoise policy. Do NOT fall back to OpenImageDenoise: on Kaggle
    # GPU may run on CPU and dominate render time. If OptiX denoise is not
    # definitely available, render without denoising and let sample count/scene
    # settings control quality/performance.
    backend = bpy.context.preferences.addons["cycles"].preferences.compute_device_type
    optix_weights_ok = os.path.exists("/usr/share/nvidia/nvoptix.bin")
    if backend == "OPTIX" and optix_weights_ok:
        try:
            scene.cycles.denoiser = "OPTIX"
            actual = str(scene.cycles.denoiser)
            if actual == "OPTIX":
                scene.cycles.use_denoising = True
                print("  [denoiser] OPTIX active", flush=True)
            else:
                print(f"  [denoiser] disabled: wanted OPTIX but got {actual}",
                      flush=True)
        except Exception:
            print("  [denoiser] disabled: OPTIX denoiser assignment failed",
                  flush=True)
    else:
        reason = "CUDA backend" if backend != "OPTIX" else "missing nvoptix.bin"
        print(f"  [denoiser] disabled ({reason}; no CPU/OIDN fallback)",
              flush=True)
    # Resolution / fps / output format
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95
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
    frame_end = max(0, scenario["duration_frames"] - 1)
    scenario_dir = scenario.get("_base_dir", os.path.dirname(os.path.abspath(out_blend)))
    traj_index = load_trajectory_index(scenario, scenario_dir)
    n_visible_candidates = 0
    print(f"  [build] placing vehicles for {camera_tag} "
          f"(scanning {len(scenario['vehicles'])} total)...", flush=True)
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
        n_visible_candidates += 1
        # D2: progress every 5th vehicle so a 100-vehicle scenario doesn't
        # look frozen during the place+keyframe loop. Report id/approach/
        # lane/turn so a stuck vehicle is identifiable in the log.
        if n_visible_candidates % 5 == 0 or n_visible_candidates == 1:
            print(f"    [{camera_tag}] V{n_visible_candidates}: "
                  f"{veh['id']} {veh['approach']}/{veh.get('lane')}/"
                  f"{veh.get('turn')}", flush=True)
        anchor_loc, anchor_rot_z = ENV.lane_default_anchor(env, anchor_lane)
        # Structural failures (missing .blend, plan_motion error, bpy parenting
        # error) MUST propagate — a silently-dropped vehicle is a correctness
        # bug, not data quality. Plate/color failures are caught inside
        # make_vehicle_instance (data-quality: vehicle renders without plate).
        empty, motion = make_vehicle_instance(
            veh, veh_manifest, plates_dir,
            anchor_loc=anchor_loc, anchor_rot_z=anchor_rot_z,
            is_in_camera=is_in, road_meta=road_meta,
            trajectory_samples=traj_index.get(veh["id"]),
            out_blend_dir=os.path.dirname(out_blend))
        keyframe_motion(empty, motion, is_in_camera=is_in, frame_end=frame_end)
        scene_objs.append(empty)
        motions.append((veh, motion))

    # 3. Camera + render setup (GPU: Cycles + OPTIX)
    place_camera(approach, is_in, road_meta, env=env)
    # Note when the env camera overrides the geometry default — this is the
    # intended override layer for framing, NOT an inconsistency: metadata.json
    # records the SAME resolved camera (envfile.resolve_camera), so render and
    # metadata agree by construction. The note is informational only.
    drift = ENV.camera_drift(env, scenario, camera_tag, road_meta)
    if drift:
        print(f"  [env] camera override active for {camera_tag} ({drift})")
    configure_gpu()
    setup_render(env_lights=env.get("lights"))
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, scenario["duration_frames"] - 1)

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
    scenario["_base_dir"] = os.path.dirname(os.path.abspath(ns.scenario))
    build_shot(scenario, ns.camera, ns.out)


if __name__ == "__main__":
    main()
