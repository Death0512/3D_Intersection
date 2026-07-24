#!/usr/bin/env python3
"""Blender-side helper: render one camera from unified_scene.blend to JPEGs."""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

import bpy

import build_scene as BS


_VEHICLE_ALLOWED_ANIM_PATHS = {
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "scale",
    "hide_render",
    "hide_viewport",
}

_FRAME_DEPENDENT_MODIFIERS = {
    "BUILD",
    "CLOTH",
    "DYNAMIC_PAINT",
    "FLUID",
    "MESH_SEQUENCE_CACHE",
    "NODES",
    "OCEAN",
    "PARTICLE_SYSTEM",
    "SOFT_BODY",
    "WAVE",
}


def _frame_path(frames_dir: str, index: int) -> str:
    return os.path.join(frames_dir, f"f_{index:04d}.jpg")


def _verify_jpeg(path: str) -> None:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError(f"render output missing/empty: {path}")


def _has_action_or_drivers(id_data) -> bool:
    ad = getattr(id_data, "animation_data", None)
    if ad is None:
        return False
    if getattr(ad, "action", None) is not None:
        return True
    if getattr(ad, "drivers", None):
        return True
    return bool(getattr(ad, "nla_tracks", None))


def _action_fcurves(action, owner_name: str = "<unknown>"):
    if action is None:
        return []
    fcurves = []
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        fcurves.extend(legacy)
    layers = getattr(action, "layers", None)
    slots = getattr(action, "slots", None)
    if not layers or not slots:
        return fcurves
    try:
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                for slot in slots:
                    cb = strip.channelbag(slot)
                    if cb:
                        fcurves.extend(cb.fcurves)
    except Exception as e:
        raise RuntimeError(
            f"[unified] keyframe API error for {owner_name!r} "
            f"(action={getattr(action, 'name', '?')!r}): {type(e).__name__}: {e}. "
            "This usually means the Blender slotted-action API changed."
        ) from e
    return fcurves


def _animation_paths(id_data) -> list[str]:
    ad = getattr(id_data, "animation_data", None)
    paths = []
    if ad is None:
        return paths
    action = getattr(ad, "action", None)
    if action is not None:
        paths.extend(fc.data_path for fc in _action_fcurves(action, getattr(id_data, "name", "<unknown>")))
    paths.extend(fc.data_path for fc in getattr(ad, "drivers", []) or [])
    for track in getattr(ad, "nla_tracks", []) or []:
        for strip in getattr(track, "strips", []) or []:
            action = getattr(strip, "action", None)
            if action is not None:
                paths.extend(fc.data_path for fc in _action_fcurves(action, getattr(id_data, "name", "<unknown>")))
    return paths


def _is_vehicle_object(obj) -> bool:
    if obj.name.startswith("VEH_"):
        return True
    parent = obj.parent
    while parent is not None:
        if parent.name.startswith("VEH_"):
            return True
        parent = parent.parent
    return False


def _vehicle_roots():
    roots = []
    for obj in bpy.data.objects:
        if not obj.name.startswith("VEH_"):
            continue
        if obj.parent is not None and obj.parent.name.startswith("VEH_"):
            continue
        roots.append(obj)
    return sorted(roots, key=lambda o: o.name)


def _vehicle_signature_objects():
    objs = []
    seen = set()
    for root in _vehicle_roots():
        for obj in [root] + list(root.children_recursive):
            if obj.name in seen:
                continue
            seen.add(obj.name)
            objs.append(obj)
    return sorted(objs, key=lambda o: o.name)


def _matrix_tuple(obj):
    return tuple(round(float(v), 6) for row in obj.matrix_world for v in row)


def _effectively_render_visible(obj) -> bool:
    if obj.hide_render:
        return False
    parent = obj.parent
    while parent is not None:
        if parent.hide_render:
            return False
        parent = parent.parent
    return True


def _frame_signature():
    sig = []
    for obj in _vehicle_signature_objects():
        visible = _effectively_render_visible(obj)
        sig.append((obj.name, obj.type, visible, bool(obj.hide_render), _matrix_tuple(obj)))
    return tuple(sig)


def _unsupported_reuse_reason(scene) -> str | None:
    if getattr(scene.render, "use_motion_blur", False):
        return "motion blur enabled"
    cycles = getattr(scene, "cycles", None)
    if cycles is not None and getattr(cycles, "use_animated_seed", False):
        return "Cycles animated seed enabled"

    cam = scene.camera
    if cam is None:
        return "no active camera"
    if cam.parent is not None or len(getattr(cam, "constraints", []) or []) > 0:
        return "active camera has parent/constraints"
    if _has_action_or_drivers(cam):
        return "active camera has animation/drivers"
    if getattr(cam, "data", None) is not None and _has_action_or_drivers(cam.data):
        return "active camera data has animation/drivers"

    for obj in bpy.data.objects:
        is_vehicle = _is_vehicle_object(obj)
        if obj.type == "LIGHT":
            if obj.parent is not None or len(getattr(obj, "constraints", []) or []) > 0:
                return f"light {obj.name} has parent/constraints"
            if _has_action_or_drivers(obj):
                return f"light {obj.name} has animation/drivers"
            if getattr(obj, "data", None) is not None and _has_action_or_drivers(obj.data):
                return f"light data {obj.name} has animation/drivers"
        if is_vehicle:
            bad = [p for p in _animation_paths(obj) if p not in _VEHICLE_ALLOWED_ANIM_PATHS]
            if bad:
                return f"vehicle {obj.name} has unsupported animation path {bad[0]}"
        elif _has_action_or_drivers(obj):
            return f"non-vehicle object {obj.name} has animation/drivers"

        data = getattr(obj, "data", None)
        if data is not None:
            if _has_action_or_drivers(data):
                return f"object data {data.name} has animation/drivers"
            shape_keys = getattr(data, "shape_keys", None)
            if shape_keys is not None and _has_action_or_drivers(shape_keys):
                return f"shape keys {shape_keys.name} have animation/drivers"

        for mod in getattr(obj, "modifiers", []) or []:
            if _has_action_or_drivers(mod):
                return f"modifier {obj.name}.{mod.name} has animation/drivers"
            if mod.type in _FRAME_DEPENDENT_MODIFIERS:
                return f"modifier {obj.name}.{mod.name} may be frame-dependent"

    for mat in bpy.data.materials:
        if _has_action_or_drivers(mat):
            return f"material {mat.name} has animation/drivers"
        nt = getattr(mat, "node_tree", None)
        if nt is not None and _has_action_or_drivers(nt):
            return f"material node tree {mat.name} has animation/drivers"

    for node_group in bpy.data.node_groups:
        if _has_action_or_drivers(node_group):
            return f"node group {node_group.name} has animation/drivers"

    world = scene.world
    if world is not None:
        if _has_action_or_drivers(world):
            return "world has animation/drivers"
        nt = getattr(world, "node_tree", None)
        if nt is not None and _has_action_or_drivers(nt):
            return "world node tree has animation/drivers"

    if _has_action_or_drivers(scene):
        return "scene has animation/drivers"
    return None


def _render_frame(scene, frame: int, out_path: str) -> None:
    scene.frame_set(frame)
    scene.render.filepath = os.path.splitext(out_path)[0]
    bpy.ops.render.render(write_still=True)
    _verify_jpeg(out_path)


def _render_manual(scene, frames_dir: str, reuse: bool) -> tuple[int, int]:
    rendered = copied = 0
    prev_sig = None
    prev_path = None
    total = scene.frame_end - scene.frame_start + 1
    for index in range(total):
        frame = scene.frame_start + index
        out_path = _frame_path(frames_dir, index)
        scene.frame_set(frame)
        sig = _frame_signature() if reuse else None
        if reuse and prev_sig == sig and prev_path is not None:
            shutil.copyfile(prev_path, out_path)
            _verify_jpeg(out_path)
            copied += 1
            continue
        _render_frame(scene, frame, out_path)
        rendered += 1
        prev_sig = sig
        prev_path = out_path
    return rendered, copied


def main():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender -b --python render_unified_camera.py -- --scene X --camera in_N --out output/run1 --samples 48")
    post = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--no-frame-reuse", action="store_true")
    ns = ap.parse_args(post)

    bpy.ops.wm.open_mainfile(filepath=ns.scene)
    BS.configure_gpu()
    BS.setup_render(samples=ns.samples)

    cam = bpy.data.objects.get(f"Camera_{ns.camera}") or bpy.data.objects.get(ns.camera)
    if cam is None:
        raise SystemExit(f"FAIL: camera not found in unified scene: {ns.camera}")
    bpy.context.scene.camera = cam

    frames_dir = os.path.join(ns.out, f"frames_{ns.camera}")
    os.makedirs(frames_dir, exist_ok=True)
    for fn in os.listdir(frames_dir):
        if fn.endswith(".jpg"):
            os.remove(os.path.join(frames_dir, fn))
    scene = bpy.context.scene
    scene.render.filepath = os.path.join(frames_dir, "f_")
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 95
    if ns.no_frame_reuse:
        reuse = False
        reason = "disabled by --no-frame-reuse"
    else:
        reason = _unsupported_reuse_reason(scene)
        reuse = reason is None
    mode = "reuse" if reuse else f"no-reuse ({reason})"
    print(f"[unified:{ns.camera}] rendering frames {scene.frame_start}..{scene.frame_end} -> {frames_dir} [{mode}]", flush=True)
    rendered, copied = _render_manual(scene, frames_dir, reuse=reuse)
    print(f"[unified:{ns.camera}] frames ready (rendered={rendered}, copied={copied})", flush=True)


if __name__ == "__main__":
    main()
