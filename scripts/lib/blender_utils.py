"""Reusable helpers for headless bpy scripts.

All functions assume they are called from inside a running Blender instance
(invoked via `blender -b ... --python ...`). They never import bpy at module
import time in a way that breaks plain-python linting; bpy is expected to be
present in the interpreter.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# Scene clearing / setup
# ---------------------------------------------------------------------------

def reset_scene() -> None:
    """Wipe the current .blend to a clean state (no objects, no orphan data)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def remove_objects(names: Iterable[str]) -> None:
    for n in list(names):
        obj = bpy.data.objects.get(n)
        if obj is None:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)


def purge_orphans() -> None:
    """Remove all orphan data-blocks (meshes, materials, images, ...)."""
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


# ---------------------------------------------------------------------------
# Bounding box / origin / scale helpers
# ---------------------------------------------------------------------------

def world_bbox(objs: Iterable[bpy.types.Object]) -> Tuple[Vector, Vector]:
    """Return (min_corner, max_corner) in world space across the given objects."""
    mn = Vector((1e18, 1e18, 1e18))
    mx = Vector((-1e18, -1e18, -1e18))
    for o in objs:
        if o.type != "MESH":
            continue
        for v in o.bound_box:
            wc = o.matrix_world @ Vector(v)
            for i in range(3):
                if wc[i] < mn[i]:
                    mn[i] = wc[i]
                if wc[i] > mx[i]:
                    mx[i] = wc[i]
    return mn, mx


def dims_of(objs: Iterable[bpy.types.Object]) -> Vector:
    mn, mx = world_bbox(objs)
    return mx - mn


def apply_all_transforms(obj: bpy.types.Object) -> None:
    """Apply location, rotation, scale of a single object."""
    for attr in ("location", "rotation", "scale"):
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            pass
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=attr == "location", rotation=attr == "rotation", scale=attr == "scale")


def set_origin_to_ground_center(objs: List[bpy.types.Object]) -> None:
    """Set the origin of each object so that the group's min-Z is 0 and the
    origin lies at the ground-center of the group. We do this by moving the
    3D cursor and using origin_set on each mesh.
    """
    if not objs:
        return
    mn, mx = world_bbox(objs)
    center = Vector(((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, mn.z))
    bpy.context.scene.cursor.location = center
    for o in objs:
        if o.type != "MESH":
            continue
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")


# ---------------------------------------------------------------------------
# Texture remapping
# ---------------------------------------------------------------------------

def images_used_by_collection(coll) -> list:
    """Return the distinct Image data-blocks referenced by TEX_IMAGE nodes in
    any material assigned to any mesh object in ``coll``.

    Used to scope ``remap_textures_to_local`` to just one appended vehicle's
    own images instead of rescanning every image in the .blend (which is
    O(n^2) when called once per vehicle in a many-vehicle scene).
    """
    seen = set()
    images = []
    for obj in coll.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    img = node.image
                    if img.name not in seen:
                        seen.add(img.name)
                        images.append(img)
    return images


def remap_textures_to_local(tex_dir: str, missing_log: Optional[List[str]] = None,
                            images=None) -> int:
    """Remap image-block filepaths to files inside tex_dir.

    By default scans every image in the .blend (``bpy.data.images``) — fine
    for a one-time whole-file pass (e.g. asset_prep). Pass an explicit
    ``images`` iterable (e.g. just one vehicle's own images) to remap only
    those, which avoids O(n^2) rescans when called once per appended vehicle
    in a scene with many vehicles.

    Matching is extension-insensitive on the base name. Returns the number of
    images successfully remapped. Unresolved paths are appended to missing_log.
    """
    if not os.path.isdir(tex_dir):
        if missing_log is not None:
            missing_log.append(f"tex_dir not found: {tex_dir}")
        return 0

    # Index local files by lowercased basename without extension.
    local: dict = {}
    for fn in os.listdir(tex_dir):
        base, ext = os.path.splitext(fn)
        key = base.lower()
        local.setdefault(key, []).append(os.path.join(tex_dir, fn))

    remapped = 0
    for img in (images if images is not None else bpy.data.images):
        if img.name == "Render Result":
            continue
        filepath = bpy.path.abspath(img.filepath)
        base = os.path.splitext(os.path.basename(filepath))[0]
        key = base.lower()
        candidates = local.get(key, [])
        if not candidates:
            # Try the image's own name as a fallback (sometimes filepath is empty).
            base2 = os.path.splitext(img.name)[0]
            candidates = local.get(base2.lower(), [])
        if candidates:
            img.filepath = candidates[0]
            try:
                img.reload()
            except RuntimeError:
                pass
            remapped += 1
        else:
            if missing_log is not None:
                missing_log.append(f"{img.name} <- {filepath}")
    return remapped


# ---------------------------------------------------------------------------
# Material / node helpers
# ---------------------------------------------------------------------------

def ensure_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    return mat


def find_image_texture_node(mat: bpy.types.Material):
    """Return the first TEX_IMAGE node in mat.node_tree, or None."""
    if not mat or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE":
            return n
    return None


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def move_objects_to_collection(objs: Iterable[bpy.types.Object], coll: bpy.types.Collection) -> None:
    for o in objs:
        # unlink from all other collections
        for c in list(o.users_collection):
            c.objects.unlink(o)
        if o.name not in coll.objects:
            coll.objects.link(o)


# ---------------------------------------------------------------------------
# Geometry / math
# ---------------------------------------------------------------------------

def radians(deg: float) -> float:
    return math.radians(deg)
