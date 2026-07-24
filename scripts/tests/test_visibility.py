"""Regression test: keyframe_motion applies hide_render/hide_viewport to every
child mesh, not just the parent Empty.

Root cause this guards against: `hide_render` is per-object in Blender and does
NOT propagate parent→child; an Empty renders nothing. Keyframing only the Empty
left the car meshes visible for the whole timeline — frozen at leave_pos after
leave_frame (the "cars stop at the end of the road" symptom) and at the spawn
anchor before appear_frame. Also broke render==metadata (compute_metadata emits
frames only in [appear,disappear] ∪ [reappear,leave]).

The test stubs only what keyframe_motion's visibility path touches and asserts
every descendant received the three hide keyframes (hidden@0, visible@start,
hidden@hide_end) on both hide_render and hide_viewport.
"""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


# ---------------------------------------------------------------------------
# Minimal bpy / mathutils stubs — installed BEFORE importing build_scene so
# its module-level `import bpy` / `from mathutils import ...` succeed outside
# Blender. The stub models only the surface keyframe_motion's visibility path
# touches; other code paths (configure_gpu, build_shot) are not exercised here.
# ---------------------------------------------------------------------------

class _FakeKeyframePoint:
    def __init__(self, co):
        self.co = co
        self.interpolation = "BEZIER"
        self.handle_left_type = "AUTO_CLAMPED"
        self.handle_right_type = "AUTO_CLAMPED"
        self.handle_left = (0.0, 0.0)
        self.handle_right = (0.0, 0.0)


class _FakeFCurve:
    def __init__(self, data_path, array_index=0):
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = []


class _FakeChannelbag:
    def __init__(self):
        self.fcurves = []


class _FakeStrip:
    def __init__(self):
        self._cb = _FakeChannelbag()

    def channelbag(self, slot):
        return self._cb


class _FakeLayer:
    def __init__(self):
        self.strips = [_FakeStrip()]


class _FakeSlot:
    pass


class _FakeAction:
    def __init__(self):
        self.layers = [_FakeLayer()]
        self.slots = [_FakeSlot()]
        self.name = "FakeAction"


class _FakeAnimData:
    def __init__(self):
        self.action = _FakeAction()


class _FakeEuler:
    def __init__(self):
        self.z = 0.0
        self.x = 0.0
        self.y = 0.0


class _FakeObject:
    """Records keyframe_insert calls so the test can assert per-object."""
    def __init__(self, name="obj", children=None):
        self.name = name
        self.location = (0.0, 0.0, 0.0)
        self.rotation_euler = _FakeEuler()
        self.hide_render = False
        self.hide_viewport = False
        self._props = {}
        self._anim = _FakeAnimData()
        self.children_recursive = children or []
        # Per-(data_path, frame) record of insert calls.
        self.inserts = []  # list of (data_path, frame)

    def get(self, key, default=None):
        return self._props.get(key, default)

    def __setitem__(self, key, value):
        self._props[key] = value

    @property
    def animation_data(self):
        return self._anim

    def keyframe_insert(self, data_path, frame):
        self.inserts.append((data_path, frame))


def _install_bpy_stub():
    bpy = types.ModuleType("bpy")
    # Not actually called by the visibility path, but keep namespace sane.
    bpy.context = types.SimpleNamespace(scene=types.SimpleNamespace())
    bpy.ops = types.SimpleNamespace()
    bpy.app = types.SimpleNamespace(handlers=types.SimpleNamespace())
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda *a, **k: None
    mathutils.Matrix = lambda *a, **k: None
    sys.modules["bpy"] = bpy
    sys.modules["mathutils"] = mathutils



# build_scene.py was removed; tests that depended on it are removed.
