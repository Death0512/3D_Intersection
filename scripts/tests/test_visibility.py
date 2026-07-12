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


@pytest.fixture
def build_scene_module():
    _install_bpy_stub()
    # Drop any cached build_scene so the stub takes effect on re-import.
    sys.modules.pop("build_scene", None)
    import build_scene as BS
    yield BS
    # Don't leak the stubs into other tests.
    sys.modules.pop("build_scene", None)
    sys.modules.pop("bpy", None)
    sys.modules.pop("mathutils", None)


def _visibility_inserts(obj, data_path):
    return [f for (dp, f) in obj.inserts if dp == data_path]


def test_keyframe_motion_applies_hide_to_child_meshes(build_scene_module):
    """Every descendant mesh must receive hide_render + hide_viewport keyframes
    at frame 0 (hidden), vis_start (visible), and hide_end (hidden). Without
    this, the parent Empty's keyframes are inert (Empty renders nothing,
    hide_render doesn't propagate) and the car geometry stays frozen at the
    track endpoints for the whole timeline."""
    BS = build_scene_module

    # Build a parent Empty with two child meshes + one grandchild to verify
    # children_recursive traversal reaches nested rigs. In real Blender,
    # `obj.children_recursive` returns ALL descendants flattened (child +
    # grandchild + ...), so the parent's list includes the grandchild too.
    grandchild = _FakeObject("Wheel")
    child2 = _FakeObject("BodyMesh")  # grandchild is parented to child2,
    child1 = _FakeObject("Chassis")   # but child2.children_recursive is not
                                      # read — only the root's is. Model the
                                      # flattened list Blender would return.
    parent = _FakeObject("VEH_001",
                         children=[child1, child2, grandchild])

    # A minimal 2-point track: appear at frame 10 at (0,0), leave at frame 50
    # at (40,0). TrackPoint(frame, x, y, visible, interp, cp1, cp2).
    from geometry import TrackPoint
    track = [
        TrackPoint(frame=10, x=0.0, y=0.0, visible=True, interp="LINEAR"),
        TrackPoint(frame=50, x=40.0, y=0.0, visible=True, interp="LINEAR"),
    ]

    # Stub a VehicleMotion with just the fields keyframe_motion reads.
    class _M:
        track_in = track
        track_out = track
    motion = _M()

    BS.keyframe_motion(parent, motion, is_in_camera=True, frame_end=60)

    vis_start = 10
    hide_end = 51  # end_frame(50) + 1
    all_objs = [parent, child1, child2, grandchild]

    for o in all_objs:
        hr = _visibility_inserts(o, "hide_render")
        hv = _visibility_inserts(o, "hide_viewport")
        assert hr == [0, vis_start, hide_end], (
            f"{o.name} hide_render keyframes = {hr}, expected [0, {vis_start}, {hide_end}]")
        assert hv == [0, vis_start, hide_end], (
            f"{o.name} hide_viewport keyframes = {hv}, expected [0, {vis_start}, {hide_end}]")
        # Final state must be hidden (last keyframe set hide_render=True).
        assert o.hide_render is True, f"{o.name}.hide_render not left hidden"
        assert o.hide_viewport is True, f"{o.name}.hide_viewport not left hidden"

    # The Empty itself must still be keyframed (kept for future-proofing / any
    # renderable parent), in addition to the children.
    assert _visibility_inserts(parent, "hide_render"), "parent Empty lost its keyframes"


def test_keyframe_motion_hide_end_clamps_to_frame_end(build_scene_module):
    """When end_frame > frame_end (vehicle can't finish before the window), the
    hide keyframe lands at frame_end+1, not end_frame+1 — the vehicle vanishes
    at the window edge, not beyond it. Applies to children just like the parent."""
    BS = build_scene_module
    child = _FakeObject("Body")
    parent = _FakeObject("VEH_002", children=[child])

    from geometry import TrackPoint
    track = [
        TrackPoint(frame=10, x=0.0, y=0.0, visible=True, interp="LINEAR"),
        TrackPoint(frame=100, x=80.0, y=0.0, visible=True, interp="LINEAR"),
    ]

    class _M:
        track_in = track
        track_out = track
    BS.keyframe_motion(parent, _M(), is_in_camera=True, frame_end=60)

    # end_frame=100 > frame_end=60 → hide_end = 60+1 = 61, not 101.
    for o in (parent, child):
        hr = _visibility_inserts(o, "hide_render")
        assert hr == [0, 10, 61], f"{o.name} hide_end not clamped: {hr}"


def test_keyframe_motion_vis_start_never_negative(build_scene_module):
    """vis_start = max(0, start_frame); a track starting at a negative frame
    (shouldn't normally happen, but the clamp guards against Blender refusing
    a negative keyframe) is already visible at frame 0 and should not receive a
    hidden@0 keyframe followed by visible@0."""
    BS = build_scene_module
    child = _FakeObject("Body")
    parent = _FakeObject("VEH_003", children=[child])

    from geometry import TrackPoint
    track = [
        TrackPoint(frame=-5, x=0.0, y=0.0, visible=True, interp="LINEAR"),
        TrackPoint(frame=40, x=40.0, y=0.0, visible=True, interp="LINEAR"),
    ]

    class _M:
        track_in = track
        track_out = track
    BS.keyframe_motion(parent, _M(), is_in_camera=True, frame_end=50)

    for o in (parent, child):
        hr = _visibility_inserts(o, "hide_render")
        # vis_start = max(0, -5) = 0; hide_end = 40+1 = 41
        assert hr == [0, 41], f"{o.name} warm-up visibility wrong: {hr}"
