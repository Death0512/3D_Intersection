#!/usr/bin/env python3
"""Blender-side helper: render one camera from unified_scene.blend to JPEGs."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

import bpy

import build_scene as BS


def main():
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender -b --python render_unified_camera.py -- --scene X --camera in_N --out output/run1 --samples 48")
    post = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=48)
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
    print(f"[unified:{ns.camera}] rendering frames {scene.frame_start}..{scene.frame_end} -> {frames_dir}", flush=True)
    bpy.ops.render.render(animation=True)
    print(f"[unified:{ns.camera}] frames ready", flush=True)


if __name__ == "__main__":
    main()
