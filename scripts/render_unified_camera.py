#!/usr/bin/env python3
"""Blender-side helper: render one camera from a chunk .blend to one MP4 segment,
or from a legacy unified scene to JPEGs + batch-encode + concat final video."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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





def _encode_segment(frames_dir: str, seg_path: str, fps: int,
                    start_frame: int, batch_size: int, tag: str,
                    bitrate: str, segment_limit_bytes: int = 0) -> None:
    """Encode a contiguous batch of JPEGs to an MP4 segment via ffmpeg NVENC CBR."""
    cmd = ["ffmpeg", "-y",
           "-framerate", str(fps),
           "-start_number", str(start_frame),
           "-i", os.path.join(frames_dir, "f_%04d.jpg"),
           "-frames:v", str(batch_size),
           "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
           "-rc", "cbr", "-b:v", bitrate, "-maxrate", bitrate,
           "-bufsize", bitrate]
    if segment_limit_bytes > 0:
        cmd += ["-fs", str(segment_limit_bytes)]
    cmd.append(seg_path)
    print(f"  [unified:{tag}] encode batch {start_frame}..{start_frame + batch_size - 1}: "
          f"{' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-30:])
        raise RuntimeError(f"ffmpeg segment encode failed for {seg_path}:\n{tail}")
    # `-fs` may write its final packet past the nominal limit; frame validation
    # below rejects a truncated segment/final while static budget headroom absorbs it.
    if not os.path.isfile(seg_path) or os.path.getsize(seg_path) <= 0:
        raise RuntimeError(f"segment missing/empty after encode: {seg_path}")


def _concat_segments(segments: list[str], video_path: str,
                     concat_limit_bytes: int = 0) -> str:
    """Concatenate segments using ffmpeg concat demuxer -c copy."""
    concat_list = os.path.join(
        os.path.dirname(video_path), f"_{os.path.basename(video_path)}.concat.txt")
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{os.path.relpath(seg, os.path.dirname(video_path))}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", concat_list, "-c", "copy"]
    if concat_limit_bytes > 0:
        cmd += ["-fs", str(concat_limit_bytes)]
    cmd.append(video_path)
    print(f"  [encode] concat segments -> {video_path}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-30:])
        raise RuntimeError(f"ffmpeg concat failed for {video_path}:\n{tail}")
    return concat_list


def _video_valid(video_path: str, expected_fps: int,
                 expected_frames: int, frame_tol: int = 1) -> bool:
    """Return True if video exists and matches expected fps/frame count."""
    if not os.path.isfile(video_path):
        return False
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_frames,r_frame_rate",
                            "-of", "csv=p=0", video_path],
                           capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if p.returncode != 0 or not p.stdout.strip():
        return False
    parts = p.stdout.strip().split(",")
    if len(parts) < 2:
        return False
    fps_str, frames_str = parts[0], parts[-1]
    try:
        num, den = fps_str.split("/", 1)
        actual_fps = float(int(num)) / float(int(den))
    except (ValueError, ZeroDivisionError):
        return False
    try:
        actual_frames = int(frames_str)
    except ValueError:
        return False
    if abs(actual_fps - float(expected_fps)) > 0.51:
        return False
    if abs(actual_frames - expected_frames) > frame_tol:
        return False
    return True


def _fps_from_scenario(scenario_path: str | None) -> int:
    """Read fps from scenario JSON, default 30."""
    if scenario_path and os.path.isfile(scenario_path):
        import ijson
        with open(scenario_path, "rb") as f:
            for prefix, event, value in ijson.parse(f, use_float=True):
                if prefix == "fps" and event == "number":
                    return int(value)
                if prefix.startswith("vehicles."):
                    break
    return 30


def _check_usage(out_dir: str, tag: str, cap_bytes: int, label: str) -> None:
    """Check tag-scoped usage (frames_<tag>/ + segments_<tag>/ + video_<tag>.mp4 +
    concat manifest) against cap (0 = unlimited). Preserves artifacts."""
    if cap_bytes <= 0:
        return
    used = 0
    for sub in [f"frames_{tag}", f"segments_{tag}"]:
        p = os.path.join(out_dir, sub)
        if os.path.isdir(p):
            used += _dir_usage_bytes(p)
    video_path = os.path.join(out_dir, f"video_{tag}.mp4")
    if os.path.isfile(video_path):
        used += os.path.getsize(video_path)
    concat_list = os.path.join(out_dir, f"_video_{tag}.mp4.concat.txt")
    if os.path.isfile(concat_list):
        used += os.path.getsize(concat_list)
    if used > cap_bytes:
        raise RuntimeError(
            f"[unified:{tag}] usage cap exceeded {label}: "
            f"{used} bytes used > {cap_bytes} cap — "
            f"aborting without deleting artifacts")


def _dir_usage_bytes(path: str) -> int:
    """Return total bytes consumed by path and all children via du -sb."""
    p = subprocess.run(["du", "-sb", path],
                       capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        # ponytail: fallback to shutil when du fails (non-POSIX, permission)
        return _walk_size_bytes(path)
    try:
        return int(p.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return _walk_size_bytes(path)


def _walk_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _clean_stale(segments_dir: str, frames_dir: str) -> None:
    """Remove stale frames/segments from a previous incomplete run."""
    if os.path.isdir(segments_dir):
        shutil.rmtree(segments_dir)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)


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
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="frames per batch segment (default 1000)")
    ap.add_argument("--bitrate", type=str, default="5M",
                    help="ffmpeg NVENC CBR video bitrate, e.g. 5M (default 5M)")
    ap.add_argument("--storage-cap-bytes", type=int, default=0,
                    help="per-camera storage budget in bytes (0 = unlimited)")
    ap.add_argument("--segment-limit-bytes", type=int, default=0,
                    help="ffmpeg -fs bound per segment in bytes (0 = unlimited)")
    ap.add_argument("--concat-limit-bytes", type=int, default=0,
                    help="ffmpeg -fs bound for concat output in bytes (0 = unlimited)")
    ap.add_argument("--chunk-idx", type=int, default=None,
                    help="chunk index for time-chunk mode (renders single segment)")
    ns = ap.parse_args(post)

    chunk_mode = ns.chunk_idx is not None
    fps = _fps_from_scenario(ns.scenario)

    bpy.ops.wm.open_mainfile(filepath=ns.scene)
    BS.configure_gpu()
    BS.setup_render(samples=ns.samples)

    cam = bpy.data.objects.get(f"Camera_{ns.camera}") or bpy.data.objects.get(ns.camera)
    if cam is None:
        raise SystemExit(f"FAIL: camera not found in scene: {ns.camera}")
    bpy.context.scene.camera = cam

    out_dir = ns.out
    scene = bpy.context.scene
    chunk_frames = scene.frame_end - scene.frame_start + 1

    batch_size = int(ns.batch_size)
    if not 1 <= batch_size <= 9_999:
        raise SystemExit("--batch-size must be between 1 and 9999")
    storage_cap_bytes = ns.storage_cap_bytes
    segment_limit_bytes = ns.segment_limit_bytes
    concat_limit_bytes = ns.concat_limit_bytes

    # ---- Chunk mode: render ONE chunk scene → ONE segment ----
    if chunk_mode:
        chunk_idx = int(ns.chunk_idx)
        segments_dir = os.path.join(ns.out, f"segments_{ns.camera}")
        seg_path = os.path.join(segments_dir, f"seg_{chunk_idx:04d}.mp4")

        # Resume: skip if segment already valid
        if os.path.isfile(seg_path) and _video_valid(seg_path, fps, chunk_frames):
            print(f"[unified:{ns.camera}] SKIP chunk {chunk_idx}: segment already valid "
                  f"({chunk_frames}f@{fps}fps)", flush=True)
            return

        os.makedirs(segments_dir, exist_ok=True)
        # Isolated temp frames dir per chunk to avoid collisions
        frames_dir = os.path.join(ns.out, f"frames_{ns.camera}_chunk{chunk_idx:04d}")
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        scene.render.filepath = os.path.join(frames_dir, "f_")
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 95

        reuse = not ns.no_frame_reuse
        if not reuse:
            print(f"[unified:{ns.camera}] rendering chunk {chunk_idx} "
                  f"[{scene.frame_start},{scene.frame_end}] ({chunk_frames}f) no-reuse",
                  flush=True)
        else:
            reason = _unsupported_reuse_reason(scene)
            reuse = reason is None
            mode = "reuse" if reuse else f"no-reuse ({reason})"
            print(f"[unified:{ns.camera}] rendering chunk {chunk_idx} "
                  f"[{scene.frame_start},{scene.frame_end}] ({chunk_frames}f) [{mode}]",
                  flush=True)

        rendered = copied = 0
        prev_sig = None
        prev_path = None
        for index in range(chunk_frames):
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

        print(f"  [unified@{ns.camera}] chunk {chunk_idx}: rendered={rendered} copied={copied}",
              flush=True)

        # Encode chunk to one segment
        _check_usage(ns.out, ns.camera, storage_cap_bytes, f"before seg {chunk_idx}")
        _encode_segment(frames_dir, seg_path, fps, 0,
                        chunk_frames, ns.camera, ns.bitrate,
                        segment_limit_bytes)
        _check_usage(ns.out, ns.camera, storage_cap_bytes, f"after seg {chunk_idx}")

        # Validate segment
        if not os.path.isfile(seg_path) or os.path.getsize(seg_path) <= 0:
            raise RuntimeError(f"segment validation failed {seg_path}: missing/empty after encode")

        # Clean temp JPEGs for this chunk
        for fn_ in os.listdir(frames_dir):
            if fn_.endswith(".jpg"):
                os.remove(os.path.join(frames_dir, fn_))
        os.rmdir(frames_dir)

        print(f"[unified:{ns.camera}] chunk {chunk_idx} segment ready: {seg_path}", flush=True)
        return

    # ---- Legacy full-scene mode (unchanged) ----
    frames_dir = os.path.join(ns.out, f"frames_{ns.camera}")
    segments_dir = os.path.join(ns.out, f"segments_{ns.camera}")
    video_path = os.path.join(ns.out, f"video_{ns.camera}.mp4")
    total_frames = chunk_frames
    total_batches = (total_frames + batch_size - 1) // batch_size

    # Resumability: valid final video → skip.
    if os.path.isfile(video_path) and _video_valid(video_path, fps, total_frames):
        print(f"[unified:{ns.camera}] SKIP: valid video exists ({total_frames}f@{fps}fps)",
              flush=True)
        _clean_stale(segments_dir, frames_dir)
        return

    # Clean stale from prior incomplete run
    _clean_stale(segments_dir, frames_dir)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)

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
    print(f"[unified:{ns.camera}] rendering {total_frames} frames in "
          f"{total_batches} batch(es) ({batch_size}/batch) -> {frames_dir} [{mode}]",
          flush=True)

    segments: list[str] = []
    rendered_total = copied_total = 0
    first_frame = scene.frame_start

    for batch_idx in range(total_batches):
        batch_start_frame = first_frame + batch_idx * batch_size
        batch_end_frame = min(first_frame + (batch_idx + 1) * batch_size - 1,
                              scene.frame_end)
        actual_count = batch_end_frame - batch_start_frame + 1

        print(f"[unified:{ns.camera}] batch {batch_idx + 1}/{total_batches}: "
              f"frames {batch_start_frame}..{batch_end_frame} ({actual_count} frames)",
              flush=True)

        # Clean frames dir of prior batch JPEGs before rendering this batch
        for fn in os.listdir(frames_dir):
            if fn.endswith(".jpg"):
                os.remove(os.path.join(frames_dir, fn))

        # Render just this batch's frames
        rendered = copied = 0
        prev_sig = None
        prev_path = None
        for index in range(actual_count):
            frame = batch_start_frame + index
            # Frames are deleted after each batch, so local numbering keeps the
            # ffmpeg input pattern valid for arbitrarily long videos.
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

        rendered_total += rendered
        copied_total += copied
        print(f"  [unified:{ns.camera}] batch {batch_idx + 1}: "
              f"rendered={rendered} copied={copied}", flush=True)

        # Encode batch to segment MP4
        seg_path = os.path.join(segments_dir, f"seg_{batch_idx:04d}.mp4")
        _check_usage(ns.out, ns.camera, storage_cap_bytes,
                      f"before seg {batch_idx}")
        _encode_segment(frames_dir, seg_path, fps, 0,
                        actual_count, ns.camera, ns.bitrate,
                        segment_limit_bytes)
        _check_usage(ns.out, ns.camera, storage_cap_bytes,
                      f"after seg {batch_idx}")

        # Validate segment before deleting JPEGs
        if not os.path.isfile(seg_path) or os.path.getsize(seg_path) <= 0:
            raise RuntimeError(
                f"segment validation failed {seg_path}: missing/empty after encode")

        # Delete batch JPEGs only after successful segment encode
        for fn in os.listdir(frames_dir):
            if fn.endswith(".jpg"):
                os.remove(os.path.join(frames_dir, fn))

        segments.append(seg_path)

    # Concatenate all segments into final video
    _check_usage(ns.out, ns.camera, storage_cap_bytes, "before concat")
    concat_list = _concat_segments(segments, video_path, concat_limit_bytes)
    _check_usage(ns.out, ns.camera, storage_cap_bytes, "after concat")

    # Validate final video
    if not _video_valid(video_path, fps, total_frames):
        raise RuntimeError(
            f"final video validation failed: {video_path} "
            f"(expected {total_frames}f@{fps}fps)")

    # Clean up segments and frames after success
    _clean_stale(segments_dir, frames_dir)
    os.remove(concat_list)

    print(f"[unified:{ns.camera}] video ready: {video_path} "
          f"(rendered={rendered_total}, copied={copied_total})", flush=True)


if __name__ == "__main__":
    main()
