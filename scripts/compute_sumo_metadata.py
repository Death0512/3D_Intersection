#!/usr/bin/env python3
"""Write metadata.json directly from SUMO unified trajectories.
Uses ijson streaming to avoid a giant in-memory vehicles list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import ijson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import envfile as ENV
import geometry as G


def _camera_tags(only: str | None) -> list[str]:
    if not only:
        return G.camera_names()
    tags = [t.strip() for t in only.split(",") if t.strip()]
    valid = set(G.camera_names())
    bad = [t for t in tags if t not in valid]
    if bad:
        raise SystemExit(f"FAIL: invalid camera tag(s): {', '.join(bad)}")
    return tags


def _stream_scenario_metadata(scenario_path: str) -> dict:
    """Stream-read top-level scalar fields without loading vehicles array."""
    meta = {"fps": 30, "duration_frames": 0}
    with open(scenario_path, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        for prefix, event, value in parser:
            if prefix == "fps" and event == "number":
                meta["fps"] = int(value)
            elif prefix == "duration_frames" and event == "number":
                meta["duration_frames"] = int(value)
            elif prefix.startswith("vehicles."):
                break
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--expected-videos", action="store_true")
    ns = ap.parse_args()

    scenario_meta = _stream_scenario_metadata(ns.scenario)
    tags = _camera_tags(ns.only)
    videos = [f"video_{t}.mp4" for t in tags]

    cameras = {}
    with open(os.path.join(ROOT, "assets", "road.json")) as f:
        road_meta = json.load(f)
    for tag in G.camera_names():
        cameras[tag] = ENV.resolve_camera(ENV.load_env(tag, ROOT), road_meta, unified=True)

    def _camera_for_edge(edge_id: str | None) -> str | None:
        """Map SUMO edge id to camera name. Internal junction edges start with ':'. ponytail:"""
        if not edge_id or edge_id.startswith(":"):
            return None
        parts = edge_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("in", "out"):
            d, role = parts
            cam = f"{'in' if role == 'in' else 'out'}_{d}"
            if cam in cameras:
                return cam
        return None

    out_path = os.path.join(ns.out, "metadata.json")

    # Stream-write metadata.json: write prefix, then append each vehicle as we
    # stream it, then close.  Valid JSON, no giant in-memory list.
    with open(out_path, "w") as outf:
        # Opening brace + fixed fields (cameras is small, json.dump it inline)
        outf.write('{\n  "schema": "metadata.sumo_unified.v1",\n')
        outf.write(f'  "simulator": "sumo",\n')
        outf.write(f'  "fps": {json.dumps(scenario_meta["fps"])},\n')
        outf.write(f'  "duration_frames": {json.dumps(scenario_meta["duration_frames"])},\n')
        outf.write(f'  "cameras": {json.dumps(cameras, indent=6)},\n')
        outf.write('  "vehicles": [\n')

        first = True
        count = 0
        with open(ns.scenario, "rb") as inf:
            for v in ijson.items(inf, "vehicles.item", use_float=True):
                pts = v.get("trajectory") or []
                frames = [{
                    "frame": int(p["frame"]),
                    "visible": True,
                    "camera": _camera_for_edge(p.get("edge_id")),
                    "pose": {"x": p["x"], "y": p["y"], "z": p.get("z", 0.0), "rot_z": p.get("rot_z", 0.0)},
                    "speed": p.get("speed", 0.0),
                    "lane_id": p.get("lane_id"),
                    "edge_id": p.get("edge_id"),
                } for p in pts]

                approach = v.get("approach")
                turn = v.get("turn")
                ex_cam = None
                if approach and turn:
                    try:
                        ex_dir, _ = G.exit_lane_for_movement(G.Direction(approach), v.get("lane", 0), G.Turn(turn))
                        ex_cam = f"out_{ex_dir.value}"
                    except Exception:
                        pass

                entry = {
                    "id": v.get("id"), "class": v.get("class", "car"),
                    "plate": v.get("plate"), "color": v.get("color"),
                    "approach": approach, "lane": v.get("lane"), "turn": turn,
                    "in_camera": f"in_{approach}" if approach else None,
                    "out_camera": ex_cam,
                    "speed_ms": v.get("speed_ms", 0.0),
                    "appear_frame": v.get("appear_frame", pts[0]["frame"] if pts else 0),
                    "disappear_frame": v.get("disappear_frame", pts[-1]["frame"] if pts else 0),
                    "frames": frames,
                }

                if first:
                    first = False
                else:
                    outf.write(",\n")
                # Serialize with consistent indent (4 spaces per level, starting at 4)
                json.dump(entry, outf, indent=4)
                count += 1

                if count % 100 == 0:
                    print(f"  [metadata] streamed {count} vehicles", flush=True)

        outf.write("\n  ],\n")
        outf.write(f'  "videos": {json.dumps(videos)}\n')
        outf.write("}\n")

    print(f"[metadata] wrote {out_path} ({count} vehicles)", flush=True)


if __name__ == "__main__":
    main()
