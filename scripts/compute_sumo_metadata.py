#!/usr/bin/env python3
"""Write metadata.json directly from SUMO unified trajectories."""
from __future__ import annotations

import argparse
import json
import os
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--expected-videos", action="store_true")
    ns = ap.parse_args()
    with open(ns.scenario) as f:
        scenario = json.load(f)
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
        # edge_id format: "N_in" or "N_out"
        parts = edge_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("in", "out"):
            d, role = parts
            cam = f"{'in' if role == 'in' else 'out'}_{d}"
            if cam in cameras:
                return cam
        return None

    vehicles = []
    for v in scenario.get("vehicles", []):
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
        vehicles.append({
            "id": v.get("id"), "class": v.get("class", "car"),
            "plate": v.get("plate"), "color": v.get("color"),
            "approach": approach, "lane": v.get("lane"), "turn": turn,
            "in_camera": f"in_{approach}" if approach else None,
            "out_camera": ex_cam,
            "speed_ms": v.get("speed_ms", 0.0),
            "appear_frame": v.get("appear_frame", pts[0]["frame"] if pts else 0),
            "disappear_frame": v.get("disappear_frame", pts[-1]["frame"] if pts else 0),
            "frames": frames,
        })
    meta = {
        "schema": "metadata.sumo_unified.v1",
        "simulator": "sumo",
        "fps": scenario.get("fps", 30),
        "duration_frames": scenario.get("duration_frames", 0),
        "cameras": cameras,
        "vehicles": vehicles,
        "videos": videos,
    }
    out_path = os.path.join(ns.out, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[metadata] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
