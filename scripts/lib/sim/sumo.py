"""Phase 7 — SUMO export and comparison helpers.

The project does not depend on SUMO at runtime.  These helpers create a small
SUMO-compatible input package and compute metrics from either this simulator's
scenario JSON or a SUMO ``tripinfo.xml`` file when the user runs SUMO outside
the Python test environment.
"""
from __future__ import annotations

import csv
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Optional


def scenario_metrics(scenario: dict) -> Dict[str, float]:
    fps = float(scenario.get("fps", 30) or 30)
    duration_s = float(scenario.get("duration_frames", 0)) / fps if fps > 0 else 0.0
    vehicles = list(scenario.get("vehicles", []))
    n = len(vehicles)
    waits = [float(v.get("wait_frames", 0) or 0) / fps for v in vehicles]
    speeds = [float(v.get("speed_ms", 0) or 0) for v in vehicles]
    released = [v for v in vehicles if v.get("release_frame") is not None]
    throughput_vph = (len(released) / duration_s * 3600.0) if duration_s > 0 else 0.0
    return {
        "vehicle_count": float(n),
        "released_count": float(len(released)),
        "throughput_vph": throughput_vph,
        "mean_wait_s": sum(waits) / n if n else 0.0,
        "max_wait_s": max(waits) if waits else 0.0,
        "mean_desired_speed_mps": sum(speeds) / n if n else 0.0,
    }


def parse_sumo_tripinfo(path: str) -> Dict[str, float]:
    root = ET.parse(path).getroot()
    trips = list(root.findall("tripinfo"))
    n = len(trips)
    durations = [float(t.get("duration", 0.0)) for t in trips]
    waits = [float(t.get("waitingTime", 0.0)) for t in trips]
    losses = [float(t.get("timeLoss", 0.0)) for t in trips]
    return {
        "vehicle_count": float(n),
        "mean_travel_time_s": sum(durations) / n if n else 0.0,
        "mean_wait_s": sum(waits) / n if n else 0.0,
        "max_wait_s": max(waits) if waits else 0.0,
        "mean_time_loss_s": sum(losses) / n if n else 0.0,
    }


def _xml_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _route_id(v: dict) -> str:
    return f"{v['approach']}_{v.get('turn', 'straight')}"


def export_sumo_files(scenario: dict, out_dir: str) -> Dict[str, str]:
    """Write a minimal SUMO package for external validation.

    The generated files are intentionally simple and portable.  They can be fed
    to ``netconvert``/``sumo`` by researchers who have SUMO installed; unit tests
    only verify structural validity and do not require SUMO.
    """
    sumo_dir = os.path.join(out_dir, "sumo")
    os.makedirs(sumo_dir, exist_ok=True)
    fps = float(scenario.get("fps", 30) or 30)
    directions = ["N", "E", "S", "W"]
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    left = {"N": "W", "W": "S", "S": "E", "E": "N"}
    right = {"N": "E", "E": "S", "S": "W", "W": "N"}
    turn_to_exit = {"straight": opposite, "left": left, "right": right}

    nodes_path = os.path.join(sumo_dir, "intersection.nod.xml")
    edges_path = os.path.join(sumo_dir, "intersection.edg.xml")
    routes_path = os.path.join(sumo_dir, "routes.rou.xml")
    cfg_path = os.path.join(sumo_dir, "run.sumocfg")
    readme_path = os.path.join(sumo_dir, "README.md")

    with open(nodes_path, "w") as f:
        f.write("<nodes>\n")
        f.write('  <node id="C" x="0" y="0" type="traffic_light"/>\n')
        coords = {"N": (0, 120), "S": (0, -120), "E": (120, 0), "W": (-120, 0)}
        for d, (x, y) in coords.items():
            f.write(f'  <node id="{d}" x="{x}" y="{y}" type="priority"/>\n')
        f.write("</nodes>\n")

    with open(edges_path, "w") as f:
        f.write("<edges>\n")
        for d in directions:
            f.write(f'  <edge id="{d}_in" from="{d}" to="C" numLanes="4" speed="22.22"/>\n')
            f.write(f'  <edge id="{d}_out" from="C" to="{d}" numLanes="4" speed="22.22"/>\n')
        f.write("</edges>\n")

    routes = {}
    for v in scenario.get("vehicles", []):
        rid = _route_id(v)
        if rid in routes:
            continue
        app = v["approach"]
        turn = v.get("turn", "straight")
        ex = turn_to_exit.get(turn, opposite)[app]
        routes[rid] = f"{app}_in {ex}_out"

    with open(routes_path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" accel="2.5" decel="2.0" sigma="0.5" length="4.5" maxSpeed="22.22"/>\n')
        for rid, edges in sorted(routes.items()):
            f.write(f'  <route id="{_xml_escape(rid)}" edges="{_xml_escape(edges)}"/>\n')
        for v in sorted(scenario.get("vehicles", []), key=lambda x: x.get("depart_frame", 0)):
            depart = float(v.get("depart_frame", 0)) / fps
            f.write(
                f'  <vehicle id="{_xml_escape(v["id"])}" type="car" '
                f'route="{_xml_escape(_route_id(v))}" depart="{depart:.3f}" '\
                f'departLane="{int(v.get("lane", 0))}"/>\n')
        f.write("</routes>\n")

    with open(cfg_path, "w") as f:
        f.write("<configuration>\n")
        f.write('  <input net-file="net.net.xml" route-files="routes.rou.xml"/>\n')
        f.write('  <output tripinfo-output="tripinfo.xml"/>\n')
        f.write("</configuration>\n")

    with open(readme_path, "w") as f:
        f.write("# SUMO validation package\n\n")
        f.write("Build a SUMO net, then run SUMO externally:\n\n")
        f.write("```bash\n")
        f.write("netconvert --node-files intersection.nod.xml --edge-files intersection.edg.xml --output-file net.net.xml\n")
        f.write("sumo -c run.sumocfg\n")
        f.write("python3 scripts/compare_sumo.py --scenario scenario.json --sumo-tripinfo sumo/tripinfo.xml --out comparison\n")
        f.write("```\n")

    return {
        "sumo_dir": "sumo",
        "nodes": "sumo/intersection.nod.xml",
        "edges": "sumo/intersection.edg.xml",
        "routes": "sumo/routes.rou.xml",
        "config": "sumo/run.sumocfg",
        "readme": "sumo/README.md",
    }


def write_comparison_report(out_dir: str,
                            ours: Dict[str, float],
                            sumo: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    report = {"ours": ours, "sumo": sumo or {}}
    json_path = os.path.join(out_dir, "comparison_report.json")
    csv_path = os.path.join(out_dir, "comparison_report.csv")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    keys = sorted(set(ours) | set((sumo or {}).keys()))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "ours", "sumo", "delta_ours_minus_sumo"])
        for k in keys:
            ov = ours.get(k, "")
            sv = (sumo or {}).get(k, "")
            delta = ov - sv if isinstance(ov, (int, float)) and isinstance(sv, (int, float)) else ""
            writer.writerow([k, ov, sv, delta])
    return {"json": json_path, "csv": csv_path}


__all__ = [
    "scenario_metrics",
    "parse_sumo_tripinfo",
    "export_sumo_files",
    "write_comparison_report",
]
