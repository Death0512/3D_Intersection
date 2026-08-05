#!/usr/bin/env python3
"""Generate a SUMO-backed unified traffic scenario for Blender.

This replaces the project's internal traffic-motion algorithm for the new
unified pipeline.  SUMO owns vehicle insertion, car-following, lane changing,
intersection traversal, and signal compliance.  This script only:

  1. writes a compact four-arm SUMO network + routes,
  2. runs SUMO once through TraCI at the target video FPS,
  3. records every visible vehicle pose per frame,
  4. writes ``scenario.json`` with full trajectories for Blender.

The output schema intentionally stays close to the existing scenario fields so
plate generation, manifests, and downstream tooling remain usable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import string
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "lib"))

import geometry as G


DIRECTIONS = ["N", "E", "S", "W"]
TURN_SPLIT = {"left": 0.15, "straight": 0.70, "right": 0.15}
COLOR_LIST = [
    ((0.90, 0.10, 0.08, 1.0), "red"),
    ((0.08, 0.18, 0.85, 1.0), "blue"),
    ((0.08, 0.55, 0.12, 1.0), "green"),
    ((0.95, 0.88, 0.12, 1.0), "yellow"),
    ((0.92, 0.92, 0.92, 1.0), "white"),
    ((0.08, 0.08, 0.08, 1.0), "black"),
    ((0.55, 0.55, 0.55, 1.0), "gray"),
]


def _xml(s: object) -> str:
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _random_plate(rng: random.Random) -> str:
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


def _turn_exit(approach: str, turn: str) -> str:
    ap = G.Direction(approach)
    ex, _ = G.exit_lane_for_movement(ap, 1, G.Turn(turn))
    return ex.value


def _route_id(approach: str, turn: str) -> str:
    return f"{approach}_{turn}"


def _allowed_lanes_for_turn(turn: str) -> List[int]:
    t = G.Turn(turn)
    return [i for i in range(G.NUM_LANES) if t in G.allowed_turns(i)]


def _pick_turn_and_lane(rng: random.Random) -> Tuple[str, int]:
    turns = list(TURN_SPLIT)
    weights = [TURN_SPLIT[t] for t in turns]
    turn = rng.choices(turns, weights=weights, k=1)[0]
    lanes = _allowed_lanes_for_turn(turn)
    if turn == "left":
        lane = 0
    elif turn == "right":
        lane = G.NUM_LANES - 1
    else:
        lane = rng.choice(lanes)
    return turn, lane


def _write_xml(path: str, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_sumo_network(sumo_dir: str, speed_ms: float = 16.67,
                       edge_len: float = 120.0) -> str:
    """Write SUMO node/edge/connection/config files and run netconvert."""
    os.makedirs(sumo_dir, exist_ok=True)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="C", x="0", y="0", type="traffic_light")
    # Project convention: approach N vehicles move +Y into the box, so N_in
    # starts south of the centre and N_out leaves north of it.
    outer = {
        "N_from": (0, -edge_len), "N_to": (0, edge_len),
        "S_from": (0, edge_len), "S_to": (0, -edge_len),
        "E_from": (-edge_len, 0), "E_to": (edge_len, 0),
        "W_from": (edge_len, 0), "W_to": (-edge_len, 0),
    }
    for nid, (x, y) in outer.items():
        ET.SubElement(nodes, "node", id=nid, x=f"{x:.3f}", y=f"{y:.3f}", type="priority")

    edges = ET.Element("edges")
    for d in DIRECTIONS:
        ET.SubElement(edges, "edge", {
            "id": f"{d}_in", "from": f"{d}_from", "to": "C",
            "numLanes": str(G.NUM_LANES), "speed": f"{speed_ms:.3f}",
        })
        ET.SubElement(edges, "edge", {
            "id": f"{d}_out", "from": "C", "to": f"{d}_to",
            "numLanes": str(G.NUM_LANES), "speed": f"{speed_ms:.3f}",
        })

    cons = ET.Element("connections")
    for ap in DIRECTIONS:
        for lane in range(G.NUM_LANES):
            for turn in G.allowed_turns(lane):
                ex, ex_lane = G.exit_lane_for_movement(G.Direction(ap), lane, turn)
                ET.SubElement(cons, "connection", {
                    "from": f"{ap}_in", "to": f"{ex.value}_out",
                    "fromLane": str(lane), "toLane": str(ex_lane),
                    "dir": {G.Turn.LEFT: "l", G.Turn.STRAIGHT: "s", G.Turn.RIGHT: "r"}[turn],
                })

    nod = os.path.join(sumo_dir, "intersection.nod.xml")
    edg = os.path.join(sumo_dir, "intersection.edg.xml")
    con = os.path.join(sumo_dir, "intersection.con.xml")
    net = os.path.join(sumo_dir, "intersection.net.xml")
    _write_xml(nod, nodes)
    _write_xml(edg, edges)
    _write_xml(con, cons)

    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise SystemExit("FAIL: netconvert not found. Install SUMO or put it on PATH.")
    cmd = [
        netconvert,
        "--node-files", nod,
        "--edge-files", edg,
        "--connection-files", con,
        "--tls.default-type", "static",
        "--tls.cycle.time", "70",
        "--tls.yellow.time", "4",
        "--tls.allred.time", "2",
        "--no-turnarounds", "true",
        "--output-file", net,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # ponytail: netconvert 1.21.1 can SIGSEGV with explicit TLS cycle timing
        # options; retry without them and keep static TLS defaults.
        if e.returncode == -signal.SIGSEGV:
            # Remove any partial output from the crashed run.
            if os.path.exists(net):
                os.unlink(net)
            print("[sumo] WARNING: netconvert crashed (SIGSEGV) with --tls.cycle.time, "
                  "--tls.yellow.time, --tls.allred.time; retrying without them "
                  "(static TLS remains)", flush=True)
            fallback_cmd = [
                netconvert,
                "--node-files", nod,
                "--edge-files", edg,
                "--connection-files", con,
                "--tls.default-type", "static",
                "--no-turnarounds", "true",
                "--output-file", net,
            ]
            subprocess.run(fallback_cmd, check=True)
        else:
            raise
    if not os.path.isfile(net) or os.path.getsize(net) == 0:
        raise SystemExit(f"FAIL: netconvert produced no output at {net}")
    return net


def _parse_spike_windows(rest: str) -> list[tuple[float, float, float]]:
    """Parse semicolon-separated spike windows: start=55,end=65,scale=20;start=90,end=100,scale=30"""
    windows: list[tuple[float, float, float]] = []
    for segment in rest.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        opts = {"start": 55.0, "end": 65.0, "scale": 20.0}
        for part in segment.split(","):
            if not part.strip():
                continue
            k, sep, v = part.partition("=")
            if not sep or k.strip() not in opts:
                raise SystemExit(f"FAIL: bad --demand-profile component {part!r}")
            opts[k.strip()] = float(v)
        s, e, sc = opts["start"], opts["end"], opts["scale"]
        if e <= s:
            raise SystemExit("FAIL: demand spike end must be greater than start")
        if sc < 0:
            raise SystemExit("FAIL: demand spike scale must be >= 0")
        windows.append((s, e, sc))
    return windows


def _demand_multiplier(profile: str | None) -> tuple[Callable[[float], float], list[float]]:
    """Return (multiplier_fn, probe_times) for time-varying demand.

    Supported syntax:
      spike:start=55,end=65,scale=20               # single spike
      spike:start=55,end=65,scale=20;start=90,end=100,scale=30  # multiple
    """
    if not profile:
        return lambda _t: 1.0, [0.0]
    kind, _, rest = profile.partition(":")
    if kind.strip().lower() != "spike":
        raise SystemExit(f"FAIL: unsupported --demand-profile {profile!r}; expected spike:start=55,end=65,scale=20")
    windows = _parse_spike_windows(rest)
    if not windows:
        return lambda _t: 1.0, [0.0]
    # ponytail: linear scan over small window list; full partition table is overkill.
    def multi_spike(t: float) -> float:
        for s, e, sc in windows:
            if s <= t < e:
                return sc
        return 1.0
    probes = [1.0] + [s for s, _, _ in windows] + [e for _, e, _ in windows]
    return multi_spike, probes


def write_routes(sumo_dir: str, rng: random.Random, seconds: float,
                 flow_vph: Dict[str, float], demand_profile: str | None = None) -> Tuple[str, Dict[str, dict]]:
    routes_path = os.path.join(sumo_dir, "routes.rou.xml")
    vehicles_meta: Dict[str, dict] = {}
    per_route_counter = defaultdict(int)
    vehicles = []

    multiplier, spike_probes = _demand_multiplier(demand_profile)
    probe_times = [0.0, seconds] + spike_probes
    max_mult = max(1.0, *(multiplier(t) for t in probe_times))

    for ap in DIRECTIONS:
        base_rate = max(0.0, float(flow_vph.get(ap, 0.0))) / 3600.0
        proposal_rate = base_rate * max_mult
        if proposal_rate <= 0:
            continue
        t = 0.0
        while t < seconds:
            t += rng.expovariate(proposal_rate)
            if t >= seconds:
                break
            accept_p = (base_rate * multiplier(t)) / proposal_rate if proposal_rate > 0 else 0.0
            if rng.random() > accept_p:
                continue
            turn, lane = _pick_turn_and_lane(rng)
            rid = _route_id(ap, turn)
            idx = per_route_counter[rid]
            per_route_counter[rid] += 1
            vid = f"{rid}.{idx}"
            rgba, cname = rng.choice(COLOR_LIST)
            meta = {
                "id": vid, "class": "car", "approach": ap, "turn": turn,
                "lane": lane, "depart": t, "depart_frame": None,
                "route": rid, "speed_ms": 0.0, "speed_kmh": 0.0,
                "length": 4.5, "plate": _random_plate(rng),
                "color": list(rgba), "color_name": cname,
            }
            vehicles_meta[vid] = meta
            vehicles.append(meta)

    vehicles.sort(key=lambda v: (v["depart"], v["id"]))
    with open(routes_path, "w") as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" vClass="passenger" length="4.5" width="1.8" '
                'accel="2.6" decel="4.5" sigma="0.5" tau="1.0" '
                'minGap="2.5" maxSpeed="16.67" speedFactor="normc(1.0,0.10,0.8,1.2)" '\
                'carFollowModel="Krauss"/>\n')
        for ap in DIRECTIONS:
            for turn in ("left", "straight", "right"):
                ex = _turn_exit(ap, turn)
                rid = _route_id(ap, turn)
                f.write(f'  <route id="{_xml(rid)}" edges="{ap}_in {ex}_out"/>\n')
        for v in vehicles:
            f.write(f'  <vehicle id="{_xml(v["id"])}" type="car" route="{_xml(v["route"])}" '
                    f'depart="{v["depart"]:.3f}" departLane="{v["lane"]}" '\
                    'departSpeed="max"/>\n')
        f.write('</routes>\n')
    return routes_path, vehicles_meta


def write_config(sumo_dir: str, step_length: float, seconds: float) -> str:
    cfg = os.path.join(sumo_dir, "run.sumocfg")
    with open(cfg, "w") as f:
        f.write('<configuration>\n')
        f.write('  <input>\n')
        f.write('    <net-file value="intersection.net.xml"/>\n')
        f.write('    <route-files value="routes.rou.xml"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write('    <begin value="0"/>\n')
        f.write(f'    <end value="{seconds:.3f}"/>\n')
        f.write(f'    <step-length value="{step_length:.8f}"/>\n')
        f.write('  </time>\n')
        f.write('  <processing>\n')
        f.write('    <collision.action value="warn"/>\n')
        f.write('    <time-to-teleport value="-1"/>\n')
        f.write('  </processing>\n')
        f.write('  <report>\n')
        f.write('    <verbose value="false"/>\n')
        f.write('    <no-step-log value="true"/>\n')
        f.write('  </report>\n')
        f.write('</configuration>\n')
    return cfg


def _sumo_angle_to_blender_rot_z(angle_deg: float) -> float:
    # SUMO angle: 0=north/+Y, 90=east/+X, clockwise. Blender positive Z rotation
    # maps local +Y to world (-sin(z), cos(z)), so clockwise SUMO headings need a
    # negative Blender Z angle. ponytail: one convention shared with geometry.py.
    return -math.radians(float(angle_deg))


def _unwrap_angle(prev: float | None, current: float) -> float:
    if prev is None:
        return current
    while current - prev > math.pi:
        current -= 2.0 * math.pi
    while current - prev < -math.pi:
        current += 2.0 * math.pi
    return current


def _motion_delta_to_blender_rot_z(dx: float, dy: float) -> float:
    """Blender Z rotation that points a local +Y vehicle nose along ``(dx, dy)``.

    Blender positive Z rotation maps local +Y to world ``(-sin(z), cos(z))``.
    Solving that for a world movement vector gives ``atan2(-dx, dy)``.
    """
    return math.atan2(-float(dx), float(dy))


def _apply_motion_derived_rot_z(points: List[dict], epsilon: float = 0.01) -> None:
    """Replace visual ``rot_z`` in-place using trajectory deltas.

    SUMO ``heading_deg`` is preserved as raw/debug data, but it can lag the actual
    corrected XY motion on internal junction edges.  The rendered vehicle should
    visually face the path it follows, so derive yaw from adjacent position deltas.
    Stationary samples inherit the nearest moving yaw; all-stationary trajectories
    keep their existing heading-derived fallback.  The final sequence is unwrapped
    to avoid 0/360 spin jumps in Blender interpolation.
    """
    if not points:
        return
    eps2 = float(epsilon) * float(epsilon)
    yaws: List[float | None] = [None] * len(points)
    for i in range(len(points) - 1):
        dx = float(points[i + 1]["x"]) - float(points[i]["x"])
        dy = float(points[i + 1]["y"]) - float(points[i]["y"])
        if dx * dx + dy * dy >= eps2:
            yaws[i] = _motion_delta_to_blender_rot_z(dx, dy)

    # Carry yaw through stationary runs after movement.
    last: float | None = None
    for i, yaw in enumerate(yaws):
        if yaw is None:
            if last is not None:
                yaws[i] = last
        else:
            last = yaw

    # Fill leading stationary samples from the first later moving yaw.
    nxt: float | None = None
    for i in range(len(yaws) - 1, -1, -1):
        if yaws[i] is None:
            if nxt is not None:
                yaws[i] = nxt
        else:
            nxt = yaws[i]

    # All-stationary fallback: retain the existing SUMO-heading-derived rot_z.
    prev: float | None = None
    for i, point in enumerate(points):
        yaw = yaws[i]
        if yaw is None:
            yaw = float(point.get("rot_z", _sumo_angle_to_blender_rot_z(point.get("heading_deg", 0.0))))
        yaw = _unwrap_angle(prev, yaw)
        point["rot_z"] = round(yaw, 8)
        prev = yaw


def _read_net_offset(net_xml: str) -> Tuple[float, float]:
    """Parse netOffset from net.xml so we can shift SUMO world → Blender world.
    netconvert shifts all node coordinates so the network fits in positive space;
    we subtract that offset to re-centre the intersection at (0, 0)."""
    tree = ET.parse(net_xml)
    loc = tree.getroot().find("location")
    if loc is None:
        return (0.0, 0.0)
    raw = loc.get("netOffset", "0,0")
    ox, oy = raw.split(",")
    return (float(ox), float(oy))


def run_traci(cfg: str, fps: int, duration_frames: int,
              vehicles_meta: Dict[str, dict],
              net_offset: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, List[dict]]:
    try:
        import traci  # type: ignore
    except ImportError:
        # SUMO from distro packages often ships TraCI under SUMO_HOME/tools or
        # /usr/share/sumo/tools instead of installing a pip package. ponytail:
        for tools_dir in (os.path.join(os.environ.get("SUMO_HOME", ""), "tools"),
                          "/usr/share/sumo/tools"):
            if tools_dir and os.path.isdir(tools_dir) and tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
        try:
            import traci  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "FAIL: Python package 'traci' not importable. "
                "Install python3-traci/pip traci or set SUMO_HOME so SUMO_HOME/tools is importable."
            ) from e

    sumo = shutil.which("sumo")
    if not sumo:
        raise SystemExit("FAIL: sumo binary not found on PATH")
    traci.start([sumo, "-c", cfg, "--step-length", f"{1.0 / fps:.8f}",
                 "--no-step-log", "true"])
    trajectories: Dict[str, List[dict]] = defaultdict(list)
    last_heading_rot_z: Dict[str, float] = {}
    # Per-vehicle state for trajectory decimation (ponytail: same logic as
    # _keyframe_sumo_trajectory in build_unified_scene.py — reduce 6-10x RAM
    # by storing only keyframe candidates, not every frame).
    _prev_pt: Dict[str, dict | None] = {}
    last_pt: Dict[str, dict] = {}
    try:
        for frame in range(duration_frames):
            traci.simulationStep()
            for vid in traci.vehicle.getIDList():
                pos = traci.vehicle.getPosition(vid)
                # Shift SUMO world coords → Blender world (intersection centre = 0,0)
                x, y = float(pos[0]) - net_offset[0], float(pos[1]) - net_offset[1]
                angle = float(traci.vehicle.getAngle(vid))
                speed = float(traci.vehicle.getSpeed(vid))
                accel = float(traci.vehicle.getAcceleration(vid))
                lane_id = traci.vehicle.getLaneID(vid)
                edge_id = traci.vehicle.getRoadID(vid)
                heading_rot_z = _unwrap_angle(last_heading_rot_z.get(vid), _sumo_angle_to_blender_rot_z(angle))
                last_heading_rot_z[vid] = heading_rot_z
                pt = {
                    "frame": frame,
                    "time": round(frame / float(fps), 6),
                    "x": round(x, 4), "y": round(y, 4), "z": 0.0,
                    "heading_deg": round(angle, 4),
                    "rot_z": round(heading_rot_z, 8),
                    "speed": round(speed, 4),
                    "accel": round(accel, 4),
                    "lane_id": lane_id, "edge_id": edge_id,
                }
                # ponytail: decimate trajectory at source — only keep keyframe
                # candidates (always first sample, always stride-6 hits, and
                # heading/speed-change triggers).  Cuts JSON size and RAM 6-10x
                # for long (3600s) renders.  Blender linearly interpolates
                # between keyframes — visually identical to storing every frame.
                prev = _prev_pt.get(vid)
                keep = (prev is None
                        or (int(pt["frame"]) % 6 == 0)
                        or abs(float(pt.get("rot_z", 0.0)) - float(prev.get("rot_z", 0.0))) > 0.0175   # ~1 deg
                        or abs(float(pt["speed"]) - float(prev["speed"])) > 0.8)  # m/s
                if keep:
                    trajectories[vid].append(pt)
                    _prev_pt[vid] = pt
                # Always record the LAST sample per vehicle so the final frame
                # at disappear_frame is present (Blender needs the endpoint).
                last_pt[vid] = pt
                meta = vehicles_meta.get(vid)
                if meta is not None:
                    meta["speed_ms"] = max(float(meta.get("speed_ms", 0.0)), speed)
                    if meta.get("depart_frame") is None:
                        meta["depart_frame"] = frame
        # Ensure each vehicle's last collected point is in its trajectory
        # (might have been skipped by decimation stride).
        for vid, pt in last_pt.items():
            traj = trajectories.get(vid)
            if traj and traj[-1]["frame"] != pt["frame"]:
                traj.append(pt)
        for pts in trajectories.values():
            _apply_motion_derived_rot_z(pts)
        return trajectories
    finally:
        traci.close(False)


def _flow_dict(ns) -> Dict[str, float]:
    if ns.flow_json:
        with open(ns.flow_json) as f:
            data = json.load(f)
        flows = data.get("flows", data)
        return {d: float(flows.get(d, flows.get(d.lower(), ns.flow))) for d in DIRECTIONS}
    base = float(ns.flow) * float(ns.demand_scale)
    return {"N": base, "E": base, "S": base, "W": base}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flow", type=float, default=400.0,
                    help="base demand veh/h per approach")
    ap.add_argument("--demand-scale", type=float, default=1.0)
    ap.add_argument("--demand-profile", default=None,
                    help="optional time-varying demand, e.g. spike:start=55,end=65,scale=20")
    ap.add_argument("--flow-json", default=None,
                    help="optional JSON with flows per approach: {N,E,S,W}")
    ns = ap.parse_args(argv)

    rng = random.Random(ns.seed)
    out_dir = os.path.abspath(ns.out)
    sumo_dir = os.path.join(out_dir, "sumo_unified")
    os.makedirs(out_dir, exist_ok=True)
    fps = int(ns.fps)
    duration_frames = int(round(float(ns.seconds) * fps))
    step = 1.0 / fps

    print(f"[sumo] writing network/routes in {sumo_dir}", flush=True)
    net_xml = write_sumo_network(sumo_dir)
    net_offset = _read_net_offset(net_xml)
    print(f"[sumo] netOffset={net_offset} (will be subtracted from all coordinates)", flush=True)
    flow_vph = _flow_dict(ns)
    routes_path, vehicles_meta = write_routes(sumo_dir, rng, ns.seconds, flow_vph,
                                              demand_profile=ns.demand_profile)
    cfg = write_config(sumo_dir, step, ns.seconds)

    print(f"[sumo] running TraCI at {fps} FPS for {duration_frames} frames "
          f"({len(vehicles_meta)} scheduled vehicles)", flush=True)
    trajectories = run_traci(cfg, fps, duration_frames, vehicles_meta, net_offset)

    vehicles = []
    for vid, pts in sorted(trajectories.items(), key=lambda kv: (kv[1][0]["frame"], kv[0])):
        meta = vehicles_meta.get(vid, {"id": vid, "class": "car"})
        if not pts:
            continue
        speeds = [p["speed"] for p in pts]
        meta["depart_frame"] = pts[0]["frame"]
        meta["appear_frame"] = pts[0]["frame"]
        meta["disappear_frame"] = pts[-1]["frame"]
        meta["speed_ms"] = round(max(speeds) if speeds else 0.0, 3)
        meta["speed_kmh"] = round(meta["speed_ms"] * 3.6, 2)
        meta["trajectory"] = pts
        meta["trajectory_source"] = "sumo_traci"
        meta.setdefault("plate", _random_plate(rng))
        meta.setdefault("color", list(rng.choice(COLOR_LIST)[0]))
        meta.setdefault("color_name", "unknown")
        meta.setdefault("length", 4.5)
        meta.setdefault("approach", "N")
        meta.setdefault("turn", "straight")
        meta.setdefault("lane", 0)
        vehicles.append(meta)

    scenario = {
        "schema": "sumo_unified.v1",
        "simulator": "sumo",
        "fps": fps,
        "duration_frames": duration_frames,
        "seconds": float(ns.seconds),
        "seed": ns.seed,
        "flow_vph": flow_vph,
        "demand_profile": ns.demand_profile,
        "coordinate_system": "Blender world; intersection centre at (0,0,0); +X east, +Y north, Z up. SUMO netOffset subtracted.",
        "net_offset": list(net_offset),
        "sumo": {
            "dir": os.path.relpath(sumo_dir, out_dir),
            "routes": os.path.relpath(routes_path, out_dir),
            "config": os.path.relpath(cfg, out_dir),
        },
        "vehicles": vehicles,
    }
    out_path = os.path.join(out_dir, "scenario.json")
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"[sumo] wrote {out_path}: {len(vehicles)} vehicles with trajectories", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
