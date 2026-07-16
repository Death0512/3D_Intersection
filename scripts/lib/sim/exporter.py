"""Phase 6 — simulation artifact exporter.

The renderer should consume traffic state produced by the simulation engine, not
recompute traffic behavior.  This module writes standalone simulation artifacts
beside ``scenario.json`` while keeping the legacy scenario fields intact for
backward-compatible rendering.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Tuple


def _jsonable_arrival_events(arrivals: Dict) -> Dict[str, list]:
    out = {}
    for key, frames in arrivals.items():
        if isinstance(key, tuple) and len(key) == 2:
            a, t = key
            a = getattr(a, "value", a)
            t = getattr(t, "value", t)
            out[f"{a}_{t}"] = list(frames)
        else:
            out[str(key)] = list(frames)
    return out


def write_simulation_artifacts(out_dir: str, scenario: Dict,
                               sim_meta: Dict) -> Dict[str, str]:
    """Write trajectory/metrics/meta JSON files and return relative paths.

    Returned paths are relative to ``out_dir`` so ``scenario.json`` remains
    portable when a Kaggle output directory is zipped or moved.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "trajectory": "trajectory.json",
        "lane_metrics": "lane_metrics.json",
        "simulation_meta": "simulation_meta.json",
    }

    trajectory = {
        "schema": sim_meta.get("trajectory_schema", "trajectory.v1"),
        "fps": scenario.get("fps"),
        "duration_frames": scenario.get("duration_frames"),
        "simulator": scenario.get("simulator"),
        "coordinate": "world+lane-longitudinal",
        "description": (
            "Per-frame simulation trace before Blender visualization. "
            "v2 samples keep legacy lane-longitudinal s and add world-space "
            "position plus velocity-vector fields. Per-vehicle constants live "
            "in the vehicles side table."
        ),
        "vehicles": sim_meta.get("trajectory_vehicles", {}),
        "samples": sim_meta.get("trajectory_samples", []),
    }
    lane_summary = sim_meta.get("lane_metrics", {})
    lane_timeseries = sim_meta.get("lane_metrics_timeseries", {})
    if lane_timeseries:
        lane_metrics = {
            "schema": "lane_metrics.v2",
            "fps": scenario.get("fps"),
            "duration_frames": scenario.get("duration_frames"),
            "simulator": scenario.get("simulator"),
            "description": (
                "Lane analytics with backward-compatible final summary and "
                "per-frame timeseries snapshots from the simulation engine. "
                "Flow/arrival/discharge rates are one-second window-finalized "
                "values and may remain stale between window boundaries."
            ),
            "lanes": {
                key: {
                    "summary": lane_summary.get(key, {}),
                    "timeseries": lane_timeseries.get(key, []),
                }
                for key in sorted(set(lane_summary) | set(lane_timeseries))
            },
        }
    else:
        lane_metrics = {
            "schema": "lane_metrics.v1",
            "fps": scenario.get("fps"),
            "duration_frames": scenario.get("duration_frames"),
            "simulator": scenario.get("simulator"),
            "lanes": lane_summary,
        }
    simulation_meta = {
        "schema": "simulation_meta.v1",
        "fps": scenario.get("fps"),
        "duration_frames": scenario.get("duration_frames"),
        "simulator": scenario.get("simulator"),
        "arrival_events": _jsonable_arrival_events(
            sim_meta.get("arrival_events", {})),
        "adaptive_intervals": sim_meta.get("adaptive_intervals", []),
        "adaptive_clearances": sim_meta.get("adaptive_clearances", []),
    }

    for rel, payload in (
        (paths["trajectory"], trajectory),
        (paths["lane_metrics"], lane_metrics),
        (paths["simulation_meta"], simulation_meta),
    ):
        with open(os.path.join(out_dir, rel), "w") as f:
            json.dump(payload, f, indent=2)
    return paths


__all__ = ["write_simulation_artifacts"]
