#!/usr/bin/env python3
"""Phase 7 — prepare and compare SUMO validation runs.

This script does not require SUMO.  It always exports a minimal SUMO input
package from ``scenario.json`` and writes this simulator's metrics.  If a SUMO
``tripinfo.xml`` file is provided, it also parses SUMO metrics and writes a
side-by-side JSON/CSV comparison report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))

from sim.sumo import (
    export_sumo_files,
    parse_sumo_tripinfo,
    scenario_metrics,
    write_comparison_report,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="path to scenario.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sumo-tripinfo", default=None,
                    help="optional SUMO tripinfo.xml produced by sumo -c run.sumocfg")
    ns = ap.parse_args(argv)

    with open(ns.scenario) as f:
        scenario = json.load(f)

    os.makedirs(ns.out, exist_ok=True)
    export_paths = export_sumo_files(scenario, ns.out)
    ours = scenario_metrics(scenario)
    sumo = parse_sumo_tripinfo(ns.sumo_tripinfo) if ns.sumo_tripinfo else None
    report_paths = write_comparison_report(ns.out, ours, sumo)

    manifest = {
        "schema": "sumo_comparison.v1",
        "scenario": os.path.abspath(ns.scenario),
        "sumo_inputs": export_paths,
        "reports": {
            "json": os.path.basename(report_paths["json"]),
            "csv": os.path.basename(report_paths["csv"]),
        },
        "sumo_tripinfo": os.path.abspath(ns.sumo_tripinfo) if ns.sumo_tripinfo else None,
    }
    manifest_path = os.path.join(ns.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote SUMO package: {os.path.join(ns.out, 'sumo')}")
    print(f"Wrote comparison report: {report_paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
