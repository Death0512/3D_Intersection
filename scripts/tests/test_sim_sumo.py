"""Tests for Phase 7 SUMO export/comparison scaffolding."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

from compare_sumo import main as compare_main
from sim.sumo import (
    export_sumo_files,
    parse_sumo_tripinfo,
    scenario_metrics,
    write_comparison_report,
)


def _scenario():
    return {
        "fps": 30,
        "duration_frames": 300,
        "vehicles": [
            {"id": "V000", "approach": "N", "lane": 1, "turn": "straight",
             "depart_frame": 0, "release_frame": 100, "wait_frames": 30,
             "speed_ms": 12.0},
            {"id": "V001", "approach": "E", "lane": 0, "turn": "right",
             "depart_frame": 30, "release_frame": 130, "wait_frames": 0,
             "speed_ms": 10.0},
        ],
    }


class TestSUMOComparison(unittest.TestCase):
    def test_scenario_metrics(self):
        m = scenario_metrics(_scenario())
        self.assertEqual(m["vehicle_count"], 2.0)
        self.assertEqual(m["released_count"], 2.0)
        self.assertAlmostEqual(m["mean_wait_s"], 0.5)
        self.assertAlmostEqual(m["throughput_vph"], 720.0)

    def test_parse_sumo_tripinfo(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "tripinfo.xml")
            with open(p, "w") as f:
                f.write('<tripinfos>\n')
                f.write('  <tripinfo id="a" duration="10" waitingTime="2" timeLoss="3"/>\n')
                f.write('  <tripinfo id="b" duration="20" waitingTime="4" timeLoss="5"/>\n')
                f.write('</tripinfos>\n')
            m = parse_sumo_tripinfo(p)
        self.assertEqual(m["vehicle_count"], 2.0)
        self.assertAlmostEqual(m["mean_travel_time_s"], 15.0)
        self.assertAlmostEqual(m["mean_wait_s"], 3.0)

    def test_export_sumo_files(self):
        with tempfile.TemporaryDirectory() as td:
            paths = export_sumo_files(_scenario(), td)
            for rel in paths.values():
                self.assertTrue(os.path.exists(os.path.join(td, rel)))
            with open(os.path.join(td, paths["routes"])) as f:
                routes = f.read()
        self.assertIn('<route id="N_straight"', routes)
        self.assertIn('<vehicle id="V000"', routes)

    def test_write_comparison_report(self):
        with tempfile.TemporaryDirectory() as td:
            paths = write_comparison_report(
                td,
                {"mean_wait_s": 1.0},
                {"mean_wait_s": 2.5},
            )
            self.assertTrue(os.path.exists(paths["json"]))
            self.assertTrue(os.path.exists(paths["csv"]))
            with open(paths["json"]) as f:
                report = json.load(f)
        self.assertEqual(report["ours"]["mean_wait_s"], 1.0)

    def test_compare_sumo_script_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            scenario_path = os.path.join(td, "scenario.json")
            with open(scenario_path, "w") as f:
                json.dump(_scenario(), f)
            out = os.path.join(td, "cmp")
            rc = compare_main(["--scenario", scenario_path, "--out", out])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "sumo", "routes.rou.xml")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
