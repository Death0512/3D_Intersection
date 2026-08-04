"""Tests for SUMO export/comparison scaffolding."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))

from run_sumo_unified import (
    _apply_motion_derived_rot_z,
    _motion_delta_to_blender_rot_z,
    _sumo_angle_to_blender_rot_z,
    _unwrap_angle,
    write_sumo_network,
)
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
    def test_sumo_heading_conversion_uses_blender_z_convention(self):
        self.assertAlmostEqual(_sumo_angle_to_blender_rot_z(0.0), 0.0)
        self.assertAlmostEqual(_sumo_angle_to_blender_rot_z(90.0), -1.5707963267948966)
        self.assertAlmostEqual(abs(_sumo_angle_to_blender_rot_z(180.0)), 3.141592653589793)
        self.assertAlmostEqual(_sumo_angle_to_blender_rot_z(270.0), -4.71238898038469)

    def test_sumo_heading_unwrap_prevents_zero_360_spin(self):
        prev = _sumo_angle_to_blender_rot_z(0.0)
        cur = _unwrap_angle(prev, _sumo_angle_to_blender_rot_z(359.9))
        self.assertLess(abs(cur - prev), 0.01)

    def test_motion_delta_rot_z_maps_local_y_to_motion(self):
        cases = [
            ((0.0, 1.0), (0.0, 1.0)),
            ((1.0, 0.0), (1.0, 0.0)),
            ((0.0, -1.0), (0.0, -1.0)),
            ((-1.0, 0.0), (-1.0, 0.0)),
            ((1.0, 1.0), (2 ** -0.5, 2 ** -0.5)),
        ]
        for (dx, dy), expected in cases:
            z = _motion_delta_to_blender_rot_z(dx, dy)
            front = (-math.sin(z), math.cos(z))
            self.assertAlmostEqual(front[0], expected[0], places=6)
            self.assertAlmostEqual(front[1], expected[1], places=6)

    def test_motion_derived_rot_z_carries_stationary_samples(self):
        pts = [
            {"x": 0.0, "y": 0.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
            {"x": 0.0, "y": 0.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
            {"x": 0.0, "y": 1.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
            {"x": 0.0, "y": 1.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
        ]
        _apply_motion_derived_rot_z(pts)
        for pt in pts:
            self.assertAlmostEqual(pt["rot_z"], 0.0, places=6)
            self.assertEqual(pt["heading_deg"], 90.0)

    def test_motion_derived_rot_z_all_stationary_keeps_heading_fallback(self):
        pts = [
            {"x": 1.0, "y": 2.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
            {"x": 1.0, "y": 2.0, "heading_deg": 90.0, "rot_z": _sumo_angle_to_blender_rot_z(90.0)},
        ]
        _apply_motion_derived_rot_z(pts)
        self.assertAlmostEqual(pts[0]["rot_z"], -1.57079633, places=6)
        self.assertAlmostEqual(pts[1]["rot_z"], -1.57079633, places=6)

    def test_motion_derived_rot_z_unwraps_turn_continuity(self):
        pts = [
            {"x": 0.0, "y": 0.0, "heading_deg": 0.0, "rot_z": 0.0},
            {"x": -1.0, "y": -0.01, "heading_deg": 0.0, "rot_z": 0.0},
            {"x": -2.0, "y": 0.01, "heading_deg": 0.0, "rot_z": 0.0},
        ]
        _apply_motion_derived_rot_z(pts)
        self.assertLess(abs(pts[1]["rot_z"] - pts[0]["rot_z"]), 0.1)

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


class TestNetconvertRetry(unittest.TestCase):
    def setUp(self):
        self._netconvert_patcher = mock.patch(
            "shutil.which", return_value="/fake/netconvert"
        )
        self.mock_which = self._netconvert_patcher.start()
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._netconvert_patcher.stop()
        self.tempdir.cleanup()

    def _touch_nonempty(self, path):
        """Create a non-empty mock .net.xml for post-run assertions."""
        with open(path, "w") as f:
            f.write("<net/>\n")

    def _call(self, *, first_crash: bool):
        side_effects = []
        if first_crash:
            side_effects.append(
                subprocess.CalledProcessError(-signal.SIGSEGV, ["netconvert"])
            )
        side_effects.append(0)  # fallback (or first) success
        with (
            mock.patch("run_sumo_unified.subprocess.run") as mock_run,
            mock.patch("run_sumo_unified._write_xml"),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.path.getsize", return_value=1024),
        ):
            mock_run.side_effect = side_effects
            result = write_sumo_network(self.tempdir.name)

        self.assertEqual(result, os.path.join(self.tempdir.name, "intersection.net.xml"))

        if first_crash:
            self.assertEqual(mock_run.call_count, 2,
                             "expected retry after SIGSEGV")
        else:
            self.assertEqual(mock_run.call_count, 1,
                             "expected single call on clean success")

    def test_netconvert_retries_on_sigsegv(self):
        self._touch_nonempty(
            os.path.join(self.tempdir.name, "intersection.net.xml")
        )
        self._call(first_crash=True)

    def test_netconvert_no_retry_on_nonzero(self):
        # Non-SIGSEGV failures must raise immediately.
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch("shutil.which", return_value="/fake/netconvert"),
                mock.patch("run_sumo_unified.subprocess.run") as mock_run,
                mock.patch("run_sumo_unified._write_xml"),
            ):
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, ["netconvert"]
                )
                with self.assertRaises(subprocess.CalledProcessError):
                    write_sumo_network(td)
                self.assertEqual(mock_run.call_count, 1, "no retry for non-SIGSEGV")

    def test_clean_first_success(self):
        with tempfile.TemporaryDirectory() as td:
            net_path = os.path.join(td, "intersection.net.xml")
            with (
                mock.patch("shutil.which", return_value="/fake/netconvert"),
                mock.patch("run_sumo_unified.subprocess.run",
                           return_value=mock.DEFAULT),
                mock.patch("run_sumo_unified._write_xml"),
                mock.patch("os.path.isfile", return_value=True),
                mock.patch("os.path.getsize", return_value=1024),
            ):
                result = write_sumo_network(td)
                self.assertEqual(result, net_path)

    def test_fallback_omits_timing_options(self):
        with (
            mock.patch("run_sumo_unified.subprocess.run") as mock_run,
            mock.patch("run_sumo_unified._write_xml"),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.path.getsize", return_value=1024),
        ):
            mock_run.side_effect = [
                subprocess.CalledProcessError(-signal.SIGSEGV, ["netconvert"]),
                0,
            ]
            write_sumo_network(self.tempdir.name)

        fallback = mock_run.call_args_list[1].args[0]
        self.assertIn("--tls.default-type", fallback)
        self.assertIn("static", fallback)
        self.assertNotIn("--tls.cycle.time", fallback)
        self.assertNotIn("--tls.yellow.time", fallback)
        self.assertNotIn("--tls.allred.time", fallback)

    def test_missing_output_after_success_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch("shutil.which", return_value="/fake/netconvert"),
                mock.patch("run_sumo_unified.subprocess.run",
                           return_value=mock.DEFAULT),
                mock.patch("run_sumo_unified._write_xml"),
                mock.patch("os.path.isfile", return_value=False),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    write_sumo_network(td)
                self.assertIn("produced no output", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
