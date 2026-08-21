import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plot_scaling import inference_bucket, inference_seconds, main


class PlotScalingTest(unittest.TestCase):
    def test_inference_runtime_from_log(self):
        self.assertEqual(
            inference_seconds("Running model inference with seed 1 took 62.04 seconds.", 100),
            62.04,
        )

    def test_inference_runtime_falls_back_to_wall(self):
        self.assertEqual(inference_seconds("no timing", 100), 100)

    def test_inference_bucket_from_log(self):
        self.assertEqual(
            inference_bucket("Got bucket size 4096 for input with 4011 tokens"),
            4096,
        )

    def test_plot_with_bucket_strip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "scaling-repeat-bmrc-a100-128-device-1"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({
                "run_id": run.name,
                "status": "completed",
                "exit_status": 0,
                "start_time_utc": "2026-01-01T00:00:00+00:00",
                "wall_clock_seconds": 10,
                "input": {"jobs": [{"polymer_residues": 128}]},
            }))
            (run / "gpu.csv").write_text(
                "timestamp,index,memory_used_mib\n2026/01/01 00:00:00.000,0,1024\n"
            )
            (run / "alphafold.log").write_text(
                "Got bucket size 256 for input with 128 tokens\n"
                "Running model inference with seed 1 took 9.0 seconds.\n"
            )
            output = root / "plot.png"
            with mock.patch("sys.argv", ["plot_scaling.py", str(root), "--output", str(output)]):
                main()
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
