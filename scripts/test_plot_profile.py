import tempfile
import unittest
from pathlib import Path

from plot_profile import (
    active_window,
    load_gpu_csv,
    memory_exceedance,
    parse_phase_durations,
    model_load_at,
    plot_diagnostics,
    plot_timeseries,
)

CSV = """timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,utilization_memory_percent,power_draw_w,temperature_gpu_c,clocks_sm_mhz,clocks_memory_mhz
2026/08/18 02:08:50.820, 0, 0, 40442, 0, 0, 33.04, 28, 210, 1215
2026/08/18 02:09:00.820, 0, 10240, 30202, 50, 25, 150.5, 42, 1200, 1215
2026/08/18 02:09:10.820, 0, 20480, 19962, 100, 50, 250.1, 55, 1400, 1215
"""


class PlotProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / "gpu.csv"
        self.csv.write_text(CSV)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_gpu_csv_parses_space_padded_values(self):
        df = load_gpu_csv(self.csv)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["memory_used_mib"].tolist(), [0, 10240, 20480])
        self.assertAlmostEqual(df["elapsed_min"].iloc[0], 0)
        self.assertAlmostEqual(df["elapsed_min"].iloc[2], 20.0 / 60.0, places=3)

    def test_active_window_uses_utilization(self):
        df = load_gpu_csv(self.csv)
        start, end = active_window(df)
        self.assertAlmostEqual(start, 10.0 / 60.0, places=3)
        self.assertAlmostEqual(end, 20.0 / 60.0, places=3)
        self.assertEqual(active_window(df, util_threshold=200), (None, None))

    def test_model_load_at(self):
        df = load_gpu_csv(self.csv)
        self.assertAlmostEqual(model_load_at(df), 10.0 / 60.0, places=3)
        self.assertIsNone(model_load_at(df, threshold_mib=50000))

    def test_parse_phase_durations(self):
        log = Path(self.tmp.name) / "alphafold.log"
        log.write_text(
            "Running data pipeline for chain A took 2861.09 seconds\n"
            "Running model inference and extracting output structures with 1 seed(s) took "
            "62.41 seconds\n"
        )
        self.assertEqual(parse_phase_durations(log), {
            "data pipeline": 2861.09,
            "inference": 62.41,
        })

    def test_peak_memory_at(self):
        df = load_gpu_csv(self.csv)
        self.assertEqual(df.loc[df["memory_used_mib"].idxmax(), "elapsed_min"], 20.0 / 60.0)

    def test_memory_exceedance_is_monotone_decreasing(self):
        df = load_gpu_csv(self.csv)
        mem, exc = memory_exceedance(df)
        self.assertEqual(exc[-1], 0.0)
        self.assertEqual(list(mem), [0, 10240, 20480])
        for actual, expected in zip(exc, [2 / 3, 1 / 3, 0.0]):
            self.assertAlmostEqual(actual, expected)

    def test_plots_are_written(self):
        df = load_gpu_csv(self.csv)
        title = "test run"
        plot_timeseries(df, Path(self.tmp.name) / "ts.png", title)
        plot_diagnostics(df, Path(self.tmp.name) / "diag.png", title)
        self.assertTrue((Path(self.tmp.name) / "ts.png").stat().st_size > 0)
        self.assertTrue((Path(self.tmp.name) / "diag.png").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
