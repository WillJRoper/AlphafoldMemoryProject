import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_pipeline_scaling import parse_pipeline_phases, plot_length, plot_threads


class PlotPipelineScalingTest(unittest.TestCase):
    def test_pipeline_phase_parsing(self):
        phases = parse_pipeline_phases(
            "Getting protein MSAs took 10.5 seconds\n"
            "Deduplicating MSAs took 1.2 seconds\n"
            "Getting 4 protein templates took 3.4 seconds\n"
            "Running data pipeline for chain A took 15.6 seconds\n"
        )
        self.assertEqual(
            phases,
            {"pipeline_s": 15.6, "msa_s": 10.5, "dedup_s": 1.2, "templates_s": 3.4},
        )

    def test_scaling_plots(self):
        rows = []
        for accession, residues in (("P60174", 249), ("Q8I3Z1", 10061)):
            for cpus in (4, 8):
                rows.append({
                    "accession": accession,
                    "residues": residues,
                    "target": residues,
                    "cpus": cpus,
                    "runtime_s": residues / cpus,
                    "max_rss_gib": residues / 1000,
                })
        data = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            length_output = Path(directory) / "length.png"
            threads_output = Path(directory) / "threads.png"
            plot_length(data, length_output)
            plot_threads(data, threads_output)
            self.assertTrue(length_output.exists())
            self.assertTrue(threads_output.exists())


if __name__ == "__main__":
    unittest.main()
