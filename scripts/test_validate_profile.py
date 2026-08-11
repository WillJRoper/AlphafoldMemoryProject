import unittest

from validate_profile import parse_usalign


class ParseUSAlignTest(unittest.TestCase):
    def test_metrics(self):
        metrics = parse_usalign(
            "Aligned length= 129, RMSD= 1.25, Seq_ID=n_identical/n_aligned= 1.000\n"
            "TM-score= 0.9123 (if normalized by length of Structure_1)\n"
            "TM-score= 0.9234 (if normalized by length of Structure_2)\n"
        )

        self.assertEqual(metrics["aligned_residues"], 129)
        self.assertEqual(metrics["rmsd_angstrom"], 1.25)
        self.assertEqual(metrics["tm_score_reference"], 0.9234)


if __name__ == "__main__":
    unittest.main()
