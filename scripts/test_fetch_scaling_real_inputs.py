import unittest

from fetch_scaling_real_inputs import make_input, parse_fasta


class FetchScalingRealInputsTest(unittest.TestCase):
    def test_fasta_and_input(self):
        sequence = parse_fasta(">sp|P1|TEST\nACDE\nFGHI\n")
        row = {"target_tokens": "8", "accession": "P1"}
        data = make_input(row, sequence)
        self.assertEqual(sequence, "ACDEFGHI")
        self.assertEqual(data["name"], "real_8_p1")
        self.assertEqual(data["sequences"][0]["protein"]["sequence"], sequence)


if __name__ == "__main__":
    unittest.main()
