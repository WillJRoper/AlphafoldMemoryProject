import unittest

from profile_metadata import summarise_input


class SummariseInputTest(unittest.TestCase):
    def test_alphafold3_entities_and_seeds(self):
        summary = summarise_input(
            {
                "name": "complex",
                "dialect": "alphafold3",
                "version": 2,
                "modelSeeds": [7, 11],
                "sequences": [
                    {"protein": {"id": "A", "sequence": "ACDE"}},
                    {"rna": {"id": "B", "sequence": "AUG"}},
                    {"ligand": {"id": "C", "ccdCodes": ["ATP"]}},
                ],
            }
        )[0]

        self.assertEqual(summary["model_seeds"], [7, 11])
        self.assertEqual(summary["sequence_entities"], 3)
        self.assertEqual(summary["polymer_entities"], 2)
        self.assertEqual(summary["polymer_residues"], 7)
        self.assertEqual(summary["ligand_entities"], 1)


if __name__ == "__main__":
    unittest.main()
