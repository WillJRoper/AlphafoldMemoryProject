import unittest

from generate_scaling_input import BASE_SEQUENCE, make_input


class GenerateScalingInputTest(unittest.TestCase):
    def test_exact_single_chain_token_count(self):
        data = make_input(300)
        protein = data["sequences"][0]["protein"]
        self.assertEqual(data["name"], "scaling_300")
        self.assertEqual(len(protein["sequence"]), 300)
        self.assertEqual(protein["sequence"], (BASE_SEQUENCE * 3)[:300])
        self.assertEqual(protein["templates"], [])
        self.assertEqual(protein["unpairedMsa"], f">query\n{protein['sequence']}\n")


if __name__ == "__main__":
    unittest.main()
