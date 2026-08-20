import io
import unittest
import urllib.error

from fetch_scaling_real_inputs import fetch_sequence, make_input, parse_fasta


class FetchScalingRealInputsTest(unittest.TestCase):
    def test_fasta_and_input(self):
        sequence = parse_fasta(">sp|P1|TEST\nACDE\nFGHI\n")
        row = {"target_tokens": "8", "accession": "P1"}
        data = make_input(row, sequence)
        self.assertEqual(sequence, "ACDEFGHI")
        self.assertEqual(data["name"], "real_8_p1")
        self.assertEqual(data["sequences"][0]["protein"]["sequence"], sequence)

    def test_fetch_retries_connection_resets(self):
        calls = []
        delays = []

        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            if len(calls) < 3:
                raise urllib.error.URLError(ConnectionResetError("reset"))
            return io.BytesIO(b">sp|P1|TEST\nACDE\n")

        sequence = fetch_sequence("P1", opener=opener, sleep=delays.append)

        self.assertEqual(sequence, "ACDE")
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [1, 2])


if __name__ == "__main__":
    unittest.main()
