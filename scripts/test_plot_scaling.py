import unittest

from plot_scaling import inference_seconds


class PlotScalingTest(unittest.TestCase):
    def test_inference_runtime_from_log(self):
        self.assertEqual(
            inference_seconds("Running model inference with seed 1 took 62.04 seconds.", 100),
            62.04,
        )

    def test_inference_runtime_falls_back_to_wall(self):
        self.assertEqual(inference_seconds("no timing", 100), 100)


if __name__ == "__main__":
    unittest.main()
