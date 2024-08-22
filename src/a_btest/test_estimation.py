import unittest

from estimation import ABTEST


class TestABTEST(unittest.TestCase):
    def setUp(self):
        self.abtest = ABTEST()

    def test_get_sample_size(self):
        sample_size = self.abtest.get_sample_size("One-sided Test")
        self.assertTrue(23000 <= sample_size//2 <= 31000, f"Sample size {sample_size} is not within the expected range (380-390).")

    def test_calculate_duration(self):
        duration=self.abtest.calculate_duration(1000, "One-sided Test")
        self.assertTrue(40<= duration <=60, f"Duration {duration} is not within the expected range (23-31).")

    def test_calculate_mde(self):
        mde = self.abtest.calculate_mde()
        self.assertTrue(80<= mde*100 <=98, f"M.D.E {mde} is not within the expected range (70-80).")

    def test_generate_plot(self):
        """Test that the plot generation works and returns a Matplotlib figure object."""
        fig = self.abtest.generate_plot("One-sided Test")
        self.assertIsNotNone(fig)
        self.assertEqual(str(type(fig)), "<class 'matplotlib.figure.Figure'>")


if __name__ == "__main__":
    unittest.main()
