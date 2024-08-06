import unittest
from estimation import ABTEST

class TestABTEST(unittest.TestCase):
    def setUp(self):
        self.abtest = ABTEST()

    def test_get_sample_size(self):
        self.assertEqual(self.abtest.get_sample_size("One-sided Test"), 384)
        self.assertEqual(self.abtest.get_sample_size("Two-sided Test"), 482)

    def test_calculate_duration(self):
        self.assertEqual(self.abtest.calculate_duration(100, "One-sided Test"), 4)
        self.assertEqual(self.abtest.calculate_duration(100, "Two-sided Test"), 5)

    def test_calculate_mde(self):
        self.assertAlmostEqual(self.abtest.calculate_mde(), 10.0, places=2)

    def test_plot_distributions(self):
        plt = self.abtest.plot_distributions(500)
        self.assertIsNotNone(plt)

if __name__ == '__main__':
    unittest.main()