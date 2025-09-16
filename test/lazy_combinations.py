import unittest

from src.lazy_combinations import LazyCombinations


class TestLazyCombinations(unittest.TestCase):

    def test_basic(self):
        data = [1, 2, 3, 4]
        result = list(LazyCombinations(data, 2))
        expected = [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4)
        ]
        self.assertEqual(result, expected)

    def test_n_equals_length(self):
        data = [1, 2, 3]
        result = list(LazyCombinations(data, 3))
        self.assertEqual(result, [(1, 2, 3)])

    def test_n_zero(self):
        data = [1, 2, 3]
        result = list(LazyCombinations(data, 0))
        self.assertEqual(result, [()])  # leeres Tupel ist die einzige Kombination

    def test_n_greater_than_length(self):
        data = [1, 2]
        result = list(LazyCombinations(data, 3))
        self.assertEqual(result, [])

    def test_empty_input(self):
        data = []
        result = list(LazyCombinations(data, 0))
        self.assertEqual(result, [()])  # leere Kombination
        result = list(LazyCombinations(data, 1))
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
