import unittest
import numpy as np
from connected_k_community import find_connected_k_community

class TestConnectedKCommunity(unittest.TestCase):

    def test_small_connected_graph(self):
        """
        Test a small fully connected graph with k = 2 communities.
        """
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        k = 2
        result = find_connected_k_community(A, k)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), k)

    def test_disconnected_graph(self):
        """
        Test a disconnected graph where k-communities cannot be formed.
        """
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        k = 2
        result = find_connected_k_community(A, k)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), k)

    def test_sparse_graph(self):
        """
        Test a sparse graph where k-communities may not exist.
        """
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        k = 3
        result = find_connected_k_community(A, k)
        self.assertIsInstance(result, str)
        self.assertIn("No connected", result)

    def test_single_community(self):
        """
        Test a fully connected graph where k = 1.
        """
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        k = 1
        result = find_connected_k_community(A, k)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)

    def test_invalid_input_non_square(self):
        """
        Test with a non-square adjacency matrix.
        """
        A = np.array([
            [0, 1, 0],
            [1, 0, 1]
        ])
        k = 2
        with self.assertRaises(AssertionError):
            find_connected_k_community(A, k)

    def test_invalid_input_non_symmetric(self):
        """
        Test with a non-symmetric adjacency matrix.
        """
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        k = 2
        with self.assertRaises(AssertionError):
            find_connected_k_community(A, k)

    def test_large_graph(self):
        """
        Test a large random graph.
        """
        np.random.seed(42)
        n = 10
        A = np.random.randint(0, 2, (n, n))
        A = np.triu(A, 1)  # Ensure no self-loops and upper triangular
        A += A.T  # Make symmetric
        k = 3
        result = find_connected_k_community(A, k)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), k)

if __name__ == "__main__":
    unittest.main()
