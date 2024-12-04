import unittest
import numpy as np
from max_community import find_max_community 

class TestFindMaxCommunity(unittest.TestCase):

    def test_fully_connected_graph(self):
        """
        Test a small fully connected graph.
        """
        A = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ])
        result = find_max_community(A)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["size"], 2)

    def test_sparse_graph(self):
        """
        Test a sparse graph.
        """
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        result = find_max_community(A)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["size"], 2)

    def test_disconnected_graph(self):
        """
        Test a disconnected graph.
        """
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        result = find_max_community(A)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["size"], 2)

    def test_small_graph_with_no_solution(self):
        """
        Test a graph with fewer than 3 vertices.
        """
        A = np.array([
            [0, 1],
            [1, 0]
        ])
        result = find_max_community(A)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Graph has less than 3 vertices.")

    def test_invalid_input_non_square(self):
        """
        Test with a non-square adjacency matrix.
        """
        A = np.array([
            [0, 1, 0],
            [1, 0, 1]
        ])
        with self.assertRaises(AssertionError):
            find_max_community(A)

    def test_invalid_input_non_symmetric(self):
        """
        Test with a non-symmetric adjacency matrix.
        """
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        with self.assertRaises(AssertionError):
            find_max_community(A)

    def test_large_graph(self):
        """
        Test a larger graph for scalability.
        """
        np.random.seed(42)
        n = 10
        A = np.random.randint(0, 2, (n, n))
        A = np.triu(A, 1)  # Ensure no self-loops and upper triangular
        A += A.T  # Make symmetric
        result = find_max_community(A)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["size"], 2)

if __name__ == "__main__":
    unittest.main()
