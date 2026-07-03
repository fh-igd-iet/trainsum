import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, exact_toeplitz, assert_exact


class TestToeplitz(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [2, 8, 12]
        self.modes = ("full", "lower", "upper", "circular")

    def test_toeplitz(self):
        for ts, size, mode in product(self.trainsum, self.sizes, self.modes):
            dim = ts.dimension(size)

            train = ts.toeplitz(dim, mode)
            exact = exact_toeplitz(ts.namespace, size, mode)
            assert_exact(self, ts, exact, train.to_tensor())


if __name__ == "__main__":
    unittest.main()
