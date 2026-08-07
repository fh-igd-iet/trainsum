import unittest
from itertools import product

import numpy as np

from trainsum import TrainSum
from utils import backends, assert_exact


class TestConvolution(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(2, 3), (4, 3), (5, 5)]

    def test_convolve(self):
        for ts, (size1, size2) in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim1 = ts.dimension(size1)
            dim2 = ts.dimension(size2)
            data1 = xp.asarray(np.arange(1, size1 + 1, dtype=float))
            data2 = xp.asarray(np.arange(1, size2 + 1, dtype=float))

            train1 = ts.tensortrain(ts.trainshape(dim1), data1)
            train2 = ts.tensortrain(ts.trainshape(dim2), data2)

            with ts.exact():
                conv = ts.convolve(train1, train2)

            exact = xp.asarray(np.convolve(np.asarray(data1), np.asarray(data2)))
            assert_exact(self, ts, exact, conv.to_tensor())


if __name__ == "__main__":
    unittest.main()
