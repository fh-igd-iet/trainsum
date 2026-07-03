import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, assert_exact, exact_slice_operator


class TestSlice(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [64, 120]

    def test_slice_vector(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)
            slices = [slice(None), slice(size - 1, size, 1)]

            for slc in slices:
                train = ts.slice_vector(dim, slc)
                exact = xp.zeros((size,))
                exact[slc] = 1.0
                assert_exact(self, ts, exact, train.to_tensor())

    def test_slice_vector_with_offset_and_stride(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)
            slc = slice(2, size, 3)

            train = ts.slice_vector(dim, slc)
            exact = xp.zeros((size,))
            exact[slc] = 1.0
            assert_exact(self, ts, exact, train.to_tensor())

    def test_slice_operator(self):
        for ts, size in product(self.trainsum, self.sizes):
            dim = ts.dimension(size)
            slices = [slice(None), slice(2, size, 3)]

            for slc in slices:
                train = ts.slice_operator(dim, slc)
                exact = exact_slice_operator(ts.namespace, size, slc)
                assert_exact(self, ts, exact, train.to_tensor())

    @unittest.expectedFailure
    def test_slice_operator_singleton_slice(self):
        for ts, size in product(self.trainsum, self.sizes):
            dim = ts.dimension(size)
            slc = slice(size - 1, size, 1)

            train = ts.slice_operator(dim, slc)
            exact = exact_slice_operator(ts.namespace, size, slc)
            assert_exact(self, ts, exact, train.to_tensor())

if __name__ == "__main__":
    unittest.main()
