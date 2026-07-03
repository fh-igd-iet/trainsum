import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, shift_matrix, assert_exact


class TestShift(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [20, 64, 18]

    def test_shift(self):
        for ts, size in product(self.trainsum, self.sizes):
            shifts = list(range(-size + 1, size - 1))
            circular = [False, True]
            dim = ts.dimension(size)

            for shift, circ in product(shifts, circular):
                exact = shift_matrix(ts.namespace, dim.size(), dim.size(), shift)
                #if circ:
                #    tmp = -size+shift if shift >= 0 else size+shift
                #    exact += self.shift_matrix(xp, dim.size(), tmp)

                train = ts.shift(dim, shift)
                assert_exact(self, ts, exact, train.to_tensor())

    def test_shift_overloads(self):
        for ts in self.trainsum:
            rows = ts.dimension(18)
            cols = ts.dimension(20)
            for shift in (2, -3):
                exact = shift_matrix(ts.namespace, rows.size(), cols.size(), shift)

                train = ts.shift((rows, cols), shift)
                assert_exact(self, ts, exact, train.to_tensor())

                shape = ts.trainshape(rows, cols, mode="interleaved_rear")
                train = ts.shift(shape, shift)
                assert_exact(self, ts, exact, train.to_tensor())


if __name__ == "__main__":
    unittest.main()
